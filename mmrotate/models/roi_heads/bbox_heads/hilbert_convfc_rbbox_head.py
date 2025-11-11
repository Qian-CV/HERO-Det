# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
from mmdet.models.losses import accuracy
from mmdet.structures.bbox import get_box_tensor
from torch import Tensor

from mmrotate.models.roi_heads.bbox_heads import RotatedShared2FCBBoxHead
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead, ConvFCBBoxHead
from mmrotate.registry import MODELS

from projects.HERO.hero.HPFormer import FastHilbertTransform
from projects.HERO.hero.hilbert_cyclic_shift import CyclicShiftConv
from projects.HERO.hero.cyclic_shift_direct import CyclicShiftConvDirect

from projects.HERO.hero.ResidualORN import ResidualORN
from ...utils import ORConv2d, RotationInvariantPooling
import torch.nn.functional as F


@MODELS.register_module()
class HilbertRotatedShared2FCBBoxHead(Shared2FCBBoxHead):
    """Rotated Shared2FC RBBox head.

    Args:
        loss_bbox_type (str): Set the input type of ``loss_bbox``.
            Defaults to 'normal'.
    """

    def __init__(self,
                 loss_bbox_type: str = 'normal',
                 use_cyclic_shift: bool = False,
                 use_HCS_Direct: bool = True,
                 ORN_type: str = '0',
                 Orientation_cfg: tuple = (4, 4),
                 rotation_angles=(0, 90, 180, 270),
                 H_roi_scale_factor: float = 1.0,
                 *args,
                 **kwargs) -> None:
        super().__init__(
            *args,
            **kwargs)
        self.loss_bbox_type = loss_bbox_type
        self.use_cyclic_shift = use_cyclic_shift
        self.use_HCS_Direct = use_HCS_Direct
        self.rotation_angles = rotation_angles
        self.H_roi_scale_factor = H_roi_scale_factor
        
        self.hor_size = int(8 * self.H_roi_scale_factor)
        
        if self.use_cyclic_shift:
            if self.use_HCS_Direct:
                self.cyclic_shift_fuse = CyclicShiftConvDirect(8, 8, 256, rotation_angles=self.rotation_angles)
            else:
                self.cyclic_shift_fuse = CyclicShiftConv(8, 8, 256, rotation_angles=self.rotation_angles)

        self.ORN_type = ORN_type
        self.nOrientation, self.nRotation = Orientation_cfg

        self.HC_unfolding = FastHilbertTransform(8, 8)

        if self.ORN_type == '2':
            self.or_conv = ResidualORN(
                channels=self.in_channels,
                kernel_size=3,
                padding=1,
                nOrientation=self.nOrientation,
                nRotation=self.nRotation,
                alpha=0.8,
                bias=True)
        elif self.ORN_type == '1':
            self.or_conv = ORConv2d(
                int(self.in_channels / self.nOrientation),
                int(self.in_channels / self.nRotation),
                kernel_size=3,
                padding=1,
                arf_config=(self.nOrientation, self.nRotation),
                groups=1,
                bias=True,
            )
        if self.ORN_type != '0':
            _, self.reg_2fc, _ = self._add_conv_fc_branch(
                num_branch_convs=0,
                num_branch_fcs=self.num_shared_fcs,
                in_channels=self.in_channels,
                is_shared=True)

    def forward(self, x: Tuple[Tensor]) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): A tuple that may contain two feature tensors:
                - rot_feats: rotated ROI features with shape [k, C, 8, 8]
                - hor_feats: horizontal ROI features with shape [k, C, hor_size, hor_size]

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        # Unpack input features
        if len(x) == 2:
            rot_feats, hor_feats = x
        else:
            rot_feats = x

        # Process rotated features (8×8) separately
        # Apply Hilbert flatten to rotated features
        rot_x = self.HC_unfolding.flatten(rot_feats)  # B, C, 64
        if self.use_cyclic_shift:
            # Cyclic shift for the rotated branch
            rot_x = self.cyclic_shift_fuse(rot_x).flatten(1)  # B, C*64
        else:
            rot_x = rot_x.flatten(1)  # B, C*64

            # Classification branch uses only rotated features
        for fc in self.shared_fcs:
            rot_x = self.relu(fc(rot_x))
        x_cls = rot_x

        # Process horizontal features (hor_size×hor_size) separately
        if self.ORN_type != '0':
            or_feat = self.or_conv(hor_feats)  # ORN processing for horizontal features
            # Resize horizontal ORN features to 8×8 to fit existing FC layers
            or_feat_resized = F.adaptive_avg_pool2d(or_feat, (8, 8))
            x_reg = self.HC_unfolding.flatten(or_feat_resized).flatten(1)
            # Regression branch processing
            for fc in self.reg_2fc:
                x_reg = self.relu(fc(x_reg))
        else:
            x_reg = x_cls  # Use rotated features for regression

        # Output layers
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred

    def loss(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the network predictions and targets.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            labels (Tensor): Gt_labels for all proposals in a batch, has
                shape (batch_size * num_proposals_single_image, ).
            label_weights (Tensor): Labels_weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, ).
            bbox_targets (Tensor): Regression target for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4).
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss.
        """

        losses = dict()

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox and (self.loss_bbox_type != 'kfiou'):
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    bbox_pred = get_box_tensor(bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), self.num_classes,
                        -1)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]
                if self.loss_bbox_type == 'normal':
                    losses['loss_bbox'] = self.loss_bbox(
                        pos_bbox_pred,
                        bbox_targets[pos_inds.type(torch.bool)],
                        bbox_weights[pos_inds.type(torch.bool)],
                        avg_factor=bbox_targets.size(0),
                        reduction_override=reduction_override)
                elif self.loss_bbox_type == 'kfiou':
                    # When the regression loss (e.g. `KFLoss`)
                    # is applied on both the delta and decoded boxes.
                    bbox_pred_decode = self.bbox_coder.decode(
                        rois[:, 1:], bbox_pred)
                    bbox_pred_decode = get_box_tensor(bbox_pred_decode)
                    bbox_targets_decode = self.bbox_coder.decode(
                        rois[:, 1:], bbox_targets)
                    bbox_targets_decode = get_box_tensor(bbox_targets_decode)

                    if self.reg_class_agnostic:
                        pos_bbox_pred_decode = bbox_pred_decode.view(
                            bbox_pred_decode.size(0),
                            5)[pos_inds.type(torch.bool)]
                    else:
                        pos_bbox_pred_decode = bbox_pred_decode.view(
                            bbox_pred_decode.size(0), -1,
                            5)[pos_inds.type(torch.bool),
                               labels[pos_inds.type(torch.bool)]]

                    losses['loss_bbox'] = self.loss_bbox(
                        pos_bbox_pred,
                        bbox_targets[pos_inds.type(torch.bool)],
                        bbox_weights[pos_inds.type(torch.bool)],
                        pred_decode=pos_bbox_pred_decode,
                        targets_decode=bbox_targets_decode[pos_inds.type(
                            torch.bool)],
                        avg_factor=bbox_targets.size(0),
                        reduction_override=reduction_override)
                else:
                    raise NotImplementedError
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses
