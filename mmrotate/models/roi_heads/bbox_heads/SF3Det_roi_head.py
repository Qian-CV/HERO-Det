# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
from mmdet.models.losses import accuracy
from mmdet.structures.bbox import get_box_tensor
from torch import Tensor

from mmrotate.models.roi_heads.bbox_heads import RotatedShared2FCBBoxHead
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead, ConvFCBBoxHead
from mmrotate.registry import MODELS

from projects.HERO.hero.HPFormer import FastHilbertTransform, FastRowMajorTransform, FastSnakeTransform, \
    FastMortonTransform, FastPeanoTransform
from projects.HERO.hero.hilbert_cyclic_shift import CyclicShiftConv
from projects.HERO.hero.cyclic_shift_direct import CyclicShiftConvDirect

from projects.HERO.hero.ResidualORN import ResidualORN
from ...utils import ORConv2d, RotationInvariantPooling
import torch.nn.functional as F
import torch.nn as nn


@MODELS.register_module()
class SF3DetBBoxHead(Shared2FCBBoxHead):
    """Rotated Shared2FC RBBox head.

    Args:
        loss_bbox_type (str): Set the input type of ``loss_bbox``.
            Defaults to 'normal'.
    """

    def __init__(self,
                 loss_bbox_type: str = 'normal',
                 use_JCA_loss: bool = False,
                 *args,
                 **kwargs) -> None:
        super().__init__(
            *args,
            **kwargs)
        self.loss_bbox_type = loss_bbox_type
        self.use_JCA_loss = use_JCA_loss

        self.HC_unfolding = FastHilbertTransform(8, 8)
        self.extra_HC_unfolding = FastHilbertTransform(8, 8)
        self.hilbert_conv = nn.Sequential(
            # nn.Conv1d(self.in_channels, self.in_channels, kernel_size=3, padding=1),
            # nn.Conv1d(self.in_channels, self.in_channels, kernel_size=5, padding=2),
            # nn.Conv1d(self.in_channels, self.in_channels, kernel_size=7, padding=3),
            nn.Conv1d(self.in_channels, self.in_channels, kernel_size=9, padding=4),
            # nn.Conv1d(self.in_channels, self.in_channels, kernel_size=11, padding=5),
            # nn.Conv1d(self.in_channels, self.in_channels, kernel_size=15, padding=7),
            # nn.Conv1d(self.in_channels, self.in_channels, kernel_size=3, padding=2, dilation=2),  # 5
            # nn.Conv1d(self.in_channels, self.in_channels, kernel_size=3, padding=3, dilation=3), # 7
            # nn.Conv1d(self.in_channels, self.in_channels, kernel_size=5, padding=4, dilation=2), # 9
            # nn.Conv1d(self.in_channels, self.in_channels, kernel_size=3, padding=4, dilation=4), # 9
            # nn.Conv1d(self.in_channels, self.in_channels, kernel_size=9, padding=8, dilation=2), # 17
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): 包含两个特征张量的元组：
                - rot_feats: 旋转ROI特征，形状为[k, C, 8, 8]
                - hor_feats: 水平ROI特征，形状为[k, C, hor_size, hor_size]

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        # 单独处理旋转特征(8×8)
        # Hilbert flatten旋转特征

        # rot_x = x.flatten(1)

        rot_x = self.HC_unfolding.flatten(x)  # B, C, 64
        # rot_x = self.HC_unfolding.flatten(x)+self.extra_HC_unfolding.flatten(x)  # B, C, 64
        #######################################
        rot_x = self.hilbert_conv(rot_x)
        #######################################
        rot_x = rot_x.flatten(1)  # B, C*64
        for fc in self.shared_fcs:
            rot_x = self.relu(fc(rot_x))
        x_cls = rot_x
        x_reg = rot_x

        # 输出层
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
            if self.use_JCA_loss:
                # 1.直接加
                loss_cls = losses['loss_cls']
                loss_bbox = losses['loss_bbox']
                if self.use_JCA_loss:
                    br = 1 + torch.exp(-loss_bbox)
                    bc = 1 + torch.exp(-loss_cls)
                    loss_cls = br * loss_cls
                    loss_bbox = bc * loss_bbox
                losses['loss_cls'] = loss_cls
                losses['loss_bbox'] = loss_bbox
                # 2.系数不参与.detach()

        return losses
