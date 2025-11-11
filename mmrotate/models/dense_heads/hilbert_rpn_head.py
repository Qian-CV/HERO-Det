# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms
from mmdet.models.dense_heads import RPNHead
from mmdet.models.utils import multi_apply
from mmdet.structures.bbox import (BaseBoxes, get_box_tensor, get_box_wh,
                                   scale_boxes)
from mmdet.utils import MultiConfig
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import MODELS
from mmrotate.structures.bbox import rbox2hbox
from projects.HERO.hero.hilbert_cross_attention import HilbertCrossScaleAttention
from projects.HERO.hero.HPFormer import FastHilbertTransform, FastRowMajorTransform, FastSnakeTransform, \
    FastMortonTransform, FastPeanoTransform


# from projects.HERO.hero.hilbert_cyclic_shift import CyclicShiftConv


@MODELS.register_module()
class HilbertRPNHead(RPNHead):
    """Oriented RPN head for Oriented R-CNN."""

    def __init__(self,
                 in_channels: int,
                 num_classes: int = 1,
                 init_cfg: MultiConfig = dict(
                     type='Normal', layer='Conv2d', std=0.01),
                 num_convs: int = 1,
                 attn_cfg: Dict = None,
                 use_hpformer_unflatten: bool = False,  # hilbert flatten
                 **kwargs) -> None:
        #
        rpn_kwargs = {k: v for k, v in kwargs.items() 
                     if k not in ['attn_cfg', 'use_hpformer_unflatten']}
        
        #
        super().__init__(in_channels, num_classes, init_cfg, num_convs, **rpn_kwargs)
        
        #
        default_attn_cfg = dict(
            use_cross_attention=False,
            use_pos_embed=True,
            bidirectional=True,
            use_enhanced_attn=True)
        
        #
        if attn_cfg is not None:
            default_attn_cfg.update(attn_cfg)
        
        #
        self.use_cross_attention = default_attn_cfg['use_cross_attention']
        
        if self.use_cross_attention:
            self.hilbert_cross_attention = HilbertCrossScaleAttention(
                in_channels,
                use_pos_embed=default_attn_cfg['use_pos_embed'],
                bidirectional=default_attn_cfg['bidirectional'],
                use_enhanced_attn=default_attn_cfg['use_enhanced_attn']
            )
            
        #
        self.use_hpformer_unflatten = use_hpformer_unflatten

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        if self.num_convs > 1:
            rpn_convs = []
            for i in range(self.num_convs):
                if i == 0:
                    in_channels = self.in_channels
                else:
                    in_channels = self.feat_channels
                # use ``inplace=False`` to avoid error: one of the variables
                # needed for gradient computation has been modified by an
                # inplace operation.
                rpn_convs.append(
                    ConvModule(
                        in_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        inplace=False))
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            # self.rpn_conv = nn.Conv2d(
            #     self.in_channels, self.feat_channels, 3, padding=1)
            # todo:  init hilbert conv
            self.hilbert_conv = nn.Sequential(
                # nn.Conv1d(self.in_channels, self.feat_channels, kernel_size=3, padding=1),
                # nn.Conv1d(self.in_channels, self.feat_channels, kernel_size=5, padding=2),
                # nn.Conv1d(self.in_channels, self.feat_channels, kernel_size=7, padding=3),
                nn.Conv1d(self.in_channels, self.feat_channels, kernel_size=9, padding=4),
                # nn.Conv1d(self.in_channels, self.feat_channels, kernel_size=3, padding=2, dilation=2), # 5
                # nn.Conv1d(self.in_channels, self.feat_channels, kernel_size=3, padding=3, dilation=3), # 7
                nn.ReLU(inplace=True)
            )
        # todo:
        self.HPFormer = nn.ModuleList([
            FastHilbertTransform(256 // (2 ** i), 256 // (2 ** i)) for i in range(5)
        ])
        self.hilbert_convs = nn.ModuleList([
            self.hilbert_conv for _ in range(len(self.anchor_generator.strides))
        ])
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * self.cls_out_channels,
                                 1)
        reg_dim = self.bbox_coder.encode_size
        self.rpn_reg = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * reg_dim, 1)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_base_priors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_base_priors * 4.
        """
        #
        if not hasattr(self, '_layer_idx') or self._layer_idx >= len(self.HPFormer):
            self._layer_idx = 0
        # 1)
        B, C, L = x.shape
        H = int(math.sqrt(L))
        W = H
        if self.use_hpformer_unflatten:  # todo: hilbert填数
            x_rec = self.HPFormer[self._layer_idx].unflatten(x)
            self._layer_idx += 1
        else:
            x_rec = x.view(B, C, H, W)

        # 2) RPN cls/reg
        rpn_cls_score = self.rpn_cls(x_rec)
        rpn_bbox_pred = self.rpn_reg(x_rec)
        return rpn_cls_score, rpn_bbox_pred

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        hilbert_seq_list = []
        for feat_map, hpformer, conv1d in zip(x,
                                              self.HPFormer,
                                              self.hilbert_convs):
            x_hilbert = hpformer.flatten(feat_map)

            x_hilbert = conv1d(x_hilbert)
            hilbert_seq_list.append(x_hilbert)
        if self.use_cross_attention:
            hilbert_seq_list = self.hilbert_cross_attention(hilbert_seq_list)
        self._layer_idx = 0
        return multi_apply(self.forward_single, hilbert_seq_list)

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method, which use horizontal bboxes for NMS,
        but return the rotated bboxes result.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert with_nms, '`with_nms` must be True in RPNHead'
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        if results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            hbboxes = rbox2hbox(bboxes)
            det_bboxes, keep_idxs = batched_nms(hbboxes, results.scores,
                                                results.level_ids, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]
            # TODO: This would unreasonably show the 0th class label
            #  in visualization
            results.labels = results.scores.new_zeros(
                len(results), dtype=torch.long)
            del results.level_ids
        else:
            # To avoid some potential error
            results_ = InstanceData()
            if isinstance(results.bboxes, BaseBoxes):
                results_.bboxes = results.bboxes.empty_boxes()
            else:
                results_.bboxes = results.scores.new_zeros(0, 4)
            results_.scores = results.scores.new_zeros(0)
            results_.labels = results.scores.new_zeros(0)
            results = results_
        return results
