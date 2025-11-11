# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv import ops
from mmdet.models.roi_heads.roi_extractors.base_roi_extractor import \
    BaseRoIExtractor
from mmengine.utils import to_2tuple

from mmrotate.registry import MODELS
from mmrotate.models.roi_heads.roi_extractors import RotatedSingleRoIExtractor
import torch.nn.functional as F
from mmcv.ops import roi_align


@MODELS.register_module()
class RotatedSingleHandRroiExtractor(RotatedSingleRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (List[int]): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
        H_roi_scale_factor (float): Scale factor for horizontal ROI size. Default: 1.0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56,
                 H_roi_scale_factor=1.0,
                 init_cfg=None):
        super(RotatedSingleHandRroiExtractor,
              self).__init__(roi_layer, out_channels, featmap_strides,
                             init_cfg)
        self.finest_scale = finest_scale
        self.fp16_enabled = False
        self.H_roi_scale_factor = H_roi_scale_factor
        
        # Calculate ROI size
        self.hor_size = int(8 * self.H_roi_scale_factor)  # default 8×8

    def build_roi_layers(self, layer_cfg, featmap_strides):
        """Build RoI operator to extract feature from each level feature map.

        Args:
            layer_cfg (dict): Dictionary to construct and config RoI layer
                operation. Options are modules under ``mmcv/ops`` such as
                ``RoIAlign``.
            featmap_strides (List[int]): The stride of input feature map w.r.t
                to the original image size, which would be used to scale RoI
                coordinate (original image coordinate system) to feature
                coordinate system.

        Returns:
            nn.ModuleList: The RoI extractor modules for each level feature \
                map.
        """

        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')

        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (torch.Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        # scale = torch.sqrt(
        #     (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        scale = torch.sqrt(rois[:, 3] * rois[:, 4])
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def obb2xyxy(self, rbboxes):
        """
        Args:
            rbboxes: Tensor[K,6] = [batch_idx, cx, cy, w, h, theta]
        Returns:
            Tensor[K,5] = [batch_idx, x1, y1, x2, y2]
        """
        batch_idx = rbboxes[:, :1]
        cx, cy, w, h, a = (rbboxes[:, i] for i in (1, 2, 3, 4, 5))
        #
        cosa = torch.cos(a).abs()
        sina = torch.sin(a).abs()
        # new half-width/half-height in axis-aligned frame
        hw = 0.5 * (cosa * w + sina * h)
        hh = 0.5 * (sina * w + cosa * h)
        # Calculate x1,y1,x2,y2
        x1 = cx - hw
        y1 = cy - hh
        x2 = cx + hw
        y2 = cy + hh
        return torch.cat([batch_idx, x1.unsqueeze(1), y1.unsqueeze(1),
                          x2.unsqueeze(1), y2.unsqueeze(1)], dim=1)

    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function. Outputs both rotated RoI features (8×8) and horizontal bounding RoI features (hor_size×hor_size):

        Args:
            feats (torch.Tensor): Input features.
            rois (torch.Tensor): Input RoIs, shape (k, 5).
            scale_factor (float): Scale factor that RoI will be multiplied by.

        Returns:
            tuple(torch.Tensor, torch.Tensor): 
                - rot_feats: rotated RoI features with shape [k, out_channels, 8, 8]
                - hor_feats: horizontal RoI features with shape [k, out_channels, hor_size, hor_size]
        """
        # 1) Extract rotated RoI features
        rot_feats = super().forward(feats, rois, roi_scale_factor)  # [K, C, 8, 8]

        # 2) Convert rotated RoIs to their enclosing horizontal RoIs
        hor_rois = self.obb2xyxy(rois)  # [K,5]

        # 3) Assign to feature levels, consistent with map_roi_levels
        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)

        # Prepare tensor for horizontal features (hor_size×hor_size)
        K, C, _, _ = rot_feats.shape
        hor_feats = rot_feats.new_zeros(K, C, self.hor_size, self.hor_size)

        # 4) Perform horizontal roi_align on each level with (hor_size×hor_size) output size
        for lvl in range(num_levels):
            inds = (target_lvls == lvl).nonzero(as_tuple=False).squeeze(1)
            if inds.numel() == 0:
                continue
            rois_lvl = hor_rois[inds]
            feat_lvl = feats[lvl]
            spatial_scale = 1.0 / self.featmap_strides[lvl]
            aligned = roi_align(
                feat_lvl, rois_lvl,
                (self.hor_size, self.hor_size),  # Use the configured horizontal ROI size
                spatial_scale,
                0
            )
            hor_feats[inds] = aligned

        # 5) Return rotated and horizontal features separately (no channel concatenation)
        return rot_feats, hor_feats

    def roi_rescale(self, rois, scale_factor):
        """Scale RoI coordinates by scale factor.

        Args:
            rois (torch.Tensor): RoI (Region of Interest), shape (n, 6)
            scale_factor (float): Scale factor that RoI will be multiplied by.

        Returns:
            torch.Tensor: Scaled RoI.
        """
        if scale_factor is None:
            return rois
        h_scale_factor, w_scale_factor = to_2tuple(scale_factor)
        new_rois = rois.clone()
        new_rois[:, 3] = w_scale_factor * new_rois[:, 3]
        new_rois[:, 4] = h_scale_factor * new_rois[:, 4]
        return new_rois
