import torch
import torch.nn.functional as F
from torch import nn
from mmrotate.models.utils.orconv import ORConv2d


class ResidualORN(nn.Module):
    """
    Decompose ORConv2d(arf_config=(nOrientation, nRotation)) into:
      1. Learn filters for nOrientation base directions;
      2. For each base direction, perform nRotation rotated convolution samples;
      3. Average outputs of all rotated convolutions as the increment d_mean;
      4. Residual fusion: out = x + alpha * d_mean;
      5. Apply BN + ReLU at the end.

    Args:
        channels (int): Input/output channels (kept equal for residual addition)
        kernel_size (int): Convolution kernel size
        padding (int): Convolution padding
        nOrientation (int): Number of base orientations
        nRotation (int): Number of rotations per base orientation
        alpha (float): Residual fusion strength coefficient
        bias (bool): Whether to use bias
    """

    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 nOrientation: int = 8,
                 nRotation: int = 2,
                 alpha: float = 1.0,
                 bias: bool = True):
        super().__init__()
        # Store parameters
        self.C = channels
        self.nOri = nOrientation
        self.nRot = nRotation
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)

        # Use ORConv2d to generate nOri × nRot filters
        self.base_or = ORConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            arf_config=(nOrientation, nRotation),
            bias=bias
        )
        # Record conv parameters for F.conv2d
        self.stride = self.base_or.stride
        self.padding = self.base_or.padding
        self.dilation = self.base_or.dilation
        self.groups = self.base_or.groups

        # BN after fusion
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 1) Get convolution kernels of all orientations and rotations
        weight_all = self.base_or.rotate_arf()
        bias_all = self.base_or.bias  # [C * nRot] or None

        # 2) For each orientation, perform nRot convolutions and average
        d_sum = 0
        for ori in range(self.nOri):
            # Each orientation contains nRot × C kernels
            start = ori * C
            end = (ori + 1) * C
            w_slice = weight_all[:, start:end, ...]  # [nRot*C, C, k, k]

            for i in range(self.nRot):
                w_i = w_slice[i*C:(i+1)*C, ...]
                b_i = (bias_all[i*C:(i+1)*C] if bias_all is not None else None)
                out = F.conv2d(
                    x, w_i, b_i,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups
                )
                d_sum += out

        d_mean = d_sum / float(self.nOri*self.nRot)

        # 3) Residual fusion + BN
        out = x + self.alpha * d_mean
        out = self.bn(out)
        
        return out
