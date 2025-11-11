import math
import torch
import torch.nn as nn
from hilbertcurve.hilbertcurve import HilbertCurve
from projects.HERO.hero.cyclic_shift_methods import RotationChannelAttention


def make_coords(N):
    """Return a list of (row, col) with length N*N in Hilbert curve order."""
    p = HilbertCurve(int(math.log2(N)), 2)
    coords = []
    for d in range(N * N):
        x, y = p.point_from_distance(d)
        # point_from_distance returns (x, y); we need (row=y, col=x)
        coords.append((y, x))
    return coords  # list of length N*N


class CyclicShiftConvDirect(nn.Module):
    """
    Apply Hilbert flatten + direct sequence shift to simulate rotation + 1D Conv fusion.
    Optional angle sets: (0, 90); (0, 90, 180); (0, 90, 180, 270)

    Input: x [B, C, H*W]
    Output: x_f [B, C, H*W]
    """

    def __init__(self, height: int, width: int, in_channels: int, rotation_angles=(0, 90, 180, 270)):
        super().__init__()
        # Store angle set
        self.rotation_angles = rotation_angles
        self.num_rotations = len(rotation_angles)

        # Convolution: merge channels C×R → C (R is the number of rotations)
        self.fuse = RotationChannelAttention(in_channels, height * width, self.num_rotations)

        # Compute Hilbert curve length
        L = height * width
        N = int(math.sqrt(L))

        # Compute shift for each rotation angle
        shifts = []
        for angle in self.rotation_angles:
            if angle == 0:
                shift = 0
            elif angle == 90:
                shift = L // 4  # 90 degrees equals shifting right by L/4
            elif angle == 180:
                shift = L // 2  # 180 degrees equals shifting right by L/2
            else:  # 270 degrees
                shift = 3 * L // 4  # 270 degrees equals shifting right by 3L/4
            shifts.append(shift)

        self.register_buffer('shifts', torch.tensor(shifts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape

        # Create tensor to store shifted sequences
        x_rot = torch.zeros(B, C, self.num_rotations, L, device=x.device)

        # Apply shift for each angle
        for i, shift in enumerate(self.shifts):
            if shift == 0:
                x_rot[:, :, i, :] = x
            else:
                # Use torch.roll for circular shift
                x_rot[:, :, i, :] = torch.roll(x, shifts=int(shift), dims=2)

        # Feature fusion
        x_f = self.fuse(x_rot)  # [B, C, L]
        return x_f


def direct_flatten(img, coords):
    """
    Directly flatten [1,1,H,W] image to [1,1,L] sequence according to coords,
    for comparison with simulation results.
    """
    B, C, H, W = img.shape
    assert B == 1 and C == 1 and len(coords) == H * W
    flat = [img[0, 0, r, c].item() for (r, c) in coords]
    return torch.tensor(flat, device=img.device).view(1, 1, -1)