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


class CyclicShiftConv(nn.Module):
    """
    Apply Hilbert flatten + simulate rotation at specified angles + 1D Conv fusion.
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
        # self.fuse = DynamicRotationFusion(in_channels, height * width)
        # self.fuse = RotationAttentionFusion(in_channels, height * width, self.num_rotations)
        # self.fuse = DirectionalChannelAttention(in_channels, height * width, self.num_rotations)
        self.fuse = RotationChannelAttention(in_channels, height * width, self.num_rotations)

        # Pre-compute index buffer for specified angles
        L = height * width
        N = int(math.sqrt(L))
        coords = make_coords(N)
        # Cache coordinates and mapping indices
        idx_map = {coord: i for i, coord in enumerate(coords)}
        all_inv = []
        for angle in self.rotation_angles:  # Use the provided angle set
            inv = []
            for (r, c) in coords:
                if angle == 0:
                    r0, c0 = r, c
                elif angle == 90:
                    r0, c0 = c, N - 1 - r
                elif angle == 180:
                    r0, c0 = N - 1 - r, N - 1 - c
                else:  # 270 degrees
                    r0, c0 = N - 1 - c, r
                inv.append(idx_map[(r0, c0)])
            all_inv.append(inv)
        self.register_buffer('rot_idx', torch.LongTensor(all_inv))  # [R, L], R is the number of angles

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        # 1) For each angle, reorder the sequence
        # x: [B, C, L] → expand to [B, C, R, L] (R is the number of angles)
        x_expand = x.unsqueeze(2).expand(B, C, self.num_rotations, L)  # [B,C,R,L]
        # rot_idx: [R, L] → [1,1,R,L] → [B,C,R,L]
        idx = self.rot_idx.view(1, 1, self.num_rotations, L).expand(B, C, self.num_rotations, L)
        # Gather sequences for the specified angle
        x_rot = torch.gather(x_expand, 3, idx)  # [B, C, R, L]
        # 2) Feature fusion
        x_f = self.fuse(x_rot)  # [B, C, L]
        return x_f


def direct_flatten(img, coords):
    """
    Directly flatten [1,1,H, W] image to [1,1,L] sequence according to coords,
    for comparison with simulation results.
    """
    B, C, H, W = img.shape
    assert B == 1 and C == 1 and len(coords) == H * W
    flat = [img[0, 0, r, c].item() for (r, c) in coords]
    return torch.tensor(flat, device=img.device).view(1, 1, -1)


def main():
    # ---- Configuration ----
    N = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Build Hilbert coords
    coords = make_coords(N)

    # 2) Build test image: 0,1,...,L-1
    L = N * N
    img = torch.arange(L, dtype=torch.float32, device=device).view(1, 1, N, N)
    print("Original image:")
    print(img[0, 0])

    # 3) Reference flatten
    seq_ref = direct_flatten(img, coords)  # [1,1,L]

    # Test different angle sets
    for angles in [(0, 90), (0, 90, 180), (0, 90, 180, 270)]:
        print(f"\nTesting angle set: {angles}")
        # 4) Cyclic shift simulation + gather
        conv = CyclicShiftConv(N, N, 1, rotation_angles=angles).to(device)
        # Note: here we only take rot_idx during forward and directly gather
        # Expand seq_ref to [1, 1*len(angles), L] then run fuse if necessary. Here we only validate rot_idx:
        seq_expand = seq_ref.expand(1, len(angles), L)
        seq_sim = seq_expand.gather(2, conv.rot_idx.unsqueeze(0))  # [1,len(angles),L]

        # 5) Compare each angle
        for i, angle in enumerate(angles):
            img_rot = torch.rot90(img, k=angle // 90, dims=[-2, -1])
            seq_img = direct_flatten(img_rot, coords).view(1, -1)  # [1, L]
            seq_gath = seq_sim[0, i].view(1, -1)  # [1, L]
            ok = torch.equal(seq_img, seq_gath)
            print(f"Rotation {angle}° → Match: {ok}")
            if not ok:
                print("  Expected first 16:", seq_img[0, :16].tolist())
                print("  Simulated first 16:", seq_gath[0, :16].tolist())
                break


if __name__ == "__main__":
    main()
