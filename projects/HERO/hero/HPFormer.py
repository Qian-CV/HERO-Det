import math
import torch
import torch.nn as nn
from hilbertcurve.hilbertcurve import HilbertCurve


def make_coords(N):
    """Return a list of (row, col) with length N*N in Hilbert curve order."""
    p = HilbertCurve(int(math.log2(N)), 2)
    coords = []
    for d in range(N * N):
        x, y = p.point_from_distance(d)
        # point_from_distance returns (x, y); we need (row=y, col=x)
        coords.append((y, x))
    return coords  # list of length N*N


class FastHilbertTransform(nn.Module):
    """
    Efficient Hilbert flatten and reconstruction operator.

    Precompute and cache the Hilbert curve order indices and their inverse.
    Provides two methods: flatten and unflatten:
      - flatten(x): [B, C, H, W] -> [B, C, L]
      - unflatten(x): [B, C, L]  -> [B, C, H, W]
    """

    def __init__(self, height: int, width: int):
        super().__init__()
        L = height * width
        N = int(math.sqrt(L))
        assert N * N == L, 'height*width must be a perfect square'
        # 1) Compute Hilbert order indices
        coords = make_coords(N)
        idx_map = {coord: i for i, coord in enumerate(coords)}
        hilbert_idx = [idx_map[c] for c in coords]  # length L
        self.register_buffer('hilbert_idx', torch.LongTensor(hilbert_idx))

        # 2) Compute inverse mapping: linear index -> sequence position
        inv_idx = [0] * L
        for seq_pos, lin in enumerate(hilbert_idx):
            inv_idx[lin] = seq_pos
        self.register_buffer('inverse_idx', torch.LongTensor(inv_idx))  # [L]
        # Save original sizes
        self.height = height
        self.width = width

    def flatten(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flatten feature map in Hilbert order.
        Args:
            x: [B, C, H, W]
        Returns:
            x_h: [B, C, L]
        """
        B, C, H, W = x.shape
        assert H == self.height and W == self.width
        # First, flatten to linear order [B, C, H*W]
        x_flat = x.reshape(B, C, -1)
        # Ensure hilbert_idx is on the correct device
        hilbert_idx = self.hilbert_idx.to(x.device)
        # Reorder by cached hilbert_idx to [B, C, L]
        x_h = x_flat.index_select(dim=2, index=hilbert_idx)
        return x_h

    def unflatten(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct feature map from Hilbert-flattened sequence.
        Args:
            x: [B, C, L]
        Returns:
            x_rec: [B, C, H, W]
        """
        B, C, L = x.shape
        assert L == self.height * self.width
        # Ensure inverse_idx is on the correct device
        inverse_idx = self.inverse_idx.to(x.device)
        # x: [B, C, L], reorder by inverse_idx to linear order [B, C, H*W]
        x_lin = x.index_select(dim=2, index=inverse_idx)
        # Reshape back to [B, C, H, W]
        x_rec = x_lin.reshape(B, C, self.height, self.width)
        return x_rec


class FastRowMajorTransform(nn.Module):
    """Flatten and reconstruct in standard row-major order."""

    def __init__(self, height: int, width: int):
        super().__init__()
        L = height * width
        # Continuous linear order is 0,1,2...L-1
        self.register_buffer('idx', torch.arange(L, dtype=torch.long))
        self.register_buffer('inv_idx', torch.arange(L, dtype=torch.long))
        self.height = height
        self.width = width

    def flatten(self, x):
        B, C, H, W = x.shape
        assert H == self.height and W == self.width
        x_flat = x.reshape(B, C, -1)
        return x_flat.index_select(2, self.idx)

    def unflatten(self, x):
        B, C, L = x.shape
        assert L == self.height * self.width
        x_lin = x.index_select(2, self.inv_idx)
        return x_lin.view(B, C, self.height, self.width)


class FastSnakeTransform(nn.Module):
    """Flatten and reconstruct in snake (serpentine) row order with alternating directions per row."""

    def __init__(self, height: int, width: int):
        super().__init__()
        L = height * width
        coords = []
        for r in range(height):
            if r % 2 == 0:
                cols = range(width)
            else:
                cols = range(width - 1, -1, -1)
            for c in cols:
                coords.append(r * width + c)
        idx = torch.LongTensor(coords)
        # Inverse index
        inv = torch.zeros(L, dtype=torch.long)
        for seq, i in enumerate(coords):
            inv[i] = seq
        self.register_buffer('idx', idx)
        self.register_buffer('inv_idx', inv)
        self.height = height
        self.width = width

    def flatten(self, x):
        B, C, H, W = x.shape
        assert H == self.height and W == self.width
        x_flat = x.reshape(B, C, -1)
        return x_flat.index_select(2, self.idx)

    def unflatten(self, x):
        B, C, L = x.shape
        assert L == self.height * self.width
        x_lin = x.index_select(2, self.inv_idx)
        return x_lin.view(B, C, self.height, self.width)


# Morton / Z-order flatten
class FastMortonTransform(nn.Module):
    """Flatten and reconstruct by Morton / Z-order space-filling curve (supports 2^k x 2^k sizes)."""

    def __init__(self, height: int, width: int):
        super().__init__()
        assert height == width and ((height & (height - 1)) == 0), 'Morton supports only sizes of 2^k'
        self.N = height
        L = height * width
        # Compute Morton indices
        idx = []
        for y in range(height):
            for x in range(width):
                code = 0
                for b in range(int(math.log2(self.N))):
                    code |= ((y >> b) & 1) << (2 * b + 1)
                    code |= ((x >> b) & 1) << (2 * b)
                idx.append(code)
        # Morton sequence idx[pos] -> Hilbert sequence position
        inv = [0] * L
        for seq, mort in enumerate(idx):
            inv[mort] = seq
        self.register_buffer('idx', torch.LongTensor(idx))
        self.register_buffer('inv_idx', torch.LongTensor(inv))
        self.height = height;
        self.width = width

    def flatten(self, x):
        B, C, H, W = x.shape
        assert H == self.height and W == self.width
        x_flat = x.reshape(B, C, -1)
        return x_flat.index_select(2, self.idx)

    def unflatten(self, x):
        B, C, L = x.shape
        assert L == self.height * self.width
        x_lin = x.index_select(2, self.inv_idx)
        return x_lin.view(B, C, self.height, self.width)


# Peano flatten supporting arbitrary sizes (pad to nearest 3^k)
class FastPeanoTransform(nn.Module):
    """Flatten and reconstruct by Peano curve; arbitrary square sizes are supported via padding to 3^k."""

    def __init__(self, height: int, width: int):
        super().__init__()
        assert height == width, 'Peano supports only square shapes'
        N = height
        k = 0
        while 3 ** k < N:
            k += 1
        pad = 3 ** k
        L_full = pad * pad

        # Peano coordinate computation
        def compute_coords(level):
            if level == 0:
                return [(0, 0)]
            sub = compute_coords(level - 1)
            size = 3 ** (level - 1)
            blocks = [
                (0, 0, 0), (0, 1, 0), (0, 2, 0),
                (1, 2, 1), (1, 1, 1), (1, 0, 1),
                (2, 0, 0), (2, 1, 0), (2, 2, 0),
            ]
            coords_list = []
            for bx, by, rot in blocks:
                for x, y in sub:
                    if rot:
                        x, y = y, x
                    coords_list.append((bx * size + x, by * size + y))
            return coords_list

        coords_full = compute_coords(k)
        linear_idx = [r * pad + c for (r, c) in coords_full]
        inv_full = [0] * L_full
        for seq, lin in enumerate(linear_idx):
            inv_full[lin] = seq
        self.register_buffer('idx_full', torch.LongTensor(linear_idx))
        self.register_buffer('inv_full', torch.LongTensor(inv_full))
        self.orig = N
        self.pad = pad

    def flatten(self, x):
        B, C, H, W = x.shape
        assert H == self.orig and W == self.orig
        x_pad = torch.nn.functional.pad(x, (0, self.pad - W, 0, self.pad - H))
        seq_full = x_pad.reshape(B, C, -1).index_select(2, self.idx_full)
        return seq_full[..., : self.orig * self.orig]

    def unflatten(self, x):
        B, C, L = x.shape
        full = self.orig * self.orig
        x_full = x.new_zeros((B, C, self.pad * self.pad))
        x_full[..., :full] = x
        x_lin = x_full.index_select(2, self.inv_full)
        x_rec = x_lin.view(B, C, self.pad, self.pad)
        return x_rec[..., : self.orig, : self.orig]
