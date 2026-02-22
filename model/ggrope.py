"""
GGRoPE — Golden-ratio Group Rotary Position Embeddings.

2D positional encoding designed for grid-structured data (ARC grids).
- Even frequency dimensions encode column position
- Odd frequency dimensions encode row position scaled by the golden ratio
- Compatible with Grouped Query Attention (different num Q vs KV heads)

Usage:
    ggrope = GGRoPE(head_dim=128, max_seq_len=2048)
    q_rot, k_rot = ggrope.apply_ggrope(q, k, row_ids, col_ids)
"""

import torch
import torch.nn as nn
from torch import Tensor


class GGRoPE(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        golden_ratio: float = 1.618033988749895,
        base: float = 10000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.golden_ratio = golden_ratio
        self.base = base

        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        half_dim = head_dim // 2

        # Frequency bands: theta_i = 1 / (base^(2i/head_dim))
        # We have half_dim frequency bands, each applied as a rotation pair
        freq_indices = torch.arange(0, half_dim, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (freq_indices / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute masks for even/odd frequency splitting
        # Even indices (0, 2, 4, ...) -> column encoding
        # Odd indices (1, 3, 5, ...) -> row encoding (scaled by golden ratio)
        even_mask = torch.zeros(half_dim, dtype=torch.bool)
        odd_mask = torch.zeros(half_dim, dtype=torch.bool)
        even_mask[0::2] = True  # frequency bands 0, 2, 4, ...
        odd_mask[1::2] = True   # frequency bands 1, 3, 5, ...
        self.register_buffer("even_mask", even_mask, persistent=False)
        self.register_buffer("odd_mask", odd_mask, persistent=False)

    def _compute_freqs(
        self, row_ids: Tensor, col_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Compute cos and sin rotation matrices from 2D grid positions.

        Args:
            row_ids: (batch, seq_len) integer row positions
            col_ids: (batch, seq_len) integer column positions

        Returns:
            cos, sin: (batch, seq_len, 1, half_dim) rotation values
        """
        # Scale row positions by golden ratio for incommensurable encoding
        row_float = row_ids.float() * self.golden_ratio  # (B, T)
        col_float = col_ids.float()                       # (B, T)

        # Compute angles for each frequency band
        # inv_freq: (half_dim,)
        # Position * frequency -> angle
        # col_angles: (B, T, half_dim) — only even frequency indices used
        # row_angles: (B, T, half_dim) — only odd frequency indices used
        col_angles = col_float.unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)
        row_angles = row_float.unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)

        # Combine: even frequencies get column, odd frequencies get row
        angles = torch.where(
            self.even_mask.unsqueeze(0).unsqueeze(0),
            col_angles,
            row_angles,
        )  # (B, T, half_dim)

        cos = angles.cos()  # (B, T, half_dim)
        sin = angles.sin()  # (B, T, half_dim)

        # Add head dimension for broadcasting: (B, T, 1, half_dim)
        return cos.unsqueeze(2), sin.unsqueeze(2)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        """Rotate pairs: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]"""
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def _apply_rotary(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """
        Apply rotary embeddings to a tensor.

        Args:
            x: (..., head_dim) tensor
            cos: (B, T, 1, half_dim)
            sin: (B, T, 1, half_dim)

        Returns:
            Rotated tensor with same shape as x
        """
        # Split into pairs and apply rotation
        half = self.head_dim // 2
        x_pairs = x.unflatten(-1, (half, 2))  # (..., half_dim, 2)
        x_real = x_pairs[..., 0]  # (..., half_dim)
        x_imag = x_pairs[..., 1]  # (..., half_dim)

        # Complex rotation: (a + bi)(cos + sin*i) = (a*cos - b*sin) + (a*sin + b*cos)i
        out_real = x_real * cos - x_imag * sin
        out_imag = x_real * sin + x_imag * cos

        # Interleave back
        return torch.stack((out_real, out_imag), dim=-1).flatten(-2)

    def apply_ggrope(
        self,
        q: Tensor,
        k: Tensor,
        row_ids: Tensor,
        col_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Apply GGRoPE to query and key tensors.

        Args:
            q: (batch, seq_len, num_q_heads, head_dim)
            k: (batch, seq_len, num_kv_heads, head_dim)
            row_ids: (batch, seq_len) — row positions in the grid
            col_ids: (batch, seq_len) — column positions in the grid

        Returns:
            q_rot: (batch, seq_len, num_q_heads, head_dim)
            k_rot: (batch, seq_len, num_kv_heads, head_dim)
        """
        cos, sin = self._compute_freqs(row_ids, col_ids)
        # cos, sin are (B, T, 1, half_dim) — broadcast over head dimension

        q_rot = self._apply_rotary(q, cos, sin)
        k_rot = self._apply_rotary(k, cos, sin)

        return q_rot.type_as(q), k_rot.type_as(k)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        row_ids: Tensor,
        col_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.apply_ggrope(q, k, row_ids, col_ids)
