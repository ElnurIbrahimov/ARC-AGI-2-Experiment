"""
RMSNorm — Root Mean Square Layer Normalization.

Simpler and faster than LayerNorm: no mean centering, no bias.
Always computed in float32 for numerical stability.
"""

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Always compute norm in float32
        input_dtype = x.dtype
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        output = (x_float * rms).to(input_dtype)
        return output * self.weight
