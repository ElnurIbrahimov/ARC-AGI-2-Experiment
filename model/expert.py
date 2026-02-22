"""
Single SwiGLU Expert MLP for Mixture of Experts.

Architecture:
    gate = Linear(hidden_dim, expert_dim)
    up   = Linear(hidden_dim, expert_dim)
    out  = SiLU(gate) * up
    down = Linear(expert_dim, hidden_dim)

No bias anywhere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Expert(nn.Module):
    def __init__(self, hidden_dim: int, expert_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, expert_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, expert_dim, bias=False)
        self.down_proj = nn.Linear(expert_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., hidden_dim)

        Returns:
            output: (..., hidden_dim)
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
