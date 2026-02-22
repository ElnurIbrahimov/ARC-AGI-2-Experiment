"""
Mamba-2 Selective State Space Model block.

Pure PyTorch implementation (no custom CUDA kernels).

Architecture:
    Input -> RMSNorm -> Input Projection (x, z paths)
    x path: Conv1D -> SiLU -> SSM (selective scan with dt, B, C projections)
    z path: SiLU gate
    Output: (SSM output * z gate) -> Output Projection -> Residual Add

The selective scan is the core innovation: dt, B, C are input-dependent,
allowing the model to selectively remember or forget information.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.model_config import ModelConfig
from model.rmsnorm import RMSNorm


class Mamba2Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.d_state = config.mamba_d_state
        self.d_conv = config.mamba_d_conv
        self.expand = config.mamba_expand
        self.inner_dim = config.mamba_inner_dim  # expand * hidden_dim
        self.dt_rank = getattr(config, 'mamba_dt_rank', self.inner_dim // 16)

        # Pre-norm
        self.norm = RMSNorm(self.hidden_dim, eps=config.rms_norm_eps)

        # Input projection: hidden_dim -> 2 * inner_dim (x and z paths)
        self.in_proj = nn.Linear(self.hidden_dim, 2 * self.inner_dim, bias=False)

        # 1D depthwise convolution on the x path
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.inner_dim,  # depthwise
            bias=True,
        )

        # SSM parameter projections from x (combined, original Mamba style)
        # Projects to dt_rank (for low-rank dt) + d_state (B) + d_state (C)
        self.x_proj = nn.Linear(self.inner_dim, self.dt_rank + 2 * self.d_state, bias=False)
        # Low-rank dt projection: dt_rank -> inner_dim (much smaller than inner_dim -> inner_dim)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim, bias=True)

        # A parameter (log-space for stability): (inner_dim, d_state)
        # Initialize as negative values (decay)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.inner_dim, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D parameter (skip connection within SSM)
        self.D = nn.Parameter(torch.ones(self.inner_dim))

        # Output projection
        self.out_proj = nn.Linear(self.inner_dim, self.hidden_dim, bias=False)

        # Initialize dt bias to ensure positive dt values
        with torch.no_grad():
            dt_init_std = self.inner_dim ** -0.5
            nn.init.uniform_(self.dt_proj.bias, 0.001, 0.1)
            nn.init.normal_(self.dt_proj.weight, std=dt_init_std)

    def _selective_scan(
        self,
        x: Tensor,
        dt: Tensor,
        B: Tensor,
        C: Tensor,
    ) -> Tensor:
        """
        Selective scan (the core Mamba operation).

        Discretizes A and B with dt, then performs a linear recurrence:
            h_t = A_bar * h_{t-1} + B_bar * x_t
            y_t = C_t @ h_t

        Args:
            x:  (batch, seq_len, inner_dim)
            dt: (batch, seq_len, inner_dim) — discretization time steps
            B:  (batch, seq_len, d_state)
            C:  (batch, seq_len, d_state)

        Returns:
            y: (batch, seq_len, inner_dim)
        """
        input_dtype = x.dtype
        batch, seq_len, d = x.shape

        # Compute scan in float32 for numerical stability
        x_f = x.float()
        dt_f = dt.float()
        B_f = B.float()
        C_f = C.float()

        # Get A from log-space
        A = -torch.exp(self.A_log.float())  # (inner_dim, d_state), negative for decay

        # Discretize: A_bar = exp(dt * A), B_bar = dt * B
        # dt: (B, T, D) -> (B, T, D, 1)
        dt_f = dt_f.unsqueeze(-1)  # (B, T, D, 1)
        # A: (D, N) -> (1, 1, D, N)
        A = A.unsqueeze(0).unsqueeze(0)

        # dA = exp(dt * A): (B, T, D, N)
        dA = torch.exp(dt_f * A)

        # dB = dt * B: (B, T, D, N)
        # B: (B, T, N) -> (B, T, 1, N)
        dB = dt_f * B_f.unsqueeze(2)

        # x: (B, T, D) -> (B, T, D, 1)
        x_expanded = x_f.unsqueeze(-1)

        # Sequential scan
        # h: (B, D, N) — hidden state
        h = torch.zeros(batch, d, self.d_state, device=x.device, dtype=torch.float32)
        ys = []

        for t in range(seq_len):
            # h = dA_t * h + dB_t * x_t
            h = dA[:, t] * h + dB[:, t] * x_expanded[:, t]
            # y_t = (C_t @ h).sum over state dim
            # C: (B, T, N) -> C[:, t]: (B, N) -> (B, 1, N)
            y_t = (h * C_f[:, t].unsqueeze(1)).sum(dim=-1)  # (B, D)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, T, D)

        # Add skip connection: y = y + D * x
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_f

        # Cast back to input dtype
        return y.to(input_dtype)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)

        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        residual = x
        x = self.norm(x)

        # Input projection -> x_path and z_path
        xz = self.in_proj(x)  # (B, T, 2 * inner_dim)
        x_path, z_path = xz.chunk(2, dim=-1)  # each (B, T, inner_dim)

        # Conv1D on x path (transpose for conv: B, D, T)
        x_path = x_path.transpose(1, 2)  # (B, inner_dim, T)
        x_path = self.conv1d(x_path)[:, :, :x.shape[1]]  # trim padding
        x_path = x_path.transpose(1, 2)  # (B, T, inner_dim)

        # SiLU activation on x path
        x_path = F.silu(x_path)

        # SSM parameter projections (combined low-rank, original Mamba style)
        x_dbl = self.x_proj(x_path)  # (B, T, dt_rank + 2 * d_state)
        dt_x = x_dbl[:, :, :self.dt_rank]  # (B, T, dt_rank)
        B = x_dbl[:, :, self.dt_rank:self.dt_rank + self.d_state]  # (B, T, d_state)
        C = x_dbl[:, :, self.dt_rank + self.d_state:]  # (B, T, d_state)
        dt = F.softplus(self.dt_proj(dt_x))  # (B, T, inner_dim), positive

        # Selective scan
        y = self._selective_scan(x_path, dt, B, C)

        # Output gate via z path
        z = F.silu(z_path)
        y = y * z

        # Output projection
        output = self.out_proj(y)

        # Residual connection
        return output + residual
