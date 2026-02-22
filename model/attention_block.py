"""
Grouped Query Attention block with SSMax, QK-Norm, and GGRoPE.

Architecture:
    Input -> RMSNorm -> GQA (32 Q heads, 8 KV heads)
                          - QK-Norm: RMSNorm on Q and K before attention
                          - GGRoPE applied to Q and K
                          - SSMax: logits scaled by log(seq_len)
                          - Standard softmax attention
    -> Residual Add
    -> RMSNorm -> SwiGLU MLP (NOT MoE — MoE is a separate wrapper)
    -> Residual Add

Patterns from MoR ablation study (train_gpt_ablation.py):
    SSMax: attn_scale * log(max(T, 2))
    QK-Norm: RMSNorm on Q and K projections
    GQA: 4:1 ratio of Q to KV heads
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
from model.ggrope import GGRoPE


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network (used within attention blocks, not MoE)."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        # SwiGLU: SiLU(xW_gate) * (xW_up), then down projection
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GQAAttention(nn.Module):
    """
    Grouped Query Attention with SSMax, QK-Norm, and GGRoPE.

    - 32 query heads, 8 KV heads (4:1 ratio)
    - QK-Norm applied to Q and K before attention
    - GGRoPE for 2D positional encoding
    - SSMax scales attention logits by log(seq_len)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_q_heads = config.num_query_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.gqa_group_size = self.num_q_heads // self.num_kv_heads

        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        # Projections (no bias)
        self.q_proj = nn.Linear(self.hidden_dim, q_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, self.hidden_dim, bias=False)

        # QK-Norm: separate RMSNorm for Q and K
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # GGRoPE
        self.ggrope = GGRoPE(
            head_dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            golden_ratio=config.golden_ratio,
            base=config.rope_base,
        )

        # Base attention scale
        self.base_scale = self.head_dim ** -0.5

    def forward(
        self,
        x: Tensor,
        row_ids: Tensor,
        col_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            row_ids: (batch, seq_len) — grid row positions
            col_ids: (batch, seq_len) — grid column positions
            attention_mask: (batch, 1, seq_len, seq_len) or None — additive mask

        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        B, T, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        # QK-Norm: normalize Q and K per head
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply GGRoPE to Q and K
        q, k = self.ggrope.apply_ggrope(q, k, row_ids, col_ids)

        # Expand KV heads for GQA: (B, T, num_kv_heads, D) -> (B, T, num_q_heads, D)
        k = k.unsqueeze(3).expand(B, T, self.num_kv_heads, self.gqa_group_size, self.head_dim)
        k = k.reshape(B, T, self.num_q_heads, self.head_dim)
        v = v.unsqueeze(3).expand(B, T, self.num_kv_heads, self.gqa_group_size, self.head_dim)
        v = v.reshape(B, T, self.num_q_heads, self.head_dim)

        # Transpose to (B, num_heads, T, head_dim) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores: Q @ K^T
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, T)

        # SSMax: scale by base_scale * log(seq_len)
        ssmax_scale = self.base_scale * math.log(max(T, 2))
        attn_weights = attn_weights * ssmax_scale

        # Apply attention mask (additive: -inf for masked positions)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, v)  # (B, H, T, D)

        # Reshape back: (B, H, T, D) -> (B, T, H*D)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, T, self.num_q_heads * self.head_dim)

        # Output projection
        return self.o_proj(attn_output)


class AttentionBlock(nn.Module):
    """
    Full attention block: Pre-RMSNorm -> GQA -> Residual -> Pre-RMSNorm -> SwiGLU MLP -> Residual.

    The MLP here is a standard SwiGLU (not MoE). MoE is applied as a separate
    layer wrapper in the full model assembly.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Pre-norm for attention
        self.attn_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.attn = GQAAttention(config)

        # Pre-norm for MLP
        self.mlp_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        # SwiGLU intermediate dim: 4 * hidden_dim (standard ratio)
        # This is the attention block's own MLP, separate from MoE
        self.mlp = SwiGLUMLP(
            hidden_dim=config.hidden_dim,
            intermediate_dim=4 * config.hidden_dim,
        )

    def forward(
        self,
        x: Tensor,
        row_ids: Tensor,
        col_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            row_ids: (batch, seq_len)
            col_ids: (batch, seq_len)
            attention_mask: optional additive mask (batch, 1, seq_len, seq_len)

        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        # Attention with pre-norm and residual
        x = x + self.attn(self.attn_norm(x), row_ids, col_ids, attention_mask)

        # MLP with pre-norm and residual
        x = x + self.mlp(self.mlp_norm(x))

        return x
