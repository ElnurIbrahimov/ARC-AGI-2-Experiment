"""
ModelConfig — single source of truth for the ARC-AGI-2 hybrid Mamba-2 + Transformer MoE model.

~7B total parameters, ~1B active per token.
24 Mamba-2 layers + 4 GQA attention layers at positions [7, 15, 23, 31].
64 SwiGLU experts with top-8 routing.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    # ---------- Global ----------
    hidden_dim: int = 4096
    num_layers: int = 32

    # ---------- Layer types ----------
    # 0-indexed positions where attention layers are placed (rest are Mamba-2)
    attention_layer_positions: List[int] = field(
        default_factory=lambda: [7, 15, 23, 31]
    )

    # ---------- Vocabulary ----------
    # 10 grid colors (0-9) + ~200 DSL tokens + 4 special tokens (PAD, BOS, EOS, SEP)
    num_grid_colors: int = 10
    num_dsl_tokens: int = 200
    num_special_tokens: int = 4  # PAD=0, BOS=1, EOS=2, SEP=3
    # Total vocab: special + grid colors + DSL tokens
    # Ordered as: [PAD, BOS, EOS, SEP, color_0..color_9, dsl_0..dsl_199]

    @property
    def vocab_size(self) -> int:
        return self.num_special_tokens + self.num_grid_colors + self.num_dsl_tokens

    # ---------- Sequence ----------
    max_seq_len: int = 2048

    # ---------- GQA Attention ----------
    num_query_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128  # hidden_dim // num_query_heads

    # ---------- GGRoPE ----------
    golden_ratio: float = 1.618033988749895
    rope_base: float = 10000.0

    # ---------- Mamba-2 SSM ----------
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2  # inner_dim = expand * hidden_dim
    mamba_dt_rank: int = 256  # low-rank dt projection (original Mamba style)

    # ---------- MoE ----------
    num_experts: int = 64
    top_k: int = 8
    expert_dim: int = 1280  # SwiGLU intermediate dimension
    moe_aux_loss_weight: float = 0.01  # load balancing loss coefficient

    # ---------- Normalization ----------
    rms_norm_eps: float = 1e-5

    # ---------- Regularization ----------
    dropout: float = 0.0  # no dropout

    @property
    def num_mamba_layers(self) -> int:
        return self.num_layers - len(self.attention_layer_positions)

    @property
    def mamba_inner_dim(self) -> int:
        return self.mamba_expand * self.hidden_dim

    def is_attention_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.attention_layer_positions

    def validate(self) -> None:
        assert self.hidden_dim == self.num_query_heads * self.head_dim, (
            f"hidden_dim ({self.hidden_dim}) must equal "
            f"num_query_heads ({self.num_query_heads}) * head_dim ({self.head_dim})"
        )
        assert self.num_query_heads % self.num_kv_heads == 0, (
            f"num_query_heads ({self.num_query_heads}) must be divisible by "
            f"num_kv_heads ({self.num_kv_heads})"
        )
        assert all(0 <= pos < self.num_layers for pos in self.attention_layer_positions), (
            f"All attention layer positions must be in [0, {self.num_layers})"
        )
        assert self.top_k <= self.num_experts, (
            f"top_k ({self.top_k}) must be <= num_experts ({self.num_experts})"
        )
