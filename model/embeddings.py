"""
ARC grid + DSL token embeddings.

Token layout (matching ModelConfig):
    [PAD=0, BOS=1, EOS=2, SEP=3, color_0..color_9, dsl_0..dsl_199]

No positional embedding — handled separately by GGRoPE.
"""

import torch
import torch.nn as nn
from torch import Tensor

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.model_config import ModelConfig


class ARCEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_dim = config.hidden_dim

        # Single embedding table for all token types
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)

        # Initialize with small values (scaled by hidden_dim)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=self.hidden_dim ** -0.5)

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Args:
            token_ids: (batch, seq_len) integer token indices in [0, vocab_size)

        Returns:
            embeddings: (batch, seq_len, hidden_dim)
        """
        return self.token_embedding(token_ids)

    # --- Convenience token ID helpers ---
    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def bos_token_id(self) -> int:
        return 1

    @property
    def eos_token_id(self) -> int:
        return 2

    @property
    def sep_token_id(self) -> int:
        return 3

    @property
    def grid_color_offset(self) -> int:
        return self.config.num_special_tokens  # 4

    @property
    def dsl_token_offset(self) -> int:
        return self.config.num_special_tokens + self.config.num_grid_colors  # 14

    def grid_color_id(self, color: int) -> int:
        """Map grid color (0-9) to token ID."""
        assert 0 <= color <= 9
        return self.grid_color_offset + color

    def dsl_token_id(self, dsl_idx: int) -> int:
        """Map DSL token index to token ID."""
        assert 0 <= dsl_idx < self.config.num_dsl_tokens
        return self.dsl_token_offset + dsl_idx
