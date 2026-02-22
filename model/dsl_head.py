"""
DSL Output Head — maps hidden states to DSL token logits.

Architecture:
    RMSNorm -> Linear(hidden_dim, vocab_size) -> optional temperature scaling

No bias.
"""

import torch
import torch.nn as nn
from torch import Tensor

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.model_config import ModelConfig
from model.rmsnorm import RMSNorm


class DSLHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.proj = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(
        self,
        x: Tensor,
        temperature: float | None = None,
    ) -> Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim) — hidden states from the last layer
            temperature: optional temperature for scaling logits (> 0).
                         None means no scaling (equivalent to temperature=1.0).

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        x = self.norm(x)
        logits = self.proj(x)

        if temperature is not None and temperature != 1.0:
            logits = logits / temperature

        return logits
