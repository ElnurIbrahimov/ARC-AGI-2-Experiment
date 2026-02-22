"""
Shared test fixtures and small test config for unit tests.
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from config.model_config import ModelConfig


def make_test_config(**overrides) -> ModelConfig:
    """
    Create a small ModelConfig suitable for CPU testing.

    4 layers total: 3 Mamba + 1 Attention (at position 3).
    4 experts with top-2 routing.
    hidden_dim=64, head_dim=16, 4 Q heads, 2 KV heads.
    """
    defaults = dict(
        hidden_dim=64,
        num_layers=4,
        attention_layer_positions=[3],
        num_grid_colors=10,
        num_dsl_tokens=20,
        num_special_tokens=4,
        max_seq_len=128,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=16,
        golden_ratio=1.618033988749895,
        rope_base=10000.0,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank=16,
        num_experts=4,
        top_k=2,
        expert_dim=32,
        moe_aux_loss_weight=0.01,
        rms_norm_eps=1e-5,
        dropout=0.0,
    )
    defaults.update(overrides)
    config = ModelConfig(**defaults)
    config.validate()
    return config


@pytest.fixture
def test_config():
    """Small config for fast CPU tests."""
    return make_test_config()


@pytest.fixture
def batch_data(test_config):
    """Random batch of input data matching test_config."""
    B, T = 2, 16
    vocab_size = test_config.vocab_size
    token_ids = torch.randint(0, vocab_size, (B, T))
    row_ids = torch.randint(0, 5, (B, T))
    col_ids = torch.randint(0, 5, (B, T))
    return token_ids, row_ids, col_ids
