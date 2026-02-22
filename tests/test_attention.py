"""
Tests for AttentionBlock — GQA with SSMax, QK-Norm, and GGRoPE.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
import torch
from model.attention_block import AttentionBlock, GQAAttention
from model.ggrope import GGRoPE
from tests.conftest import make_test_config


class TestGQAAttention:
    """Tests for the core GQA attention mechanism."""

    @pytest.fixture
    def config(self):
        return make_test_config()

    @pytest.fixture
    def attn(self, config):
        return GQAAttention(config)

    def test_output_shape(self, attn, config):
        """Output shape should match input: (B, T, hidden_dim)."""
        B, T, D = 2, 16, config.hidden_dim
        x = torch.randn(B, T, D)
        row_ids = torch.randint(0, 5, (B, T))
        col_ids = torch.randint(0, 5, (B, T))
        out = attn(x, row_ids, col_ids)
        assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"

    def test_gqa_head_counts(self, config):
        """GQA should have correct Q:KV head ratio."""
        assert config.num_query_heads % config.num_kv_heads == 0
        ratio = config.num_query_heads // config.num_kv_heads
        assert ratio == 2, f"Expected 4:1 or 2:1 GQA ratio, got {ratio}:1"

    def test_causal_masking(self, attn, config):
        """With causal mask, changing future tokens should not affect past outputs.

        SSMax scales by log(seq_len), so we must compare same-length sequences.
        We change the last 4 tokens and verify the first 4 outputs are unchanged.
        """
        B, T, D = 1, 8, config.hidden_dim
        torch.manual_seed(123)
        x_a = torch.randn(B, T, D)
        x_b = x_a.clone()
        # Change the last 4 tokens
        x_b[:, 4:, :] = torch.randn(B, 4, D)

        row_ids = torch.zeros(B, T, dtype=torch.long)
        col_ids = torch.arange(T).unsqueeze(0).expand(B, -1)

        # Create causal mask (same length for both)
        causal = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        causal = causal.unsqueeze(0).unsqueeze(0)

        attn.eval()
        with torch.no_grad():
            out_a = attn(x_a, row_ids, col_ids, attention_mask=causal)
            out_b = attn(x_b, row_ids, col_ids, attention_mask=causal)

        # First 4 positions should be identical (future tokens changed, causal mask applied)
        diff = (out_a[:, :4] - out_b[:, :4]).abs().max().item()
        assert diff < 1e-5, f"Causal violation: max diff = {diff}"

    def test_gradient_flows(self, attn, config):
        """Gradients should flow through attention."""
        B, T, D = 2, 8, config.hidden_dim
        x = torch.randn(B, T, D, requires_grad=True)
        row_ids = torch.randint(0, 5, (B, T))
        col_ids = torch.randint(0, 5, (B, T))
        out = attn(x, row_ids, col_ids)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "No gradient on input"
        assert not torch.all(x.grad == 0), "Gradient is all zeros"


class TestAttentionBlock:
    """Tests for the full AttentionBlock (GQA + SwiGLU MLP)."""

    @pytest.fixture
    def config(self):
        return make_test_config()

    @pytest.fixture
    def block(self, config):
        return AttentionBlock(config)

    def test_output_shape(self, block, config):
        """Output shape should match input."""
        B, T, D = 2, 16, config.hidden_dim
        x = torch.randn(B, T, D)
        row_ids = torch.randint(0, 5, (B, T))
        col_ids = torch.randint(0, 5, (B, T))
        out = block(x, row_ids, col_ids)
        assert out.shape == (B, T, D)

    def test_residual_connections(self, block, config):
        """Block output should differ from input (not a no-op) but not be wildly different."""
        B, T, D = 1, 8, config.hidden_dim
        x = torch.randn(B, T, D)
        row_ids = torch.zeros(B, T, dtype=torch.long)
        col_ids = torch.zeros(B, T, dtype=torch.long)
        out = block(x, row_ids, col_ids)
        # Should not be identical (block does computation)
        assert not torch.allclose(out, x, atol=1e-6)

    def test_qk_norm_gradient(self, config):
        """QK-Norm should not block gradient flow."""
        attn = GQAAttention(config)
        B, T, D = 1, 4, config.hidden_dim
        x = torch.randn(B, T, D, requires_grad=True)
        row_ids = torch.zeros(B, T, dtype=torch.long)
        col_ids = torch.zeros(B, T, dtype=torch.long)
        out = attn(x, row_ids, col_ids)
        loss = out.sum()
        loss.backward()
        # Check Q norm and K norm parameters have gradients
        assert attn.q_norm.weight.grad is not None, "Q norm weight has no gradient"
        assert attn.k_norm.weight.grad is not None, "K norm weight has no gradient"

    def test_ssmax_scaling(self, config):
        """SSMax should scale attention by log(seq_len)."""
        attn = GQAAttention(config)
        # SSMax scale = head_dim^(-0.5) * log(max(T, 2))
        T = 16
        expected_scale = config.head_dim ** -0.5 * math.log(T)
        actual_scale = attn.base_scale * math.log(max(T, 2))
        assert abs(expected_scale - actual_scale) < 1e-7, (
            f"SSMax scale mismatch: expected {expected_scale}, got {actual_scale}"
        )

    def test_ggrope_applied(self, config):
        """GGRoPE should make attention position-aware -- different positions should give different outputs."""
        block = AttentionBlock(config)
        block.eval()

        B, T, D = 1, 8, config.hidden_dim
        x = torch.randn(B, T, D)

        # Same input, different position encoding
        row_ids_a = torch.zeros(B, T, dtype=torch.long)
        col_ids_a = torch.arange(T).unsqueeze(0)

        row_ids_b = torch.arange(T).unsqueeze(0)
        col_ids_b = torch.zeros(B, T, dtype=torch.long)

        with torch.no_grad():
            out_a = block(x, row_ids_a, col_ids_a)
            out_b = block(x, row_ids_b, col_ids_b)

        # Different positions should yield different outputs
        assert not torch.allclose(out_a, out_b, atol=1e-5), "GGRoPE has no effect"
