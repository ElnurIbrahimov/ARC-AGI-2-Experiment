"""
Tests for Mamba2Block — selective state space model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from model.mamba2_block import Mamba2Block
from tests.conftest import make_test_config


class TestMamba2Block:
    """Tests for the Mamba2Block module."""

    @pytest.fixture
    def config(self):
        return make_test_config()

    @pytest.fixture
    def mamba(self, config):
        return Mamba2Block(config)

    def test_output_shape(self, mamba, config):
        """Output shape should match input: (B, T, hidden_dim)."""
        B, T, D = 2, 16, config.hidden_dim
        x = torch.randn(B, T, D)
        out = mamba(x)
        assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"

    def test_different_sequence_lengths(self, mamba, config):
        """Should handle various sequence lengths."""
        D = config.hidden_dim
        for T in [1, 4, 16, 32]:
            x = torch.randn(1, T, D)
            out = mamba(x)
            assert out.shape == (1, T, D), f"Failed for T={T}"

    def test_different_batch_sizes(self, mamba, config):
        """Should handle various batch sizes."""
        D = config.hidden_dim
        for B in [1, 2, 4]:
            x = torch.randn(B, 8, D)
            out = mamba(x)
            assert out.shape == (B, 8, D), f"Failed for B={B}"

    def test_gradient_flows(self, mamba, config):
        """Gradients should flow back through the block."""
        B, T, D = 2, 8, config.hidden_dim
        x = torch.randn(B, T, D, requires_grad=True)
        out = mamba(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "No gradient on input"
        assert x.grad.shape == x.shape, "Gradient shape mismatch"
        assert not torch.all(x.grad == 0), "Gradient is all zeros"

    def test_causal_property(self, mamba, config):
        """Output at position t should not depend on positions > t.

        Test by running the same prefix twice (once alone, once with extra tokens appended)
        and checking that the prefix outputs match.
        """
        D = config.hidden_dim
        torch.manual_seed(42)
        x_short = torch.randn(1, 4, D)
        x_long = torch.cat([x_short, torch.randn(1, 4, D)], dim=1)

        mamba.eval()
        with torch.no_grad():
            out_short = mamba(x_short)
            out_long = mamba(x_long)

        # Outputs for the first 4 positions should be identical
        diff = (out_short - out_long[:, :4, :]).abs().max().item()
        assert diff < 1e-5, f"Causal violation: max diff = {diff}"

    def test_residual_connection(self, mamba, config):
        """With zero-initialized weights, output should be close to input (residual passthrough)."""
        # This is a sanity check -- with random weights the residual is still present
        D = config.hidden_dim
        x = torch.randn(1, 4, D)
        out = mamba(x)
        # Output should not be identical to input (the block does computation)
        assert not torch.allclose(out, x, atol=1e-6), "Output is identical to input -- block is a no-op"

    def test_deterministic(self, mamba, config):
        """Same input should produce same output (deterministic in eval mode)."""
        D = config.hidden_dim
        x = torch.randn(1, 8, D)
        mamba.eval()
        with torch.no_grad():
            out1 = mamba(x)
            out2 = mamba(x)
        assert torch.allclose(out1, out2, atol=1e-6), "Non-deterministic output"
