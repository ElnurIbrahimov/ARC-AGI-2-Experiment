"""
Tests for MoELayer — Mixture of Experts with top-k routing.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from model.moe_layer import MoELayer
from tests.conftest import make_test_config


class TestMoELayer:
    """Tests for the MoE routing and expert dispatch."""

    @pytest.fixture
    def config(self):
        return make_test_config()

    @pytest.fixture
    def moe(self, config):
        return MoELayer(config)

    def test_output_shape(self, moe, config):
        """Output shape should match input: (B, T, hidden_dim)."""
        B, T, D = 2, 8, config.hidden_dim
        x = torch.randn(B, T, D)
        moe.train()
        out, aux_loss = moe(x)
        assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"

    def test_aux_loss_positive_during_training(self, moe, config):
        """Auxiliary loss should be non-zero and positive during training."""
        B, T, D = 2, 8, config.hidden_dim
        x = torch.randn(B, T, D)
        moe.train()
        _, aux_loss = moe(x)
        assert aux_loss.item() > 0, f"Aux loss should be positive, got {aux_loss.item()}"

    def test_aux_loss_zero_during_eval(self, moe, config):
        """Auxiliary loss should be zero during evaluation."""
        B, T, D = 2, 8, config.hidden_dim
        x = torch.randn(B, T, D)
        moe.eval()
        with torch.no_grad():
            _, aux_loss = moe(x)
        assert aux_loss.item() == 0.0, f"Aux loss should be 0 in eval, got {aux_loss.item()}"

    def test_top_k_routing(self, config):
        """Only top_k experts should be selected per token."""
        moe = MoELayer(config)
        B, T, D = 1, 4, config.hidden_dim
        x = torch.randn(B, T, D)
        x_flat = x.reshape(-1, D)
        x_normed = moe.norm(x_flat)

        routing_weights, expert_indices, _ = moe._compute_routing(x_normed)

        # Each token should have exactly top_k expert indices
        assert expert_indices.shape == (B * T, config.top_k), (
            f"Expected {(B * T, config.top_k)}, got {expert_indices.shape}"
        )
        # All indices should be valid expert IDs
        assert expert_indices.min() >= 0
        assert expert_indices.max() < config.num_experts

    def test_routing_weights_sum_to_one(self, config):
        """Routing weights for each token should sum to ~1 (softmax over top-k)."""
        moe = MoELayer(config)
        B, T, D = 1, 8, config.hidden_dim
        x = torch.randn(B, T, D)
        x_flat = x.reshape(-1, D)
        x_normed = moe.norm(x_flat)

        routing_weights, _, _ = moe._compute_routing(x_normed)
        weight_sums = routing_weights.sum(dim=-1)

        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), (
            f"Routing weights don't sum to 1: {weight_sums}"
        )

    def test_different_batch_sizes(self, moe, config):
        """Should handle various batch sizes."""
        D = config.hidden_dim
        moe.eval()
        for B in [1, 2, 4]:
            x = torch.randn(B, 8, D)
            with torch.no_grad():
                out, _ = moe(x)
            assert out.shape == (B, 8, D), f"Failed for B={B}"

    def test_gradient_flows(self, moe, config):
        """Gradients should flow through MoE."""
        B, T, D = 2, 8, config.hidden_dim
        x = torch.randn(B, T, D, requires_grad=True)
        moe.train()
        out, aux_loss = moe(x)
        loss = out.sum() + aux_loss
        loss.backward()
        assert x.grad is not None, "No gradient on input"
        assert not torch.all(x.grad == 0), "Gradient is all zeros"

    def test_load_balance_over_many_inputs(self, config):
        """Over many random inputs, experts should get roughly equal load."""
        moe = MoELayer(config)
        moe.eval()

        D = config.hidden_dim
        expert_counts = torch.zeros(config.num_experts)

        # Run many batches
        for _ in range(50):
            x = torch.randn(4, 16, D)
            x_flat = x.reshape(-1, D)
            x_normed = moe.norm(x_flat)
            _, expert_indices, _ = moe._compute_routing(x_normed)
            for idx in expert_indices.flatten().tolist():
                expert_counts[idx] += 1

        # Check that no expert is completely starved
        total_assignments = expert_counts.sum().item()
        expected_per_expert = total_assignments / config.num_experts
        min_count = expert_counts.min().item()

        # Each expert should get at least 10% of the expected load
        assert min_count > expected_per_expert * 0.1, (
            f"Expert load imbalance: min={min_count}, expected~{expected_per_expert:.0f}, "
            f"counts={expert_counts.tolist()}"
        )

    def test_num_experts_correct(self, moe, config):
        """MoE should have exactly num_experts experts."""
        assert len(moe.experts) == config.num_experts
