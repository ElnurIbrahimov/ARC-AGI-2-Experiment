"""
Mixture of Experts (MoE) layer.

64 SwiGLU experts with top-8 routing.
Includes load balancing auxiliary loss (importance + load balance).

This is applied after each Mamba-2 or Attention block as the feed-forward layer.
For attention blocks that already have a SwiGLU MLP, the MoE replaces it
(handled at model assembly time).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.model_config import ModelConfig
from model.expert import Expert
from model.rmsnorm import RMSNorm


class MoELayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.aux_loss_weight = config.moe_aux_loss_weight

        # Pre-norm
        self.norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

        # Router: projects hidden state to expert logits
        self.router = nn.Linear(config.hidden_dim, self.num_experts, bias=False)

        # Expert pool
        self.experts = nn.ModuleList([
            Expert(config.hidden_dim, config.expert_dim)
            for _ in range(self.num_experts)
        ])

    def _compute_routing(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute top-k routing weights and indices.

        Args:
            x: (batch * seq_len, hidden_dim)

        Returns:
            routing_weights: (batch * seq_len, top_k) — normalized weights
            expert_indices:  (batch * seq_len, top_k) — selected expert indices
            router_logits:   (batch * seq_len, num_experts) — raw logits for aux loss
        """
        router_logits = self.router(x)  # (N, num_experts)

        # Top-k selection
        top_k_logits, expert_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )  # both (N, top_k)

        # Softmax over selected experts only (not all experts)
        routing_weights = F.softmax(top_k_logits, dim=-1, dtype=torch.float32).type_as(x)

        return routing_weights, expert_indices, router_logits

    def _compute_aux_loss(
        self, router_logits: Tensor
    ) -> Tensor:
        """
        Load balancing auxiliary loss.

        Combines importance loss (variance in total routing probability per expert)
        and load loss (variance in number of tokens per expert).
        This encourages balanced expert utilization.

        Args:
            router_logits: (N, num_experts) — raw router logits

        Returns:
            aux_loss: scalar
        """
        if not self.training:
            return torch.tensor(0.0, device=router_logits.device)

        num_tokens = router_logits.shape[0]

        # Router probabilities over all experts
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        # Importance loss: variance in mean routing probability per expert
        # Ideal: each expert gets equal probability mass
        importance = router_probs.mean(dim=0)  # (num_experts,)
        importance_loss = (importance * self.num_experts).pow(2).mean()

        # Load loss: variance in token assignment count per expert
        # Use straight-through estimator: argmax for counting, but differentiable
        _, top1_indices = router_logits.topk(1, dim=-1)
        expert_mask = F.one_hot(top1_indices.squeeze(-1), self.num_experts).float()
        load = expert_mask.mean(dim=0)  # (num_experts,)
        load_loss = (load * self.num_experts).pow(2).mean()

        # Combined aux loss
        aux_loss = self.aux_loss_weight * (importance_loss + load_loss)
        return aux_loss

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_dim)

        Returns:
            output: (batch, seq_len, hidden_dim)
            aux_loss: scalar load-balancing loss
        """
        residual = x
        x = self.norm(x)

        B, T, D = x.shape
        # Flatten to (N, D) for routing
        x_flat = x.reshape(-1, D)
        N = x_flat.shape[0]

        routing_weights, expert_indices, router_logits = self._compute_routing(x_flat)

        # Compute auxiliary loss
        aux_loss = self._compute_aux_loss(router_logits)

        # Dispatch tokens to experts and combine outputs
        # Simple loop implementation — sufficient for training without expert parallelism
        output = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert and which top-k slot they're in
            # expert_indices: (N, top_k), check each slot
            expert_mask = (expert_indices == expert_idx)  # (N, top_k)

            if not expert_mask.any():
                continue

            # Get token indices that use this expert (any top-k slot)
            token_mask = expert_mask.any(dim=-1)  # (N,)
            token_indices = token_mask.nonzero(as_tuple=True)[0]

            if token_indices.numel() == 0:
                continue

            # Gather input tokens for this expert
            expert_input = x_flat[token_indices]  # (num_selected, D)

            # Run expert
            expert_output = self.experts[expert_idx](expert_input)  # (num_selected, D)

            # Get routing weights for this expert
            # For each selected token, sum weights across all top-k slots matching this expert
            token_expert_mask = expert_mask[token_indices]  # (num_selected, top_k)
            token_weights = routing_weights[token_indices]  # (num_selected, top_k)
            combined_weight = (token_weights * token_expert_mask.to(token_weights.dtype)).sum(dim=-1, keepdim=True)

            # Accumulate weighted expert output (ensure matching dtype)
            weighted_output = (expert_output * combined_weight).to(output.dtype)
            output.index_add_(0, token_indices, weighted_output)

        # Reshape back and add residual
        output = output.view(B, T, D) + residual
        return output, aux_loss

    # --- Expert parallelism API stubs ---
    def get_expert_assignments(self, x: Tensor) -> dict:
        """
        Returns routing information for expert parallelism.
        Actual distribution handled in training code.

        Returns dict with:
            - expert_indices: which experts each token is routed to
            - routing_weights: the normalized weights
            - router_logits: raw logits for loss computation
        """
        x_flat = x.reshape(-1, x.shape[-1])
        routing_weights, expert_indices, router_logits = self._compute_routing(x_flat)
        return {
            "expert_indices": expert_indices,
            "routing_weights": routing_weights,
            "router_logits": router_logits,
        }
