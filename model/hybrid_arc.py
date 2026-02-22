"""
HybridARC — Top-level 7B Hybrid Mamba-2 + Transformer MoE model for ARC-AGI-2.

Architecture:
    - 32 layers total
    - 24 Mamba-2 layers (for O(N) sequence modeling)
    - 4 GQA Attention layers at positions [7, 15, 23, 31] (0-indexed)
    - After EVERY attention layer: MoE layer (64 experts, top-8)
    - After Mamba layers: simple SwiGLU FFN (no MoE, for efficiency)
    - Pre-RMSNorm everywhere
    - Residual connections throughout

Layer structure:
    - Mamba layer:     Mamba2Block (has its own norm+residual) -> RMSNorm -> SwiGLU FFN -> residual
    - Attention layer: RMSNorm -> GQAAttention -> residual -> MoELayer (has its own norm+residual)

Input:
    - token_ids: (B, T) int tensor of grid/DSL tokens
    - row_ids:   (B, T) int tensor of row positions (for GGRoPE)
    - col_ids:   (B, T) int tensor of column positions (for GGRoPE)

Output:
    - ModelOutput with logits, hidden_states, aux_loss, optional all_hidden_states
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.model_config import ModelConfig
from model.rmsnorm import RMSNorm
from model.embeddings import ARCEmbedding
from model.mamba2_block import Mamba2Block
from model.attention_block import GQAAttention
from model.moe_layer import MoELayer
from model.dsl_head import DSLHead


@dataclass
class ModelOutput:
    logits: Tensor              # (B, T, vocab_size) DSL token logits
    hidden_states: Tensor       # (B, T, hidden_dim) last layer hidden states
    aux_loss: Tensor            # scalar MoE auxiliary loss (sum over all MoE layers)
    all_hidden_states: Optional[List[Tensor]] = None  # per-layer (if requested)


class SwiGLUFFN(nn.Module):
    """Simple SwiGLU feed-forward network for Mamba layers (no MoE overhead)."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MambaFFNBlock(nn.Module):
    """
    Single Mamba-2 layer (no separate FFN — Mamba's gated projection acts as the channel mixer).

    Structure:
        Mamba2Block (contains its own pre-norm + residual)

    Note: Unlike Transformer blocks, Mamba blocks don't need a separate FFN because
    the in_proj -> SSM -> gate -> out_proj already provides channel mixing through
    the gated SiLU activation. This follows the original Mamba architecture.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.mamba = Mamba2Block(config)

    def forward(self, x: Tensor, row_ids: Tensor = None, col_ids: Tensor = None):
        """
        Args:
            x: (B, T, hidden_dim)
            row_ids, col_ids: unused by Mamba, included for uniform interface

        Returns:
            output: (B, T, hidden_dim)
        """
        return self.mamba(x)


class AttentionMoEBlock(nn.Module):
    """
    GQA Attention layer + MoE FFN with pre-norm and residuals.

    Structure:
        RMSNorm -> GQAAttention -> residual add
        -> MoELayer (contains its own pre-norm + residual)

    Note: We use GQAAttention (not AttentionBlock) directly because
    AttentionBlock includes its own SwiGLU MLP that we don't want --
    the MoE layer replaces that standard FFN.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Attention with pre-norm
        self.attn_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.attn = GQAAttention(config)

        # MoE layer replaces the standard SwiGLU FFN
        # MoELayer already does: norm -> routing -> experts -> residual internally
        self.moe = MoELayer(config)

    def forward(self, x: Tensor, row_ids: Tensor, col_ids: Tensor):
        """
        Args:
            x: (B, T, hidden_dim)
            row_ids: (B, T) grid row positions for GGRoPE
            col_ids: (B, T) grid column positions for GGRoPE

        Returns:
            output: (B, T, hidden_dim)
            aux_loss: scalar MoE load-balancing loss
        """
        # Build causal mask
        T = x.shape[1]
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        # Attention: pre-norm -> GQA -> residual
        x = x + self.attn(self.attn_norm(x), row_ids, col_ids, attention_mask=causal_mask)

        # MoE: norm -> routing -> experts -> residual (internal)
        x, aux_loss = self.moe(x)

        return x, aux_loss


class HybridARC(nn.Module):
    """
    7B Hybrid Mamba-2 + Transformer MoE model for ARC-AGI-2.

    Architecture:
    - 32 layers total
    - 24 Mamba-2 layers (for O(N) sequence modeling)
    - 4 GQA Attention layers at positions [7, 15, 23, 31] (0-indexed)
      (every 8th layer starting from 7, i.e. at roughly 1/4, 1/2, 3/4, end)
    - After EVERY attention layer: MoE layer (64 experts, top-8)
    - After Mamba layers: simple SwiGLU FFN (no MoE, for efficiency)
    - Pre-RMSNorm everywhere
    - Residual connections
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        config.validate()

        # Token embeddings
        self.embeddings = ARCEmbedding(config)

        # Build layers
        self.layers = nn.ModuleList()
        self.attention_layer_indices = set(config.attention_layer_positions)

        for i in range(config.num_layers):
            if i in self.attention_layer_indices:
                self.layers.append(AttentionMoEBlock(config, layer_idx=i))
            else:
                self.layers.append(MambaFFNBlock(config, layer_idx=i))

        # Final norm
        self.final_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

        # Output head
        self.dsl_head = DSLHead(config)

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights: normal(0, 0.02) for linear layers, ones for norm weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def forward(
        self,
        token_ids: Tensor,
        row_ids: Tensor,
        col_ids: Tensor,
        output_hidden_states: bool = False,
    ) -> ModelOutput:
        """
        Forward pass.

        Args:
            token_ids: (B, T) integer token indices
            row_ids:   (B, T) integer row positions
            col_ids:   (B, T) integer column positions
            output_hidden_states: if True, collect hidden states from every layer

        Returns:
            ModelOutput with logits, hidden_states, aux_loss, and optionally all_hidden_states
        """
        # Embed tokens
        x = self.embeddings(token_ids)  # (B, T, hidden_dim)

        all_hidden_states = [] if output_hidden_states else None
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Process through layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(x)

            if i in self.attention_layer_indices:
                # AttentionMoEBlock returns (output, aux_loss)
                if self.gradient_checkpointing and self.training:
                    # checkpoint doesn't support tuple returns directly,
                    # so we wrap in a helper
                    x, aux_loss = self._checkpoint_attn_moe(layer, x, row_ids, col_ids)
                else:
                    x, aux_loss = layer(x, row_ids, col_ids)
                total_aux_loss = total_aux_loss + aux_loss
            else:
                # MambaFFNBlock returns output only
                if self.gradient_checkpointing and self.training:
                    x = checkpoint(
                        layer, x, row_ids, col_ids,
                        use_reentrant=False,
                    )
                else:
                    x = layer(x, row_ids, col_ids)

        # Final norm
        x = self.final_norm(x)

        # Collect final hidden states if requested
        if output_hidden_states:
            all_hidden_states.append(x)

        # Output head
        logits = self.dsl_head(x)

        return ModelOutput(
            logits=logits,
            hidden_states=x,
            aux_loss=total_aux_loss,
            all_hidden_states=all_hidden_states,
        )

    @staticmethod
    def _checkpoint_attn_moe(layer, x, row_ids, col_ids):
        """Helper to gradient-checkpoint an AttentionMoEBlock that returns a tuple."""
        # Pack outputs into a single tensor, then unpack
        # We pass aux_loss as a 1-element tensor appended to hidden states
        def run_layer(x_, row_ids_, col_ids_):
            out, aux = layer(x_, row_ids_, col_ids_)
            return out, aux

        return checkpoint(
            run_layer, x, row_ids, col_ids,
            use_reentrant=False,
        )

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency during training."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def count_parameters(self) -> Dict[str, int]:
        """Count total, active-per-token, and per-component parameters."""
        total = sum(p.numel() for p in self.parameters())

        # Per-component breakdown
        embedding_params = sum(p.numel() for p in self.embeddings.parameters())
        head_params = sum(p.numel() for p in self.dsl_head.parameters())
        norm_params = sum(p.numel() for p in self.final_norm.parameters())

        mamba_params = 0
        attention_params = 0
        moe_params = 0

        for i, layer in enumerate(self.layers):
            layer_params = sum(p.numel() for p in layer.parameters())
            if i in self.attention_layer_indices:
                # Split attention vs MoE params
                attn_p = sum(p.numel() for p in layer.attn_norm.parameters())
                attn_p += sum(p.numel() for p in layer.attn.parameters())
                moe_p = sum(p.numel() for p in layer.moe.parameters())
                attention_params += attn_p
                moe_params += moe_p
            else:
                mamba_params += layer_params

        # Active params per token: total minus inactive expert params
        # Each MoE layer has num_experts experts, but only top_k are active
        # Active fraction of MoE = top_k / num_experts
        num_moe_layers = len(self.attention_layer_indices)
        first_attn_idx = next(iter(self.attention_layer_indices), None)
        if num_moe_layers > 0 and first_attn_idx is not None and hasattr(self.layers[first_attn_idx], 'moe'):
            # Find first MoE layer to compute expert params
            for i in self.attention_layer_indices:
                moe_layer = self.layers[i].moe
                total_expert_params = sum(
                    p.numel() for expert in moe_layer.experts for p in expert.parameters()
                )
                router_and_norm_params = sum(p.numel() for p in moe_layer.parameters()) - total_expert_params
                active_expert_params = total_expert_params * self.config.top_k // self.config.num_experts
                break
            inactive_expert_params = num_moe_layers * (total_expert_params - active_expert_params)
        else:
            inactive_expert_params = 0

        active_per_token = total - inactive_expert_params

        return {
            "total": total,
            "active_per_token": active_per_token,
            "embedding": embedding_params,
            "mamba_layers": mamba_params,
            "attention_layers": attention_params,
            "moe_layers": moe_params,
            "output_head": head_params,
            "final_norm": norm_params,
        }

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        row_ids: Tensor,
        col_ids: Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> Tensor:
        """
        Autoregressive generation for DSL tokens.

        Args:
            token_ids: (B, T) initial token sequence
            row_ids:   (B, T) initial row positions
            col_ids:   (B, T) initial column positions
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature (> 0)
            top_k: top-k filtering (0 = no filtering)

        Returns:
            generated_ids: (B, T + max_new_tokens) full sequence including prompt
        """
        self.eval()
        B, T = token_ids.shape
        device = token_ids.device

        # We'll grow these tensors as we generate
        ids = token_ids.clone()
        rows = row_ids.clone()
        cols = col_ids.clone()

        # Default row/col for generated tokens (0, 0) -- DSL tokens aren't grid-positioned
        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            seq_len = ids.shape[1]
            if seq_len > self.config.max_seq_len:
                ids_input = ids[:, -self.config.max_seq_len:]
                rows_input = rows[:, -self.config.max_seq_len:]
                cols_input = cols[:, -self.config.max_seq_len:]
            else:
                ids_input = ids
                rows_input = rows
                cols_input = cols

            # Forward pass
            output = self.forward(ids_input, rows_input, cols_input)

            # Get logits for the last position
            next_logits = output.logits[:, -1, :]  # (B, vocab_size)

            # Temperature scaling
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                threshold = v[:, -1].unsqueeze(-1)
                next_logits[next_logits < threshold] = float("-inf")

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append
            ids = torch.cat([ids, next_token], dim=1)
            # Generated DSL tokens get row=0, col=0
            rows = torch.cat([rows, torch.zeros(B, 1, dtype=torch.long, device=device)], dim=1)
            cols = torch.cat([cols, torch.zeros(B, 1, dtype=torch.long, device=device)], dim=1)

            # Stop if all sequences have generated EOS (token id 2)
            if (next_token == 2).all():
                break

        return ids
