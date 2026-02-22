"""
CandidateRanker — uses CausewayAdapter to rank candidate DSL programs
by predicted improvement WITHOUT executing them.

This is the key efficiency mechanism: instead of executing all N candidate
programs (expensive symbolic execution), we use Causeway's causal reasoning
to predict which candidates are most likely to improve, then only execute
the top fraction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.dsl_config import VOCAB_SIZE


class CandidateRanker:
    """
    Ranks DSL program candidates using Causeway's causal reasoning.

    Given the current best program and a set of candidate token sequences,
    predicts which candidates are most likely to improve the program
    without executing them.
    """

    def __init__(
        self,
        causeway_adapter=None,
        model=None,
        top_fraction: float = 0.5,
        d_model: int = 4096,
    ):
        """
        Args:
            causeway_adapter: CausewayAdapter instance. If None, falls back to
                              random ranking (graceful degradation).
            model: optional, the backbone model for getting hidden states and embeddings.
            top_fraction: fraction of candidates to keep after filtering (0-1).
            d_model: hidden dimension, used for encoding candidates when no model is provided.
        """
        self.causeway = causeway_adapter
        self.model = model
        self.top_fraction = max(0.1, min(1.0, top_fraction))
        self.d_model = d_model
        self._available = causeway_adapter is not None

    def rank(
        self,
        hidden_states: torch.Tensor,
        candidate_tokens: List[List[int]],
        current_score: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """
        Rank candidates by predicted improvement.

        Args:
            hidden_states: (1, T, H) or (T, H) model hidden states from the
                           last forward pass on the current task. We pool to (1, H).
            candidate_tokens: list of candidate DSL token sequences.
            current_score: score of current best program (for relative scoring).

        Returns:
            List of (candidate_idx, predicted_score) sorted descending.
            Scores are absolute (not relative to current_score).
        """
        if not candidate_tokens:
            return []

        # Fallback: no Causeway available -> return candidates in original order
        # with uniform scores. Still functional, just unranked.
        if not self._available:
            return [(i, 0.0) for i in range(len(candidate_tokens))]

        # Pool hidden states to a single vector
        h = self._pool_hidden_states(hidden_states)  # (1, H)

        # Encode each candidate as an action embedding
        candidate_actions = []
        for tokens in candidate_tokens:
            embedding = self._tokens_to_embedding(tokens, h.device)  # (1, d_model)
            action = self.causeway.encode_action(embedding)  # (1, d_action)
            candidate_actions.append(action)

        # Use Causeway to rank
        ranked = self.causeway.rank_candidates(h, candidate_actions)
        return ranked

    def filter_candidates(
        self,
        hidden_states: torch.Tensor,
        candidate_tokens: List[List[int]],
        current_score: float = 0.0,
    ) -> List[int]:
        """
        Return indices of top candidates worth executing.

        Keeps top_fraction of candidates (at least 1).

        Args:
            hidden_states: model hidden states from the current task.
            candidate_tokens: list of candidate DSL token sequences.
            current_score: score of current best program.

        Returns:
            List of candidate indices to execute, sorted by predicted quality.
        """
        if not candidate_tokens:
            return []

        ranked = self.rank(hidden_states, candidate_tokens, current_score)

        # Keep top fraction, minimum 1
        keep_count = max(1, int(len(ranked) * self.top_fraction))
        top_indices = [idx for idx, _score in ranked[:keep_count]]

        return top_indices

    def encode_candidates(
        self,
        candidate_tokens: List[List[int]],
        device: str = 'cuda',
    ) -> torch.Tensor:
        """
        Convert candidate token sequences to embeddings for Causeway.

        Each candidate is converted to a fixed-size (d_model,) embedding
        using bag-of-tokens with learned position weighting.

        Args:
            candidate_tokens: list of token sequences.
            device: target device.

        Returns:
            (N, d_model) tensor of candidate embeddings.
        """
        embeddings = []
        for tokens in candidate_tokens:
            emb = self._tokens_to_embedding(tokens, device)  # (1, d_model)
            embeddings.append(emb)

        if not embeddings:
            return torch.zeros(0, self.d_model, device=device)

        return torch.cat(embeddings, dim=0)  # (N, d_model)

    # ── Internal helpers ────────────────────────────────────────────────

    def _pool_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Pool hidden states to a single (1, H) vector.

        Accepts (B, T, H) or (T, H). Uses mean pooling over the sequence dimension.
        """
        if hidden_states.dim() == 2:
            # (T, H) -> (1, H)
            return hidden_states.mean(dim=0, keepdim=True)
        elif hidden_states.dim() == 3:
            # (B, T, H) -> (B, H), take first batch element
            pooled = hidden_states.mean(dim=1)  # (B, H)
            return pooled[:1]  # (1, H)
        else:
            raise ValueError(
                f"Expected hidden_states with 2 or 3 dims, got {hidden_states.dim()}"
            )

    def _tokens_to_embedding(
        self,
        tokens: List[int],
        device: str = 'cuda',
    ) -> torch.Tensor:
        """
        Convert a token sequence to a fixed-size (1, d_model) embedding.

        Strategy: position-weighted bag-of-tokens. Each token gets a one-hot-like
        embedding scaled by inverse position (earlier tokens weighted more heavily).
        This is projected into d_model space.

        When the backbone model is available and has an embedding layer, we use
        the model's own embeddings for higher quality.
        """
        if not tokens:
            return torch.zeros(1, self.d_model, device=device)

        # Try to use model embeddings if available
        if self.model is not None and hasattr(self.model, 'token_embedding'):
            token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            with torch.no_grad():
                embs = self.model.token_embedding(token_tensor)  # (L, H)
            # Position-weighted mean: earlier tokens matter more
            weights = torch.arange(len(tokens), 0, -1, dtype=torch.float32, device=device)
            weights = weights / weights.sum()
            weighted = embs * weights.unsqueeze(1)  # (L, H)
            pooled = weighted.sum(dim=0, keepdim=True)  # (1, H)
            return pooled

        # Fallback: sparse bag-of-tokens projected to d_model
        # Create a feature vector from token IDs using sinusoidal encoding
        max_len = len(tokens)
        feature = torch.zeros(self.d_model, device=device)

        for pos, token_id in enumerate(tokens):
            # Position weight: earlier tokens weighted more
            weight = 1.0 / (pos + 1)
            # Distribute token signal across feature dimensions using hashing
            # This spreads different tokens across different dimensions
            for d in range(min(32, self.d_model)):
                idx = (token_id * 31 + d * 7) % self.d_model
                feature[idx] += weight * (1.0 if (token_id + d) % 2 == 0 else -1.0)

        # Normalize to unit length
        norm = feature.norm()
        if norm > 0:
            feature = feature / norm

        return feature.unsqueeze(0)  # (1, d_model)
