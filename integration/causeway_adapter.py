"""
Causeway Adapter for ARC-AGI-2
==============================

Wraps Causeway's causal counterfactual reasoning module for the ARC grid domain.

Original Causeway outputs a 5D DeltaVector with generic dimensions
(risk_shift, goal_progress, constraint_violation, resource_cost, success_probability).

We re-map to ARC-specific dimensions:
  0: spatial_correctness   — are objects in the right positions?
  1: color_correctness     — are colors right?
  2: structural_integrity  — right number/size of objects?
  3: pattern_consistency   — does the pattern repeat correctly?
  4: overall_improvement   — is this modification better overall?

The adapter projects from our 7B model's hidden_dim (4096) down to Causeway's
causal variable space, and re-interprets the 5D output for ARC scoring.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Try importing Causeway from source
try:
    sys.path.insert(0, r'C:\Users\asus\Desktop\causeway')
    from causeway.causeway_module import Causeway
    from causeway.delta_predictor import DeltaVector
    CAUSEWAY_AVAILABLE = True
except ImportError:
    CAUSEWAY_AVAILABLE = False
    # Stub DeltaVector for type hints when Causeway isn't installed
    @dataclass
    class DeltaVector:
        values: torch.Tensor
        dim_names: list
        confidence: torch.Tensor
        def to_dict(self, batch_idx=0): return {}

    class Causeway(nn.Module):
        """Stub Causeway when source is not available."""
        def __init__(self, d_model, d_causal=32, d_action=64, **kwargs):
            super().__init__()
            self.d_model = d_model
            self.d_causal = d_causal
            self.d_action = d_action
            self.state_encoder = nn.Linear(d_model, d_causal)
            self._action_proj = nn.Linear(d_action, d_causal)
            self._delta_proj = nn.Linear(d_causal * 3, 5)
            self._conf_proj = nn.Linear(d_causal * 3, 5)

        def forward(self, h, action):
            z = self.state_encoder(h)
            a = self._action_proj(action)
            z_cf = z + a
            ctx = torch.cat([z, z_cf, z_cf - z], dim=-1)
            values = self._delta_proj(ctx)
            confidence = torch.sigmoid(self._conf_proj(ctx))
            return DeltaVector(
                values=values,
                dim_names=["d0", "d1", "d2", "d3", "d4"],
                confidence=confidence,
            )

        def get_regularization_losses(self):
            return {
                "acyclicity": torch.tensor(0.0),
                "sparsity": torch.tensor(0.0),
                "edge_count": torch.tensor(0.0),
                "orthogonality": torch.tensor(0.0),
            }


# ─── ARC-specific delta dimensions ──────────────────────────────────

ARC_DELTA_DIMS = [
    "spatial_correctness",
    "color_correctness",
    "structural_integrity",
    "pattern_consistency",
    "overall_improvement",
]


# ─── Output container ───────────────────────────────────────────────

@dataclass
class ARCDelta:
    """ARC-specific structured delta from Causeway."""
    spatial_correctness: torch.Tensor     # (batch,)
    color_correctness: torch.Tensor       # (batch,)
    structural_integrity: torch.Tensor    # (batch,)
    pattern_consistency: torch.Tensor     # (batch,)
    overall_improvement: torch.Tensor     # (batch,)
    confidence: torch.Tensor             # (batch, 5)
    raw_delta: object                    # Original DeltaVector from Causeway

    def score(self) -> torch.Tensor:
        """Weighted overall score: (batch,). Higher = better modification."""
        # Weight spatial and color more heavily for ARC
        weights = torch.tensor(
            [0.25, 0.25, 0.20, 0.15, 0.15],
            device=self.confidence.device,
        )
        stacked = torch.stack([
            self.spatial_correctness,
            self.color_correctness,
            self.structural_integrity,
            self.pattern_consistency,
            self.overall_improvement,
        ], dim=-1)  # (batch, 5)
        # Confidence-weighted
        weighted = stacked * self.confidence * weights.unsqueeze(0)
        return weighted.sum(dim=-1)  # (batch,)


# ─── Adapter ────────────────────────────────────────────────────────

class CausewayAdapter(nn.Module):
    """
    Wraps Causeway for ARC-AGI-2.

    Causeway operates in a causal variable space (d_causal) much smaller than
    our 7B model's hidden_dim (4096). This adapter:
      1. Instantiates Causeway with scaled parameters
      2. Provides an action encoder for DSL modification embeddings
      3. Re-maps Causeway's 5D output to ARC-specific dimensions
      4. Exposes rank_candidates() for batch scoring without execution

    Parameter budget: ~2-4M (mostly the action encoder and re-mapping MLP).
    """

    def __init__(
        self,
        d_model: int = 4096,
        d_causal: int = 128,
        d_action: int = 128,
        graph_layers: int = 2,
        propagation_steps: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model:  Hidden dim of the 7B backbone.
            d_causal: Causal variable count (scaled up from 48 for larger model).
            d_action: Action embedding dimension for DSL modifications.
            graph_layers: Message-passing layers in Causeway's causal graph.
            propagation_steps: Intervention propagation depth.
            dropout: Dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.d_causal = d_causal
        self.d_action = d_action

        # Core Causeway module
        self.causeway = Causeway(
            d_model=d_model,
            d_causal=d_causal,
            d_action=d_action,
            graph_layers=graph_layers,
            propagation_steps=propagation_steps,
            delta_dims=ARC_DELTA_DIMS,
            dropout=dropout,
        )

        # Action encoder: project DSL modification embeddings to d_action
        # Input is a modification embedding from the backbone (d_model) or
        # a smaller pre-processed embedding. We support both via a two-layer MLP.
        self.action_encoder = nn.Sequential(
            nn.Linear(d_model, d_action * 2),
            nn.LayerNorm(d_action * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_action * 2, d_action),
            nn.LayerNorm(d_action),
        )

        # ARC re-mapping: Causeway outputs 5 generic dims, we project to
        # ARC-specific interpretation. This is a learned linear re-weighting
        # (not just a rename) because the semantic meaning differs.
        # Input: 5 values + 5 confidences = 10
        self.arc_remap = nn.Sequential(
            nn.Linear(10, 64),
            nn.GELU(),
            nn.Linear(64, 5),
        )

        # Confidence calibrator for ARC domain
        self.arc_confidence = nn.Sequential(
            nn.Linear(10, 64),
            nn.GELU(),
            nn.Linear(64, 5),
            nn.Sigmoid(),
        )

    def encode_action(self, modification_embedding: torch.Tensor) -> torch.Tensor:
        """
        Encode a DSL program modification as an action vector.

        Args:
            modification_embedding: (batch, d_model) from the backbone, or
                                    any embedding describing the modification.

        Returns:
            action: (batch, d_action) encoded action for Causeway.
        """
        return self.action_encoder(modification_embedding)

    def forward(
        self,
        h: torch.Tensor,
        action: torch.Tensor,
    ) -> ARCDelta:
        """
        Run Causeway and return ARC-specific delta.

        Args:
            h:      (batch, d_model) backbone hidden state.
            action: (batch, d_action) encoded action (from encode_action or external).

        Returns:
            ARCDelta with per-dimension scores, confidence, and raw DeltaVector.
        """
        # Run Causeway's full pipeline
        raw_delta = self.causeway(h, action)

        # Combine raw values and confidence for re-mapping
        combined = torch.cat([raw_delta.values, raw_delta.confidence], dim=-1)  # (batch, 10)

        # ARC-specific re-interpretation
        arc_values = self.arc_remap(combined)         # (batch, 5)
        arc_conf = self.arc_confidence(combined)       # (batch, 5)

        return ARCDelta(
            spatial_correctness=arc_values[:, 0],
            color_correctness=arc_values[:, 1],
            structural_integrity=arc_values[:, 2],
            pattern_consistency=arc_values[:, 3],
            overall_improvement=arc_values[:, 4],
            confidence=arc_conf,
            raw_delta=raw_delta,
        )

    def rank_candidates(
        self,
        h: torch.Tensor,
        candidate_actions: List[torch.Tensor],
    ) -> List[Tuple[int, float]]:
        """
        Score multiple candidate modifications WITHOUT executing them.

        This is the key API used by the refinement ranker. Given a hidden state
        and a list of candidate action vectors, returns ranked candidates by
        predicted improvement score.

        Args:
            h: (batch, d_model) backbone hidden state. Typically batch=1 for ranking.
            candidate_actions: List of (batch, d_action) tensors, one per candidate.

        Returns:
            List of (candidate_idx, predicted_improvement_score), sorted descending.
        """
        scores = []
        with torch.no_grad():
            for idx, action in enumerate(candidate_actions):
                arc_delta = self.forward(h, action)
                score = arc_delta.score()  # (batch,)
                # Average over batch (typically batch=1 for ranking)
                scores.append((idx, score.mean().item()))

        # Sort by score descending (higher = better predicted modification)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def get_regularization_losses(self) -> Dict[str, torch.Tensor]:
        """Forward Causeway's regularization losses for training."""
        return self.causeway.get_regularization_losses()

    def get_diagnostics(self) -> Dict:
        """Return adapter diagnostics."""
        n_adapter_params = sum(p.numel() for p in self.parameters())
        n_causeway_params = sum(p.numel() for p in self.causeway.parameters())
        result = {
            "adapter_total_params": n_adapter_params,
            "adapter_total_params_human": f"{n_adapter_params / 1e6:.2f}M",
            "causeway_params": n_causeway_params,
            "causeway_available": CAUSEWAY_AVAILABLE,
            "d_model": self.d_model,
            "d_causal": self.d_causal,
            "d_action": self.d_action,
        }
        if CAUSEWAY_AVAILABLE and hasattr(self.causeway, 'get_diagnostics'):
            result["causeway_diagnostics"] = self.causeway.get_diagnostics()
        return result
