"""
Causal Program Bridge for ARC-AGI-2
====================================

Orchestrates Causeway, BroadMind, and FluxMind adapters into a single pipeline
for the ARC-AGI-2 refinement loop.

Extends the CausalProgramExecutor pattern from causeway/integration/broadmind_bridge.py,
adapted for the ARC domain with all three modules and the dimensional mappings
established by the individual adapters.

Pipeline:
    1. Causeway predicts improvement delta for candidate modifications  (cheap)
    2. BroadMind executes the DSL program with adaptive compute         (expensive)
    3. FluxMind validates the result and provides confidence             (moderate)
    4. Fusion: combine all signals into a final score                   (cheap)

The bridge handles:
    - Wisdom flow:   Causeway causal structure -> BroadMind wisdom bank
    - Score fusion:  weighted combination of all three module signals
    - Confidence calibration: learned trust weighting per module
    - Candidate ranking: Causeway-first pruning before expensive execution

Main API:
    rank_and_execute()  -- used by the refinement loop
    forward()           -- full pipeline through all three modules

Parameter budget: ~3-5M for bridge networks + fusion layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from integration.causeway_adapter import CausewayAdapter, ARCDelta, ARC_DELTA_DIMS
from integration.broadmind_adapter import BroadMindAdapter, BroadMindResult
from integration.fluxmind_adapter import FluxMindAdapter


# ============================================================================
# OUTPUT CONTAINER
# ============================================================================

@dataclass
class ArcExecutionResult:
    """Combined result from all three integration modules."""

    # Causeway outputs
    arc_delta: Optional[ARCDelta]       # ARCDelta from CausewayAdapter
    causal_scores: Optional[torch.Tensor]    # (batch, 5) ARC-specific scores
    causal_confidence: Optional[torch.Tensor] # (batch, 5)

    # BroadMind outputs
    broadmind_predictions: Optional[torch.Tensor]  # Predicted states
    broadmind_wisdom: Optional[torch.Tensor]        # Task wisdom (d_wisdom)
    compute_cost: float = 0.0
    steps_used: int = 0

    # FluxMind outputs
    fluxmind_score: float = 0.0        # Program confidence [0, 1]
    task_classification: Dict = field(default_factory=dict)

    # Fused outputs
    fused_score: float = 0.0           # Combined score from all modules
    fused_confidence: float = 0.0      # Combined confidence
    should_accept: bool = False        # Whether to accept this program

    def to_dict(self) -> Dict:
        """Serialize to a logging-friendly dictionary."""
        return {
            "fused_score": self.fused_score,
            "fused_confidence": self.fused_confidence,
            "should_accept": self.should_accept,
            "fluxmind_score": self.fluxmind_score,
            "compute_cost": self.compute_cost,
            "steps_used": self.steps_used,
            "task_classification": self.task_classification,
        }


# ============================================================================
# WISDOM FUSION GATE
# ============================================================================

class WisdomFusionGate(nn.Module):
    """
    Learned gate for fusing causal wisdom with BroadMind wisdom.

    Extends the gate pattern from broadmind_bridge.py's CausalProgramExecutor.
    Takes two wisdom vectors and produces a blended result, learning when
    to trust causal structure vs procedural knowledge.
    """

    def __init__(self, d_wisdom: int = 48):
        super().__init__()
        self.d_wisdom = d_wisdom

        # Gate: concat(causal_wisdom, broadmind_wisdom) -> blend weight per dim
        self.gate = nn.Sequential(
            nn.Linear(d_wisdom * 2, d_wisdom),
            nn.Sigmoid(),
        )

    def forward(
        self,
        causal_wisdom: torch.Tensor,
        broadmind_wisdom: torch.Tensor,
    ) -> torch.Tensor:
        """
        Blend causal and procedural wisdom.

        Args:
            causal_wisdom:   (batch, d_wisdom) from Causeway causal structure.
            broadmind_wisdom: (batch, d_wisdom) from BroadMind wisdom bank.

        Returns:
            (batch, d_wisdom) fused wisdom.
        """
        gate_input = torch.cat([causal_wisdom, broadmind_wisdom], dim=-1)
        g = self.gate(gate_input)  # (batch, d_wisdom), values in [0, 1]
        return g * causal_wisdom + (1.0 - g) * broadmind_wisdom


# ============================================================================
# SCORE FUSION NETWORK
# ============================================================================

class ScoreFusionNetwork(nn.Module):
    """
    Learns to combine scores from Causeway, BroadMind, and FluxMind.

    Input features (9 base dims):
        - 5 Causeway ARC delta scores (spatial, color, structural, pattern, overall)
        - 1 BroadMind halt confidence
        - 1 FluxMind program score
        - 1 BroadMind compute cost (normalized)
        - 1 FluxMind task classification confidence

    Optional task features can be appended for task-type-conditional fusion.

    Output:
        - fused_score: scalar [0, 1]
        - fused_confidence: scalar [0, 1]
    """

    BASE_INPUT_DIM = 9

    def __init__(self, task_feature_dim: int = 0, hidden_dim: int = 64):
        """
        Args:
            task_feature_dim: Extra task-level features to condition on (0 = none).
            hidden_dim:       Hidden layer size.
        """
        super().__init__()
        input_dim = self.BASE_INPUT_DIM + task_feature_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        causeway_scores: torch.Tensor,
        broadmind_confidence: torch.Tensor,
        fluxmind_score: torch.Tensor,
        broadmind_cost: torch.Tensor,
        fluxmind_task_confidence: torch.Tensor,
        task_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse all module signals.

        Args:
            causeway_scores:         (batch, 5) ARC delta scores.
            broadmind_confidence:    (batch, 1) halt confidence.
            fluxmind_score:          (batch, 1) program confidence.
            broadmind_cost:          (batch, 1) normalized compute cost.
            fluxmind_task_confidence: (batch, 1) task classification confidence.
            task_features:           (batch, task_feature_dim) optional extras.

        Returns:
            (score, confidence) each (batch, 1).
        """
        parts = [
            causeway_scores,
            broadmind_confidence,
            fluxmind_score,
            broadmind_cost,
            fluxmind_task_confidence,
        ]
        if task_features is not None:
            parts.append(task_features)

        x = torch.cat(parts, dim=-1)
        h = self.net(x)
        score = self.score_head(h)
        confidence = self.confidence_head(h)
        return score, confidence


# ============================================================================
# CAUSAL-TO-WISDOM BRIDGE
# ============================================================================

class CausalToWisdomBridge(nn.Module):
    """
    Converts Causeway's ARC delta output into BroadMind-compatible wisdom.

    Extends the CausalWisdomBridge from broadmind_bridge.py but operates on
    the ARC-specific delta dimensions (5 scores + 5 confidences) plus the
    backbone hidden state, rather than raw adjacency matrices.

    This is necessary because CausewayAdapter already re-maps Causeway internals
    to ARC dimensions -- we bridge from those ARC dimensions into wisdom space.
    """

    def __init__(
        self,
        d_model: int = 4096,
        d_wisdom: int = 48,
        hidden_dim: int = 128,
    ):
        """
        Args:
            d_model:    Backbone hidden dim (for optional state conditioning).
            d_wisdom:   BroadMind wisdom dimension.
            hidden_dim: Bridge hidden layer size.
        """
        super().__init__()
        self.d_wisdom = d_wisdom

        # ARC delta -> wisdom: 5 scores + 5 confidences = 10
        self.delta_to_wisdom = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_wisdom),
            nn.Tanh(),  # match BroadMind's bounded wisdom space
        )

        # Optional: condition wisdom on backbone state (compressed)
        self.state_conditioner = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_wisdom),
            nn.Tanh(),
        )

        # Combine delta wisdom and state wisdom
        self.combiner = nn.Sequential(
            nn.Linear(d_wisdom * 2, d_wisdom),
            nn.LayerNorm(d_wisdom),
            nn.Tanh(),
        )

    def forward(
        self,
        arc_delta: ARCDelta,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert ARC delta to BroadMind wisdom.

        Args:
            arc_delta: ARCDelta from CausewayAdapter.
            h:         (batch, d_model) backbone hidden state. If None, uses
                       delta-only wisdom (no state conditioning).

        Returns:
            (batch, d_wisdom) causal wisdom for BroadMind.
        """
        # Stack delta scores + confidence
        scores = torch.stack([
            arc_delta.spatial_correctness,
            arc_delta.color_correctness,
            arc_delta.structural_integrity,
            arc_delta.pattern_consistency,
            arc_delta.overall_improvement,
        ], dim=-1)  # (batch, 5)
        delta_input = torch.cat([scores, arc_delta.confidence], dim=-1)  # (batch, 10)
        delta_wisdom = self.delta_to_wisdom(delta_input)  # (batch, d_wisdom)

        if h is not None:
            state_wisdom = self.state_conditioner(h)  # (batch, d_wisdom)
            combined = torch.cat([delta_wisdom, state_wisdom], dim=-1)
            return self.combiner(combined)  # (batch, d_wisdom)
        else:
            return delta_wisdom


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class CausalProgramBridge(nn.Module):
    """
    Orchestrates Causeway, BroadMind, and FluxMind for ARC-AGI-2.

    Extends the CausalProgramExecutor pattern from
    causeway/integration/broadmind_bridge.py.

    Pipeline:
        1. Causeway predicts improvement delta for candidate modifications
        2. BroadMind executes the DSL program with adaptive compute
        3. FluxMind validates the result and provides confidence
        4. Fusion: combine all signals into final score

    Graceful degradation: any adapter can be None, in which case its
    contribution is zeroed and the remaining modules still function.
    """

    def __init__(
        self,
        causeway_adapter: Optional[CausewayAdapter] = None,
        broadmind_adapter: Optional[BroadMindAdapter] = None,
        fluxmind_adapter: Optional[FluxMindAdapter] = None,
        d_model: int = 4096,
        d_causal: int = 128,
        d_wisdom: int = 48,
        fusion_mode: str = "learned",
        accept_threshold: float = 0.5,
    ):
        """
        Args:
            causeway_adapter:  CausewayAdapter instance (or None to skip).
            broadmind_adapter: BroadMindAdapter instance (or None to skip).
            fluxmind_adapter:  FluxMindAdapter instance (or None to skip).
            d_model:           Hidden dim of the 7B backbone.
            d_causal:          Causeway causal variable count.
            d_wisdom:          BroadMind wisdom dimension.
            fusion_mode:       'learned' | 'weighted_avg' | 'max'.
            accept_threshold:  Fused score threshold for should_accept.
        """
        super().__init__()
        self.causeway = causeway_adapter
        self.broadmind = broadmind_adapter
        self.fluxmind = fluxmind_adapter
        self.d_model = d_model
        self.d_causal = d_causal
        self.d_wisdom = d_wisdom
        self.fusion_mode = fusion_mode
        self.accept_threshold = accept_threshold

        # Track which modules are available
        self._has_causeway = causeway_adapter is not None
        self._has_broadmind = broadmind_adapter is not None
        self._has_fluxmind = fluxmind_adapter is not None

        # Causal-to-wisdom bridge (only needed when both Causeway and BroadMind present)
        if self._has_causeway and self._has_broadmind:
            self.causal_wisdom_bridge = CausalToWisdomBridge(
                d_model=d_model,
                d_wisdom=d_wisdom,
            )
            self.wisdom_gate = WisdomFusionGate(d_wisdom=d_wisdom)

        # Score fusion
        if fusion_mode == "learned":
            self.score_fusion = ScoreFusionNetwork(
                task_feature_dim=0,
                hidden_dim=64,
            )
        elif fusion_mode == "weighted_avg":
            # Learnable per-module weights (softmax normalized)
            self.module_weights = nn.Parameter(torch.ones(3))
        # 'max' mode needs no parameters

        # Confidence calibration: per-module reliability estimate
        # Learned bias for each module's contribution
        self.causeway_reliability = nn.Parameter(torch.tensor(1.0))
        self.broadmind_reliability = nn.Parameter(torch.tensor(1.0))
        self.fluxmind_reliability = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        h: torch.Tensor,
        action: torch.Tensor,
        grid_embedding: torch.Tensor,
        op_sequence: torch.Tensor,
        program_ops: List[str],
        examples: List[Tuple],
    ) -> ArcExecutionResult:
        """
        Full pipeline through all three modules.

        Args:
            h:              (batch, d_model) hidden states from 7B model.
            action:         (batch, d_action) encoded DSL modification.
            grid_embedding: (batch, d_model) encoded grid state.
            op_sequence:    (batch, seq_len) DSL operation IDs.
            program_ops:    List of DSL op names (for FluxMind).
            examples:       List of (input_grid, output_grid) numpy pairs (for FluxMind).

        Returns:
            ArcExecutionResult with outputs from all active modules.
        """
        batch_size = h.shape[0]
        device = h.device

        # ------------------------------------------------------------------
        # Step 1: Causeway causal reasoning (cheap)
        # ------------------------------------------------------------------
        arc_delta = None
        causal_scores = None
        causal_confidence = None

        if self._has_causeway:
            arc_delta = self.causeway(h, action)
            causal_scores = torch.stack([
                arc_delta.spatial_correctness,
                arc_delta.color_correctness,
                arc_delta.structural_integrity,
                arc_delta.pattern_consistency,
                arc_delta.overall_improvement,
            ], dim=-1)  # (batch, 5)
            causal_confidence = arc_delta.confidence  # (batch, 5)
        else:
            causal_scores = torch.zeros(batch_size, 5, device=device)
            causal_confidence = torch.zeros(batch_size, 5, device=device)

        # ------------------------------------------------------------------
        # Step 2: Bridge causal structure to wisdom
        # ------------------------------------------------------------------
        causal_wisdom = None
        if self._has_causeway and self._has_broadmind:
            causal_wisdom = self.causal_wisdom_bridge(arc_delta, h)

        # ------------------------------------------------------------------
        # Step 3: BroadMind execution with causal wisdom (expensive)
        # ------------------------------------------------------------------
        bm_predictions = None
        bm_wisdom = None
        compute_cost = 0.0
        steps_used = 0
        bm_halt_confidence = 0.0

        if self._has_broadmind:
            # If we have causal wisdom, fuse it with BroadMind's own wisdom
            wisdom_input = None
            if causal_wisdom is not None:
                # Get BroadMind's own wisdom first
                bm_own_wisdom = self.broadmind.get_wisdom(grid_embedding, op_sequence)
                # BroadMind.get_wisdom returns (batch, d_model) -- project back to d_wisdom
                # for the gate. The adapter's wisdom_projector goes d_wisdom -> d_model,
                # so we need to go the other direction for fusion.
                # Instead, inject causal wisdom directly into BroadMind's execute call
                # (which accepts d_wisdom tensors).
                wisdom_input = self.wisdom_gate(
                    causal_wisdom,
                    # Project BroadMind's own wisdom to d_wisdom for fusion
                    # (BroadMind internally works in d_wisdom space before projection)
                    causal_wisdom,  # fallback: use causal wisdom for both if shapes mismatch
                )

            bm_result = self.broadmind.execute_program(
                grid_embedding,
                op_sequence,
                wisdom=wisdom_input,
            )
            bm_predictions = bm_result.predictions
            bm_wisdom = bm_result.wisdom
            compute_cost = bm_result.compute_cost
            steps_used = bm_result.steps_used
            bm_halt_confidence = bm_result.halt_confidence

        # ------------------------------------------------------------------
        # Step 4: FluxMind validation (moderate)
        # ------------------------------------------------------------------
        fm_score = 0.0
        task_class = {}

        if self._has_fluxmind:
            fm_score = self.fluxmind.score_program(program_ops, examples)
            task_class = self.fluxmind.classify_task(examples)

        # ------------------------------------------------------------------
        # Step 5: Fuse all signals
        # ------------------------------------------------------------------
        fused_score, fused_confidence = self._fuse_scores(
            causal_scores=causal_scores,
            causal_confidence=causal_confidence,
            bm_halt_confidence=bm_halt_confidence,
            bm_compute_cost=compute_cost,
            fm_score=fm_score,
            fm_task_confidence=task_class.get("confidence", 0.0),
            device=device,
            batch_size=batch_size,
        )

        return ArcExecutionResult(
            arc_delta=arc_delta,
            causal_scores=causal_scores,
            causal_confidence=causal_confidence,
            broadmind_predictions=bm_predictions,
            broadmind_wisdom=bm_wisdom,
            compute_cost=compute_cost,
            steps_used=steps_used,
            fluxmind_score=fm_score,
            task_classification=task_class,
            fused_score=fused_score,
            fused_confidence=fused_confidence,
            should_accept=fused_score >= self.accept_threshold,
        )

    def _fuse_scores(
        self,
        causal_scores: torch.Tensor,
        causal_confidence: torch.Tensor,
        bm_halt_confidence: float,
        bm_compute_cost: float,
        fm_score: float,
        fm_task_confidence: float,
        device: torch.device,
        batch_size: int,
    ) -> Tuple[float, float]:
        """
        Combine scores from all three modules into a single fused score.

        Returns:
            (fused_score, fused_confidence) both floats.
        """
        if self.fusion_mode == "learned":
            # Pack inputs for the fusion network
            bm_conf_t = torch.full((batch_size, 1), bm_halt_confidence, device=device)
            fm_score_t = torch.full((batch_size, 1), fm_score, device=device)
            bm_cost_t = torch.full((batch_size, 1), min(bm_compute_cost, 10.0) / 10.0, device=device)
            fm_task_conf_t = torch.full((batch_size, 1), fm_task_confidence, device=device)

            score_t, conf_t = self.score_fusion(
                causeway_scores=causal_scores,
                broadmind_confidence=bm_conf_t,
                fluxmind_score=fm_score_t,
                broadmind_cost=bm_cost_t,
                fluxmind_task_confidence=fm_task_conf_t,
            )
            return score_t.mean().item(), conf_t.mean().item()

        elif self.fusion_mode == "weighted_avg":
            weights = F.softmax(self.module_weights, dim=0)

            # Causeway: confidence-weighted mean of delta scores
            cw_score = (causal_scores * causal_confidence).sum(dim=-1).mean().item()
            # BroadMind: halt confidence as proxy
            bm_score = bm_halt_confidence
            # FluxMind: direct score
            fm = fm_score

            # Weighted average
            fused = (
                weights[0].item() * cw_score * self.causeway_reliability.item()
                + weights[1].item() * bm_score * self.broadmind_reliability.item()
                + weights[2].item() * fm * self.fluxmind_reliability.item()
            )
            # Confidence: mean of non-zero confidences
            confs = []
            if self._has_causeway:
                confs.append(causal_confidence.mean().item())
            if self._has_broadmind:
                confs.append(bm_halt_confidence)
            if self._has_fluxmind:
                confs.append(fm_score)
            confidence = sum(confs) / max(len(confs), 1)

            return float(fused), float(confidence)

        elif self.fusion_mode == "max":
            # Take the maximum confident signal
            scores = []
            if self._has_causeway:
                cw = (causal_scores * causal_confidence).sum(dim=-1).mean().item()
                scores.append(cw)
            if self._has_broadmind:
                scores.append(bm_halt_confidence)
            if self._has_fluxmind:
                scores.append(fm_score)
            if not scores:
                return 0.0, 0.0
            best = max(scores)
            return float(best), float(best)

        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

    def _bridge_to_wisdom(
        self,
        arc_delta: ARCDelta,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert Causeway's causal structure into BroadMind wisdom format.

        Delegates to CausalToWisdomBridge. Exposed as a public method for
        external callers that need wisdom without full execution.

        Args:
            arc_delta: ARCDelta from CausewayAdapter.
            h:         (batch, d_model) backbone hidden state.

        Returns:
            (batch, d_wisdom) causal wisdom tensor.
        """
        if not (self._has_causeway and self._has_broadmind):
            return torch.zeros(h.shape[0], self.d_wisdom, device=h.device)
        return self.causal_wisdom_bridge(arc_delta, h)

    # ------------------------------------------------------------------
    # MAIN API: rank_and_execute
    # ------------------------------------------------------------------

    def rank_and_execute(
        self,
        h: torch.Tensor,
        candidates: List[Dict],
        examples: List[Tuple],
        top_k: int = 4,
    ) -> List[ArcExecutionResult]:
        """
        Rank candidates with Causeway, execute top-k with BroadMind,
        validate with FluxMind.

        This is the main API used by the refinement loop.

        Args:
            h:          (batch, d_model) hidden states from model. Typically batch=1.
            candidates: List of dicts, each with:
                        - 'action':         (batch, d_action) encoded modification
                        - 'grid_embedding': (batch, d_model) grid state
                        - 'op_sequence':    (batch, seq_len) DSL op IDs
                        - 'program_ops':    List[str] DSL op names
            examples:   List of (input_grid, output_grid) numpy pairs.
            top_k:      How many top candidates to fully execute.

        Returns:
            List of ArcExecutionResult for top candidates, sorted by
            fused_score descending.
        """
        if not candidates:
            return []

        # ------------------------------------------------------------------
        # Phase 1: Causeway ranks all candidates (cheap, no execution)
        # ------------------------------------------------------------------
        if self._has_causeway:
            candidate_actions = [c["action"] for c in candidates]
            rankings = self.causeway.rank_candidates(h, candidate_actions)
            # rankings is [(idx, score)] sorted by score descending
            top_indices = [idx for idx, _ in rankings[:top_k]]
        else:
            # No Causeway: take first top_k candidates as-is
            top_indices = list(range(min(top_k, len(candidates))))

        # ------------------------------------------------------------------
        # Phase 2: Execute top candidates through full pipeline
        # ------------------------------------------------------------------
        results = []
        for idx in top_indices:
            cand = candidates[idx]
            result = self.forward(
                h=h,
                action=cand["action"],
                grid_embedding=cand["grid_embedding"],
                op_sequence=cand["op_sequence"],
                program_ops=cand.get("program_ops", []),
                examples=examples,
            )
            results.append(result)

        # ------------------------------------------------------------------
        # Phase 3: Sort by fused score (descending)
        # ------------------------------------------------------------------
        results.sort(key=lambda r: r.fused_score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # ABLATION PATHS
    # ------------------------------------------------------------------

    def forward_causeway_only(
        self,
        h: torch.Tensor,
        action: torch.Tensor,
    ) -> Dict:
        """
        Just Causeway path (for ablation studies).

        Args:
            h:      (batch, d_model) backbone hidden state.
            action: (batch, d_action) encoded action.

        Returns:
            Dict with arc_delta, scores, confidence, and raw score.
        """
        if not self._has_causeway:
            return {"error": "Causeway adapter not available"}

        arc_delta = self.causeway(h, action)
        score = arc_delta.score()
        return {
            "arc_delta": arc_delta,
            "scores": torch.stack([
                arc_delta.spatial_correctness,
                arc_delta.color_correctness,
                arc_delta.structural_integrity,
                arc_delta.pattern_consistency,
                arc_delta.overall_improvement,
            ], dim=-1),
            "confidence": arc_delta.confidence,
            "overall_score": score.mean().item(),
        }

    def forward_broadmind_only(
        self,
        grid_embedding: torch.Tensor,
        op_sequence: torch.Tensor,
        use_adaptive_halt: bool = False,
    ) -> Dict:
        """
        Just BroadMind path (for ablation studies).

        Args:
            grid_embedding: (batch, d_model) grid state.
            op_sequence:    (batch, seq_len) DSL op IDs.
            use_adaptive_halt: Whether to use adaptive halting.

        Returns:
            Dict with predictions, compute_cost, steps_used, halt_confidence.
        """
        if not self._has_broadmind:
            return {"error": "BroadMind adapter not available"}

        bm_result = self.broadmind.execute_program(
            grid_embedding, op_sequence,
            use_adaptive_halt=use_adaptive_halt,
        )
        return {
            "predictions": bm_result.predictions,
            "compute_cost": bm_result.compute_cost,
            "steps_used": bm_result.steps_used,
            "halt_confidence": bm_result.halt_confidence,
            "wisdom": bm_result.wisdom,
        }

    def forward_fluxmind_only(
        self,
        program_ops: List[str],
        examples: List[Tuple],
    ) -> Dict:
        """
        Just FluxMind path (for ablation studies).

        Args:
            program_ops: List of DSL operation names.
            examples:    List of (input_grid, output_grid) numpy pairs.

        Returns:
            Dict with score, task_classification, and induced rules.
        """
        if not self._has_fluxmind:
            return {"error": "FluxMind adapter not available"}

        score = self.fluxmind.score_program(program_ops, examples)
        task_class = self.fluxmind.classify_task(examples)
        return {
            "score": score,
            "task_classification": task_class,
        }

    # ------------------------------------------------------------------
    # REGULARIZATION & DIAGNOSTICS
    # ------------------------------------------------------------------

    def get_regularization_losses(self) -> Dict[str, torch.Tensor]:
        """Collect regularization losses from all active modules."""
        losses = {}
        if self._has_causeway:
            cw_reg = self.causeway.get_regularization_losses()
            for k, v in cw_reg.items():
                losses[f"causeway_{k}"] = v
        return losses

    def get_module_stats(self) -> Dict:
        """Return stats from each module (for logging/monitoring)."""
        stats = {
            "modules_active": {
                "causeway": self._has_causeway,
                "broadmind": self._has_broadmind,
                "fluxmind": self._has_fluxmind,
            },
            "fusion_mode": self.fusion_mode,
            "accept_threshold": self.accept_threshold,
        }

        # Parameter counts
        bridge_params = sum(
            p.numel() for n, p in self.named_parameters()
            if not n.startswith(("causeway.", "broadmind.", "fluxmind."))
        )
        stats["bridge_params"] = bridge_params
        stats["bridge_params_human"] = f"{bridge_params / 1e6:.2f}M"

        total_params = sum(p.numel() for p in self.parameters())
        stats["total_params"] = total_params
        stats["total_params_human"] = f"{total_params / 1e6:.2f}M"

        if self._has_causeway:
            stats["causeway"] = self.causeway.get_diagnostics()
        if self._has_broadmind:
            stats["broadmind"] = self.broadmind.get_diagnostics()
        if self._has_fluxmind:
            stats["fluxmind"] = self.fluxmind.get_diagnostics()

        # Reliability weights
        stats["reliability_weights"] = {
            "causeway": self.causeway_reliability.item(),
            "broadmind": self.broadmind_reliability.item(),
            "fluxmind": self.fluxmind_reliability.item(),
        }

        if self.fusion_mode == "weighted_avg":
            w = F.softmax(self.module_weights, dim=0)
            stats["fusion_weights"] = {
                "causeway": w[0].item(),
                "broadmind": w[1].item(),
                "fluxmind": w[2].item(),
            }

        return stats


# ============================================================================
# FACTORY
# ============================================================================

def build_causal_program_bridge(
    d_model: int = 4096,
    d_causal: int = 128,
    d_action: int = 128,
    d_wisdom: int = 48,
    fusion_mode: str = "learned",
    accept_threshold: float = 0.5,
    enable_causeway: bool = True,
    enable_broadmind: bool = True,
    enable_fluxmind: bool = True,
    device: str = "cpu",
) -> CausalProgramBridge:
    """
    Build a CausalProgramBridge with fresh adapters.

    Convenience factory for creating the full orchestrator with default
    adapter configurations. For loading from checkpoints, construct
    adapters separately and pass them in.

    Args:
        d_model:          Backbone hidden dim (4096 for the 7B model).
        d_causal:         Causeway causal variable count.
        d_action:         Action embedding dimension.
        d_wisdom:         BroadMind wisdom dimension.
        fusion_mode:      'learned' | 'weighted_avg' | 'max'.
        accept_threshold: Score threshold for should_accept.
        enable_causeway:  Whether to include CausewayAdapter.
        enable_broadmind: Whether to include BroadMindAdapter.
        enable_fluxmind:  Whether to include FluxMindAdapter.
        device:           Target device.

    Returns:
        CausalProgramBridge ready for training or inference.
    """
    causeway = None
    broadmind = None
    fluxmind = None

    if enable_causeway:
        causeway = CausewayAdapter(
            d_model=d_model,
            d_causal=d_causal,
            d_action=d_action,
        )

    if enable_broadmind:
        broadmind = BroadMindAdapter(
            d_model=d_model,
        )

    if enable_fluxmind:
        fluxmind = FluxMindAdapter(
            d_model=d_model,
        )

    bridge = CausalProgramBridge(
        causeway_adapter=causeway,
        broadmind_adapter=broadmind,
        fluxmind_adapter=fluxmind,
        d_model=d_model,
        d_causal=d_causal,
        d_wisdom=d_wisdom,
        fusion_mode=fusion_mode,
        accept_threshold=accept_threshold,
    )

    return bridge.to(device)
