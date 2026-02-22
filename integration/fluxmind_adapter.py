"""
FluxMind Adapter for ARC-AGI-2
===============================

Wraps HybridFluxMind v0.79 for ARC program validation and task classification.

HybridFluxMind combines:
  - v0.76.4 Fixed DSLs (99% accuracy on known patterns via DSL embedding lookup)
  - v0.78.1 Meta-Learning (learns new patterns from examples via attention)
  - DSLClassifier (routes to fixed or meta backend based on prototype matching)

For ARC-AGI-2, we use FluxMind's meta-learning capability to:
  1. Score how well a DSL program matches observed input-output patterns
  2. Provide confidence estimates beyond exact match
  3. Classify which "type" of transformation a task involves
  4. Induce transformation rules from examples

FluxMind operates in a discrete state space:
  state_dim=4, state_range=15, num_operations=8

We map ARC grids to this space by extracting key features (dominant color,
object count, grid size ratio, symmetry score) and map our 50 DSL ops
to FluxMind's 8-op space via learned category clustering.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Try importing FluxMind from source
try:
    sys.path.insert(0, r'C:\Users\asus\Desktop\FluxMind paper MetaLearning\files')
    from hybrid_fluxmind import HybridFluxMind, HybridConfig, RoutingDecision
    FLUXMIND_AVAILABLE = True
except ImportError:
    FLUXMIND_AVAILABLE = False

    @dataclass
    class HybridConfig:
        """Stub config matching HybridFluxMind defaults."""
        state_dim: int = 4
        state_range: int = 15
        num_operations: int = 8
        num_known_dsls: int = 5
        known_dsl_names: tuple = (
            "Additive", "Multiplicative", "XOR", "MinMax", "Modular"
        )
        classifier_embed_dim: int = 128
        classifier_hidden_dim: int = 256
        prototype_dim: int = 128
        routing_confidence_threshold: float = 0.85
        min_examples_for_classification: int = 8
        meta_example_embed_dim: int = 192
        meta_context_dim: int = 192
        meta_hidden_dim: int = 384
        meta_num_attention_heads: int = 6
        fixed_latent_dim: int = 128
        fixed_hidden_dim: int = 256

    class RoutingDecision:
        FIXED = "fixed"
        META = "meta"
        UNCERTAIN = "uncertain"

    class HybridFluxMind(nn.Module):
        """Stub HybridFluxMind when source is not available."""
        def __init__(self, config=None):
            super().__init__()
            self.config = config or HybridConfig()
            self._dummy = nn.Linear(1, 1)

        def step(self, state, op, examples, force_backend=None):
            return {
                'next_state': state,
                'confidence': 0.5,
                'backend': 'stub',
                'routing': {'decision': 'stub', 'matched_dsl': None,
                            'classification_confidence': 0.0}
            }

        def classify_examples(self, examples):
            return RoutingDecision.META, -1, 0.0

        def induce_dsl(self, examples, name="induced"):
            return _StubInducedDSL(self, examples, name)

    class _StubInducedDSL:
        def __init__(self, model, examples, name):
            self.model = model
            self.examples = examples
            self.name = name
        def step(self, state, op):
            return {'next_state': state, 'confidence': 0.5, 'backend': 'stub'}
        def execute(self, state, ops):
            return {
                'trajectory': [state], 'final_state': state,
                'confidences': [0.5] * len(ops),
                'mean_confidence': 0.5, 'should_trust': False,
                'backend': 'stub', 'matched_dsl': None,
            }


# ─── DSL operation categories for mapping to FluxMind's 8 ops ───────

# We cluster our 50 DSL ops into 8 categories that map to FluxMind's operations.
# This mirrors FluxMind's internal op structure where ops within a category
# share similar state-transition patterns.
DSL_OP_CATEGORIES = {
    # Category 0: Spatial transforms (rotations, mirrors)
    "rot90": 0, "rot180": 0, "rot270": 0,
    "hmirror": 0, "vmirror": 0, "transpose": 0, "shift": 0,
    # Category 1: Color operations
    "fill": 1, "recolor": 1, "flood_fill": 1, "color_map": 1,
    "most_common_color": 1, "least_common_color": 1, "count_color": 1,
    # Category 2: Object operations
    "find_objects": 2, "isolate_object": 2, "extract_largest": 2,
    "extract_smallest": 2, "bounding_box": 2, "move_object": 2, "count_objects": 2,
    # Category 3: Grid manipulation
    "crop": 3, "trim": 3, "pad": 3, "concat_h": 3,
    "concat_v": 3, "tile": 3, "resize": 3, "overlay": 3,
    # Category 4: Set operations
    "intersection": 4, "union": 4, "difference": 4, "xor": 4,
    # Category 5: Logic/filtering
    "if_color": 5, "filter_by_size": 5, "select_by_color": 5,
    # Category 6: Composition
    "sequence": 6, "iterate_until_fixpoint": 6, "apply_to_each": 6,
    # Category 7: Pattern operations
    "detect_period": 7, "extend_pattern": 7, "symmetry_type": 7,
}


# ─── Adapter ────────────────────────────────────────────────────────

class FluxMindAdapter(nn.Module):
    """
    Wraps HybridFluxMind for ARC-AGI-2 program validation.

    Uses FluxMind's meta-learning to:
    - Score DSL programs against input-output examples
    - Classify task types (spatial, color, object, etc.)
    - Induce transformation rules from examples

    FluxMind operates in a compressed state space (state_dim=4, values 1-15).
    We extract grid features to map ARC grids into this space.

    Parameter budget: ~1-2M (grid encoder + op mapper + FluxMind itself).
    """

    def __init__(
        self,
        d_model: int = 4096,
        state_dim: int = 4,
        state_range: int = 15,
        num_operations: int = 8,
        fluxmind_config: object = None,
    ):
        """
        Args:
            d_model:        Hidden dim of our 7B backbone.
            state_dim:      FluxMind state dimensionality (4).
            state_range:    Max state value (15).
            num_operations: FluxMind operation count (8).
            fluxmind_config: HybridConfig object. None uses defaults.
        """
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.state_range = state_range
        self.num_operations = num_operations

        # Initialize FluxMind
        self.fm_config = fluxmind_config or HybridConfig(
            state_dim=state_dim,
            state_range=state_range,
            num_operations=num_operations,
        )
        self.fluxmind = HybridFluxMind(self.fm_config)

        # Grid feature encoder: extract state_dim features from a raw ARC grid.
        # This converts a variable-size grid into FluxMind's fixed state space.
        # Uses a small neural net on hand-crafted features for robustness.
        # Input features: 10 color counts + 4 spatial features = 14
        grid_feature_dim = 14
        self.grid_feature_net = nn.Sequential(
            nn.Linear(grid_feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, state_dim),
        )

        # For backbone-space grid embeddings (d_model -> state_dim)
        self.embedding_to_state = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, state_dim),
        )

        # Op name -> FluxMind op ID mapping (static, from DSL_OP_CATEGORIES)
        self._op_map = DSL_OP_CATEGORIES.copy()

    def _extract_grid_features(self, grid: np.ndarray) -> List[float]:
        """
        Extract features from a raw ARC grid for FluxMind's state space.

        Features (14-dim):
          - Color histogram: 10 values (normalized count of each color 0-9)
          - Grid height / 30.0 (normalized)
          - Grid width / 30.0 (normalized)
          - Object count estimate (unique non-zero colors)
          - Symmetry score (horizontal mirror similarity)

        Args:
            grid: 2D numpy array, values 0-9.

        Returns:
            List of 14 floats.
        """
        grid = np.asarray(grid)
        h, w = grid.shape
        total_cells = max(h * w, 1)

        # Color histogram (normalized)
        color_counts = np.zeros(10, dtype=np.float32)
        for c in range(10):
            color_counts[c] = np.sum(grid == c) / total_cells

        # Spatial features
        height_norm = h / 30.0
        width_norm = w / 30.0

        # Object count estimate: number of distinct non-zero colors
        unique_nonzero = len(set(grid.flatten()) - {0})
        obj_count_norm = unique_nonzero / 9.0  # max 9 non-zero colors

        # Symmetry score: horizontal mirror similarity
        flipped = np.fliplr(grid)
        symmetry = np.mean(grid == flipped)

        features = list(color_counts) + [height_norm, width_norm, obj_count_norm, symmetry]
        return features

    def encode_grid_as_state(self, grid: np.ndarray) -> List[int]:
        """
        Compress a grid into FluxMind's discrete state representation.

        Extracts features, runs through grid_feature_net, then quantizes
        to integers in [1, state_range].

        Args:
            grid: 2D numpy array, values 0-9.

        Returns:
            List of state_dim integers, each in [1, state_range].
        """
        features = self._extract_grid_features(grid)
        features_t = torch.tensor([features], dtype=torch.float32)

        device = next(self.grid_feature_net.parameters()).device
        features_t = features_t.to(device)

        with torch.no_grad():
            raw_state = self.grid_feature_net(features_t)  # (1, state_dim)
            # Sigmoid -> scale to [1, state_range] -> round to int
            scaled = torch.sigmoid(raw_state) * (self.state_range - 1) + 1
            quantized = scaled.round().long().clamp(1, self.state_range)

        return quantized[0].cpu().tolist()

    def encode_grid_from_embedding(self, grid_embedding: torch.Tensor) -> List[int]:
        """
        Compress a backbone grid embedding into FluxMind's state space.

        Args:
            grid_embedding: (d_model,) or (1, d_model) tensor.

        Returns:
            List of state_dim integers, each in [1, state_range].
        """
        if grid_embedding.dim() == 1:
            grid_embedding = grid_embedding.unsqueeze(0)

        with torch.no_grad():
            raw_state = self.embedding_to_state(grid_embedding)
            scaled = torch.sigmoid(raw_state) * (self.state_range - 1) + 1
            quantized = scaled.round().long().clamp(1, self.state_range)

        return quantized[0].cpu().tolist()

    def encode_dsl_op(self, op_name: str) -> int:
        """
        Map a DSL operation name to FluxMind's operation space (0-7).

        Args:
            op_name: DSL primitive name (e.g., "rot90", "fill", "crop").

        Returns:
            int in [0, num_operations-1].
        """
        return self._op_map.get(op_name, 0)  # Default to category 0 (spatial)

    def score_program(
        self,
        program_ops: List[str],
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> float:
        """
        Score a DSL program against input-output examples using FluxMind.

        Converts each example pair to (state_before, op, state_after) format,
        then uses FluxMind's step() to predict state_after. Compares prediction
        confidence across all ops and examples.

        Args:
            program_ops: List of DSL operation names (e.g., ["rot90", "recolor"]).
            examples:    List of (input_grid, output_grid) numpy arrays.

        Returns:
            float in [0, 1] — confidence score. Higher = better program match.
        """
        if not program_ops or not examples:
            return 0.0

        # Build FluxMind-compatible examples from the input-output pairs
        fm_examples = []
        for input_grid, output_grid in examples:
            state_before = self.encode_grid_as_state(input_grid)
            state_after = self.encode_grid_as_state(output_grid)
            # Use the first op in the program as the representative op
            op_id = self.encode_dsl_op(program_ops[0])
            fm_examples.append((state_before, op_id, state_after))

        if len(fm_examples) < 2:
            # FluxMind needs multiple examples for reliable meta-learning
            return 0.0

        # Score: for each example, use the others as support and predict
        total_confidence = 0.0
        n_scored = 0

        for i, (state_before, op_id, state_after_true) in enumerate(fm_examples):
            # Support set: all examples except the current one
            support = [ex for j, ex in enumerate(fm_examples) if j != i]

            # Test each op in the program sequence
            current_state = state_before
            step_confidences = []

            for op_name in program_ops:
                fm_op = self.encode_dsl_op(op_name)
                result = self.fluxmind.step(current_state, fm_op, support)
                step_confidences.append(result['confidence'])
                current_state = result['next_state']

            # Compare final predicted state to true output state
            predicted = current_state
            match_score = sum(
                1.0 for a, b in zip(predicted, state_after_true) if a == b
            ) / self.state_dim

            # Combine step confidence with match accuracy
            mean_step_conf = sum(step_confidences) / len(step_confidences)
            example_score = 0.5 * mean_step_conf + 0.5 * match_score

            total_confidence += example_score
            n_scored += 1

        return total_confidence / max(n_scored, 1)

    def classify_task(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict:
        """
        Classify the task type using FluxMind's DSL classifier.

        Builds a set of (state_before, op, state_after) transitions from
        the examples, then uses FluxMind's classify_examples() to determine
        which known DSL pattern (if any) the task matches.

        Args:
            examples: List of (input_grid, output_grid) numpy arrays.

        Returns:
            Dict with:
              - 'type': str — matched DSL name or 'unknown'
              - 'confidence': float — classification confidence
              - 'suggested_primitives': List[str] — likely DSL ops for this task type
        """
        if not examples:
            return {
                'type': 'unknown',
                'confidence': 0.0,
                'suggested_primitives': [],
            }

        # Build transition examples by trying all op categories
        fm_examples = []
        for input_grid, output_grid in examples:
            state_before = self.encode_grid_as_state(input_grid)
            state_after = self.encode_grid_as_state(output_grid)
            # Try each op category (0-7) and include all as examples
            for op_id in range(self.num_operations):
                fm_examples.append((state_before, op_id, state_after))

        # Classify
        decision, dsl_id, confidence = self.fluxmind.classify_examples(fm_examples)

        # Map classification result to task type and suggested primitives
        task_type = 'unknown'
        suggested = []

        if hasattr(decision, 'value'):
            decision_val = decision.value if hasattr(decision, 'value') else decision
        else:
            decision_val = str(decision)

        if dsl_id >= 0 and dsl_id < len(self.fm_config.known_dsl_names):
            task_type = self.fm_config.known_dsl_names[dsl_id]

        # Suggest primitives based on task type pattern
        type_to_primitives = {
            'Additive': ['fill', 'recolor', 'pad', 'extend_pattern'],
            'Multiplicative': ['tile', 'resize', 'overlay'],
            'XOR': ['xor', 'difference', 'intersection'],
            'MinMax': ['extract_largest', 'extract_smallest', 'filter_by_size'],
            'Modular': ['detect_period', 'extend_pattern', 'iterate_until_fixpoint'],
        }
        suggested = type_to_primitives.get(task_type, [])

        return {
            'type': task_type,
            'confidence': confidence,
            'suggested_primitives': suggested,
            'routing_decision': decision_val,
        }

    def induce_rules(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        name: str = "arc_task",
    ) -> object:
        """
        Use FluxMind's induce_dsl() to learn transformation rules from examples.

        Returns a HybridInducedDSL object that can predict transformations
        for new inputs.

        Args:
            examples: List of (input_grid, output_grid) numpy arrays.
            name:     Name for the induced DSL.

        Returns:
            Induced DSL object with .step() and .execute() methods.
        """
        # Convert ARC examples to FluxMind format
        fm_examples = []
        for input_grid, output_grid in examples:
            state_before = self.encode_grid_as_state(input_grid)
            state_after = self.encode_grid_as_state(output_grid)
            # Use op=0 as default since ARC tasks don't specify individual ops
            fm_examples.append((state_before, 0, state_after))

        return self.fluxmind.induce_dsl(fm_examples, name)

    def score_program_from_embeddings(
        self,
        program_ops: List[str],
        input_embeddings: List[torch.Tensor],
        output_embeddings: List[torch.Tensor],
    ) -> float:
        """
        Score using backbone embeddings instead of raw grids.

        Convenience method when grids are already encoded by the backbone.

        Args:
            program_ops:       List of DSL operation names.
            input_embeddings:  List of (d_model,) tensors for input grids.
            output_embeddings: List of (d_model,) tensors for output grids.

        Returns:
            float in [0, 1] confidence score.
        """
        fm_examples = []
        for inp_emb, out_emb in zip(input_embeddings, output_embeddings):
            state_before = self.encode_grid_from_embedding(inp_emb)
            state_after = self.encode_grid_from_embedding(out_emb)
            op_id = self.encode_dsl_op(program_ops[0]) if program_ops else 0
            fm_examples.append((state_before, op_id, state_after))

        if len(fm_examples) < 2:
            return 0.0

        total_confidence = 0.0
        n_scored = 0

        for i, (state_before, op_id, state_after_true) in enumerate(fm_examples):
            support = [ex for j, ex in enumerate(fm_examples) if j != i]
            current_state = state_before

            for op_name in program_ops:
                fm_op = self.encode_dsl_op(op_name)
                result = self.fluxmind.step(current_state, fm_op, support)
                current_state = result['next_state']

            match_score = sum(
                1.0 for a, b in zip(current_state, state_after_true) if a == b
            ) / self.state_dim

            total_confidence += match_score
            n_scored += 1

        return total_confidence / max(n_scored, 1)

    def get_diagnostics(self) -> Dict:
        """Return adapter diagnostics."""
        n_adapter_params = sum(p.numel() for p in self.parameters())
        n_fluxmind_params = sum(p.numel() for p in self.fluxmind.parameters())
        return {
            "adapter_total_params": n_adapter_params,
            "adapter_total_params_human": f"{n_adapter_params / 1e6:.2f}M",
            "fluxmind_params": n_fluxmind_params,
            "fluxmind_available": FLUXMIND_AVAILABLE,
            "state_dim": self.state_dim,
            "state_range": self.state_range,
            "num_operations": self.num_operations,
            "d_model": self.d_model,
        }
