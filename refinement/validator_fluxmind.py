"""
FluxMindValidator — uses FluxMind to provide confidence scores for DSL programs
beyond exact grid match.

FluxMind's meta-learning can detect if a program's transformation pattern is
"on the right track" even when the output doesn't exactly match. This is critical
for the refinement loop: a program that gets 80% of cells right and has high
FluxMind confidence is a better starting point for refinement than one that gets
50% right.

Also provides task classification and DSL primitive suggestions to guide
the generator toward promising search directions.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.dsl_config import DSL_REGISTRY, DSLCategory

# Category -> suggested primitives mapping (broader than FluxMind's classify_task)
_CATEGORY_PRIMITIVES = {
    DSLCategory.SPATIAL: [
        "rot90", "rot180", "rot270", "hmirror", "vmirror", "transpose", "shift",
    ],
    DSLCategory.COLOR: [
        "fill", "recolor", "flood_fill", "color_map",
        "most_common_color", "least_common_color",
    ],
    DSLCategory.OBJECT: [
        "find_objects", "isolate_object", "extract_largest",
        "extract_smallest", "move_object",
    ],
    DSLCategory.GRID: [
        "crop", "trim", "pad", "concat_h", "concat_v", "tile", "resize", "overlay",
    ],
    DSLCategory.SET: [
        "intersection", "union", "difference", "xor",
    ],
    DSLCategory.PATTERN: [
        "detect_period", "extend_pattern", "symmetry_type",
    ],
}


class FluxMindValidator:
    """
    Uses FluxMind to provide confidence scores for DSL programs.

    Goes beyond exact grid match by:
    1. Scoring how well the program's transformation pattern matches examples
    2. Providing confidence that the program generalizes to unseen test inputs
    3. Detecting if the program is "on the right track" even if not perfect
    """

    def __init__(
        self,
        fluxmind_adapter=None,
        exact_match_bonus: float = 0.5,
        trust_threshold: float = 0.85,
    ):
        """
        Args:
            fluxmind_adapter: FluxMindAdapter instance. If None, falls back to
                              exact-match-only scoring (graceful degradation).
            exact_match_bonus: added to FluxMind score when execution produces
                               an exact grid match.
            trust_threshold: minimum combined score to trust a program.
        """
        self.fluxmind = fluxmind_adapter
        self.exact_match_bonus = exact_match_bonus
        self.trust_threshold = trust_threshold
        self._available = fluxmind_adapter is not None

    def score(
        self,
        program_ops: List[str],
        examples: List[Tuple[np.ndarray, np.ndarray]],
        execution_results: List = None,
    ) -> float:
        """
        Combined score: FluxMind confidence + exact match bonus.

        Scoring formula:
            base = FluxMind.score_program(ops, examples)   [0-1]
            bonus = exact_match_bonus * (exact_matches / total_examples)
            final = clamp(base + bonus, 0, 1)

        When FluxMind is unavailable, falls back to pure execution accuracy.

        Args:
            program_ops: list of DSL operation names in the program.
            examples: list of (input_grid, output_grid) demo pairs.
            execution_results: optional pre-computed ExecutionResult objects,
                               one per example. Used to determine exact matches
                               without re-executing.

        Returns:
            float in [0, 1]. Higher = better.
        """
        if not examples:
            return 0.0

        # Compute exact match ratio from execution results if provided
        exact_ratio = 0.0
        if execution_results is not None:
            exact_count = 0
            scored_count = 0
            for i, result in enumerate(execution_results):
                scored_count += 1
                if self._is_exact_match(result, examples[i][1]):
                    exact_count += 1
            exact_ratio = exact_count / max(scored_count, 1)

        # FluxMind confidence score
        if self._available and program_ops:
            try:
                fm_score = self.fluxmind.score_program(program_ops, examples)
            except Exception:
                # If FluxMind fails, fall back to execution-only scoring
                fm_score = 0.0
        else:
            # No FluxMind: use execution accuracy as the base score
            fm_score = self._execution_accuracy(execution_results, examples)

        # Combine: FluxMind base + exact match bonus
        bonus = self.exact_match_bonus * exact_ratio
        combined = min(1.0, fm_score + bonus)

        return combined

    def should_trust_program(
        self,
        score: float,
        exact_match_count: int,
        total_examples: int,
    ) -> bool:
        """
        Decide if a program is trustworthy enough to submit.

        Trust conditions (any one sufficient):
        1. All examples pass exact match
        2. Combined score >= trust_threshold AND majority of examples pass

        Args:
            score: combined score from self.score().
            exact_match_count: number of examples that pass exact grid match.
            total_examples: total number of demo examples.

        Returns:
            True if the program should be submitted as a solution.
        """
        if total_examples == 0:
            return False

        # Perfect: all examples pass
        if exact_match_count == total_examples:
            return True

        # High confidence + majority pass
        match_ratio = exact_match_count / total_examples
        if score >= self.trust_threshold and match_ratio > 0.5:
            return True

        return False

    def classify_and_suggest(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict:
        """
        Classify the task and suggest which DSL primitives to try.

        Uses FluxMind's task classification when available, otherwise falls back
        to heuristic analysis of grid properties.

        Args:
            examples: list of (input_grid, output_grid) demo pairs.

        Returns:
            {
                'task_type': str,
                'confidence': float,
                'suggested_ops': List[str],
            }
        """
        if not examples:
            return {
                'task_type': 'unknown',
                'confidence': 0.0,
                'suggested_ops': [],
            }

        # Use FluxMind classification if available
        if self._available:
            try:
                fm_result = self.fluxmind.classify_task(examples)
                task_type = fm_result.get('type', 'unknown')
                confidence = fm_result.get('confidence', 0.0)
                suggested = fm_result.get('suggested_primitives', [])

                # Augment FluxMind suggestions with heuristic analysis
                heuristic_ops = self._heuristic_suggest(examples)
                # Merge: FluxMind suggestions first, then heuristic ones not already present
                seen = set(suggested)
                for op in heuristic_ops:
                    if op not in seen:
                        suggested.append(op)
                        seen.add(op)

                return {
                    'task_type': task_type,
                    'confidence': confidence,
                    'suggested_ops': suggested,
                }
            except Exception:
                pass

        # Fallback: pure heuristic
        task_type, confidence = self._heuristic_classify(examples)
        suggested_ops = self._heuristic_suggest(examples)

        return {
            'task_type': task_type,
            'confidence': confidence,
            'suggested_ops': suggested_ops,
        }

    # ── Internal helpers ────────────────────────────────────────────────

    def _is_exact_match(
        self,
        execution_result,
        expected: np.ndarray,
    ) -> bool:
        """Check if an execution result exactly matches the expected output."""
        if execution_result is None:
            return False

        # Handle ExecutionResult dataclass
        actual = None
        if hasattr(execution_result, 'output_grid'):
            actual = execution_result.output_grid
        elif hasattr(execution_result, 'output'):
            actual = execution_result.output

        if actual is None:
            return False

        expected = np.asarray(expected, dtype=int)
        actual = np.asarray(actual, dtype=int)

        if expected.shape != actual.shape:
            return False

        return bool(np.array_equal(expected, actual))

    def _execution_accuracy(
        self,
        execution_results: List,
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> float:
        """Compute cell-level accuracy from execution results as fallback score."""
        if execution_results is None or not examples:
            return 0.0

        total_cells = 0
        matching_cells = 0

        for i, result in enumerate(execution_results):
            if i >= len(examples):
                break

            expected = np.asarray(examples[i][1], dtype=int)

            actual = None
            if hasattr(result, 'output_grid'):
                actual = result.output_grid
            elif hasattr(result, 'output'):
                out = result.output
                if isinstance(out, np.ndarray) and out.ndim == 2:
                    actual = out

            if actual is None:
                total_cells += expected.size
                continue

            actual = np.asarray(actual, dtype=int)

            if expected.shape == actual.shape:
                total_cells += expected.size
                matching_cells += int(np.sum(expected == actual))
            else:
                # Partial credit on overlapping region
                h = min(expected.shape[0], actual.shape[0])
                w = min(expected.shape[1], actual.shape[1])
                total_cells += max(expected.size, actual.size)
                matching_cells += int(np.sum(expected[:h, :w] == actual[:h, :w]))

        return matching_cells / max(total_cells, 1)

    def _heuristic_classify(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[str, float]:
        """
        Classify task type using grid property heuristics.

        Returns (task_type_str, confidence).
        """
        if not examples:
            return 'unknown', 0.0

        # Collect observations across examples
        same_shape_count = 0
        color_change_count = 0
        structure_same_count = 0
        size_changes = []

        for inp, out in examples:
            inp = np.asarray(inp, dtype=int)
            out = np.asarray(out, dtype=int)

            if inp.shape == out.shape:
                same_shape_count += 1
                # Check if structure (non-zero mask) is preserved
                if np.array_equal(inp != 0, out != 0):
                    structure_same_count += 1
                    color_change_count += 1
            else:
                h_ratio = out.shape[0] / max(inp.shape[0], 1)
                w_ratio = out.shape[1] / max(inp.shape[1], 1)
                size_changes.append((h_ratio, w_ratio))

        n = len(examples)

        # Pure color recoloring
        if same_shape_count == n and color_change_count == n:
            return 'color', 0.7

        # Same shape but structural changes -> spatial or object
        if same_shape_count == n:
            return 'spatial', 0.5

        # Size changes -> grid manipulation or pattern
        if size_changes:
            # Check for integer scaling
            ratios = size_changes
            all_integer = all(
                (abs(hr - round(hr)) < 0.01 and abs(wr - round(wr)) < 0.01)
                for hr, wr in ratios
            )
            if all_integer:
                return 'pattern', 0.5
            return 'grid', 0.4

        return 'unknown', 0.2

    def _heuristic_suggest(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> List[str]:
        """Suggest DSL primitives based on heuristic analysis of examples."""
        if not examples:
            return []

        suggested = []

        # Analyze first example pair for quick signals
        inp = np.asarray(examples[0][0], dtype=int)
        out = np.asarray(examples[0][1], dtype=int)

        # Same shape?
        if inp.shape == out.shape:
            # Check for simple spatial transforms
            if np.array_equal(np.rot90(inp), out):
                suggested.append("rot270")  # numpy rot90 is CCW = our rot270
            elif np.array_equal(np.rot90(inp, 2), out):
                suggested.append("rot180")
            elif np.array_equal(np.rot90(inp, 3), out):
                suggested.append("rot90")
            elif np.array_equal(np.fliplr(inp), out):
                suggested.append("hmirror")
            elif np.array_equal(np.flipud(inp), out):
                suggested.append("vmirror")
            elif np.array_equal(inp.T, out):
                suggested.append("transpose")

            # Check for recoloring
            if np.array_equal(inp != 0, out != 0):
                suggested.append("recolor")
                suggested.append("color_map")

        else:
            # Different shapes
            out_h, out_w = out.shape
            inp_h, inp_w = inp.shape

            # Output is smaller -> crop or trim
            if out_h <= inp_h and out_w <= inp_w:
                suggested.extend(["crop", "trim", "extract_largest"])

            # Output is larger -> pad, tile, extend
            if out_h >= inp_h and out_w >= inp_w:
                # Check for tiling
                if out_h % inp_h == 0 and out_w % inp_w == 0:
                    suggested.append("tile")
                suggested.extend(["pad", "extend_pattern", "concat_h", "concat_v"])

            # Output is transposed shape
            if out_h == inp_w and out_w == inp_h:
                suggested.append("transpose")

        # Add generic useful ops if list is too short
        if len(suggested) < 3:
            fallbacks = ["find_objects", "recolor", "crop", "overlay"]
            for fb in fallbacks:
                if fb not in suggested:
                    suggested.append(fb)
                if len(suggested) >= 5:
                    break

        return suggested
