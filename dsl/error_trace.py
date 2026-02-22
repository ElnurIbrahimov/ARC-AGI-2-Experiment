"""
Structured error feedback for program refinement.

Compares expected vs actual grids and produces a detailed ErrorTrace
that can be fed back to the model for self-correction.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dsl.validator import ValidationResult, ExampleResult


@dataclass
class ErrorTrace:
    """Detailed error information for one example."""
    example_idx: int
    expected: np.ndarray
    actual: Optional[np.ndarray]
    diff_mask: Optional[np.ndarray]        # boolean mask of differing cells
    diff_colors: Dict[Tuple[int, int], Tuple[int, int]]  # (r,c) -> (expected, actual)
    suggested_category: str                # "color_error", "spatial_error", etc.
    summary: str                           # human-readable description

    def diff_count(self) -> int:
        if self.diff_mask is None:
            return -1
        return int(self.diff_mask.sum())


def classify_error(
    expected: np.ndarray,
    actual: Optional[np.ndarray],
) -> str:
    """
    Classify the type of error:
      - "execution_error": actual is None (program crashed)
      - "size_error": shapes differ
      - "color_error": same shape, only color values wrong
      - "spatial_error": pattern looks shifted/rotated/reflected
      - "structural_error": fundamental structure difference
    """
    if actual is None:
        return "execution_error"

    if expected.shape != actual.shape:
        return "size_error"

    diff = expected != actual
    if not np.any(diff):
        return "no_error"

    # Check if it's purely a color remapping
    expected_colors = set(np.unique(expected).tolist())
    actual_colors = set(np.unique(actual).tolist())
    # If the non-zero structure matches but colors differ -> color error
    expected_struct = expected != 0
    actual_struct = actual != 0
    if np.array_equal(expected_struct, actual_struct):
        return "color_error"

    # Check for spatial transform: compare sorted pixel coordinates
    expected_nz = set(zip(*np.where(expected != 0)))
    actual_nz = set(zip(*np.where(actual != 0)))

    if len(expected_nz) == len(actual_nz):
        # Same number of non-zero cells — likely a spatial transform
        # Check if it could be a simple shift
        if expected_nz and actual_nz:
            e_arr = np.array(sorted(expected_nz))
            a_arr = np.array(sorted(actual_nz))
            deltas = a_arr - e_arr
            if np.all(deltas == deltas[0]):
                return "spatial_error"  # uniform shift

        return "spatial_error"

    return "structural_error"


def build_error_trace(
    example_idx: int,
    expected: np.ndarray,
    actual: Optional[np.ndarray],
) -> ErrorTrace:
    """Build a detailed ErrorTrace for one example."""
    expected = np.asarray(expected, dtype=int)

    if actual is None:
        return ErrorTrace(
            example_idx=example_idx,
            expected=expected,
            actual=None,
            diff_mask=None,
            diff_colors={},
            suggested_category="execution_error",
            summary=f"Example {example_idx}: program did not produce a grid output.",
        )

    actual = np.asarray(actual, dtype=int)
    category = classify_error(expected, actual)

    if expected.shape != actual.shape:
        return ErrorTrace(
            example_idx=example_idx,
            expected=expected,
            actual=actual,
            diff_mask=None,
            diff_colors={},
            suggested_category=category,
            summary=(
                f"Example {example_idx}: shape mismatch — "
                f"expected {expected.shape}, got {actual.shape}."
            ),
        )

    diff_mask = expected != actual
    diff_colors: Dict[Tuple[int, int], Tuple[int, int]] = {}
    rows, cols = np.where(diff_mask)
    for r, c in zip(rows.tolist(), cols.tolist()):
        diff_colors[(r, c)] = (int(expected[r, c]), int(actual[r, c]))

    n_diff = int(diff_mask.sum())
    total = expected.size
    pct = n_diff / total * 100 if total > 0 else 0

    if category == "color_error":
        # Identify which color mappings are wrong
        wrong_mappings = {}
        for (r, c), (exp_c, act_c) in diff_colors.items():
            key = (exp_c, act_c)
            wrong_mappings[key] = wrong_mappings.get(key, 0) + 1
        mapping_str = ", ".join(
            f"{exp}->{act} ({cnt}x)" for (exp, act), cnt in
            sorted(wrong_mappings.items(), key=lambda x: -x[1])[:5]
        )
        summary = (
            f"Example {example_idx}: {n_diff}/{total} cells have wrong colors "
            f"({pct:.1f}%). Wrong mappings: {mapping_str}"
        )
    elif category == "spatial_error":
        summary = (
            f"Example {example_idx}: {n_diff}/{total} cells differ ({pct:.1f}%). "
            f"Structure (non-zero pattern) differs — likely a spatial transform error."
        )
    else:
        summary = (
            f"Example {example_idx}: {n_diff}/{total} cells differ ({pct:.1f}%). "
            f"Structural mismatch in grid content."
        )

    return ErrorTrace(
        example_idx=example_idx,
        expected=expected,
        actual=actual,
        diff_mask=diff_mask,
        diff_colors=diff_colors,
        suggested_category=category,
        summary=summary,
    )


def build_error_traces_from_validation(
    validation: ValidationResult,
    examples: List[Tuple[np.ndarray, np.ndarray]],
) -> List[ErrorTrace]:
    """
    Build error traces for all failing examples from a ValidationResult.
    """
    traces: List[ErrorTrace] = []
    for ex_result in validation.per_example:
        if ex_result.passed:
            continue
        idx = ex_result.example_idx
        expected = np.asarray(examples[idx][1], dtype=int)
        actual = ex_result.execution_result.output_grid
        traces.append(build_error_trace(idx, expected, actual))
    return traces


def format_error_traces(traces: List[ErrorTrace]) -> str:
    """Format error traces into a human-readable summary string."""
    if not traces:
        return "All examples passed."
    lines = [f"=== {len(traces)} failing example(s) ==="]
    for t in traces:
        lines.append(f"\n[{t.suggested_category}] {t.summary}")
        if t.diff_mask is not None:
            lines.append(f"  Differing cells: {t.diff_count()}")
    return "\n".join(lines)
