"""
DSLValidator — run a DSLProgram on all demo examples and score it.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dsl.program import DSLProgram
from dsl.engine import DSLEngine, ExecutionResult


@dataclass
class ExampleResult:
    """Result for a single input/output example."""
    example_idx: int
    passed: bool                         # exact grid match
    cell_accuracy: float                 # fraction of matching cells (0-1)
    expected_shape: Tuple[int, int]
    actual_shape: Optional[Tuple[int, int]]
    execution_result: ExecutionResult
    error: Optional[str] = None


@dataclass
class ValidationResult:
    """Aggregated validation result across all examples."""
    score: float                         # fraction of exactly matching outputs (0-1)
    cell_accuracy: float                 # average cell-level accuracy
    per_example: List[ExampleResult]
    all_passed: bool
    num_passed: int
    num_total: int
    error_details: List[str] = field(default_factory=list)


class DSLValidator:
    """
    Validate a DSLProgram against a set of (input_grid, output_grid) examples.

    Usage:
        validator = DSLValidator(timeout_ms=1000)
        result = validator.validate(program, examples)
    """

    def __init__(self, timeout_ms: float = 1000.0):
        self.engine = DSLEngine(timeout_ms=timeout_ms)

    def validate(
        self,
        program: DSLProgram,
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> ValidationResult:
        """
        Run program on all examples and compute scores.

        Args:
            program: DSLProgram to execute
            examples: list of (input_grid, expected_output_grid) pairs

        Returns:
            ValidationResult with exact-match score, cell accuracy, per-example details
        """
        per_example: List[ExampleResult] = []
        error_details: List[str] = []

        for idx, (inp, expected) in enumerate(examples):
            inp = np.asarray(inp, dtype=int)
            expected = np.asarray(expected, dtype=int)

            exec_result = self.engine.execute(program, inp)

            if not exec_result.success:
                per_example.append(ExampleResult(
                    example_idx=idx,
                    passed=False,
                    cell_accuracy=0.0,
                    expected_shape=tuple(expected.shape),
                    actual_shape=None,
                    execution_result=exec_result,
                    error=exec_result.error,
                ))
                error_details.append(
                    f"Example {idx}: execution failed — {exec_result.error}"
                )
                continue

            actual = exec_result.output_grid
            if actual is None:
                per_example.append(ExampleResult(
                    example_idx=idx,
                    passed=False,
                    cell_accuracy=0.0,
                    expected_shape=tuple(expected.shape),
                    actual_shape=None,
                    execution_result=exec_result,
                    error="Output is not a 2D grid",
                ))
                error_details.append(
                    f"Example {idx}: output is not a 2D grid "
                    f"(got {type(exec_result.output).__name__})"
                )
                continue

            passed, cell_acc = self._compare_grids(expected, actual)

            ex_result = ExampleResult(
                example_idx=idx,
                passed=passed,
                cell_accuracy=cell_acc,
                expected_shape=tuple(expected.shape),
                actual_shape=tuple(actual.shape),
                execution_result=exec_result,
            )

            if not passed:
                if expected.shape != actual.shape:
                    msg = (f"Example {idx}: shape mismatch — "
                           f"expected {expected.shape}, got {actual.shape}")
                else:
                    diff = np.sum(expected != actual)
                    total = expected.size
                    msg = (f"Example {idx}: {diff}/{total} cells differ "
                           f"(cell accuracy: {cell_acc:.2%})")
                error_details.append(msg)
                ex_result.error = msg

            per_example.append(ex_result)

        num_passed = sum(1 for r in per_example if r.passed)
        num_total = len(examples)
        score = num_passed / num_total if num_total > 0 else 0.0

        cell_accs = [r.cell_accuracy for r in per_example]
        avg_cell_acc = sum(cell_accs) / len(cell_accs) if cell_accs else 0.0

        return ValidationResult(
            score=score,
            cell_accuracy=avg_cell_acc,
            per_example=per_example,
            all_passed=(num_passed == num_total),
            num_passed=num_passed,
            num_total=num_total,
            error_details=error_details,
        )

    def _compare_grids(
        self, expected: np.ndarray, actual: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Compare two grids. Returns (exact_match, cell_accuracy).
        Cell accuracy is computed on the intersection of shapes (0 for mismatched cells).
        """
        if expected.shape == actual.shape:
            exact = bool(np.array_equal(expected, actual))
            total = expected.size
            matching = int(np.sum(expected == actual)) if total > 0 else 0
            cell_acc = matching / total if total > 0 else 1.0
            return exact, cell_acc

        # Different shapes: partial credit on overlapping region
        h = min(expected.shape[0], actual.shape[0])
        w = min(expected.shape[1], actual.shape[1])
        total = max(expected.size, actual.size)
        if total == 0:
            return True, 1.0

        overlap_match = int(np.sum(expected[:h, :w] == actual[:h, :w]))
        cell_acc = overlap_match / total
        return False, cell_acc
