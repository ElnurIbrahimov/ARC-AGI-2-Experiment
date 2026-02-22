"""
DSLEngine — deterministic symbolic executor for DSLProgram trees.

Walks the tree, calls primitives, tracks execution trace, handles errors.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import time
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dsl.program import DSLProgram, DSLNode
from dsl.primitives import PRIMITIVE_FUNCTIONS


@dataclass
class TraceStep:
    """One step of execution."""
    op: str
    args_summary: List[str]      # brief description of each arg
    result_summary: str           # brief description of result
    time_ms: float


@dataclass
class ExecutionResult:
    """Result of executing a DSLProgram."""
    output: Any                   # usually np.ndarray, but could be int/list/etc
    success: bool
    error: Optional[str] = None
    trace: List[TraceStep] = field(default_factory=list)
    total_time_ms: float = 0.0

    @property
    def output_grid(self) -> Optional[np.ndarray]:
        """Return output as a grid if it is one, else None."""
        if isinstance(self.output, np.ndarray) and self.output.ndim == 2:
            return self.output
        return None


class ExecutionTimeout(Exception):
    pass


class DSLEngine:
    """
    Deterministic symbolic executor.

    Usage:
        engine = DSLEngine(timeout_ms=1000)
        result = engine.execute(program, input_grid)
    """

    def __init__(self, timeout_ms: float = 1000.0):
        self.timeout_ms = timeout_ms
        self._start_time: float = 0.0
        self._trace: List[TraceStep] = []

    def execute(self, program: DSLProgram, input_grid: np.ndarray) -> ExecutionResult:
        """
        Execute a DSLProgram on an input grid.
        Returns ExecutionResult with output, success flag, errors, and trace.
        """
        self._start_time = time.perf_counter()
        self._trace = []

        try:
            input_grid = np.asarray(input_grid, dtype=int)
            if input_grid.ndim != 2:
                raise ValueError(f"Input grid must be 2D, got shape {input_grid.shape}")

            output = self._eval_node(program.root, input_grid)
            elapsed = (time.perf_counter() - self._start_time) * 1000

            return ExecutionResult(
                output=output,
                success=True,
                error=None,
                trace=list(self._trace),
                total_time_ms=elapsed,
            )
        except ExecutionTimeout:
            elapsed = (time.perf_counter() - self._start_time) * 1000
            return ExecutionResult(
                output=None,
                success=False,
                error=f"Execution timed out after {self.timeout_ms:.0f}ms",
                trace=list(self._trace),
                total_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - self._start_time) * 1000
            return ExecutionResult(
                output=None,
                success=False,
                error=f"{type(e).__name__}: {e}",
                trace=list(self._trace),
                total_time_ms=elapsed,
            )

    def _check_timeout(self) -> None:
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        if elapsed_ms > self.timeout_ms:
            raise ExecutionTimeout()

    def _summarize(self, val: Any) -> str:
        """Brief human-readable summary of a value."""
        if isinstance(val, np.ndarray):
            if val.ndim == 2:
                return f"grid({val.shape[0]}x{val.shape[1]})"
            return f"array(shape={val.shape})"
        if isinstance(val, list):
            return f"list(len={len(val)})"
        if isinstance(val, set):
            return f"set({val})"
        if isinstance(val, tuple):
            return f"tuple{val}"
        if isinstance(val, (bool, np.bool_)):
            return str(bool(val))
        return repr(val)

    def _eval_node(self, node: DSLNode, input_grid: np.ndarray) -> Any:
        """Recursively evaluate a DSLNode."""
        self._check_timeout()

        # Leaf: input grid reference
        if node.op == "__input__":
            return input_grid.copy()

        # Leaf: integer constant
        if node.op == "__const__":
            return int(node.args[0]) if node.args else 0

        # Leaf: color constant
        if node.op == "__color__":
            return int(node.args[0]) if node.args else 0

        # Primitive call
        if node.op not in PRIMITIVE_FUNCTIONS:
            raise ValueError(f"Unknown primitive: {node.op}")

        fn = PRIMITIVE_FUNCTIONS[node.op]
        step_start = time.perf_counter()

        # Evaluate all arguments
        eval_args = []
        arg_summaries = []
        for arg in node.args:
            if isinstance(arg, DSLNode):
                val = self._eval_node(arg, input_grid)
            elif isinstance(arg, int):
                val = arg
            else:
                val = arg
            eval_args.append(val)
            arg_summaries.append(self._summarize(val))

        self._check_timeout()

        # Call the primitive
        result = fn(*eval_args)

        step_time = (time.perf_counter() - step_start) * 1000
        self._trace.append(TraceStep(
            op=node.op,
            args_summary=arg_summaries,
            result_summary=self._summarize(result),
            time_ms=step_time,
        ))

        return result
