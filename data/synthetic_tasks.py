"""
SyntheticTaskGenerator — procedural task generator for pretraining.

Creates unlimited synthetic ARC-like tasks by composing DSL primitives.
Each task is a set of (input, output) demo pairs + one test pair,
where the output is produced by executing a random DSL program on the input.
"""

from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import torch
from torch.utils.data import IterableDataset

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.dsl_config import (
    DSL_REGISTRY, DSLCategory, ArgType,
    PRIMITIVE_NAME_TO_ID,
)
from dsl.program import DSLProgram, DSLNode, input_node, const_node, color_node, prim_node
from dsl.engine import DSLEngine
from data.grid_tokenizer import GridTokenizer
from data.grid_utils import normalize_grid

logger = logging.getLogger(__name__)


# Primitives that take a single grid and return a grid -- safe to compose freely
_UNARY_GRID_PRIMS = [
    "rot90", "rot180", "rot270", "hmirror", "vmirror", "transpose",
    "trim", "extract_largest", "extract_smallest",
]

# Primitives that need specific extra args but still produce grids
_PARAM_GRID_PRIMS = {
    "recolor": {"arity": 3, "extra_types": ["color", "color"]},
    "fill": {"arity": 2, "extra_types": ["color"]},
    "shift": {"arity": 3, "extra_types": ["int", "int"]},
    "tile": {"arity": 3, "extra_types": ["int", "int"]},
}


class SyntheticTaskGenerator:
    """
    Generates synthetic ARC-like tasks by composing DSL primitives.

    Strategy:
    1. Pick a random DSL program (composition of 1-5 primitives)
    2. Generate random input grids
    3. Execute the program to get output grids
    4. Return (inputs, outputs) as a task
    """

    def __init__(
        self,
        num_demos: int = 3,
        min_grid_size: int = 3,
        max_grid_size: int = 15,
        max_program_depth: int = 3,
        seed: Optional[int] = None,
    ):
        self.num_demos = num_demos
        self.min_grid_size = min_grid_size
        self.max_grid_size = max_grid_size
        self.max_program_depth = max_program_depth
        self.rng = np.random.default_rng(seed)
        self.engine = DSLEngine(timeout_ms=500)

    def generate_task(self) -> Dict:
        """
        Generate one synthetic task. Retries internally on failures.

        Returns: {
            'demo_inputs': list of np arrays,
            'demo_outputs': list of np arrays,
            'test_input': np array,
            'test_output': np array,
            'program': DSLProgram that transforms inputs to outputs,
            'program_tokens': list of int token IDs
        }
        """
        max_attempts = 50
        for _ in range(max_attempts):
            result = self._try_generate_task()
            if result is not None:
                return result

        # Fallback: identity task (program = just return input)
        logger.warning("Failed to generate task after %d attempts, using identity", max_attempts)
        return self._make_identity_task()

    def generate_random_grid(
        self,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        num_colors: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a random grid with some structure (not pure noise).
        Mixes strategies: random fill, objects on background, patterns, symmetric.
        """
        min_s = min_size or self.min_grid_size
        max_s = max_size or self.max_grid_size
        h = int(self.rng.integers(min_s, max_s + 1))
        w = int(self.rng.integers(min_s, max_s + 1))
        n_colors = num_colors or int(self.rng.integers(2, 7))

        strategy = int(self.rng.integers(0, 5))

        if strategy == 0:
            # Sparse random: mostly background with scattered colored cells
            grid = np.zeros((h, w), dtype=int)
            colors = self.rng.integers(1, 10, size=n_colors).tolist()
            n_cells = int(self.rng.integers(3, max(4, h * w // 3)))
            for _ in range(n_cells):
                r = int(self.rng.integers(0, h))
                c = int(self.rng.integers(0, w))
                grid[r, c] = int(self.rng.choice(colors))
            return grid

        elif strategy == 1:
            # Rectangular objects on background
            grid = np.zeros((h, w), dtype=int)
            n_objects = int(self.rng.integers(1, 5))
            for _ in range(n_objects):
                color = int(self.rng.integers(1, 10))
                oh = int(self.rng.integers(1, max(2, h // 2)))
                ow = int(self.rng.integers(1, max(2, w // 2)))
                r0 = int(self.rng.integers(0, h - oh + 1))
                c0 = int(self.rng.integers(0, w - ow + 1))
                grid[r0:r0 + oh, c0:c0 + ow] = color
            return grid

        elif strategy == 2:
            # Striped pattern
            grid = np.zeros((h, w), dtype=int)
            colors = self.rng.integers(0, 10, size=min(n_colors, 4)).tolist()
            axis = int(self.rng.integers(0, 2))
            period = int(self.rng.integers(1, max(2, min(h, w) // 2)))
            for r in range(h):
                for c in range(w):
                    idx = (r if axis == 0 else c) // max(1, period)
                    grid[r, c] = colors[idx % len(colors)]
            return grid

        elif strategy == 3:
            # Symmetric grid
            half_w = max(1, w // 2)
            half = self.rng.integers(0, 10, size=(h, half_w)).astype(int)
            grid = np.zeros((h, w), dtype=int)
            grid[:, :half_w] = half
            grid[:, w - half_w:] = np.fliplr(half)
            return grid

        else:
            # Dense random
            colors = self.rng.integers(0, min(10, n_colors + 1), size=(h, w))
            return colors.astype(int)

    def generate_random_program(self, max_depth: Optional[int] = None) -> DSLProgram:
        """
        Generate a random DSL program by composing primitives.
        Ensures the program is executable (valid arities, compatible operations).
        """
        depth = max_depth or self.max_program_depth
        root = self._random_program_node(depth)
        return DSLProgram(root=root)

    def _random_program_node(self, remaining_depth: int) -> DSLNode:
        """Build a random program tree node."""
        # Base case: return input reference
        if remaining_depth <= 1:
            return input_node()

        # Pick complexity
        use_params = self.rng.random() < 0.3 and remaining_depth >= 2

        if use_params:
            # Use a parameterized primitive
            prim_name = str(self.rng.choice(list(_PARAM_GRID_PRIMS.keys())))
            info = _PARAM_GRID_PRIMS[prim_name]

            args: list = []
            # First arg is always a grid (recurse)
            args.append(self._random_program_node(remaining_depth - 1))

            # Extra args are constants
            for etype in info["extra_types"]:
                if etype == "color":
                    args.append(color_node(int(self.rng.integers(0, 10))))
                elif etype == "int":
                    if prim_name == "shift":
                        args.append(const_node(int(self.rng.integers(-5, 6))))
                    elif prim_name == "tile":
                        args.append(const_node(int(self.rng.integers(1, 4))))
                    else:
                        args.append(const_node(int(self.rng.integers(0, 10))))

            return DSLNode(op=prim_name, args=args)
        else:
            # Use a unary grid -> grid primitive
            prim_name = str(self.rng.choice(_UNARY_GRID_PRIMS))
            child = self._random_program_node(remaining_depth - 1)
            return DSLNode(op=prim_name, args=[child])

    def _try_generate_task(self) -> Optional[Dict]:
        """
        Try to generate a task. Returns None if the generated program
        fails on the random inputs (retry with different program/inputs).
        """
        # 1. Generate random program
        depth = int(self.rng.integers(1, self.max_program_depth + 1))
        program = self.generate_random_program(max_depth=depth)

        # 2. Generate random input grids (num_demos + 1 for test)
        total_grids = self.num_demos + 1

        # Use consistent grid sizes within a task
        use_same_size = self.rng.random() < 0.5
        if use_same_size:
            h = int(self.rng.integers(self.min_grid_size, self.max_grid_size + 1))
            w = int(self.rng.integers(self.min_grid_size, self.max_grid_size + 1))
            n_colors = int(self.rng.integers(2, 7))
            inputs = [
                self.generate_random_grid(min_size=h, max_size=h, num_colors=n_colors)
                for _ in range(total_grids)
            ]
            # Force exact shape since min=max doesn't guarantee both dims
            inputs = [g[:h, :w] if g.shape[0] >= h and g.shape[1] >= w
                      else self.generate_random_grid(min_size=h, max_size=h, num_colors=n_colors)
                      for g in inputs]
        else:
            inputs = [self.generate_random_grid() for _ in range(total_grids)]

        # 3. Execute program on each input
        outputs = []
        for inp in inputs:
            result = self.engine.execute(program, inp)
            if not result.success:
                return None
            out_grid = result.output_grid
            if out_grid is None:
                return None
            # Validate output is reasonable
            if out_grid.size == 0 or out_grid.shape[0] > 30 or out_grid.shape[1] > 30:
                return None
            # Clamp to valid colors
            out_grid = np.clip(out_grid, 0, 9)
            outputs.append(out_grid)

        # 4. Check that the program actually does something
        #    (at least one output differs from its input, unless it's identity-ish)
        any_different = False
        for inp, out in zip(inputs, outputs):
            if inp.shape != out.shape or not np.array_equal(inp, out):
                any_different = True
                break
        if not any_different:
            return None

        # 5. Check outputs aren't all identical (program should depend on input)
        if all(np.array_equal(outputs[0], o) for o in outputs[1:]):
            # All outputs the same -- program is constant, not interesting
            return None

        # 6. Split into demos + test
        demo_inputs = inputs[:self.num_demos]
        demo_outputs = outputs[:self.num_demos]
        test_input = inputs[self.num_demos]
        test_output = outputs[self.num_demos]

        # 7. Tokenize program
        try:
            program_tokens = program.to_tokens()
        except Exception:
            return None

        return {
            "demo_inputs": demo_inputs,
            "demo_outputs": demo_outputs,
            "test_input": test_input,
            "test_output": test_output,
            "program": program,
            "program_tokens": program_tokens,
        }

    def _make_identity_task(self) -> Dict:
        """Fallback: generate a trivial identity task."""
        program = DSLProgram(root=input_node())
        grids = [self.generate_random_grid() for _ in range(self.num_demos + 1)]
        return {
            "demo_inputs": grids[:self.num_demos],
            "demo_outputs": [g.copy() for g in grids[:self.num_demos]],
            "test_input": grids[self.num_demos],
            "test_output": grids[self.num_demos].copy(),
            "program": program,
            "program_tokens": program.to_tokens(),
        }

    def generate_batch(self, batch_size: int) -> List[Dict]:
        """Generate a batch of tasks."""
        return [self.generate_task() for _ in range(batch_size)]


class SyntheticDataset(IterableDataset):
    """
    Infinite IterableDataset wrapper around SyntheticTaskGenerator.
    For Stage 1 pretraining.
    """

    def __init__(
        self,
        generator: SyntheticTaskGenerator,
        tokenizer: Optional[GridTokenizer] = None,
        max_seq_len: int = 2048,
    ):
        self.generator = generator
        self.tokenizer = tokenizer or GridTokenizer(max_seq_len=max_seq_len)
        self.max_seq_len = max_seq_len

    def __iter__(self):
        """Yield encoded tasks forever."""
        # Each DataLoader worker gets a different seed
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_seed = worker_info.id + int(self.generator.rng.integers(0, 2**31))
            self.generator.rng = np.random.default_rng(worker_seed)

        while True:
            task = self.generator.generate_task()

            # Encode for model
            encoded = self.tokenizer.encode_task(
                task["demo_inputs"],
                task["demo_outputs"],
                task["test_input"],
            )
            encoded = self.tokenizer.pad_to_length(encoded, self.max_seq_len)

            target_encoded = self.tokenizer.encode_target(task["test_output"])

            yield {
                "token_ids": encoded["token_ids"],
                "row_ids": encoded["row_ids"],
                "col_ids": encoded["col_ids"],
                "target_tokens": target_encoded["token_ids"],
                "program_tokens": torch.tensor(
                    task["program_tokens"], dtype=torch.long
                ),
                "demo_inputs": task["demo_inputs"],
                "demo_outputs": task["demo_outputs"],
                "test_input": task["test_input"],
                "test_output": task["test_output"],
            }
