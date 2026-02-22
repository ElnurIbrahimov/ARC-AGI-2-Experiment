"""
GridTokenizer — converts ARC grids into token sequences with 2D position IDs for GGRoPE.

Grid encoding:
  Each cell -> color token (via dsl_config.color_to_token)
  Rows separated by SEP tokens
  Position IDs track (row, col) for each token

Task encoding:
  BOS [demo1_in SEP demo1_out] SEP [demo2_in SEP demo2_out] SEP ... [test_in] EOS
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.dsl_config import (
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN,
    COLOR_OFFSET, VOCAB_SIZE,
    color_to_token, token_to_color, is_color_token,
)
from data.grid_utils import normalize_grid


class GridTokenizer:
    """
    Tokenizes ARC grids into sequences with 2D position information.

    Grid encoding:
    - Each cell becomes a token (color value 0-9 mapped to token IDs via dsl_config)
    - Rows are separated by SEP tokens
    - Position IDs track row and column for each token (for GGRoPE)

    Task encoding (full task -> token sequence):
    - BOS
    - Demo pair 1: input_grid SEP output_grid SEP
    - Demo pair 2: input_grid SEP output_grid SEP
    - ...
    - Test input: input_grid
    - EOS

    Position IDs:
    - row_ids[i] = row index of token i in its grid (0 for non-grid tokens)
    - col_ids[i] = column index of token i in its grid (0 for non-grid tokens)
    """

    def __init__(self, max_grid_size: int = 30, max_seq_len: int = 2048):
        self.max_grid_size = max_grid_size
        self.max_seq_len = max_seq_len

        # Build reverse lookup: token_id -> readable name
        self._token_names: Dict[int, str] = {
            PAD_TOKEN: "<PAD>",
            BOS_TOKEN: "<BOS>",
            EOS_TOKEN: "<EOS>",
            SEP_TOKEN: "<SEP>",
        }
        for c in range(10):
            self._token_names[color_to_token(c)] = str(c)

    def encode_grid(self, grid: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
        """
        Encode a single grid into tokens + position IDs.

        Returns: (token_ids, row_ids, col_ids)
        Grid is encoded row by row, with SEP between rows.
        """
        grid = normalize_grid(grid)
        h, w = grid.shape

        token_ids: List[int] = []
        row_ids: List[int] = []
        col_ids: List[int] = []

        for r in range(h):
            for c in range(w):
                color = int(grid[r, c])
                token_ids.append(color_to_token(color))
                row_ids.append(r)
                col_ids.append(c)

            # SEP after each row (including the last, to clearly delimit)
            if r < h - 1:
                token_ids.append(SEP_TOKEN)
                row_ids.append(r)
                col_ids.append(w)  # just past the last column

        return token_ids, row_ids, col_ids

    def encode_task(
        self,
        demo_inputs: List[np.ndarray],
        demo_outputs: List[np.ndarray],
        test_input: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a full ARC task into model input.

        Returns: {
            'token_ids': (seq_len,) int tensor,
            'row_ids': (seq_len,) int tensor,
            'col_ids': (seq_len,) int tensor,
            'grid_boundaries': list of (start, end) indices for each grid,
            'num_demos': int
        }
        """
        all_tokens: List[int] = []
        all_rows: List[int] = []
        all_cols: List[int] = []
        grid_boundaries: List[Tuple[int, int]] = []

        # BOS
        all_tokens.append(BOS_TOKEN)
        all_rows.append(0)
        all_cols.append(0)

        # Demo pairs
        for inp, out in zip(demo_inputs, demo_outputs):
            # Input grid
            start = len(all_tokens)
            toks, rows, cols = self.encode_grid(inp)
            all_tokens.extend(toks)
            all_rows.extend(rows)
            all_cols.extend(cols)
            grid_boundaries.append((start, len(all_tokens)))

            # SEP between input and output
            all_tokens.append(SEP_TOKEN)
            all_rows.append(0)
            all_cols.append(0)

            # Output grid
            start = len(all_tokens)
            toks, rows, cols = self.encode_grid(out)
            all_tokens.extend(toks)
            all_rows.extend(rows)
            all_cols.extend(cols)
            grid_boundaries.append((start, len(all_tokens)))

            # SEP after demo pair
            all_tokens.append(SEP_TOKEN)
            all_rows.append(0)
            all_cols.append(0)

        # Test input
        start = len(all_tokens)
        toks, rows, cols = self.encode_grid(test_input)
        all_tokens.extend(toks)
        all_rows.extend(rows)
        all_cols.extend(cols)
        grid_boundaries.append((start, len(all_tokens)))

        # EOS
        all_tokens.append(EOS_TOKEN)
        all_rows.append(0)
        all_cols.append(0)

        # Truncate if needed
        if len(all_tokens) > self.max_seq_len:
            all_tokens = all_tokens[: self.max_seq_len]
            all_rows = all_rows[: self.max_seq_len]
            all_cols = all_cols[: self.max_seq_len]
            # Adjust grid boundaries
            grid_boundaries = [
                (s, min(e, self.max_seq_len))
                for s, e in grid_boundaries
                if s < self.max_seq_len
            ]

        return {
            "token_ids": torch.tensor(all_tokens, dtype=torch.long),
            "row_ids": torch.tensor(all_rows, dtype=torch.long),
            "col_ids": torch.tensor(all_cols, dtype=torch.long),
            "grid_boundaries": grid_boundaries,
            "num_demos": len(demo_inputs),
        }

    def encode_target(self, target_grid: np.ndarray) -> Dict[str, torch.Tensor]:
        """Encode the target output grid as target tokens."""
        toks, rows, cols = self.encode_grid(target_grid)
        # Add EOS at end
        toks.append(EOS_TOKEN)
        rows.append(0)
        cols.append(0)
        return {
            "token_ids": torch.tensor(toks, dtype=torch.long),
            "row_ids": torch.tensor(rows, dtype=torch.long),
            "col_ids": torch.tensor(cols, dtype=torch.long),
        }

    def decode_grid(self, token_ids: List[int]) -> np.ndarray:
        """
        Decode token sequence back to a grid.

        Expects color tokens with SEP tokens as row delimiters.
        Stops at EOS, PAD, or end of sequence.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        rows: List[List[int]] = []
        current_row: List[int] = []

        for tok in token_ids:
            if tok in (EOS_TOKEN, PAD_TOKEN, BOS_TOKEN):
                if current_row:
                    rows.append(current_row)
                    current_row = []
                if tok in (EOS_TOKEN, PAD_TOKEN):
                    break
                continue

            if tok == SEP_TOKEN:
                if current_row:
                    rows.append(current_row)
                    current_row = []
                continue

            if is_color_token(tok):
                current_row.append(token_to_color(tok))
            # Skip non-color tokens silently

        if current_row:
            rows.append(current_row)

        if not rows:
            return np.zeros((1, 1), dtype=int)

        # Pad rows to max width
        max_w = max(len(r) for r in rows)
        for i in range(len(rows)):
            if len(rows[i]) < max_w:
                rows[i].extend([0] * (max_w - len(rows[i])))

        return np.array(rows, dtype=int)

    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode tokens to human-readable string."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        parts: List[str] = []
        for tok in token_ids:
            if tok in self._token_names:
                parts.append(self._token_names[tok])
            elif is_color_token(tok):
                parts.append(str(token_to_color(tok)))
            else:
                parts.append(f"[{tok}]")
        return " ".join(parts)

    def pad_to_length(self, encoded: Dict, max_len: int) -> Dict:
        """Pad all sequences to max_len with PAD tokens."""
        result = dict(encoded)
        current_len = result["token_ids"].shape[0]

        if current_len >= max_len:
            # Truncate
            result["token_ids"] = result["token_ids"][:max_len]
            result["row_ids"] = result["row_ids"][:max_len]
            result["col_ids"] = result["col_ids"][:max_len]
        else:
            pad_len = max_len - current_len
            result["token_ids"] = torch.cat([
                result["token_ids"],
                torch.full((pad_len,), PAD_TOKEN, dtype=torch.long),
            ])
            result["row_ids"] = torch.cat([
                result["row_ids"],
                torch.zeros(pad_len, dtype=torch.long),
            ])
            result["col_ids"] = torch.cat([
                result["col_ids"],
                torch.zeros(pad_len, dtype=torch.long),
            ])

        return result

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    @property
    def pad_token_id(self) -> int:
        return PAD_TOKEN

    @property
    def bos_token_id(self) -> int:
        return BOS_TOKEN

    @property
    def eos_token_id(self) -> int:
        return EOS_TOKEN

    @property
    def sep_token_id(self) -> int:
        return SEP_TOKEN
