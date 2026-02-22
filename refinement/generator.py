"""
DSLGenerator — wraps the model for autoregressive DSL program generation.

Supports multiple decoding strategies:
  - greedy: argmax at each step
  - beam_search: beam search with configurable width
  - sampling: top-k / top-p (nucleus) sampling with temperature
  - error_guided: conditions generation on error trace from a previous failed attempt

The model runs in autoregressive mode: input = task tokens (grid pairs), output = DSL
token sequence. For error_guided, error trace tokens are appended to the input context.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.dsl_config import (
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN,
    VOCAB_SIZE, PRIMITIVE_OFFSET, STRUCT_OFFSET, CONST_OFFSET,
    is_primitive_token, is_structural_token, is_const_token,
)


# Error trace is encoded as: SEP [category_token] [diff_count_token] ... SEP
# We reuse SEP as delimiter since no dedicated ERROR_START/ERROR_END tokens exist yet.
# Category is mapped to a primitive-range token offset (doesn't collide with real prims).
_ERROR_CATEGORY_MAP = {
    "execution_error": 0,
    "size_error": 1,
    "color_error": 2,
    "spatial_error": 3,
    "structural_error": 4,
    "no_error": 5,
}


@dataclass
class _BeamHypothesis:
    """One hypothesis in beam search."""
    tokens: List[int]
    log_prob: float
    finished: bool = False


class DSLGenerator:
    """
    Generates DSL programs from the model given an ARC task.

    The model is any module with:
        model.forward(token_ids, row_ids, col_ids) -> output
    where output.logits has shape (B, T, V) and output.hidden_states has shape (B, T, H).
    """

    def __init__(
        self,
        model,
        tokenizer_config: dict = None,
        device: str = 'cuda',
    ):
        """
        Args:
            model: HybridARC or any model with .forward(token_ids, row_ids, col_ids).
            tokenizer_config: dict with vocab info. Uses dsl_config defaults if None.
            device: torch device string.
        """
        self.model = model
        self.device = device

        cfg = tokenizer_config or {}
        self.vocab_size = cfg.get('vocab_size', VOCAB_SIZE)
        self.bos_token = cfg.get('bos_token', BOS_TOKEN)
        self.eos_token = cfg.get('eos_token', EOS_TOKEN)
        self.pad_token = cfg.get('pad_token', PAD_TOKEN)
        self.sep_token = cfg.get('sep_token', SEP_TOKEN)

    # ── Public API ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        task_tokens: torch.Tensor,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
        strategy: str = 'beam_search',
        num_candidates: int = 8,
        max_length: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        beam_width: int = 8,
        error_trace: object = None,
    ) -> List[List[int]]:
        """
        Generate DSL token sequences for a task.

        Args:
            task_tokens: (1, T) encoded input task (demo pairs + test input).
            row_ids:     (1, T) row position IDs for GGRoPE.
            col_ids:     (1, T) col position IDs for GGRoPE.
            strategy:    'greedy', 'beam_search', 'sampling', 'error_guided'.
            num_candidates: how many programs to return.
            max_length:  max tokens to generate per candidate.
            temperature: sampling temperature (for 'sampling' and 'error_guided').
            top_k:       top-k filtering (for 'sampling').
            top_p:       nucleus filtering (for 'sampling').
            beam_width:  beam width (for 'beam_search').
            error_trace: ErrorTrace object from a previous failed attempt.

        Returns:
            List of token sequences (each List[int]), up to num_candidates.
        """
        # Ensure tensors are on the right device
        task_tokens = task_tokens.to(self.device)
        row_ids = row_ids.to(self.device)
        col_ids = col_ids.to(self.device)

        # Ensure batch dimension
        if task_tokens.dim() == 1:
            task_tokens = task_tokens.unsqueeze(0)
            row_ids = row_ids.unsqueeze(0)
            col_ids = col_ids.unsqueeze(0)

        if strategy == 'greedy':
            seq = self._greedy_decode(task_tokens, row_ids, col_ids, max_length)
            return [seq]

        elif strategy == 'beam_search':
            seqs = self._beam_search(
                task_tokens, row_ids, col_ids,
                beam_width=max(beam_width, num_candidates),
                max_length=max_length,
            )
            return seqs[:num_candidates]

        elif strategy == 'sampling':
            seqs = self._sample(
                task_tokens, row_ids, col_ids,
                num_samples=num_candidates,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            return seqs

        elif strategy == 'error_guided':
            seqs = self._error_guided_generate(
                task_tokens, row_ids, col_ids,
                error_trace=error_trace,
                num_candidates=num_candidates,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            return seqs

        else:
            raise ValueError(f"Unknown generation strategy: {strategy}")

    # ── Greedy Decode ───────────────────────────────────────────────────

    def _greedy_decode(
        self,
        input_ids: torch.Tensor,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
        max_length: int,
    ) -> List[int]:
        """Single greedy decode: take argmax at each step."""
        generated = []
        current_ids = input_ids  # (1, T)
        current_rows = row_ids
        current_cols = col_ids

        for _ in range(max_length):
            output = self.model.forward(current_ids, current_rows, current_cols)
            logits = output.logits  # (1, T', V)
            next_logits = logits[:, -1, :]  # (1, V)

            next_token = next_logits.argmax(dim=-1).item()
            generated.append(next_token)

            if next_token == self.eos_token:
                break

            # Append token and extend positional IDs
            next_tok_t = torch.tensor([[next_token]], device=self.device)
            # Use 0 for row/col of generated DSL tokens (they're not grid positions)
            next_row = torch.zeros(1, 1, dtype=row_ids.dtype, device=self.device)
            next_col = torch.zeros(1, 1, dtype=col_ids.dtype, device=self.device)

            current_ids = torch.cat([current_ids, next_tok_t], dim=1)
            current_rows = torch.cat([current_rows, next_row], dim=1)
            current_cols = torch.cat([current_cols, next_col], dim=1)

        return generated

    # ── Beam Search ─────────────────────────────────────────────────────

    def _beam_search(
        self,
        input_ids: torch.Tensor,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
        beam_width: int,
        max_length: int,
    ) -> List[List[int]]:
        """Beam search returning top beam_width completed sequences."""
        # Initialize beams
        beams: List[_BeamHypothesis] = [
            _BeamHypothesis(tokens=[], log_prob=0.0),
        ]
        finished: List[_BeamHypothesis] = []

        for step in range(max_length):
            if not beams:
                break

            all_candidates: List[_BeamHypothesis] = []

            for beam in beams:
                # Build full sequence: input + generated so far
                gen_tensor = torch.tensor(
                    [beam.tokens], dtype=input_ids.dtype, device=self.device
                )
                full_ids = torch.cat([input_ids, gen_tensor], dim=1) if beam.tokens else input_ids
                gen_len = len(beam.tokens)
                extra_rows = torch.zeros(1, gen_len, dtype=row_ids.dtype, device=self.device)
                extra_cols = torch.zeros(1, gen_len, dtype=col_ids.dtype, device=self.device)
                full_rows = torch.cat([row_ids, extra_rows], dim=1) if gen_len > 0 else row_ids
                full_cols = torch.cat([col_ids, extra_cols], dim=1) if gen_len > 0 else col_ids

                output = self.model.forward(full_ids, full_rows, full_cols)
                logits = output.logits[:, -1, :]  # (1, V)
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # (V,)

                # Take top beam_width tokens
                topk_log_probs, topk_ids = log_probs.topk(beam_width)

                for i in range(beam_width):
                    token_id = topk_ids[i].item()
                    token_lp = topk_log_probs[i].item()
                    new_tokens = beam.tokens + [token_id]
                    new_lp = beam.log_prob + token_lp

                    if token_id == self.eos_token:
                        finished.append(_BeamHypothesis(
                            tokens=new_tokens, log_prob=new_lp, finished=True,
                        ))
                    else:
                        all_candidates.append(_BeamHypothesis(
                            tokens=new_tokens, log_prob=new_lp,
                        ))

            # Keep top beam_width active beams
            all_candidates.sort(key=lambda h: h.log_prob, reverse=True)
            beams = all_candidates[:beam_width]

            # Early stop: enough finished beams
            if len(finished) >= beam_width:
                break

        # Merge finished and remaining active
        all_results = finished + beams
        all_results.sort(key=lambda h: h.log_prob, reverse=True)

        return [h.tokens for h in all_results[:beam_width]]

    # ── Sampling ────────────────────────────────────────────────────────

    def _sample(
        self,
        input_ids: torch.Tensor,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
        num_samples: int,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> List[List[int]]:
        """Top-k/top-p sampling with temperature."""
        results: List[List[int]] = []

        for _ in range(num_samples):
            generated = []
            current_ids = input_ids
            current_rows = row_ids
            current_cols = col_ids

            for _step in range(max_length):
                output = self.model.forward(current_ids, current_rows, current_cols)
                logits = output.logits[:, -1, :]  # (1, V)

                # Temperature scaling
                if temperature > 0:
                    logits = logits / temperature

                # Apply top-k and top-p filtering
                filtered_logits = self._top_k_top_p_filtering(
                    logits.squeeze(0), top_k, top_p,
                )

                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_token)

                if next_token == self.eos_token:
                    break

                next_tok_t = torch.tensor([[next_token]], device=self.device)
                next_row = torch.zeros(1, 1, dtype=row_ids.dtype, device=self.device)
                next_col = torch.zeros(1, 1, dtype=col_ids.dtype, device=self.device)

                current_ids = torch.cat([current_ids, next_tok_t], dim=1)
                current_rows = torch.cat([current_rows, next_row], dim=1)
                current_cols = torch.cat([current_cols, next_col], dim=1)

            results.append(generated)

        return results

    # ── Error-Guided Generation ─────────────────────────────────────────

    def _error_guided_generate(
        self,
        input_ids: torch.Tensor,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
        error_trace: object,
        num_candidates: int,
        max_length: int,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> List[List[int]]:
        """
        Append error trace tokens to input, then sample.

        Error trace encoding:
            SEP [category_id] [diff_count_clipped] SEP
        where category_id and diff_count are mapped into the constant token range.

        This conditions the model on what went wrong, biasing generation toward
        corrections.
        """
        error_tokens = self._encode_error_trace(error_trace)

        if error_tokens:
            error_tensor = torch.tensor(
                [error_tokens], dtype=input_ids.dtype, device=self.device,
            )
            augmented_ids = torch.cat([input_ids, error_tensor], dim=1)
            # Extend positional IDs with zeros for the error tokens
            extra_len = len(error_tokens)
            extra_rows = torch.zeros(
                1, extra_len, dtype=row_ids.dtype, device=self.device,
            )
            extra_cols = torch.zeros(
                1, extra_len, dtype=col_ids.dtype, device=self.device,
            )
            augmented_rows = torch.cat([row_ids, extra_rows], dim=1)
            augmented_cols = torch.cat([col_ids, extra_cols], dim=1)
        else:
            augmented_ids = input_ids
            augmented_rows = row_ids
            augmented_cols = col_ids

        # Use slightly higher temperature for error-guided to encourage diversity
        adj_temperature = min(temperature * 1.1, 1.5)

        return self._sample(
            augmented_ids, augmented_rows, augmented_cols,
            num_samples=num_candidates,
            max_length=max_length,
            temperature=adj_temperature,
            top_k=top_k,
            top_p=top_p,
        )

    def _encode_error_trace(self, error_trace: object) -> List[int]:
        """
        Convert an ErrorTrace to a token sequence.

        Returns empty list if error_trace is None or invalid.
        """
        if error_trace is None:
            return []

        tokens = [self.sep_token]

        # Encode category
        category = getattr(error_trace, 'suggested_category', 'structural_error')
        cat_id = _ERROR_CATEGORY_MAP.get(category, 4)
        # Map to const token range so it doesn't collide with real DSL tokens
        tokens.append(min(CONST_OFFSET + cat_id, CONST_OFFSET + 139))

        # Encode diff count (clamp to 0-139 for the const token range)
        diff_count = 0
        if hasattr(error_trace, 'diff_count'):
            diff_count = error_trace.diff_count()
            if diff_count < 0:
                diff_count = 0
        diff_count = min(diff_count, 139)
        tokens.append(CONST_OFFSET + diff_count)

        tokens.append(self.sep_token)
        return tokens

    # ── Filtering Utilities ─────────────────────────────────────────────

    @staticmethod
    def _top_k_top_p_filtering(
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = float('-inf'),
    ) -> torch.Tensor:
        """
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.

        Args:
            logits: (V,) logits for a single position.
            top_k:  keep only top-k tokens. 0 = no filtering.
            top_p:  keep smallest set of tokens whose cumulative prob >= top_p. 1.0 = off.
            filter_value: value to assign to filtered tokens.

        Returns:
            Filtered logits tensor (V,).
        """
        logits = logits.clone()

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            # Remove tokens outside the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            # Shift right so that the first token above threshold is kept
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        return logits
