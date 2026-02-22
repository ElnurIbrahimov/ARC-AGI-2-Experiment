"""
RefinementLoop — main neurosymbolic generate->execute->validate->refine loop.

For each task:
  1. Encode task (demo pairs + test input) as tokens
  2. Generate initial DSL program candidates via the model
  3. Parse token sequences into DSLPrograms
  4. Execute programs on demo inputs via DSLEngine
  5. Validate outputs against demo outputs
  6. If not solved:
     a. Build error traces from validation failures
     b. Rank candidate modifications via Causeway (optional)
     c. Generate refined programs conditioned on error traces
     d. Repeat from step 4
  7. Apply best program(s) to test input
  8. Return top-k predictions (for Pass@2)

The loop is designed for hundreds of iterations per task,
enabled by Mamba-2's O(N) efficiency.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.grid_tokenizer import GridTokenizer
from dsl.engine import DSLEngine, ExecutionResult
from dsl.error_trace import (
    ErrorTrace,
    build_error_trace,
    build_error_traces_from_validation,
)
from dsl.parser import DSLParser, ParseResult
from dsl.program import DSLProgram
from dsl.validator import DSLValidator, ValidationResult
from refinement.budget_manager import BudgetConfig, BudgetManager
from refinement.generator import DSLGenerator
from refinement.ranker import CandidateRanker
from refinement.validator_fluxmind import FluxMindValidator

logger = logging.getLogger(__name__)


class RefinementLoop:
    """
    Main neurosymbolic refinement loop for ARC-AGI-2.

    Iteratively generates, executes, validates, and refines DSL programs
    until one passes all demo examples or the compute budget is exhausted.
    Works with or without Causeway/FluxMind (graceful degradation).
    """

    def __init__(
        self,
        model,
        generator: DSLGenerator,
        ranker: Optional[CandidateRanker] = None,
        fluxmind_validator: Optional[FluxMindValidator] = None,
        bridge=None,
        budget_config: Optional[BudgetConfig] = None,
        device: str = 'cuda',
        num_candidates: int = 8,
        max_program_length: int = 256,
        execution_timeout_ms: float = 1000.0,
    ):
        """
        Args:
            model: HybridARC model instance.
            generator: DSLGenerator wrapping the model for autoregressive generation.
            ranker: Optional CandidateRanker (Causeway-based) for pruning candidates.
            fluxmind_validator: Optional FluxMindValidator for confidence scoring.
            bridge: Optional CausalProgramBridge for full integration pipeline.
            budget_config: BudgetConfig controlling iteration/time limits.
            device: Torch device string.
            num_candidates: Number of candidate programs to generate per iteration.
            max_program_length: Max DSL tokens per candidate.
            execution_timeout_ms: Per-program execution timeout.
        """
        self.model = model
        self.generator = generator
        self.ranker = ranker
        self.fluxmind_validator = fluxmind_validator
        self.bridge = bridge
        self.budget_config = budget_config or BudgetConfig()
        self.device = device
        self.num_candidates = num_candidates
        self.max_program_length = max_program_length

        self.tokenizer = GridTokenizer()
        self.parser = DSLParser()
        self.dsl_validator = DSLValidator(timeout_ms=execution_timeout_ms)
        self.engine = DSLEngine(timeout_ms=execution_timeout_ms)
        self.budget = BudgetManager(self.budget_config)

        # Per-solve statistics
        self._solve_stats: Dict = {}

    # ── Public API ──────────────────────────────────────────────────────

    def solve(self, task: Dict) -> List[np.ndarray]:
        """
        Solve a single ARC task.

        Args:
            task: dict with keys:
                - 'demo_inputs':  list of np.ndarray grids
                - 'demo_outputs': list of np.ndarray grids
                - 'test_input':   np.ndarray grid

        Returns:
            List of predicted output grids (up to pass_at_k).
        """
        demo_inputs = [np.asarray(g, dtype=int) for g in task['demo_inputs']]
        demo_outputs = [np.asarray(g, dtype=int) for g in task['demo_outputs']]
        test_input = np.asarray(task['test_input'], dtype=int)

        examples = list(zip(demo_inputs, demo_outputs))
        solve_start = time.time()

        # Reset budget for this task
        self.budget.start_task()

        # Encode the task once (reused across iterations)
        encoded = self._encode_task(task)

        # Track the best candidates across all iterations
        all_candidates: List[Dict] = []
        best_score = -1.0
        best_error_trace: Optional[ErrorTrace] = None
        iteration_scores: List[float] = []

        logger.info(
            "Starting refinement loop: max_iter=%d, max_time=%.0fs, pass_at_k=%d",
            self.budget_config.max_iterations,
            self.budget_config.max_time_seconds,
            self.budget_config.pass_at_k,
        )

        while self.budget.should_continue():
            iteration = self.budget.iteration
            strategy = self._get_generation_strategy(iteration, best_score)

            # Generate candidate token sequences
            token_sequences = self._generate_candidates(
                encoded,
                error_trace=best_error_trace,
                strategy=strategy,
            )

            if not token_sequences:
                logger.debug("Iteration %d: no candidates generated", iteration)
                self.budget.record_iteration(0.0, 0.0)
                iteration_scores.append(0.0)
                continue

            # Optional: rank candidates with Causeway to prune before execution
            if self.ranker is not None:
                token_sequences = self._rank_and_filter(
                    encoded, token_sequences, best_score
                )

            # Parse and execute candidates on demo inputs
            candidates = self._parse_and_execute(token_sequences, demo_inputs)

            # Validate against demo outputs
            candidates = self._validate_candidates(candidates, demo_outputs)

            # Track iteration best
            iter_best = max(
                (c['score'] for c in candidates), default=0.0
            )
            iter_confidence = max(
                (c.get('confidence', 0.0) for c in candidates), default=0.0
            )

            self.budget.record_iteration(iter_best, iter_confidence)
            iteration_scores.append(iter_best)

            # Merge into all_candidates pool
            for c in candidates:
                all_candidates.append(c)

            # Update best score and error trace for next iteration
            if iter_best > best_score:
                best_score = iter_best
                # Find the best candidate this iteration for error trace
                best_cand = max(candidates, key=lambda c: c['score'])
                if best_cand.get('error_traces'):
                    best_error_trace = best_cand['error_traces'][0]
                else:
                    best_error_trace = None

                logger.info(
                    "Iteration %d: new best score=%.4f strategy=%s",
                    iteration, best_score, strategy,
                )

            # Early exit: perfect score on all demos
            if best_score >= 1.0:
                logger.info("Perfect score reached at iteration %d", iteration)
                break

            # Log periodically
            if iteration > 0 and iteration % 50 == 0:
                stats = self.budget.get_stats()
                logger.info(
                    "Iteration %d: best=%.4f, mean=%.4f, patience=%d/%d, time=%.1fs",
                    iteration,
                    stats['best_score'] or 0.0,
                    stats['mean_score'] or 0.0,
                    stats['iters_since_improvement'],
                    self.budget_config.patience,
                    stats['time_elapsed'],
                )

        # Select top-k candidates
        best_candidates = self._select_best(all_candidates)

        # Apply best programs to test input
        predictions = self._apply_to_test(best_candidates, test_input)

        # Record solve stats
        budget_stats = self.budget.get_stats()
        self._solve_stats = {
            'iterations_used': budget_stats['iterations_used'],
            'time_seconds': time.time() - solve_start,
            'best_score': budget_stats['best_score'],
            'best_confidence': budget_stats['best_confidence'],
            'stopped_reason': budget_stats['stopped_reason'],
            'total_candidates_generated': len(all_candidates),
            'num_predictions': len(predictions),
            'iteration_scores': iteration_scores,
        }

        logger.info(
            "Solve complete: %d iterations, %.1fs, best_score=%.4f, reason=%s, predictions=%d",
            self._solve_stats['iterations_used'],
            self._solve_stats['time_seconds'],
            self._solve_stats['best_score'] or 0.0,
            self._solve_stats['stopped_reason'],
            len(predictions),
        )

        return predictions

    def get_solve_stats(self) -> Dict:
        """Return detailed stats from the last solve() call."""
        return dict(self._solve_stats)

    # ── Task Encoding ───────────────────────────────────────────────────

    def _encode_task(self, task: Dict) -> Dict[str, torch.Tensor]:
        """
        Encode task grids as model input tokens with position IDs.

        Returns dict with 'token_ids', 'row_ids', 'col_ids' tensors
        (each shape (1, T) with batch dim).
        """
        encoded = self.tokenizer.encode_task(
            demo_inputs=task['demo_inputs'],
            demo_outputs=task['demo_outputs'],
            test_input=task['test_input'],
        )
        return {
            'token_ids': encoded['token_ids'].unsqueeze(0).to(self.device),
            'row_ids': encoded['row_ids'].unsqueeze(0).to(self.device),
            'col_ids': encoded['col_ids'].unsqueeze(0).to(self.device),
        }

    # ── Candidate Generation ────────────────────────────────────────────

    def _generate_candidates(
        self,
        encoded_task: Dict[str, torch.Tensor],
        error_trace: Optional[ErrorTrace] = None,
        strategy: str = 'beam_search',
    ) -> List[List[int]]:
        """
        Generate candidate DSL token sequences.

        Args:
            encoded_task: dict with 'token_ids', 'row_ids', 'col_ids'.
            error_trace: optional ErrorTrace from previous failed attempt.
            strategy: 'greedy', 'beam_search', 'sampling', 'error_guided'.

        Returns:
            List of token sequences (each List[int]).
        """
        try:
            sequences = self.generator.generate(
                task_tokens=encoded_task['token_ids'],
                row_ids=encoded_task['row_ids'],
                col_ids=encoded_task['col_ids'],
                strategy=strategy,
                num_candidates=self.num_candidates,
                max_length=self.max_program_length,
                error_trace=error_trace,
            )
            return sequences
        except Exception as e:
            logger.warning("Generation failed (strategy=%s): %s", strategy, e)
            return []

    # ── Candidate Ranking (optional) ────────────────────────────────────

    def _rank_and_filter(
        self,
        encoded_task: Dict[str, torch.Tensor],
        token_sequences: List[List[int]],
        current_score: float,
    ) -> List[List[int]]:
        """
        Use the ranker to filter candidates before expensive execution.

        Falls back to returning all candidates if ranking fails or is unavailable.
        """
        if self.ranker is None or len(token_sequences) <= 1:
            return token_sequences

        try:
            # Get hidden states from the model for ranking
            with torch.no_grad():
                output = self.model.forward(
                    encoded_task['token_ids'],
                    encoded_task['row_ids'],
                    encoded_task['col_ids'],
                )
                hidden_states = output.hidden_states

            filtered_indices = self.ranker.filter_candidates(
                hidden_states=hidden_states,
                candidate_tokens=token_sequences,
                current_score=current_score,
            )

            if filtered_indices:
                return [token_sequences[i] for i in filtered_indices]
        except Exception as e:
            logger.debug("Ranking failed, using all candidates: %s", e)

        return token_sequences

    # ── Parse and Execute ───────────────────────────────────────────────

    def _parse_and_execute(
        self,
        token_sequences: List[List[int]],
        demo_inputs: List[np.ndarray],
    ) -> List[Dict]:
        """
        Parse token sequences into programs, execute on demo inputs.

        Returns list of dicts, each with:
            'tokens': List[int] — original token sequence
            'program': DSLProgram or None
            'results': List[ExecutionResult] — one per demo input
            'success': bool — True if all executions succeeded
            'parse_error': str or None
        """
        candidates: List[Dict] = []

        for tokens in token_sequences:
            candidate: Dict = {
                'tokens': tokens,
                'program': None,
                'results': [],
                'success': False,
                'parse_error': None,
            }

            # Parse
            parse_result: ParseResult = self.parser.parse(tokens)
            if not parse_result.success:
                candidate['parse_error'] = (
                    parse_result.error.message if parse_result.error else "Unknown parse error"
                )
                candidates.append(candidate)
                continue

            program = parse_result.program
            candidate['program'] = program

            # Execute on each demo input
            all_success = True
            results: List[ExecutionResult] = []
            for inp in demo_inputs:
                exec_result = self.engine.execute(program, inp)
                results.append(exec_result)
                if not exec_result.success:
                    all_success = False

            candidate['results'] = results
            candidate['success'] = all_success
            candidates.append(candidate)

        return candidates

    # ── Validation ──────────────────────────────────────────────────────

    def _validate_candidates(
        self,
        candidates: List[Dict],
        demo_outputs: List[np.ndarray],
    ) -> List[Dict]:
        """
        Validate executed candidates against expected outputs.

        Adds to each candidate dict:
            'score': float — fraction of exact-matching demo outputs
            'cell_accuracy': float — average cell-level accuracy
            'confidence': float — combined confidence (with FluxMind if available)
            'error_traces': List[ErrorTrace] — for failing examples
            'num_passed': int
        """
        examples = []  # (input_placeholder, expected_output) for FluxMind
        for out in demo_outputs:
            # FluxMind expects (input, output) pairs but we only need outputs for scoring
            examples.append((np.zeros_like(out), out))

        for candidate in candidates:
            program = candidate.get('program')
            results = candidate.get('results', [])

            # Default scores for failed candidates
            if program is None or not results:
                candidate['score'] = 0.0
                candidate['cell_accuracy'] = 0.0
                candidate['confidence'] = 0.0
                candidate['error_traces'] = []
                candidate['num_passed'] = 0
                continue

            # Validate using DSLValidator on the program directly
            demo_pairs = list(zip(
                [np.zeros((1, 1), dtype=int)] * len(demo_outputs),  # dummy inputs
                demo_outputs,
            ))

            # Compute per-example metrics from execution results
            num_passed = 0
            cell_accs: List[float] = []
            error_traces: List[ErrorTrace] = []

            for i, (exec_result, expected) in enumerate(zip(results, demo_outputs)):
                expected = np.asarray(expected, dtype=int)
                actual = exec_result.output_grid

                if actual is not None and actual.shape == expected.shape and np.array_equal(actual, expected):
                    num_passed += 1
                    cell_accs.append(1.0)
                else:
                    # Compute cell accuracy
                    if actual is not None and actual.shape == expected.shape:
                        total = expected.size
                        matching = int(np.sum(actual == expected))
                        cell_accs.append(matching / max(total, 1))
                    elif actual is not None:
                        # Partial credit on overlapping region
                        h = min(expected.shape[0], actual.shape[0])
                        w = min(expected.shape[1], actual.shape[1])
                        total = max(expected.size, actual.size)
                        overlap = int(np.sum(expected[:h, :w] == actual[:h, :w]))
                        cell_accs.append(overlap / max(total, 1))
                    else:
                        cell_accs.append(0.0)

                    # Build error trace for this failing example
                    trace = build_error_trace(i, expected, actual)
                    error_traces.append(trace)

            num_total = len(demo_outputs)
            score = num_passed / max(num_total, 1)
            avg_cell_acc = sum(cell_accs) / max(len(cell_accs), 1)

            # FluxMind confidence (optional enhancement)
            confidence = score  # default: confidence = exact match ratio
            if self.fluxmind_validator is not None and program is not None:
                try:
                    # Extract op names from program
                    op_names = self._extract_op_names(program)
                    fm_score = self.fluxmind_validator.score(
                        program_ops=op_names,
                        examples=examples,
                        execution_results=results,
                    )
                    # Use FluxMind score as confidence
                    confidence = fm_score
                except Exception as e:
                    logger.debug("FluxMind scoring failed: %s", e)

            candidate['score'] = score
            candidate['cell_accuracy'] = avg_cell_acc
            candidate['confidence'] = confidence
            candidate['error_traces'] = error_traces
            candidate['num_passed'] = num_passed

        return candidates

    # ── Selection ───────────────────────────────────────────────────────

    def _select_best(self, candidates: List[Dict]) -> List[Dict]:
        """
        Select best candidates for final submission.

        Returns up to pass_at_k candidates, sorted by:
          1. Exact match score (descending)
          2. Cell accuracy (descending)
          3. Confidence (descending)

        Ensures diversity: avoids returning identical programs.
        """
        k = self.budget_config.pass_at_k

        # Filter to candidates that have a valid program
        valid = [c for c in candidates if c.get('program') is not None]

        if not valid:
            return []

        # Sort by (score, cell_accuracy, confidence) descending
        valid.sort(
            key=lambda c: (c.get('score', 0.0), c.get('cell_accuracy', 0.0), c.get('confidence', 0.0)),
            reverse=True,
        )

        # Deduplicate by token sequence
        seen_tokens: set = set()
        selected: List[Dict] = []

        for c in valid:
            token_key = tuple(c.get('tokens', []))
            if token_key in seen_tokens:
                continue
            seen_tokens.add(token_key)
            selected.append(c)
            if len(selected) >= k:
                break

        # If we couldn't fill k slots with unique candidates, allow duplicates
        if len(selected) < k:
            for c in valid:
                if len(selected) >= k:
                    break
                if c not in selected:
                    selected.append(c)

        return selected[:k]

    # ── Test Application ────────────────────────────────────────────────

    def _apply_to_test(
        self,
        candidates: List[Dict],
        test_input: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Apply best program(s) to test input grid.

        Returns list of predicted output grids.
        """
        predictions: List[np.ndarray] = []

        for candidate in candidates:
            program = candidate.get('program')
            if program is None:
                # Fallback: return a copy of the test input
                predictions.append(test_input.copy())
                continue

            try:
                exec_result = self.engine.execute(program, test_input)
                if exec_result.success and exec_result.output_grid is not None:
                    predictions.append(exec_result.output_grid)
                else:
                    # Execution failed on test — use test input as fallback
                    logger.debug(
                        "Program failed on test input: %s",
                        exec_result.error or "non-grid output",
                    )
                    predictions.append(test_input.copy())
            except Exception as e:
                logger.debug("Test execution exception: %s", e)
                predictions.append(test_input.copy())

        # Ensure we always return at least one prediction
        if not predictions:
            predictions.append(test_input.copy())

        return predictions

    # ── Adaptive Strategy Selection ─────────────────────────────────────

    def _get_generation_strategy(self, iteration: int, best_score: float) -> str:
        """
        Adaptive strategy selection based on iteration and progress.

        Strategy transitions:
          - Iterations 0-9:       beam_search (broad exploration)
          - Iterations 10-49:     sampling (diversity)
          - Iterations 50+:       error_guided if we have error traces
          - High score (>= 0.5):  error_guided (focus on fixing remaining errors)
          - Very early:           greedy for one quick attempt

        Within each phase, strategies alternate to maintain diversity.
        """
        # First iteration: try greedy for a quick baseline
        if iteration == 0:
            return 'greedy'

        # If we have a good score, focus on fixing errors
        if best_score >= 0.5:
            return 'error_guided' if iteration % 3 != 0 else 'sampling'

        # Early: broad beam search
        if iteration < 10:
            return 'beam_search'

        # Mid: alternate between sampling and beam search for diversity
        if iteration < 50:
            if iteration % 3 == 0:
                return 'beam_search'
            else:
                return 'sampling'

        # Late: primarily error-guided with occasional sampling
        if iteration % 4 == 0:
            return 'sampling'
        return 'error_guided'

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_op_names(program: DSLProgram) -> List[str]:
        """Extract all operation names from a DSLProgram tree."""
        ops: List[str] = []

        def _walk(node):
            if node.op not in ('__input__', '__const__', '__color__'):
                ops.append(node.op)
            for arg in node.args:
                if hasattr(arg, 'op'):
                    _walk(arg)

        _walk(program.root)
        return ops
