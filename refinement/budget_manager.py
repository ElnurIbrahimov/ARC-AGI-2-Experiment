"""
BudgetManager — controls iteration count, time limits, and confidence thresholds
for the ARC-AGI-2 refinement loop.

Tracks all candidate solutions with their scores. Supports early stopping
based on patience (no improvement for N iterations) and confidence thresholds.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BudgetConfig:
    max_iterations: int = 500          # Max refinement iterations per task
    max_time_seconds: float = 300.0    # 5 min per task
    confidence_threshold: float = 0.99 # Stop if confident enough
    pass_at_k: int = 2                 # ARC-AGI-2 allows 2 submissions
    min_iterations: int = 10           # Always run at least this many
    patience: int = 50                 # Stop if no improvement for this many iters


class BudgetManager:
    """Manages compute budget for the refinement loop on a single ARC task."""

    def __init__(self, config: BudgetConfig = None):
        self.config = config or BudgetConfig()
        # Per-task state
        self._start_time: Optional[float] = None
        self._iteration: int = 0
        self._candidates: List[Tuple[int, float, float]] = []  # (iter, score, confidence)
        self._best_score: float = -1.0
        self._best_confidence: float = 0.0
        self._iters_since_improvement: int = 0
        self._stopped_reason: Optional[str] = None

    def start_task(self) -> None:
        """Reset all state for a new task."""
        self._start_time = time.time()
        self._iteration = 0
        self._candidates = []
        self._best_score = -1.0
        self._best_confidence = 0.0
        self._iters_since_improvement = 0
        self._stopped_reason = None

    def should_continue(self) -> bool:
        """Check if we have budget remaining and should keep iterating.

        Returns False (and sets _stopped_reason) when any stopping condition is met.
        Conditions checked in order:
          1. Task not started
          2. Below min_iterations -> always continue
          3. Confidence threshold reached
          4. Patience exhausted (no improvement for N iters)
          5. Time limit exceeded
          6. Max iterations exceeded
        """
        if self._start_time is None:
            self._stopped_reason = "task_not_started"
            return False

        # Always run at least min_iterations
        if self._iteration < self.config.min_iterations:
            return True

        # Confidence threshold
        if self._best_confidence >= self.config.confidence_threshold:
            self._stopped_reason = "confidence_reached"
            return False

        # Patience — no improvement for N iterations
        if self._iters_since_improvement >= self.config.patience:
            self._stopped_reason = "patience_exhausted"
            return False

        # Time limit
        if self.remaining_time() <= 0:
            self._stopped_reason = "time_limit"
            return False

        # Iteration limit
        if self._iteration >= self.config.max_iterations:
            self._stopped_reason = "max_iterations"
            return False

        return True

    def record_iteration(self, score: float, confidence: float) -> None:
        """Record the result of one refinement iteration.

        Args:
            score: Quality score for this candidate (e.g., cell accuracy). Higher is better.
            confidence: Model's self-assessed confidence in the prediction, 0-1.
        """
        self._candidates.append((self._iteration, score, confidence))
        self._iteration += 1

        if score > self._best_score:
            self._best_score = score
            self._iters_since_improvement = 0
        else:
            self._iters_since_improvement += 1

        if confidence > self._best_confidence:
            self._best_confidence = confidence

    def get_best_candidates(self, k: int = None) -> List[Tuple[int, float]]:
        """Return top-k candidates by score.

        Args:
            k: Number of candidates to return. Defaults to pass_at_k from config.

        Returns:
            List of (iteration_index, score) tuples, sorted best-first.
        """
        if k is None:
            k = self.config.pass_at_k

        # Sort by score descending, break ties by preferring later iterations
        sorted_candidates = sorted(
            self._candidates,
            key=lambda c: (c[1], c[0]),  # (score, iter) — higher is better for both
            reverse=True,
        )
        return [(idx, score) for idx, score, _ in sorted_candidates[:k]]

    def get_stats(self) -> Dict:
        """Return budget usage stats."""
        elapsed = 0.0
        if self._start_time is not None:
            elapsed = time.time() - self._start_time

        scores = [s for _, s, _ in self._candidates]
        confidences = [c for _, _, c in self._candidates]

        return {
            "iterations_used": self._iteration,
            "iterations_remaining": self.remaining_iterations(),
            "time_elapsed": round(elapsed, 2),
            "time_remaining": round(self.remaining_time(), 2),
            "best_score": self._best_score if self._candidates else None,
            "best_confidence": self._best_confidence if self._candidates else None,
            "mean_score": sum(scores) / len(scores) if scores else None,
            "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
            "num_candidates": len(self._candidates),
            "iters_since_improvement": self._iters_since_improvement,
            "stopped_reason": self._stopped_reason,
        }

    def remaining_time(self) -> float:
        """Seconds remaining in the time budget."""
        if self._start_time is None:
            return self.config.max_time_seconds
        elapsed = time.time() - self._start_time
        return max(0.0, self.config.max_time_seconds - elapsed)

    def remaining_iterations(self) -> int:
        """Iterations remaining in the iteration budget."""
        return max(0, self.config.max_iterations - self._iteration)

    @property
    def iteration(self) -> int:
        """Current iteration count."""
        return self._iteration

    @property
    def best_score(self) -> float:
        """Best score seen so far."""
        return self._best_score

    @property
    def stopped_reason(self) -> Optional[str]:
        """Why the loop stopped, or None if still running."""
        return self._stopped_reason
