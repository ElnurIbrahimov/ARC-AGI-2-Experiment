"""
Learning rate scheduler: linear warmup + cosine annealing.
"""

import math
from typing import Dict

import torch


class WarmupCosineScheduler:
    """
    Warmup for N steps linearly from 0 to peak_lr,
    then cosine anneal to min_lr over remaining steps.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        peak_lr: float = 3e-4,
        min_lr: float = 1e-5,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.current_step = 0

        # Initialize LR to 0 (start of warmup)
        self._set_lr(0.0 if warmup_steps > 0 else peak_lr)

    def step(self) -> None:
        """Advance one step and update learning rate."""
        self.current_step += 1
        lr = self.get_lr()
        self._set_lr(lr)

    def get_lr(self) -> float:
        """Compute learning rate for the current step."""
        step = self.current_step

        if step < self.warmup_steps:
            # Linear warmup: 0 -> peak_lr
            return self.peak_lr * (step / max(1, self.warmup_steps))

        if step >= self.total_steps:
            return self.min_lr

        # Cosine annealing: peak_lr -> min_lr
        decay_steps = self.total_steps - self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, decay_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay

    def _set_lr(self, lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self) -> Dict:
        return {
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'peak_lr': self.peak_lr,
            'min_lr': self.min_lr,
            'current_step': self.current_step,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.peak_lr = state_dict['peak_lr']
        self.min_lr = state_dict['min_lr']
        self.current_step = state_dict['current_step']
        # Restore the LR corresponding to the loaded step
        self._set_lr(self.get_lr())
