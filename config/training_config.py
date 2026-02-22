"""
Training hyperparameters for all 3 stages.
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # ── Stage 1: Pretrain on synthetic tasks ──
    pretrain_lr: float = 3e-4
    pretrain_warmup_steps: int = 2000
    pretrain_total_steps: int = 100_000
    pretrain_batch_size: int = 4           # Per GPU
    pretrain_grad_accum: int = 8           # Effective batch = 4 * 8 * num_gpus

    # ── Stage 2: Finetune on ARC training set ──
    finetune_lr: float = 1e-4
    finetune_warmup_steps: int = 500
    finetune_total_steps: int = 20_000
    finetune_batch_size: int = 2
    finetune_grad_accum: int = 16

    # ── Stage 3: Train integration modules only ──
    integration_lr: float = 5e-4
    integration_warmup_steps: int = 200
    integration_total_steps: int = 10_000
    integration_batch_size: int = 4
    integration_grad_accum: int = 4

    # ── Common ──
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    mixed_precision: str = 'bf16'
    seed: int = 42

    # ── RunPod ──
    num_gpus: int = 4
    gpu_type: str = 'A100_80GB'

    # ── Checkpointing ──
    save_every_steps: int = 1000
    eval_every_steps: int = 500
    log_every_steps: int = 10
    checkpoint_dir: str = '/workspace/checkpoints'

    # ── Gradient checkpointing ──
    gradient_checkpointing: bool = True
