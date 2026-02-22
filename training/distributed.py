"""
FSDP/DDP wrappers for multi-GPU training on RunPod.
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial


def setup_distributed(backend: str = 'nccl') -> Dict:
    """
    Initialize distributed training.
    Returns dict with rank, world_size, local_rank, device.
    Uses env vars: RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT.
    """
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if world_size > 1:
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )

    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    return {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        'device': device,
    }


def cleanup_distributed() -> None:
    """Cleanup distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_fsdp(
    model: nn.Module,
    mixed_precision: bool = True,
) -> nn.Module:
    """
    Wrap model with FSDP for memory-efficient distributed training.
    - Use bf16 mixed precision
    - Shard parameters across GPUs
    - Enable gradient checkpointing on Mamba and Attention blocks
    - Use size-based auto wrapping (min 1M params per shard)
    """
    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    auto_wrap_policy = partial(
        size_based_auto_wrap_policy,
        min_num_params=1_000_000,
    )

    # Enable gradient checkpointing on eligible submodules before wrapping
    _enable_gradient_checkpointing(model)

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )

    return model


def _enable_gradient_checkpointing(model: nn.Module) -> None:
    """
    Enable gradient checkpointing on Mamba and Attention blocks.
    Looks for modules with 'mamba' or 'attention' in their class name (case-insensitive).
    """
    from torch.utils.checkpoint import checkpoint

    for name, module in model.named_modules():
        cls_name = module.__class__.__name__.lower()
        if 'mamba' in cls_name or 'attention' in cls_name:
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()


def wrap_model_ddp(model: nn.Module, device_id: int) -> nn.Module:
    """Simple DDP wrapper (fallback if FSDP not available)."""
    return DDP(
        model,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=False,
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    path: str,
    rank: int = 0,
) -> None:
    """Save distributed checkpoint (only on rank 0)."""
    if rank != 0:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Handle FSDP / DDP wrapped models
    if isinstance(model, FSDP):
        # For FSDP, we need full state dict consolidation
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            model_state = model.state_dict()
    elif isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        'step': step,
        'model': model_state,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
    }

    torch.save(checkpoint, path)

    # Also save a 'latest' symlink/copy for convenience
    latest_path = os.path.join(os.path.dirname(path), 'latest.pt')
    torch.save(checkpoint, latest_path)

    print_rank0(f"Checkpoint saved: {path} (step {step})", rank)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    path: str,
    device: torch.device,
) -> int:
    """Load checkpoint, return step number."""
    if not os.path.exists(path):
        print_rank0(f"No checkpoint found at {path}, starting from scratch.")
        return 0

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Handle wrapped models
    if isinstance(model, (FSDP, DDP)):
        model_to_load = model.module if isinstance(model, DDP) else model
    else:
        model_to_load = model

    model_to_load.load_state_dict(checkpoint['model'])

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler is not None and checkpoint.get('scheduler') is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    step = checkpoint.get('step', 0)
    print_rank0(f"Loaded checkpoint from {path} (step {step})")
    return step


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average tensor across all processes."""
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def print_rank0(msg: str, rank: int = 0) -> None:
    """Print only on rank 0."""
    if rank == 0:
        print(msg)
