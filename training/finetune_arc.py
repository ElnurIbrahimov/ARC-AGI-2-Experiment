"""
Stage 2: Finetune on ARC-AGI-2 training set.

Usage:
    torchrun --nproc_per_node=4 training/finetune_arc.py --pretrain_checkpoint /path/to/ckpt [args]

Loads pretrained checkpoint, finetunes on real ARC tasks with augmentation.
Lower learning rate, grid reconstruction loss active.
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from config.dsl_config import VOCAB_SIZE, PAD_TOKEN
from model.hybrid_arc import HybridARC, ModelOutput
from training.losses import ARCLoss
from training.scheduler import WarmupCosineScheduler
from training.distributed import (
    setup_distributed,
    cleanup_distributed,
    wrap_model_fsdp,
    save_checkpoint,
    load_checkpoint,
    all_reduce_mean,
    print_rank0,
)
from data.arc_dataset import ARCDataset
from data.grid_tokenizer import GridTokenizer

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("finetune_arc")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Finetune on ARC training set")
    tc = TrainingConfig()

    parser.add_argument("--stage", type=str, default="finetune")
    parser.add_argument("--pretrain_checkpoint", type=str, required=True,
                        help="Path to Stage 1 pretrained checkpoint")
    parser.add_argument("--batch_size", type=int, default=tc.finetune_batch_size)
    parser.add_argument("--grad_accum", type=int, default=tc.finetune_grad_accum)
    parser.add_argument("--lr", type=float, default=tc.finetune_lr)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--total_steps", type=int, default=tc.finetune_total_steps)
    parser.add_argument("--warmup_steps", type=int, default=tc.finetune_warmup_steps)
    parser.add_argument("--weight_decay", type=float, default=tc.weight_decay)
    parser.add_argument("--max_grad_norm", type=float, default=tc.max_grad_norm)
    parser.add_argument("--checkpoint_dir", type=str, default="/workspace/checkpoints/finetune")
    parser.add_argument("--data_dir", type=str, default="/workspace/data/arc-agi-2")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Enable data augmentation (dihedral + color permutation)")
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--log_every", type=int, default=tc.log_every_steps)
    parser.add_argument("--save_every", type=int, default=tc.save_every_steps)
    parser.add_argument("--eval_every", type=int, default=tc.eval_every_steps)
    parser.add_argument("--seed", type=int, default=tc.seed)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eval_split", type=str, default="evaluation",
                        help="Split to use for eval (evaluation or training subset)")
    parser.add_argument("--wandb_project", type=str, default="arc-agi-2-finetune")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()
    if args.no_augment:
        args.augment = False
    return args


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    """AdamW with no weight decay on norms and biases."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    return optimizer


def build_dataloaders(args, rank: int, world_size: int):
    """Build training and eval data loaders from ARC dataset."""
    tokenizer = GridTokenizer(max_seq_len=args.max_seq_len)

    # Training dataset
    train_dataset = ARCDataset(
        data_dir=args.data_dir,
        split="training",
        tokenizer=tokenizer,
        augment=args.augment,
        max_seq_len=args.max_seq_len,
    )

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=ARCDataset.collate_fn,
        drop_last=True,
    )

    # Eval dataset (no augmentation)
    eval_dataset = ARCDataset(
        data_dir=args.data_dir,
        split=args.eval_split,
        tokenizer=tokenizer,
        augment=False,
        max_seq_len=args.max_seq_len,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ARCDataset.collate_fn,
    )

    return train_loader, eval_loader, train_sampler


def run_eval(model, eval_loader, criterion, device, rank, max_batches=20):
    """Evaluate on held-out ARC tasks."""
    if rank != 0:
        return {}

    model.eval()
    total_loss = 0.0
    total_dsl_loss = 0.0
    total_grid_loss = 0.0
    correct_tokens = 0
    total_tokens = 0
    num_batches = 0
    grid_correct = 0
    grid_total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            if batch_idx >= max_batches:
                break

            token_ids = batch["token_ids"].to(device)
            row_ids = batch["row_ids"].to(device)
            col_ids = batch["col_ids"].to(device)
            target_tokens = batch["target_tokens"].to(device)

            output: ModelOutput = model(token_ids, row_ids, col_ids)

            B = token_ids.shape[0]
            target_len = target_tokens.shape[1]
            seq_len = output.logits.shape[1]

            if seq_len >= target_len:
                pred_logits = output.logits[:, -target_len:, :]
            else:
                pred_logits = output.logits
                target_tokens = target_tokens[:, :seq_len]

            model_out = {
                "logits": pred_logits,
                "aux_loss": output.aux_loss,
            }
            targets = {"target_tokens": target_tokens}

            loss_dict = criterion(model_out, targets)
            total_loss += loss_dict["total"].item()
            total_dsl_loss += loss_dict["dsl_token"].item()
            total_grid_loss += loss_dict["grid"].item()

            # Token accuracy
            pred_ids = pred_logits.argmax(dim=-1)
            non_pad = target_tokens != PAD_TOKEN
            correct_tokens += (pred_ids[non_pad] == target_tokens[non_pad]).sum().item()
            total_tokens += non_pad.sum().item()

            # Grid-level: check if predicted grid matches target exactly
            # (per-sample check)
            for b in range(B):
                sample_non_pad = target_tokens[b] != PAD_TOKEN
                if sample_non_pad.sum() == 0:
                    continue
                sample_pred = pred_ids[b][sample_non_pad]
                sample_target = target_tokens[b][sample_non_pad]
                grid_total += 1
                if torch.equal(sample_pred, sample_target):
                    grid_correct += 1

            num_batches += 1

    model.train()

    if num_batches == 0:
        return {"eval_note": "no eval data available"}

    return {
        "eval/total_loss": total_loss / num_batches,
        "eval/dsl_loss": total_dsl_loss / num_batches,
        "eval/grid_loss": total_grid_loss / num_batches,
        "eval/token_accuracy": correct_tokens / max(total_tokens, 1),
        "eval/grid_exact_match": grid_correct / max(grid_total, 1),
        "eval/num_tasks": grid_total,
    }


def train():
    args = parse_args()

    # Setup distributed
    dist_info = setup_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    device = dist_info["device"]

    # Set seed
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + rank)

    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    print_rank0(
        f"Stage 2: Finetune ARC | {world_size} GPUs | batch_size={args.batch_size} "
        f"| grad_accum={args.grad_accum} | effective_batch={args.batch_size * args.grad_accum * world_size} "
        f"| augment={args.augment}",
        rank,
    )

    # Build model
    config = ModelConfig()
    model = HybridARC(config)
    model.enable_gradient_checkpointing()

    # Load pretrained checkpoint (before FSDP wrapping)
    print_rank0(f"Loading pretrained checkpoint: {args.pretrain_checkpoint}", rank)
    pretrain_ckpt = torch.load(args.pretrain_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(pretrain_ckpt["model"])
    pretrain_step = pretrain_ckpt.get("step", 0)
    print_rank0(f"Loaded pretrained model from step {pretrain_step}", rank)
    del pretrain_ckpt

    # Wrap with FSDP
    model = wrap_model_fsdp(model, mixed_precision=True)
    print_rank0("FSDP wrapped", rank)

    # Build optimizer and scheduler (fresh, not from pretrain)
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        peak_lr=args.lr,
        min_lr=args.min_lr,
    )

    # Build loss (Stage 2: grid loss active)
    criterion = ARCLoss.for_stage(2)

    # Build data
    train_loader, eval_loader, train_sampler = build_dataloaders(args, rank, world_size)
    data_len = len(train_loader.dataset)
    print_rank0(f"Training dataset: {data_len} tasks", rank)
    print_rank0(f"Eval dataset: {len(eval_loader.dataset)} tasks", rank)

    # Resume from finetune checkpoint if specified
    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(model, optimizer, scheduler, args.resume_from, device)
        print_rank0(f"Resumed from finetune step {start_step}", rank)

    # Initialize wandb
    if WANDB_AVAILABLE and not args.no_wandb and rank == 0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"finetune-{world_size}gpu",
            config=vars(args),
        )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training loop
    model.train()
    step = start_step
    epoch = 0
    total_tokens_processed = 0
    log_loss_accum = 0.0
    log_dsl_loss_accum = 0.0
    log_grid_loss_accum = 0.0
    log_moe_loss_accum = 0.0
    log_steps = 0
    t_start = time.time()

    print_rank0(f"Starting finetuning from step {start_step} to {args.total_steps}", rank)

    while step < args.total_steps:
        epoch += 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            if step >= args.total_steps:
                break

            optimizer.zero_grad()
            micro_losses = []

            # For finite datasets, we may need to repeat within grad_accum
            # but for simplicity, each step processes one batch with the
            # accumulation happening over multiple batches
            # Use the current batch for all micro-steps within this iteration
            # (For proper grad accum over multiple batches, we'd need inner loop)

            token_ids = batch["token_ids"].to(device)
            row_ids = batch["row_ids"].to(device)
            col_ids = batch["col_ids"].to(device)
            target_tokens = batch["target_tokens"].to(device)

            # Split batch across grad_accum micro-steps if batch is large enough
            # Otherwise use the full batch as one micro-batch
            B = token_ids.shape[0]
            micro_batch_size = max(1, B // args.grad_accum) if B >= args.grad_accum else B
            num_micro = max(1, B // micro_batch_size)

            for micro_idx in range(num_micro):
                start_idx = micro_idx * micro_batch_size
                end_idx = min(start_idx + micro_batch_size, B)

                mb_token_ids = token_ids[start_idx:end_idx]
                mb_row_ids = row_ids[start_idx:end_idx]
                mb_col_ids = col_ids[start_idx:end_idx]
                mb_target_tokens = target_tokens[start_idx:end_idx]

                # Forward
                output: ModelOutput = model(mb_token_ids, mb_row_ids, mb_col_ids)

                target_len = mb_target_tokens.shape[1]
                seq_len = output.logits.shape[1]

                if seq_len >= target_len:
                    pred_logits = output.logits[:, -target_len:, :]
                else:
                    pred_logits = output.logits
                    mb_target_tokens = mb_target_tokens[:, :seq_len]

                model_out = {
                    "logits": pred_logits,
                    "aux_loss": output.aux_loss,
                }
                targets = {"target_tokens": mb_target_tokens}

                loss_dict = criterion(model_out, targets)
                loss = loss_dict["total"] / num_micro

                loss.backward()
                micro_losses.append(loss_dict)

                total_tokens_processed += mb_token_ids.numel()

            # Gradient clipping
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            step += 1

            # Accumulate for logging
            avg_total = sum(d["total"].item() for d in micro_losses) / len(micro_losses)
            avg_dsl = sum(d["dsl_token"].item() for d in micro_losses) / len(micro_losses)
            avg_grid = sum(d["grid"].item() for d in micro_losses) / len(micro_losses)
            avg_moe = sum(d["moe_aux"].item() for d in micro_losses) / len(micro_losses)

            log_loss_accum += avg_total
            log_dsl_loss_accum += avg_dsl
            log_grid_loss_accum += avg_grid
            log_moe_loss_accum += avg_moe
            log_steps += 1

            # Logging
            if step % args.log_every == 0:
                elapsed = time.time() - t_start
                tokens_per_sec = total_tokens_processed / max(elapsed, 1e-6)
                avg_loss = log_loss_accum / max(log_steps, 1)
                avg_d = log_dsl_loss_accum / max(log_steps, 1)
                avg_g = log_grid_loss_accum / max(log_steps, 1)
                avg_m = log_moe_loss_accum / max(log_steps, 1)
                current_lr = scheduler.get_lr()

                loss_tensor = torch.tensor(avg_loss, device=device)
                loss_tensor = all_reduce_mean(loss_tensor)

                print_rank0(
                    f"Step {step}/{args.total_steps} | epoch {epoch} | "
                    f"loss={loss_tensor.item():.4f} | "
                    f"dsl={avg_d:.4f} | grid={avg_g:.4f} | "
                    f"moe={avg_m:.6f} | lr={current_lr:.2e} | "
                    f"tok/s={tokens_per_sec:.0f}",
                    rank,
                )

                if WANDB_AVAILABLE and not args.no_wandb and rank == 0:
                    wandb.log({
                        "train/total_loss": loss_tensor.item(),
                        "train/dsl_loss": avg_d,
                        "train/grid_loss": avg_g,
                        "train/moe_aux_loss": avg_m,
                        "train/learning_rate": current_lr,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/epoch": epoch,
                        "train/step": step,
                    }, step=step)

                log_loss_accum = 0.0
                log_dsl_loss_accum = 0.0
                log_grid_loss_accum = 0.0
                log_moe_loss_accum = 0.0
                log_steps = 0

            # Eval
            if step % args.eval_every == 0:
                eval_metrics = run_eval(model, eval_loader, criterion, device, rank)
                if rank == 0 and eval_metrics:
                    metrics_str = " | ".join(
                        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in eval_metrics.items()
                    )
                    print_rank0(f"  Eval: {metrics_str}", rank)
                    if WANDB_AVAILABLE and not args.no_wandb:
                        wandb.log(eval_metrics, step=step)

            # Save checkpoint
            if step % args.save_every == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step}.pt")
                save_checkpoint(model, optimizer, scheduler, step, ckpt_path, rank)
                if world_size > 1:
                    dist.barrier()

    # Final save
    ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step}_final.pt")
    save_checkpoint(model, optimizer, scheduler, step, ckpt_path, rank)

    print_rank0(f"Finetuning complete. Final checkpoint: {ckpt_path}", rank)

    if WANDB_AVAILABLE and not args.no_wandb and rank == 0:
        wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    train()
