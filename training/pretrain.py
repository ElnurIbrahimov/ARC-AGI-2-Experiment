"""
Stage 1: Pretrain the 7B HybridARC model on synthetic DSL tasks.

Usage:
    torchrun --nproc_per_node=4 training/pretrain.py [args]

Synthetic tasks are generated on-the-fly by SyntheticTaskGenerator.
The model learns to predict DSL token sequences from input-output grid pairs.
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

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
from data.synthetic_tasks import SyntheticTaskGenerator, SyntheticDataset
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
logger = logging.getLogger("pretrain")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Pretrain on synthetic tasks")
    tc = TrainingConfig()

    parser.add_argument("--stage", type=str, default="pretrain")
    parser.add_argument("--batch_size", type=int, default=tc.pretrain_batch_size)
    parser.add_argument("--grad_accum", type=int, default=tc.pretrain_grad_accum)
    parser.add_argument("--lr", type=float, default=tc.pretrain_lr)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--total_steps", type=int, default=tc.pretrain_total_steps)
    parser.add_argument("--warmup_steps", type=int, default=tc.pretrain_warmup_steps)
    parser.add_argument("--weight_decay", type=float, default=tc.weight_decay)
    parser.add_argument("--max_grad_norm", type=float, default=tc.max_grad_norm)
    parser.add_argument("--checkpoint_dir", type=str, default="/workspace/checkpoints/pretrain")
    parser.add_argument("--log_every", type=int, default=tc.log_every_steps)
    parser.add_argument("--save_every", type=int, default=tc.save_every_steps)
    parser.add_argument("--eval_every", type=int, default=tc.eval_every_steps)
    parser.add_argument("--seed", type=int, default=tc.seed)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--wandb_project", type=str, default="arc-agi-2-pretrain")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    return parser.parse_args()


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    """AdamW with no weight decay on norms and biases."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No decay on bias, norm weights (any parameter with 'norm' or 'bias' in name)
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


def build_dataloader(args, rank: int, world_size: int):
    """Build infinite synthetic data loader."""
    generator = SyntheticTaskGenerator(
        num_demos=3,
        min_grid_size=3,
        max_grid_size=15,
        max_program_depth=3,
        seed=args.seed + rank,
    )
    tokenizer = GridTokenizer(max_seq_len=args.max_seq_len)
    dataset = SyntheticDataset(
        generator=generator,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        # No sampler needed for IterableDataset
    )
    return loader


def collate_synthetic(batch):
    """Collate function for SyntheticDataset items."""
    token_ids = torch.stack([item["token_ids"] for item in batch])
    row_ids = torch.stack([item["row_ids"] for item in batch])
    col_ids = torch.stack([item["col_ids"] for item in batch])

    # target_tokens vary in length -- pad to max in batch
    target_lens = [item["target_tokens"].shape[0] for item in batch]
    max_target_len = max(target_lens)
    padded_targets = []
    for item in batch:
        t = item["target_tokens"]
        if t.shape[0] < max_target_len:
            pad_len = max_target_len - t.shape[0]
            t = torch.cat([t, torch.zeros(pad_len, dtype=torch.long)])
        padded_targets.append(t)
    target_tokens = torch.stack(padded_targets)

    return {
        "token_ids": token_ids,
        "row_ids": row_ids,
        "col_ids": col_ids,
        "target_tokens": target_tokens,
    }


def run_quick_eval(model, tokenizer, generator, device, rank, num_tasks=4):
    """Quick eval: generate on a few synthetic tasks, check token accuracy."""
    if rank != 0:
        return {}

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(num_tasks):
            task = generator.generate_task()
            encoded = tokenizer.encode_task(
                task["demo_inputs"], task["demo_outputs"], task["test_input"]
            )
            encoded = tokenizer.pad_to_length(encoded, model.config.max_seq_len)

            token_ids = encoded["token_ids"].unsqueeze(0).to(device)
            row_ids = encoded["row_ids"].unsqueeze(0).to(device)
            col_ids = encoded["col_ids"].unsqueeze(0).to(device)

            output = model(token_ids, row_ids, col_ids)
            # Check if model can predict the target tokens
            target = tokenizer.encode_target(task["test_output"])
            target_tokens = target["token_ids"].to(device)
            target_len = target_tokens.shape[0]

            # Compare last target_len positions of logits
            if output.logits.shape[1] >= target_len:
                pred_tokens = output.logits[0, -target_len:, :].argmax(dim=-1)
                non_pad = target_tokens != PAD_TOKEN
                correct += (pred_tokens[non_pad] == target_tokens[non_pad]).sum().item()
                total += non_pad.sum().item()

    model.train()
    accuracy = correct / max(total, 1)
    return {"eval_token_accuracy": accuracy, "eval_total_tokens": total}


def train():
    args = parse_args()

    # Setup distributed
    dist_info = setup_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    device = dist_info["device"]

    # Set seed (different per rank for data diversity)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + rank)

    # Logging
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    print_rank0(f"Stage 1: Pretrain | {world_size} GPUs | batch_size={args.batch_size} "
                f"| grad_accum={args.grad_accum} | effective_batch={args.batch_size * args.grad_accum * world_size}",
                rank)

    # Build model
    config = ModelConfig()
    model = HybridARC(config)
    model.enable_gradient_checkpointing()
    print_rank0(f"Model built: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params", rank)

    # Wrap with FSDP
    model = wrap_model_fsdp(model, mixed_precision=True)
    print_rank0("FSDP wrapped", rank)

    # Build optimizer and scheduler
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        peak_lr=args.lr,
        min_lr=args.min_lr,
    )

    # Build loss
    criterion = ARCLoss.for_stage(1)

    # Build data
    loader = build_dataloader(args, rank, world_size)
    data_iter = iter(loader)

    # For quick eval
    eval_generator = SyntheticTaskGenerator(num_demos=3, seed=args.seed + 9999)
    eval_tokenizer = GridTokenizer(max_seq_len=args.max_seq_len)

    # Resume from checkpoint
    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(model, optimizer, scheduler, args.resume_from, device)
        print_rank0(f"Resumed from step {start_step}", rank)

    # Initialize wandb
    if WANDB_AVAILABLE and not args.no_wandb and rank == 0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"pretrain-{world_size}gpu",
            config=vars(args),
        )

    # Checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training loop
    model.train()
    step = start_step
    total_tokens_processed = 0
    log_loss_accum = 0.0
    log_dsl_loss_accum = 0.0
    log_moe_loss_accum = 0.0
    log_steps = 0
    t_start = time.time()

    print_rank0(f"Starting training from step {start_step} to {args.total_steps}", rank)

    while step < args.total_steps:
        optimizer.zero_grad()
        micro_losses = []

        for micro_step in range(args.grad_accum):
            # Get next batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            token_ids = batch["token_ids"].to(device)
            row_ids = batch["row_ids"].to(device)
            col_ids = batch["col_ids"].to(device)
            target_tokens = batch["target_tokens"].to(device)

            # Forward
            output: ModelOutput = model(token_ids, row_ids, col_ids)

            # Compute target logits: we align logits with target_tokens
            # The model outputs logits for each input position.
            # For pretraining, target_tokens are the DSL program tokens.
            # We use the last target_len positions of the logits sequence.
            B = token_ids.shape[0]
            target_len = target_tokens.shape[1]
            seq_len = output.logits.shape[1]

            if seq_len >= target_len:
                # Use last target_len positions
                pred_logits = output.logits[:, -target_len:, :]
            else:
                # Truncate target if sequence is shorter
                pred_logits = output.logits
                target_tokens = target_tokens[:, :seq_len]

            # Build model_output dict for ARCLoss
            model_out = {
                "logits": pred_logits,
                "aux_loss": output.aux_loss,
            }
            targets = {
                "target_tokens": target_tokens,
            }

            # Loss
            loss_dict = criterion(model_out, targets)
            loss = loss_dict["total"] / args.grad_accum

            # Backward
            loss.backward()
            micro_losses.append(loss_dict)

            total_tokens_processed += token_ids.numel()

        # Gradient clipping
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # Optimizer step
        optimizer.step()
        scheduler.step()
        step += 1

        # Accumulate for logging
        avg_total_loss = sum(d["total"].item() for d in micro_losses) / len(micro_losses)
        avg_dsl_loss = sum(d["dsl_token"].item() for d in micro_losses) / len(micro_losses)
        avg_moe_loss = sum(d["moe_aux"].item() for d in micro_losses) / len(micro_losses)

        log_loss_accum += avg_total_loss
        log_dsl_loss_accum += avg_dsl_loss
        log_moe_loss_accum += avg_moe_loss
        log_steps += 1

        # Logging
        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            tokens_per_sec = total_tokens_processed / max(elapsed, 1e-6)
            avg_loss = log_loss_accum / max(log_steps, 1)
            avg_dsl = log_dsl_loss_accum / max(log_steps, 1)
            avg_moe = log_moe_loss_accum / max(log_steps, 1)
            current_lr = scheduler.get_lr()

            # All-reduce the loss for consistent logging across ranks
            loss_tensor = torch.tensor(avg_loss, device=device)
            loss_tensor = all_reduce_mean(loss_tensor)

            print_rank0(
                f"Step {step}/{args.total_steps} | "
                f"loss={loss_tensor.item():.4f} | "
                f"dsl_loss={avg_dsl:.4f} | "
                f"moe_aux={avg_moe:.6f} | "
                f"lr={current_lr:.2e} | "
                f"tok/s={tokens_per_sec:.0f}",
                rank,
            )

            if WANDB_AVAILABLE and not args.no_wandb and rank == 0:
                wandb.log({
                    "train/total_loss": loss_tensor.item(),
                    "train/dsl_token_loss": avg_dsl,
                    "train/moe_aux_loss": avg_moe,
                    "train/learning_rate": current_lr,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step": step,
                }, step=step)

            log_loss_accum = 0.0
            log_dsl_loss_accum = 0.0
            log_moe_loss_accum = 0.0
            log_steps = 0

        # Quick eval
        if step % args.eval_every == 0:
            eval_metrics = run_quick_eval(
                model, eval_tokenizer, eval_generator, device, rank, num_tasks=4
            )
            if rank == 0 and eval_metrics:
                print_rank0(
                    f"  Eval: token_accuracy={eval_metrics['eval_token_accuracy']:.4f} "
                    f"({eval_metrics['eval_total_tokens']} tokens)",
                    rank,
                )
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

    print_rank0(f"Pretraining complete. Final checkpoint: {ckpt_path}", rank)

    if WANDB_AVAILABLE and not args.no_wandb and rank == 0:
        wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    train()
