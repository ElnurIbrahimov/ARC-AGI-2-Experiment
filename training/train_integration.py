"""
Stage 3: Train integration modules (Causeway, BroadMind, FluxMind adapters).

Usage:
    torchrun --nproc_per_node=4 training/train_integration.py --finetune_checkpoint /path/to/ckpt [args]

Freezes the 7B backbone, trains only:
- CausewayAdapter (~7.4M params)
- BroadMindAdapter (~6.5M params)
- FluxMindAdapter (~1.7M params)
- CausalProgramBridge (~0.6M params)
Total trainable: ~16.2M params
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
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
from integration.causal_program_bridge import CausalProgramBridge, build_causal_program_bridge

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
logger = logging.getLogger("train_integration")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: Train integration modules")
    tc = TrainingConfig()

    parser.add_argument("--stage", type=str, default="integration")
    parser.add_argument("--finetune_checkpoint", type=str, required=True,
                        help="Path to Stage 2 finetuned 7B checkpoint")
    parser.add_argument("--batch_size", type=int, default=tc.integration_batch_size)
    parser.add_argument("--grad_accum", type=int, default=tc.integration_grad_accum)
    parser.add_argument("--lr", type=float, default=tc.integration_lr)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--total_steps", type=int, default=tc.integration_total_steps)
    parser.add_argument("--warmup_steps", type=int, default=tc.integration_warmup_steps)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=tc.max_grad_norm)
    parser.add_argument("--checkpoint_dir", type=str, default="/workspace/checkpoints/integration")
    parser.add_argument("--data_dir", type=str, default="/workspace/data/arc-agi-2")
    parser.add_argument("--log_every", type=int, default=tc.log_every_steps)
    parser.add_argument("--save_every", type=int, default=tc.save_every_steps)
    parser.add_argument("--eval_every", type=int, default=tc.eval_every_steps)
    parser.add_argument("--seed", type=int, default=tc.seed)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=2)

    # Integration module toggles
    parser.add_argument("--enable_causeway", action="store_true", default=True)
    parser.add_argument("--no_causeway", action="store_true")
    parser.add_argument("--enable_broadmind", action="store_true", default=True)
    parser.add_argument("--no_broadmind", action="store_true")
    parser.add_argument("--enable_fluxmind", action="store_true", default=True)
    parser.add_argument("--no_fluxmind", action="store_true")
    parser.add_argument("--fusion_mode", type=str, default="learned",
                        choices=["learned", "weighted_avg", "max"])

    parser.add_argument("--wandb_project", type=str, default="arc-agi-2-integration")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()
    if args.no_causeway:
        args.enable_causeway = False
    if args.no_broadmind:
        args.enable_broadmind = False
    if args.no_fluxmind:
        args.enable_fluxmind = False
    return args


def freeze_backbone(model: HybridARC) -> None:
    """Freeze ALL backbone parameters (requires_grad = False)."""
    for param in model.parameters():
        param.requires_grad = False


def build_optimizer(bridge: nn.Module, lr: float, weight_decay: float):
    """AdamW for integration module parameters only."""
    decay_params = []
    no_decay_params = []

    for name, param in bridge.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    if not decay_params and not no_decay_params:
        raise RuntimeError("No trainable parameters found in integration modules")

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
    """Build training and eval data loaders."""
    tokenizer = GridTokenizer(max_seq_len=args.max_seq_len)

    train_dataset = ARCDataset(
        data_dir=args.data_dir,
        split="training",
        tokenizer=tokenizer,
        augment=True,
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

    eval_dataset = ARCDataset(
        data_dir=args.data_dir,
        split="evaluation",
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


def make_dummy_action(batch_size: int, d_action: int, device: torch.device) -> torch.Tensor:
    """Create a dummy action vector from hidden states for Causeway.
    In production the refinement loop provides real actions. During training
    we derive a pseudo-action from the model's hidden states."""
    return torch.randn(batch_size, d_action, device=device)


def run_eval(
    backbone: HybridARC,
    bridge: CausalProgramBridge,
    eval_loader,
    criterion: ARCLoss,
    device: torch.device,
    rank: int,
    max_batches: int = 20,
) -> Dict:
    """Evaluate integration modules on held-out ARC tasks."""
    if rank != 0:
        return {}

    backbone.eval()
    bridge.eval()
    total_loss = 0.0
    total_dsl_loss = 0.0
    total_causeway_loss = 0.0
    correct_tokens = 0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            if batch_idx >= max_batches:
                break

            token_ids = batch["token_ids"].to(device)
            row_ids = batch["row_ids"].to(device)
            col_ids = batch["col_ids"].to(device)
            target_tokens = batch["target_tokens"].to(device)

            # Forward through frozen backbone
            output: ModelOutput = backbone(token_ids, row_ids, col_ids)
            hidden_states = output.hidden_states  # (B, T, d_model)

            B = token_ids.shape[0]
            target_len = target_tokens.shape[1]
            seq_len = output.logits.shape[1]

            if seq_len >= target_len:
                pred_logits = output.logits[:, -target_len:, :]
            else:
                pred_logits = output.logits
                target_tokens = target_tokens[:, :seq_len]

            # Get bridge regularization losses
            causeway_reg = bridge.get_regularization_losses()
            causeway_reg_dict = None
            if causeway_reg:
                acyclicity = causeway_reg.get("causeway_acyclicity", torch.tensor(0.0, device=device))
                sparsity = causeway_reg.get("causeway_sparsity", torch.tensor(0.0, device=device))
                causeway_reg_dict = {"acyclicity": acyclicity, "sparsity": sparsity}

            model_out = {
                "logits": pred_logits,
                "aux_loss": output.aux_loss,
            }
            targets = {"target_tokens": target_tokens}

            loss_dict = criterion(model_out, targets, causeway_reg_losses=causeway_reg_dict)
            total_loss += loss_dict["total"].item()
            total_dsl_loss += loss_dict["dsl_token"].item()
            total_causeway_loss += loss_dict["causeway_structural"].item()

            # Token accuracy
            pred_ids = pred_logits.argmax(dim=-1)
            non_pad = target_tokens != PAD_TOKEN
            correct_tokens += (pred_ids[non_pad] == target_tokens[non_pad]).sum().item()
            total_tokens += non_pad.sum().item()

            num_batches += 1

    backbone.train()
    bridge.train()

    if num_batches == 0:
        return {"eval_note": "no eval data available"}

    return {
        "eval/total_loss": total_loss / num_batches,
        "eval/dsl_loss": total_dsl_loss / num_batches,
        "eval/causeway_loss": total_causeway_loss / num_batches,
        "eval/token_accuracy": correct_tokens / max(total_tokens, 1),
    }


def save_integration_checkpoint(
    bridge: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    path: str,
    rank: int,
) -> None:
    """Save only integration module checkpoint (not the frozen backbone)."""
    if rank != 0:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "step": step,
        "bridge": bridge.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }

    torch.save(checkpoint, path)

    latest_path = os.path.join(os.path.dirname(path), "latest.pt")
    torch.save(checkpoint, latest_path)

    print_rank0(f"Integration checkpoint saved: {path} (step {step})", rank)


def load_integration_checkpoint(
    bridge: nn.Module,
    optimizer,
    scheduler,
    path: str,
    device: torch.device,
) -> int:
    """Load integration module checkpoint."""
    if not os.path.exists(path):
        print_rank0(f"No checkpoint found at {path}, starting from scratch.")
        return 0

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    bridge.load_state_dict(checkpoint["bridge"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    step = checkpoint.get("step", 0)
    print_rank0(f"Loaded integration checkpoint from {path} (step {step})")
    return step


def train():
    args = parse_args()

    # Setup distributed
    dist_info = setup_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    device = dist_info["device"]

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + rank)

    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    modules_active = []
    if args.enable_causeway:
        modules_active.append("Causeway")
    if args.enable_broadmind:
        modules_active.append("BroadMind")
    if args.enable_fluxmind:
        modules_active.append("FluxMind")

    print_rank0(
        f"Stage 3: Train Integration | {world_size} GPUs | "
        f"modules=[{', '.join(modules_active)}] | "
        f"batch_size={args.batch_size} | grad_accum={args.grad_accum}",
        rank,
    )

    # --- Build and freeze 7B backbone ---
    config = ModelConfig()
    backbone = HybridARC(config)

    print_rank0(f"Loading finetuned checkpoint: {args.finetune_checkpoint}", rank)
    finetune_ckpt = torch.load(args.finetune_checkpoint, map_location="cpu", weights_only=False)
    backbone.load_state_dict(finetune_ckpt["model"])
    finetune_step = finetune_ckpt.get("step", 0)
    print_rank0(f"Loaded finetuned model from step {finetune_step}", rank)
    del finetune_ckpt

    # Freeze ALL backbone parameters
    freeze_backbone(backbone)
    backbone_params = sum(p.numel() for p in backbone.parameters())
    frozen_params = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)
    print_rank0(f"Backbone frozen: {frozen_params / 1e9:.2f}B / {backbone_params / 1e9:.2f}B params", rank)

    # Move backbone to device and set to eval (no dropout, etc.)
    backbone = backbone.to(device)
    backbone.eval()

    # --- Build integration modules ---
    bridge = build_causal_program_bridge(
        d_model=config.hidden_dim,
        d_causal=128,
        d_action=128,
        d_wisdom=48,
        fusion_mode=args.fusion_mode,
        enable_causeway=args.enable_causeway,
        enable_broadmind=args.enable_broadmind,
        enable_fluxmind=args.enable_fluxmind,
        device=str(device),
    )

    trainable_params = sum(p.numel() for p in bridge.parameters() if p.requires_grad)
    total_bridge_params = sum(p.numel() for p in bridge.parameters())
    print_rank0(
        f"Integration modules: {trainable_params / 1e6:.2f}M trainable / "
        f"{total_bridge_params / 1e6:.2f}M total params",
        rank,
    )

    # Log per-module breakdown
    if rank == 0:
        stats = bridge.get_module_stats()
        logger.info(f"Bridge stats: {stats}")

    # Wrap bridge with DDP (too small for FSDP -- DDP is more appropriate)
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        bridge = DDP(bridge, device_ids=[dist_info["local_rank"]], find_unused_parameters=True)
        bridge_module = bridge.module
    else:
        bridge_module = bridge

    # Build optimizer (only integration params)
    optimizer = build_optimizer(bridge, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        peak_lr=args.lr,
        min_lr=args.min_lr,
    )

    # Loss (Stage 3: causeway structural loss active)
    criterion = ARCLoss.for_stage(3)

    # Data
    train_loader, eval_loader, train_sampler = build_dataloaders(args, rank, world_size)
    print_rank0(f"Training dataset: {len(train_loader.dataset)} tasks", rank)
    print_rank0(f"Eval dataset: {len(eval_loader.dataset)} tasks", rank)

    # Resume
    start_step = 0
    if args.resume_from:
        start_step = load_integration_checkpoint(
            bridge_module, optimizer, scheduler, args.resume_from, device
        )
        print_rank0(f"Resumed from integration step {start_step}", rank)

    # wandb
    if WANDB_AVAILABLE and not args.no_wandb and rank == 0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"integration-{world_size}gpu",
            config=vars(args),
        )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training loop
    bridge.train()
    step = start_step
    epoch = 0
    total_tokens_processed = 0
    log_loss_accum = 0.0
    log_dsl_loss_accum = 0.0
    log_causeway_loss_accum = 0.0
    log_moe_loss_accum = 0.0
    log_steps = 0
    t_start = time.time()

    print_rank0(f"Starting integration training from step {start_step} to {args.total_steps}", rank)

    while step < args.total_steps:
        epoch += 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            if step >= args.total_steps:
                break

            optimizer.zero_grad()
            micro_losses = []

            token_ids = batch["token_ids"].to(device)
            row_ids = batch["row_ids"].to(device)
            col_ids = batch["col_ids"].to(device)
            target_tokens = batch["target_tokens"].to(device)

            B = token_ids.shape[0]

            # Forward through frozen backbone (no grad needed)
            with torch.no_grad():
                output: ModelOutput = backbone(token_ids, row_ids, col_ids)
                hidden_states = output.hidden_states  # (B, T, d_model)
                logits = output.logits
                aux_loss = output.aux_loss

            # Pool hidden states for bridge input: mean over non-PAD positions
            non_pad_mask = (token_ids != PAD_TOKEN).unsqueeze(-1).float()  # (B, T, 1)
            pooled_hidden = (hidden_states * non_pad_mask).sum(dim=1) / non_pad_mask.sum(dim=1).clamp(min=1)
            # pooled_hidden: (B, d_model)

            # Create pseudo-action from the hidden states
            # In the full refinement loop, actions come from DSL modifications.
            # During training, we derive them by projecting the pooled hidden state.
            d_action = 128
            action = pooled_hidden[:, :d_action]  # (B, d_action) -- simple slice

            # Grid embedding = pooled hidden state
            grid_embedding = pooled_hidden

            # Op sequence: use the token_ids for DSL operation sequence
            op_sequence = token_ids

            # Run bridge forward (trainable)
            # This exercises all integration adapters
            program_ops = []  # Would be real DSL ops in refinement loop
            examples = []     # Would be real grid pairs in refinement loop

            result = bridge_module.forward(
                h=pooled_hidden,
                action=action,
                grid_embedding=grid_embedding,
                op_sequence=op_sequence,
                program_ops=program_ops,
                examples=examples,
            )

            # Get regularization losses from Causeway
            causeway_reg = bridge_module.get_regularization_losses()
            causeway_reg_dict = None
            if causeway_reg:
                acyclicity = causeway_reg.get(
                    "causeway_acyclicity", torch.tensor(0.0, device=device)
                )
                sparsity = causeway_reg.get(
                    "causeway_sparsity", torch.tensor(0.0, device=device)
                )
                causeway_reg_dict = {"acyclicity": acyclicity, "sparsity": sparsity}

            # Compute loss using backbone logits (backbone is frozen, but its
            # logits are the baseline -- the integration modules learn to improve
            # upon the backbone's predictions through the refinement process)
            target_len = target_tokens.shape[1]
            seq_len = logits.shape[1]

            if seq_len >= target_len:
                pred_logits = logits[:, -target_len:, :]
            else:
                pred_logits = logits
                target_tokens = target_tokens[:, :seq_len]

            # The logits from backbone are frozen, so the gradient flows
            # only through the bridge's regularization and structural losses
            model_out = {
                "logits": pred_logits.detach(),  # no grad through backbone logits
                "aux_loss": aux_loss.detach(),
            }
            targets = {"target_tokens": target_tokens}

            loss_dict = criterion(model_out, targets, causeway_reg_losses=causeway_reg_dict)

            # The DSL token loss is detached (from backbone), but causeway
            # structural loss flows through the bridge. We also add a
            # bridge-specific loss: train the fusion network to predict
            # whether the backbone's prediction would be correct.
            # This is a form of self-supervised learning for the bridge.
            bridge_loss = loss_dict["causeway_structural"]

            # Add a small auxiliary loss to train the score fusion network:
            # The fused_score should correlate with actual token accuracy
            with torch.no_grad():
                pred_ids = pred_logits.argmax(dim=-1)
                non_pad = target_tokens != PAD_TOKEN
                per_sample_accuracy = []
                for b in range(B):
                    sample_mask = non_pad[b]
                    if sample_mask.sum() == 0:
                        per_sample_accuracy.append(0.0)
                    else:
                        acc = (pred_ids[b][sample_mask] == target_tokens[b][sample_mask]).float().mean().item()
                        per_sample_accuracy.append(acc)
                accuracy_target = torch.tensor(per_sample_accuracy, device=device)

            # Fused score as proxy for program quality
            fused_score_tensor = torch.tensor(result.fused_score, device=device)
            score_supervision_loss = (fused_score_tensor - accuracy_target.mean()).pow(2)

            total_bridge_loss = bridge_loss + 0.1 * score_supervision_loss

            if total_bridge_loss.requires_grad:
                total_bridge_loss.backward()
            micro_losses.append(loss_dict)

            total_tokens_processed += token_ids.numel()

            # Gradient clipping
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(bridge.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            step += 1

            # Accumulate for logging
            avg_total = loss_dict["total"].item()
            avg_dsl = loss_dict["dsl_token"].item()
            avg_causeway = loss_dict["causeway_structural"].item()
            avg_moe = loss_dict["moe_aux"].item()

            log_loss_accum += avg_total
            log_dsl_loss_accum += avg_dsl
            log_causeway_loss_accum += avg_causeway
            log_moe_loss_accum += avg_moe
            log_steps += 1

            # Logging
            if step % args.log_every == 0:
                elapsed = time.time() - t_start
                tokens_per_sec = total_tokens_processed / max(elapsed, 1e-6)
                avg_loss = log_loss_accum / max(log_steps, 1)
                avg_d = log_dsl_loss_accum / max(log_steps, 1)
                avg_c = log_causeway_loss_accum / max(log_steps, 1)
                avg_m = log_moe_loss_accum / max(log_steps, 1)
                current_lr = scheduler.get_lr()

                loss_tensor = torch.tensor(avg_loss, device=device)
                loss_tensor = all_reduce_mean(loss_tensor)

                print_rank0(
                    f"Step {step}/{args.total_steps} | epoch {epoch} | "
                    f"loss={loss_tensor.item():.4f} | "
                    f"dsl={avg_d:.4f} | causeway={avg_c:.6f} | "
                    f"moe={avg_m:.6f} | lr={current_lr:.2e} | "
                    f"fused_score={result.fused_score:.4f} | "
                    f"tok/s={tokens_per_sec:.0f}",
                    rank,
                )

                if WANDB_AVAILABLE and not args.no_wandb and rank == 0:
                    log_dict = {
                        "train/total_loss": loss_tensor.item(),
                        "train/dsl_loss": avg_d,
                        "train/causeway_loss": avg_c,
                        "train/moe_aux_loss": avg_m,
                        "train/learning_rate": current_lr,
                        "train/fused_score": result.fused_score,
                        "train/fused_confidence": result.fused_confidence,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/epoch": epoch,
                        "train/step": step,
                    }
                    # Module-specific metrics
                    if result.fluxmind_score > 0:
                        log_dict["train/fluxmind_score"] = result.fluxmind_score
                    if result.compute_cost > 0:
                        log_dict["train/broadmind_compute_cost"] = result.compute_cost
                    wandb.log(log_dict, step=step)

                log_loss_accum = 0.0
                log_dsl_loss_accum = 0.0
                log_causeway_loss_accum = 0.0
                log_moe_loss_accum = 0.0
                log_steps = 0

            # Eval
            if step % args.eval_every == 0:
                eval_metrics = run_eval(
                    backbone, bridge_module, eval_loader, criterion, device, rank
                )
                if rank == 0 and eval_metrics:
                    metrics_str = " | ".join(
                        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in eval_metrics.items()
                    )
                    print_rank0(f"  Eval: {metrics_str}", rank)
                    if WANDB_AVAILABLE and not args.no_wandb:
                        wandb.log(eval_metrics, step=step)

                # Log module diagnostics
                if rank == 0:
                    stats = bridge_module.get_module_stats()
                    logger.info(f"Module stats: {stats}")

            # Save checkpoint (only bridge, not backbone)
            if step % args.save_every == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step}.pt")
                save_integration_checkpoint(
                    bridge_module, optimizer, scheduler, step, ckpt_path, rank
                )
                if world_size > 1:
                    dist.barrier()

    # Final save
    ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step}_final.pt")
    save_integration_checkpoint(bridge_module, optimizer, scheduler, step, ckpt_path, rank)

    print_rank0(f"Integration training complete. Final checkpoint: {ckpt_path}", rank)
    print_rank0(
        f"To use: load backbone from {args.finetune_checkpoint} + "
        f"bridge from {ckpt_path}",
        rank,
    )

    if WANDB_AVAILABLE and not args.no_wandb and rank == 0:
        wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    train()
