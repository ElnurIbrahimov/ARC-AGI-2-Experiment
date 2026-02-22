#!/bin/bash
# Stage 2: Finetune on ARC training set (1xH100, budget-optimized)
set -e

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

mkdir -p /workspace/logs

cd /workspace/ARC-AGI-2-Experiment

python training/finetune_arc.py \
    --stage finetune \
    --batch_size 4 \
    --grad_accum 32 \
    --lr 1e-4 \
    --total_steps 5000 \
    --warmup_steps 300 \
    --checkpoint_dir /workspace/checkpoints/finetune \
    --pretrain_checkpoint /workspace/checkpoints/pretrain/latest.pt \
    --data_dir /workspace/data/arc-agi-2 \
    --log_every 10 \
    --save_every 500 \
    --eval_every 250 \
    2>&1 | tee /workspace/logs/finetune_$(date +%Y%m%d_%H%M%S).log
