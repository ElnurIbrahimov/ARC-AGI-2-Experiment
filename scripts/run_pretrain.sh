#!/bin/bash
# Stage 1: Pretrain on synthetic tasks (1xH100, budget-optimized)
set -e

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

mkdir -p /workspace/logs

cd /workspace/ARC-AGI-2-Experiment

python training/pretrain.py \
    --stage pretrain \
    --batch_size 8 \
    --grad_accum 16 \
    --lr 3e-4 \
    --total_steps 20000 \
    --warmup_steps 1000 \
    --checkpoint_dir /workspace/checkpoints/pretrain \
    --data_dir /workspace/data/arc-agi-2 \
    --log_every 10 \
    --save_every 2000 \
    --eval_every 1000 \
    2>&1 | tee /workspace/logs/pretrain_$(date +%Y%m%d_%H%M%S).log
