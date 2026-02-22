#!/bin/bash
# Stage 1: Pretrain on synthetic tasks
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

mkdir -p /workspace/logs

cd /workspace/ARC-AGI-2-Experiment

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    training/pretrain.py \
    --stage pretrain \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 3e-4 \
    --total_steps 100000 \
    --warmup_steps 2000 \
    --checkpoint_dir /workspace/checkpoints/pretrain \
    --data_dir /workspace/data/arc-agi-2 \
    --log_every 10 \
    --save_every 1000 \
    --eval_every 500 \
    2>&1 | tee /workspace/logs/pretrain_$(date +%Y%m%d_%H%M%S).log
