#!/bin/bash
# Evaluate on ARC-AGI-2 (1xH100)
set -e

mkdir -p /workspace/logs
mkdir -p /workspace/results

cd /workspace/ARC-AGI-2-Experiment

python eval/evaluate.py \
    --checkpoint /workspace/checkpoints/finetune/latest.pt \
    --data_dir /workspace/data/arc-agi-2 \
    --split evaluation \
    --max_iterations 300 \
    --max_time 120 \
    --pass_at_k 2 \
    --output_dir /workspace/results \
    --visualize \
    2>&1 | tee /workspace/logs/eval_$(date +%Y%m%d_%H%M%S).log
