#!/bin/bash
# RunPod setup script for ARC-AGI-2 training
# Run this once when starting a new RunPod instance

set -e

echo "=== ARC-AGI-2 RunPod Setup ==="

# Install system dependencies
apt-get update && apt-get install -y git wget tmux htop

# Install Python packages
pip install --upgrade pip
pip install -r /workspace/ARC-AGI-2-Experiment/requirements.txt

# Download ARC-AGI-2 dataset
mkdir -p /workspace/data/arc-agi-2
cd /workspace/data/arc-agi-2
if [ ! -d "training" ]; then
    echo "Downloading ARC-AGI-2 dataset..."
    git clone https://github.com/fchollet/ARC-AGI.git .
    echo "Dataset ready."
else
    echo "Dataset already exists."
fi

# Create checkpoint directory
mkdir -p /workspace/checkpoints

# Create log directory
mkdir -p /workspace/logs

# Set up wandb (optional)
# wandb login $WANDB_API_KEY

echo "=== Setup complete ==="
echo "To start training: bash /workspace/ARC-AGI-2-Experiment/scripts/run_pretrain.sh"
