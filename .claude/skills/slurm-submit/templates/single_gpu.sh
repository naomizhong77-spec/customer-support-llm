#!/bin/bash
#SBATCH --job-name=CHANGE_ME
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# === Environment setup ===
source ~/.bashrc
conda activate customer-support-llm

# === Logging ===
echo "=========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node:          $(hostname)"
echo "Start time:    $(date)"
echo "Working dir:   $(pwd)"
echo "Python:        $(which python)"
echo "=========================================="
nvidia-smi
echo "=========================================="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
echo "=========================================="

# === Run training ===
cd ~/customer-support-llm

# TODO: replace with actual script
python -u scripts/YOUR_SCRIPT.py "$@"

# === End logging ===
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
