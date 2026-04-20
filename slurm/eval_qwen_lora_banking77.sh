#!/bin/bash
#SBATCH --job-name=reval_qwen_b77
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/reval_qwen_banking77_%j.out
#SBATCH --error=logs/reval_qwen_banking77_%j.err

# Companion to scripts/eval_qwen_lora_banking77.py. No retrain — adapter is the
# source of truth. This job re-runs generation on val+test with the fixed
# parser cascade and writes artifacts under the `_reval` suffix so the buggy
# run's artifacts are preserved for the debugging retro.

set -euo pipefail

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/.conda")
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate customer-support-llm

export TOKENIZERS_PARALLELISM=false

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID:-<none>}"
echo "Node:          $(hostname)"
echo "Start time:    $(date)"
echo "Python:        $(which python)"
echo "=========================================="
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else '')"
echo "=========================================="

cd "$HOME/customer-support-llm"

python -u scripts/eval_qwen_lora_banking77.py \
    --adapter checkpoints/qwen_lora_banking77/qwen_lora_banking77_20260420_3eb1a21/final_adapter \
    --config configs/qwen_lora_banking77_config.yaml \
    --run-id-suffix _reval \
    "$@"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
