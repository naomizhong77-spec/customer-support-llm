#!/bin/bash
#SBATCH --job-name=train_qwen_lora
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                   # 7B bf16 download buffer + 4-bit quantized weights; 24G/32G OOM-killed in smoke
#SBATCH --time=04:00:00             # expected ~2-3h train + ~15min eval; buffer under 6h ceiling
#SBATCH --output=logs/train_qwen_lora_%j.out
#SBATCH --error=logs/train_qwen_lora_%j.err

set -euo pipefail

# ============================================================
# Robust conda activation (works even if ~/.bashrc doesn't init conda)
# ============================================================
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/.conda")
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate customer-support-llm

# Silence HF tokenizers fork warnings (DataLoader workers benign)
export TOKENIZERS_PARALLELISM=false

# ============================================================
# Environment debug logging (helps triage when jobs fail fast)
# ============================================================
echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID:-<none>}"
echo "Job Name:      ${SLURM_JOB_NAME:-<none>}"
echo "Node:          $(hostname)"
echo "Start time:    $(date)"
echo "Working dir:   $(pwd)"
echo "Python:        $(which python)"
echo "Python ver:    $(python --version)"
echo "=========================================="
nvidia-smi || echo "WARN: nvidia-smi failed"
echo "=========================================="
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'device_count', torch.cuda.device_count(), 'device_name', torch.cuda.get_device_name() if torch.cuda.is_available() else 'n/a')"
python -c "import transformers, datasets, peft, trl; print('transformers', transformers.__version__, 'datasets', datasets.__version__, 'peft', peft.__version__, 'trl', trl.__version__)"
python -c "from importlib.metadata import version; print('unsloth', version('unsloth'))"
python -c "import bitsandbytes as bnb; print('bitsandbytes', bnb.__version__)"
echo "=========================================="

# ============================================================
# Run training
# ============================================================
cd "$HOME/customer-support-llm"
mkdir -p logs checkpoints outputs/metrics outputs/figures outputs/error_analysis

# Unbuffered python so log lines flush immediately.
python -u scripts/train_qwen_lora.py \
    --config configs/qwen_lora_config.yaml \
    "$@"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
