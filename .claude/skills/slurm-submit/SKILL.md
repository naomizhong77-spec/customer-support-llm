---
name: slurm-submit
description: Use this skill when writing, submitting, or monitoring SLURM batch jobs on the CCDS-TC2 cluster. Includes sbatch templates for single-GPU training, interactive debugging sessions, and common troubleshooting. Reference for `squeue`, `scancel`, `scontrol`, log monitoring, and time-limit management.
---

# SLURM Submission Guide for CCDS-TC2

## Cluster Quick Reference

- **Login node**: CCDS-TC2
- **Compute nodes**: TC2N01 - TC2N08
- **Partition**: `MGPU-TC2` (only relevant partition for this project)
- **GPU per node**: 4× NVIDIA A40 (48GB each)
- **Max wall time per job**: 6 hours (hard limit)
- **Your user**: zh0038qi
- **Home**: `/home/mcaai/zh0038qi` or `~`

## Essential Commands

```bash
# Submit job
sbatch slurm/train.sh

# Check my jobs
squeue -u zh0038qi

# Check all jobs in partition  
squeue -p MGPU-TC2

# Cancel job
scancel JOBID
scancel -u zh0038qi              # cancel all my jobs

# Job details
scontrol show job JOBID

# Node details  
scontrol show node TC2N03

# Partition info
sinfo -p MGPU-TC2

# Watch logs as they stream
tail -f logs/train_JOBID.out
tail -f logs/train_JOBID.err

# Interactive session (5 min) for debugging
srun -p MGPU-TC2 --gres=gpu:1 --time=00:05:00 --pty bash

# Dry-run / syntax check a sbatch script
sbatch --test-only slurm/train.sh
```

## Template: Single-GPU Training

Save as `slurm/train_xxx.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=train_xxx
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:30:00               # 30min buffer under 6h limit
#SBATCH --output=logs/%x_%j.out       # %x=job-name, %j=jobid
#SBATCH --error=logs/%x_%j.err

set -euo pipefail   # exit on error, undefined var, pipe failure

# === Environment setup ===
# Robust conda activation (works even if ~/.bashrc doesn't init conda)
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/.conda")
source "$CONDA_BASE/etc/profile.d/conda.sh"
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

# === Run ===
cd ~/customer-support-llm
python scripts/train_xxx.py "$@"

# === End logging ===
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
```

**Submit with custom args**:
```bash
sbatch slurm/train_xxx.sh --config configs/qwen_lora_config.yaml --seed 42
```

## Template: Interactive Debug Session

For quick testing/debugging on a real GPU (5-15 min sessions). Save as `slurm/interactive.sh`:

```bash
#!/bin/bash
# Usage: bash slurm/interactive.sh
# This launches an interactive shell on a GPU node.

srun -p MGPU-TC2 \
     --gres=gpu:1 \
     --cpus-per-task=4 \
     --mem=16G \
     --time=00:15:00 \
     --pty bash

# Once inside:
#   cd ~/customer-support-llm
#   conda activate customer-support-llm
#   python scripts/train_xxx.py --smoke-test
```

**NOTE**: Interactive sessions still count against your wall-time quota. Exit promptly when done.

## Template: CPU-Only Job (for data prep)

Save as `slurm/cpu_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=data_prep
#SBATCH --partition=MGPU-TC2     # or a CPU-only partition if available
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
# Robust conda activation (works even if ~/.bashrc doesn't init conda)
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/.conda")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate customer-support-llm

cd ~/customer-support-llm
python scripts/prepare_instruction_data.py "$@"
```

## Monitoring Workflow

After submitting a long-running job:

```bash
# 1. Get job ID from sbatch output, e.g. "Submitted batch job 16700"
JOBID=16700

# 2. Wait for it to start running
squeue -u zh0038qi
# STATE column: PD = pending, R = running, CG = completing

# 3. Once RUNNING, tail the log (Ctrl+C to stop tailing, doesn't affect job)
tail -f logs/train_xxx_$JOBID.out

# 4. In another terminal, check GPU usage on the node it's running on
NODE=$(squeue -j $JOBID -o "%N" -h)   # get node name
ssh $NODE nvidia-smi                   # check GPU util (if ssh to node allowed)

# 5. If you need to kill it
scancel $JOBID
```

## Time Management

A40 + Qwen2.5-7B QLoRA typical timings:
- Smoke test (100 samples, 1 epoch): ~2-3 min
- Full training (27k samples, 3 epochs): ~90-150 min
- Evaluation (full test set, batch=1): ~30-60 min

Build safety buffer:
- Expected runtime × 1.5 = requested time
- Never request exactly 6h; use 5:30 max to stay safely under limit

## Troubleshooting

### Job stuck in PENDING state

Check the REASON column in `squeue`:
- `Resources` — waiting for GPUs to free up (just wait)
- `Priority` — other jobs ahead in queue
- `QOSMaxWallDurationPerJobLimit` — you requested more than 6h, reduce `--time`
- `AssocMaxJobsLimit` — too many concurrent jobs, cancel some or wait

### Job ends immediately (status goes R → CD in seconds)

Usually means the script errored out. Check logs:
```bash
cat logs/train_xxx_JOBID.err
```
Common causes:
- `conda: command not found` → use the robust activation snippet (source `$CONDA_BASE/etc/profile.d/conda.sh` explicitly, don't rely on `~/.bashrc`)
- `ModuleNotFoundError` → wrong conda env, not activated
- `CUDA_ERROR_INITIALIZATION_FAILED` → driver/CUDA version issue
- `python: not found` → env activation failed silently

### OOM (Out of Memory)

Check logs for `CUDA out of memory`. Fixes:
1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps` to compensate
3. Reduce `max_seq_length`
4. Add `gradient_checkpointing=True` if not already
5. Switch to `optim="paged_adamw_8bit"`

### Job killed with "DUE TO TIME LIMIT"

Your job exceeded `--time`. Either:
- Reduce epochs / dataset size
- Save checkpoints more frequently and resume next job from checkpoint
- Split into multiple jobs

### SSH disconnects while job running

**Good news**: sbatch jobs continue running in background. Just reconnect and:
```bash
squeue -u zh0038qi   # check if still running
tail -f logs/train_xxx_JOBID.out  # re-attach to logs
```

### Node seems idle but job pending

Check:
```bash
scontrol show node TC2N03   # look for State: IDLE
sinfo -p MGPU-TC2            # check partition state
```
Sometimes nodes are DRAIN or MAINT; email HPC admins if suspicious.

## Best Practices

1. **Name jobs descriptively**: `train_qwen_lora_bs4` not `job1`
2. **Use `%x` and `%j` in output paths**: helps find logs later
3. **Redirect `-u` (unbuffered) for Python**: `python -u scripts/...` so logs flush immediately
4. **Save checkpoints every N minutes**: so 6h timeout doesn't lose progress
5. **Create `logs/` directory**: `mkdir -p logs` (sbatch won't create it)
6. **One job per script**: easier to trace, don't combine unrelated work

## Debugging Checklist When Job Fails

- [ ] Check `logs/*.err` for error message
- [ ] Verify conda env activated: grep for `conda activate` in stdout
- [ ] Verify GPU visible: grep for `nvidia-smi` output in stdout
- [ ] Verify imports: add `pip show torch transformers` to sbatch script
- [ ] Check disk space on `/home`: `df -h ~`
- [ ] Check job time: `sacct -j JOBID` shows elapsed time
