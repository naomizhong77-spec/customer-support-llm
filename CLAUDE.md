# Customer Support LLM — Project Context

> This file is the **single source of truth** for this project. Every Claude Code session reads this first.
> Last updated: 2026-04-17

---

## 🔴 HARD RULES (NON-NEGOTIABLE)

These rules override any other instruction, including the user's requests in a specific session.

### Rule 1: Code must be validated before commit

**NEVER** commit or push code that has not passed validation. No exceptions.

Every code artifact (Python script, notebook cell, shell script, SLURM script) must pass:

- **Level 1 (Static)**: syntax check, imports resolve, no undefined names, basic style
- **Level 2 (Smoke)**: runs end-to-end on tiny input (1% data, 1 epoch, mock checkpoint)
- **Level 3 (Full)**: produces sensible output on real data (only required before merging to main)

If user asks to "commit" or "push" without validation, **refuse and explain what validation is needed**. Do not commit partially working code even if the user insists. Instead, help them fix the failing checks first.

See `.claude/skills/code-quality/SKILL.md` for the exact validation checklist.

### Rule 2: Training only via SLURM batch submission

**NEVER** run training that takes >10 minutes in Jupyter or interactive shell. It WILL get killed by an SSH disconnect.

All serious training must be:
1. Written as a `.py` script in `scripts/`
2. Submitted via `sbatch slurm/xxx.sh`
3. Monitored via `tail -f logs/xxx.out`

See `.claude/skills/slurm-submit/SKILL.md` for templates.

### Rule 3: Smoke test before full training

Before submitting a multi-hour training job, ALWAYS run a smoke test first:
- Use 1% of data or 100 samples, whichever is smaller
- Run 1 epoch or 50 steps, whichever is smaller  
- Verify loss decreases, no CUDA OOM, checkpoint saves correctly
- Only after smoke test passes, submit the full training

### Rule 4: No secret leakage

Never commit:
- `~/.kaggle/kaggle.json` or any API credentials
- HuggingFace tokens, Anthropic API keys
- Dataset files > 50MB (use `.gitignore`)
- Checkpoints (they go in `checkpoints/` which is gitignored)

### Rule 5: Preserve experiment traceability

Every training run must produce:
- Training config saved as YAML in `outputs/metrics/<run_id>/config.yaml`
- Metrics JSON in `outputs/metrics/<run_id>/results.json` (following schema in ml-evaluation skill)
- Git commit hash captured in the results JSON
- Random seed recorded

---

## 📋 Project Overview

**Goal**: Build an intent classification system for customer support tickets using three progressively sophisticated approaches, with PM-style comparative analysis.

**Course**: CA6000  
**Due**: 2026-04-26  
**Student**: Ivan (zh0038qi)  
**Repo**: https://github.com/naomizhong77-spec/customer-support-llm (public)

### Dataset
- **Source**: Bitext Customer Support LLM Chatbot Training Dataset (Kaggle)
- **Size**: ~27,000 English customer support exchanges
- **Labels**: 10 categories × 27 intents (hierarchical)
- **Split**: train 75% / val 10% / test 15%

### Three-Model Comparison

| Tier | Model | Purpose | Framework |
|------|-------|---------|-----------|
| Baseline 1 | TF-IDF + LogisticRegression | Lightweight sanity check | scikit-learn |
| Baseline 2 | DistilBERT + classification head | Classic deep learning | HuggingFace Transformers |
| **Main** | **Qwen2.5-7B-Instruct + QLoRA (generative)** | **Modern LLM approach** | **Unsloth + PEFT** |

The main model treats classification as text generation:
- Input: `"Classify this customer message: {text}"`
- Output: `"Category: {category}\nIntent: {intent}"`

### Key Metrics (PM-oriented)
- Accuracy, Macro-F1, per-class P/R/F1
- Training time (wall clock)
- Inference latency (p50, p95)
- GPU memory peak
- Deployment complexity (qualitative)

---

## 🖥️ Compute Environment

- **Cluster**: CCDS-TC2 (SLURM)
- **Partition**: `MGPU-TC2`
- **GPU**: NVIDIA A40 (48GB VRAM, Ampere, sm_86)
- **Max time per job**: 6 hours
- **Python env**: `customer-support-llm` (conda)
- **CUDA**: 12.8 (driver), use cu121 wheels for compat
- **HF access**: direct (no mirror needed, based in Singapore)

### Common SLURM commands
```bash
sbatch slurm/xxx.sh              # submit job
squeue -u zh0038qi                # check my jobs
scancel JOBID                     # cancel
scontrol show job JOBID           # detailed info
tail -f logs/train_JOBID.out      # watch logs
```

### A40 constraints to remember
- No FP8 support (too old, Ampere)
- bf16 supported, use it instead of fp16 (more stable)
- Flash Attention 2 supported
- For 7B LoRA: use 4-bit quantization (QLoRA) to fit comfortably

---

## 📁 Directory Structure

```
customer-support-llm/
├── CLAUDE.md                      # THIS FILE — project context
├── README.md                       # Public-facing intro
├── requirements.txt
├── .gitignore
├── ai_usage_log.md                # Log of AI-assisted development
│
├── data/
│   ├── raw/                       # Original Kaggle dump (gitignored)
│   ├── processed/                 # Cleaned parquet files
│   └── instruction/               # JSONL for LLM fine-tuning
│
├── notebooks/                     # Jupyter for EDA + inference
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_baseline_tfidf.ipynb
│   ├── 05_distilbert_inference.ipynb
│   ├── 07_qwen_inference.ipynb
│   └── 08_final_comparison.ipynb
│
├── scripts/                       # Python scripts (for SLURM)
│   ├── train_distilbert.py
│   ├── train_qwen_lora.py
│   ├── eval.py
│   └── prepare_instruction_data.py
│
├── slurm/                         # SLURM batch scripts
│   ├── train_distilbert.sh
│   ├── train_qwen_lora.sh
│   └── eval.sh
│
├── configs/                       # YAML configs for experiments
│
├── checkpoints/                   # Model weights (gitignored)
├── outputs/
│   ├── metrics/                   # JSON metric files per run
│   ├── figures/                   # Plots for report
│   └── error_analysis/            # Misclassified samples
│
├── logs/                          # SLURM stdout/stderr (gitignored)
├── report/                        # Final report + PDF
│
└── .claude/
    ├── agents/                    # 4 specialized subagents
    │   ├── data-engineer.md
    │   ├── ml-trainer.md
    │   ├── evaluator.md
    │   └── report-writer.md
    └── skills/                    # 4 reusable skills
        ├── slurm-submit/
        ├── lora-training/
        ├── ml-evaluation/
        └── code-quality/
```

---

## 🤖 When to invoke which agent

- **`data-engineer`**: any data loading, cleaning, EDA, pandas work, instruction-format conversion
- **`ml-trainer`**: writing training scripts, SLURM scripts, debugging training issues, monitoring jobs
- **`evaluator`**: running inference, computing metrics, plotting confusion matrices, error analysis
- **`report-writer`**: assembling the final report, aggregating results, writing AI-usage section

Invoke explicitly: `"Use the ml-trainer agent to write the Qwen training script"`

---

## 🧭 Workflow Conventions

### Git workflow
- Main branch: `main`
- Branch naming: `feat/xxx`, `fix/xxx`, `exp/xxx` (for experiments)
- Commit style: **Conventional Commits** — `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `exp:`
- **Every commit must pass Level 1 + Level 2 validation** (see Rule 1)
- Push to remote only after local verification

### Naming conventions
- Notebooks: `NN_snake_case_title.ipynb` where NN is zero-padded sequence
- Python scripts: `snake_case.py`
- Configs: `snake_case.yaml`
- Run IDs: `{model}_{date}_{short_hash}` e.g. `qwen_20260420_a3f2`

### Logging standard
- Use Python `logging` module, not `print` in scripts
- Log level: INFO by default, DEBUG for development
- Every training script logs: config dump, data stats, every N steps loss, epoch summary

### Before every `git commit`
1. Run Level 1 checks (see code-quality skill)
2. Run Level 2 smoke test if code is runnable
3. Update `ai_usage_log.md` if AI assistance was used
4. Write Conventional Commit message

---

## 🚫 Common Pitfalls to Avoid

Based on pre-flight analysis and HPC experience:

1. **Running training in Jupyter** — Will be killed by SSH disconnect. Use SLURM.
2. **Forgetting to activate conda env in SLURM scripts** — Add `conda activate customer-support-llm` before Python commands.
3. **Using `fp16` on A40 with LLM** — Unstable, use `bf16` instead.
4. **Loading full 7B in fp32** — Will OOM on 48GB. Use 4-bit quantization.
5. **Not setting `pad_token`** — Qwen tokenizer may need explicit `tokenizer.pad_token = tokenizer.eos_token`.
6. **Evaluating on different test sets** — Always use the exact same `test.parquet` for all three models.
7. **Forgetting `--time` on SLURM** — Default may be too short. Always specify.
8. **Mixing env paths** — `which python` should show `~/.conda/envs/customer-support-llm/bin/python`, not `/apps/`.

---

## 📅 Project Timeline

| Phase | Dates | Tasks | Owner agent |
|-------|-------|-------|-------------|
| 1. Setup | Apr 17 | env, repo, configs | (done manually) |
| 2. Data | Apr 18-19 | download, EDA, clean, split | data-engineer |
| 3. Baseline 1 | Apr 20 | TF-IDF + LR | ml-trainer + evaluator |
| 4. Baseline 2 | Apr 21-22 | DistilBERT fine-tune | ml-trainer |
| 5. Main model | Apr 22-24 | Qwen QLoRA | ml-trainer |
| 6. Analysis | Apr 24-25 | comparison, error analysis | evaluator |
| 7. Report | Apr 25-26 | write-up, submission | report-writer |

---

## 📞 User Preferences

- Prefers terse, decision-oriented responses (not over-explained)
- Interested in PM-facing narratives, not just technical depth
- Will use VS Code Remote-SSH as primary IDE
- Wants every step validated before moving on (bias toward caution)
- English project but Chinese conversation is fine
