# Customer Support Intent Classification: From Traditional ML to Fine-tuned LLM

> End-to-end comparison of three approaches to customer support ticket routing, from TF-IDF baselines to QLoRA fine-tuning of Qwen2.5-7B.

**Course**: CA6000 | **Institution**: [University] | **Author**: Ivan Zhong  
**Repo**: https://github.com/naomizhong77-spec/customer-support-llm

---

## 🎯 Problem

Customer support centers receive thousands of tickets daily across dozens of intent categories (refund, cancellation, account recovery, etc.). Routing them to the right team manually is slow, error-prone, and doesn't scale. This project benchmarks three AI approaches to automated intent classification, evaluated not just on accuracy but on the PM-relevant dimensions of latency, deployment cost, and operational complexity.

## 📊 Headline Results

_[To be filled in after experiments. Placeholder structure:]_

| Model | Macro-F1 | Accuracy | Latency (p50) | GPU Memory | Size on Disk |
|-------|----------|----------|---------------|------------|--------------|
| TF-IDF + LogReg | — | — | — | None | — |
| DistilBERT + Head | — | — | — | — | — |
| **Qwen2.5-7B QLoRA** | **—** | **—** | — | — | — |

## 🧰 Stack

- **Data**: two datasets, run in parallel to separate templated from real-user behaviour:
  1. [Bitext Customer Support dataset](https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset) — ~27k synthetic-augmented English tickets, 11 categories × 27 intents
  2. [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) — 13k real user-written banking queries, flat 77-way intent classification
- **Traditional ML**: scikit-learn
- **Deep Learning**: HuggingFace Transformers (DistilBERT)
- **LLM Fine-tuning**: Unsloth + PEFT (Qwen2.5-7B with 4-bit QLoRA)
- **Compute**: SLURM cluster, NVIDIA A40 (48GB)
- **Dev tools**: Claude Code (agent-based workflow), VS Code Remote-SSH

## 🏗️ Architecture

```
┌─────────────────────────┐
│ Kaggle: Bitext dataset  │
└───────────┬─────────────┘
            ↓
┌─────────────────────────┐
│ Data pipeline           │ ← pandas, parquet
│ - EDA                   │
│ - Clean + dedupe        │
│ - Stratified split      │
│ - Instruction format    │
└───────────┬─────────────┘
            ↓
    ┌───────┴────────────────────────┐
    ↓               ↓                ↓
┌────────┐   ┌──────────┐   ┌──────────────┐
│TF-IDF  │   │DistilBERT│   │Qwen 2.5 7B   │
│ + LR   │   │+ CLS head│   │+ QLoRA (gen) │
└───┬────┘   └────┬─────┘   └──────┬───────┘
    │             │                 │
    └─────────────┼─────────────────┘
                  ↓
          ┌────────────────┐
          │ Unified Eval   │
          │ - Macro-F1     │
          │ - Latency      │
          │ - Error analysis│
          └────────────────┘
                  ↓
          ┌────────────────┐
          │ PM Comparison  │
          └────────────────┘
```

## 📁 Repository Structure

```
customer-support-llm/
├── CLAUDE.md                   # Project context for Claude Code
├── README.md                   # This file
├── requirements.txt
├── ai_usage_log.md             # Log of AI-assisted development
│
├── data/                       # Data (raw gitignored)
│   ├── processed/              # Cleaned parquet files
│   └── instruction/            # JSONL for LLM fine-tuning
│
├── notebooks/                  # Jupyter for EDA + inference
├── scripts/                    # Python scripts for training
├── slurm/                      # SLURM batch scripts
├── configs/                    # YAML configs
├── checkpoints/                # Model weights (gitignored)
├── outputs/
│   ├── metrics/                # JSON results per run
│   ├── figures/                # Plots
│   └── error_analysis/         # Misclassified samples
├── report/                     # Final submission report
│
└── .claude/                    # Claude Code config
    ├── agents/                 # 4 specialized subagents
    └── skills/                 # 4 reusable skill bundles
```

## 🚀 Reproducing This Work

### 1. Environment Setup

```bash
git clone https://github.com/naomizhong77-spec/customer-support-llm.git
cd customer-support-llm

# Create conda env
conda create -n customer-support-llm python=3.10 -y
conda activate customer-support-llm

# Install PyTorch (CUDA 12.1 wheels work with CUDA 12.8 driver)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install project deps
pip install -r requirements.txt
```

### 2. Download Dataset

Set up Kaggle API credentials (`~/.kaggle/kaggle.json`), then:

```bash
kaggle datasets download -d bitext/bitext-gen-ai-chatbot-customer-support-dataset -p data/raw --unzip
```

### 3. Data Preparation

This project uses **two datasets** to quantify where the LLM approach earns
its keep:

#### Dataset 1: Bitext Customer Support

Templated / synthetic-augmented English customer-support corpus (~27k rows,
11 categories × 27 intents, near-perfect class balance). The canonical
**test set** is committed (`data/processed/test.parquet` and
`data/instruction/test.jsonl`) so all three models evaluate on the exact
same examples. The **train and validation splits are NOT tracked in git**
— regenerate deterministically by running notebooks 01 and 02 end-to-end
with `seed=42`:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_data_exploration.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_data_cleaning.ipynb
# notebook 02 writes data/processed/{train,val,test}.parquet and also
# invokes scripts/prepare_instruction_data.py which produces
# data/instruction/{train,val,test}.jsonl
```

The cleaning pipeline (synthetic defect injection → deterministic cleaning →
stratified 75/10/15 split) is audit-logged to
`outputs/metrics/data_stats.json`.

#### Dataset 2: BANKING77

Real user-written online banking queries from `PolyAI/banking77` on
HuggingFace (13,083 rows, flat 77-way intent classification, no category
hierarchy, no agent response). BANKING77 is the **real-user complement** to
Bitext's templated data — running the same 3-model comparison on both lets
us build a 2×3 experimental grid that separates "does the classifier work
on clean templated data?" from "does it survive noisy real-user input?".
The hypothesis is that a TF-IDF baseline is fine for Bitext but a 7B LLM
earns its keep on BANKING77.

Regenerate via notebooks 07 and 08:

```bash
jupyter nbconvert --to notebook --execute notebooks/07_banking77_exploration.ipynb
jupyter nbconvert --to notebook --execute notebooks/08_banking77_cleaning.ipynb
# notebook 08 writes data/banking77/processed/{train,val,test}.parquet and
# invokes scripts/prepare_banking77_instruction_data.py to produce
# data/banking77/instruction/{train,val,test}.jsonl (Alpaca format)
```

The HF native `train` split (10,003 rows) is stratify-split 88/12 into
train/val. The HF native `test` split (3,080 rows) is passed through the
same whitespace/NFKC cleaning so all three splits share one schema — this
surfaces 1 within-test `(text, label)` duplicate and 7 whitespace-masked
train/test leakage pairs, all dropped (see notebook 08, section 13). Final
splits: **train 8,793 / val 1,199 / test 3,079**. The cleaned test parquet
is pinned via sha256 in `outputs/metrics/banking77_test_hash.txt`.
Unlike Bitext, **no synthetic errors are injected** — BANKING77 is real
data, so notebook 08 reports genuine upstream quality issues (whitespace
artefacts, cross-split leakage hidden behind whitespace normalisation).
Audit trail: `outputs/metrics/banking77_stats.json`.

### 4. Train Models

```bash
# Baseline 1 (no GPU needed, runs in notebook)
jupyter nbconvert --to notebook --execute notebooks/03_baseline_tfidf.ipynb

# Baseline 2 (via SLURM)
sbatch slurm/train_distilbert.sh

# Main model (via SLURM)
sbatch slurm/train_qwen_lora.sh
```

### 5. Evaluate and Compare

```bash
python scripts/eval.py --all          # run eval on all three checkpoints
python scripts/compare_models.py      # generate comparison tables and figures
```

## 📜 License

Code: MIT License  
Data: Bitext dataset — see [original license](https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset)

## 🙋 About

Developed as part of the CA6000 coursework while exploring AI Product Manager roles in payments, sales, and customer service domains.

Contact: [Your Email]  
GitHub: [@naomizhong77-spec](https://github.com/naomizhong77-spec)

---

_This project was developed with substantial AI assistance via Claude Code. See `ai_usage_log.md` and report Section 7 for details on the workflow and lessons learned._
