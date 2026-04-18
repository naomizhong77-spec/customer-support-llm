"""Train Baseline 2: DistilBERT classification head for 27-intent classification.

Design (frozen by user — do not change without a new config):
  - distilbert-base-uncased + AutoModelForSequenceClassification(num_labels=27)
  - max_length=128, bf16 on A40 (NEVER fp16), lr=2e-5, bs=32, 3 epochs, warmup=0.1
  - HuggingFace Trainer, load_best_model_at_end on eval_macro_f1
  - EarlyStoppingCallback(patience=2)
  - Seed 42, save_total_limit=2 (best + last only)

Outputs (written under the repo root, or --output-root for smoke tests):
  - checkpoints/distilbert/<run_id>/              (best model, save_total_limit=2)
  - outputs/metrics/<run_id>/results.json         (schema: ml-evaluation skill)
  - outputs/metrics/<run_id>/config.yaml          (snapshot of cfg actually used)
  - outputs/metrics/<run_id>/train_log.jsonl      (Trainer's log_history)
  - outputs/metrics/<run_id>/git_info.txt
  - outputs/figures/04_distilbert_confusion_matrix.png
  - outputs/error_analysis/distilbert_errors.csv

Usage:
    python scripts/train_distilbert.py --config configs/distilbert_config.yaml
    python scripts/train_distilbert.py --config configs/distilbert_config.yaml --smoke-test

Smoke-test mode uses 100 train / 50 val / 50 test samples, writes to a separate
run_id prefixed "distilbert_smoke_" and respects --output-root so artifacts can
go to /tmp and never pollute real runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("train_distilbert")


def _setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


# ---------------------------------------------------------------------------
# Config & run identity
# ---------------------------------------------------------------------------
def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def git_short_hash(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:  # noqa: BLE001
        return "nogit"


def git_dirty_files(repo_root: Path) -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
        return [line.strip() for line in out.decode().splitlines() if line.strip()]
    except Exception:  # noqa: BLE001
        return []


def make_run_id(model_tag: str, repo_root: Path) -> str:
    """<model>_<YYYYMMDD>_<shorthash> per CLAUDE.md conventions."""
    date = datetime.now().strftime("%Y%m%d")
    return f"{model_tag}_{date}_{git_short_hash(repo_root)}"


def test_set_hash(parquet_path: Path) -> str:
    """Hash the test parquet so all 3 models can prove identical eval data.

    Matches .claude/skills/ml-evaluation/SKILL.md exactly: sort columns for
    stability, then sha256 of CSV bytes, first 16 hex chars.
    """
    df = pd.read_parquet(parquet_path)
    df = df.reindex(sorted(df.columns), axis=1)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return f"sha256:{hashlib.sha256(csv_bytes).hexdigest()[:16]}"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class LabelSpace:
    """Deterministic label <-> id mapping based on sorted training intents."""

    label2id: dict[str, int]
    id2label: dict[int, str]
    labels_sorted: list[str]

    @classmethod
    def from_train(cls, train_labels: list[str]) -> "LabelSpace":
        labels_sorted = sorted(set(train_labels))
        label2id = {lbl: i for i, lbl in enumerate(labels_sorted)}
        id2label = {i: lbl for lbl, i in label2id.items()}
        return cls(label2id=label2id, id2label=id2label, labels_sorted=labels_sorted)


def load_split(
    parquet_path: Path,
    text_col: str,
    label_col: str,
    limit: int | None = None,
) -> tuple[list[str], list[str]]:
    df = pd.read_parquet(parquet_path)
    if limit is not None:
        df = df.head(limit).copy()
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()
    return texts, labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Matches ml-evaluation skill: accuracy, macro P/R/F1, weighted F1, per-class."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        precision_recall_fscore_support,
    )

    accuracy = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=labels
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0, labels=labels
    )
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0, labels=labels
    )
    per_class: dict[str, Any] = {}
    for label_name, scores in report.items():
        if label_name in {"accuracy", "macro avg", "weighted avg"}:
            continue
        per_class[label_name] = {
            "precision": round(float(scores["precision"]), 4),
            "recall": round(float(scores["recall"]), 4),
            "f1": round(float(scores["f1-score"]), 4),
            "support": int(scores["support"]),
        }
    return {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class,
    }


def make_hf_compute_metrics(label_space: LabelSpace):
    """Return a `compute_metrics` callable for HuggingFace Trainer.

    Trainer passes an EvalPrediction with `.predictions` logits and `.label_ids`.
    We convert back to string labels to reuse the shared metric computer.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_fscore_support,
    )

    def _compute(eval_pred) -> dict[str, float]:  # type: ignore[no-untyped-def]
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        weighted_f1 = f1_score(
            labels, preds, average="weighted", zero_division=0
        )
        return {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "weighted_f1": float(weighted_f1),
        }

    return _compute


# ---------------------------------------------------------------------------
# Latency measurement (GPU, CUDA events per ml-evaluation skill)
# ---------------------------------------------------------------------------
def measure_latency_gpu(
    model,  # type: ignore[no-untyped-def]
    tokenizer,  # type: ignore[no-untyped-def]
    texts: list[str],
    max_length: int,
    warmup: int,
    n_measure: int,
    device,  # type: ignore[no-untyped-def]
) -> dict[str, Any]:
    """Per-sample forward-pass latency (batch=1). Uses CUDA events if GPU."""
    import torch

    n_measure = min(n_measure, max(0, len(texts) - warmup))
    if n_measure < 10:
        logger.warning(
            "Only %d samples available for latency after warmup; results will be noisy.",
            n_measure,
        )

    model.eval()

    def _forward(text: str) -> None:
        inputs = tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            _ = model(**inputs)

    # Warmup
    for x in texts[:warmup]:
        _forward(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times_ms: list[float] = []
    use_cuda_events = torch.cuda.is_available() and device.type == "cuda"
    for x in texts[warmup : warmup + n_measure]:
        if use_cuda_events:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _forward(x)
            end_event.record()
            torch.cuda.synchronize()
            times_ms.append(float(start_event.elapsed_time(end_event)))
        else:
            t0 = time.perf_counter()
            _forward(x)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times_ms)
    if arr.size == 0:
        return {
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "mean_ms": None,
            "std_ms": None,
            "throughput_per_sec": None,
            "batch_size": 1,
            "device": str(device),
            "precision": "bf16" if use_cuda_events else "fp32",
            "n_samples_timed": 0,
        }
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "throughput_per_sec": float(1000.0 / arr.mean()),
        "batch_size": 1,
        "device": str(device),
        "precision": "bf16" if use_cuda_events else "fp32",
        "n_samples_timed": int(arr.size),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    save_path: Path,
    title: str = "DistilBERT — Intent Confusion Matrix (normalized)",
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Results schema validation
# ---------------------------------------------------------------------------
def validate_results(results: dict[str, Any]) -> None:
    required_top = [
        "run_id",
        "model_name",
        "test_set_hash",
        "metrics",
        "latency",
        "resources",
    ]
    for k in required_top:
        assert k in results, f"Missing required field: {k}"
    acc = results["metrics"]["intent"]["accuracy"]
    assert 0 <= acc <= 1, f"Accuracy out of range: {acc}"
    assert acc != 1.0, "Suspicious perfect accuracy — likely data leak"
    assert acc > 0.05, "Suspicious near-zero accuracy — likely bug"
    f1 = results["metrics"]["intent"]["macro_f1"]
    assert f1 <= acc + 0.05, f"macro_F1 ({f1}) > accuracy ({acc}) shouldn't happen often"
    p50 = results["latency"]["p50_ms"]
    p95 = results["latency"]["p95_ms"]
    if p50 is not None and p95 is not None:
        assert p95 >= p50, "p95 must be >= p50"


# ---------------------------------------------------------------------------
# Full pipeline entry
# ---------------------------------------------------------------------------
def run_pipeline(  # noqa: C901, PLR0912, PLR0915
    config_path: Path,
    repo_root: Path,
    smoke_test: bool = False,
    seed: int | None = None,
    resume_from: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Train, evaluate, and save all artifacts. Returns the results dict."""
    # Local imports: heavy deps are imported only when actually running.
    import torch
    import transformers
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
    from transformers import set_seed as hf_set_seed

    cfg = load_config(config_path)
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    art_cfg = cfg["artifacts"]
    runtime_cfg = cfg.get("runtime", {})

    if seed is not None:
        train_cfg["seed"] = seed
    effective_seed = int(train_cfg.get("seed", 42))
    hf_set_seed(effective_seed)

    art_root = output_root if output_root is not None else repo_root

    # Resolve data paths (always relative to repo_root — data lives with the code)
    train_path = repo_root / data_cfg["train_path"]
    val_path = repo_root / data_cfg["val_path"]
    test_path = repo_root / data_cfg["test_path"]
    text_col = data_cfg["text_column"]
    label_col = data_cfg["label_column"]

    # Run identity
    model_tag = "distilbert_smoke" if smoke_test else "distilbert"
    run_id = make_run_id(model_tag, repo_root)
    logger.info(
        "run_id=%s smoke_test=%s seed=%s resume_from=%s",
        run_id, smoke_test, effective_seed, resume_from,
    )

    # Output directories
    ckpt_dir = art_root / art_cfg["checkpoint_root"] / run_id
    metrics_dir = art_root / art_cfg["metrics_root"] / run_id
    figures_dir = art_root / art_cfg["figures_dir"]
    errors_dir = art_root / art_cfg["error_analysis_dir"]
    for d in (ckpt_dir, metrics_dir, figures_dir, errors_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Persist git info early (useful if job dies)
    git_info_lines = [
        f"commit: {git_short_hash(repo_root)}",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        "dirty_files:",
        *[f"  {line}" for line in git_dirty_files(repo_root)],
    ]
    (metrics_dir / "git_info.txt").write_text("\n".join(git_info_lines), encoding="utf-8")

    # Load data
    train_limit = 100 if smoke_test else None
    eval_limit = 50 if smoke_test else None
    logger.info("Loading data (smoke_test=%s)", smoke_test)
    X_train, y_train_str = load_split(train_path, text_col, label_col, limit=train_limit)
    X_val, y_val_str = load_split(val_path, text_col, label_col, limit=eval_limit)
    X_test, y_test_str = load_split(test_path, text_col, label_col, limit=eval_limit)
    logger.info(
        "train=%d val=%d test=%d | n_intents_train=%d",
        len(X_train), len(X_val), len(X_test), len(set(y_train_str)),
    )

    # Label space: in smoke mode we may not see all 27 intents in the 100 train
    # rows. Fall back to the union of all splits so label2id is stable and test
    # labels never crash the loss computation.
    if smoke_test:
        union = sorted(set(y_train_str) | set(y_val_str) | set(y_test_str))
        label_space = LabelSpace(
            label2id={lbl: i for i, lbl in enumerate(union)},
            id2label={i: lbl for i, lbl in enumerate(union)},
            labels_sorted=union,
        )
    else:
        label_space = LabelSpace.from_train(y_train_str)
        expected = int(model_cfg["num_labels"])
        if len(label_space.labels_sorted) != expected:
            raise ValueError(
                f"Training set has {len(label_space.labels_sorted)} unique intents, "
                f"config expects num_labels={expected}. "
                f"Refusing to train with mismatched label space."
            )

    logger.info("Label space size=%d (first 3: %s)",
                len(label_space.labels_sorted), label_space.labels_sorted[:3])

    num_labels_runtime = len(label_space.labels_sorted)
    label2id = label_space.label2id
    y_train = [label2id[lbl] for lbl in y_train_str]
    y_val = [label2id[lbl] for lbl in y_val_str]
    y_test = [label2id[lbl] for lbl in y_test_str]

    # Tokenizer + model
    base_model = model_cfg["base_model"]
    max_length = int(model_cfg["max_length"])
    logger.info("Loading tokenizer + model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        # DistilBERT tokenizer already has [PAD]; guard for other bases.
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    logger.info("Tokenizer pad_token=%r model_max_length=%d",
                tokenizer.pad_token, tokenizer.model_max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_labels_runtime,
        id2label=label_space.id2label,
        label2id=label_space.label2id,
        problem_type=model_cfg.get("problem_type", "single_label_classification"),
    )

    # Build HF datasets
    def _tokenize(examples: dict) -> dict:  # type: ignore[type-arg]
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
        )

    train_ds = Dataset.from_dict({"text": X_train, "labels": y_train}).map(
        _tokenize, batched=True, remove_columns=["text"]
    )
    val_ds = Dataset.from_dict({"text": X_val, "labels": y_val}).map(
        _tokenize, batched=True, remove_columns=["text"]
    )
    test_ds = Dataset.from_dict({"text": X_test, "labels": y_test}).map(
        _tokenize, batched=True, remove_columns=["text"]
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # TrainingArguments
    epochs = 1 if smoke_test else int(train_cfg["num_train_epochs"])
    max_steps = 50 if smoke_test else -1
    ta_kwargs = dict(
        output_dir=str(ckpt_dir),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.0)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "linear")),
        bf16=bool(train_cfg.get("bf16", True)),
        fp16=bool(train_cfg.get("fp16", False)),
        logging_steps=int(train_cfg.get("logging_steps", 50)),
        eval_strategy=str(train_cfg.get("eval_strategy", "epoch")),
        save_strategy=str(train_cfg.get("save_strategy", "epoch")),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        load_best_model_at_end=bool(train_cfg.get("load_best_model_at_end", True)),
        metric_for_best_model=str(train_cfg.get("metric_for_best_model", "eval_macro_f1")),
        greater_is_better=bool(train_cfg.get("greater_is_better", True)),
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
        report_to=str(train_cfg.get("report_to", "none")),
        seed=effective_seed,
        save_safetensors=True,
    )
    # For smoke tests the eval/save cadence must fit in max_steps=50.
    if smoke_test:
        ta_kwargs["eval_strategy"] = "steps"
        ta_kwargs["save_strategy"] = "steps"
        ta_kwargs["eval_steps"] = 25
        ta_kwargs["save_steps"] = 25
        ta_kwargs["logging_steps"] = 10

    training_args = TrainingArguments(**ta_kwargs)

    early_stop_patience = int(train_cfg.get("early_stopping_patience", 2))
    callbacks = []
    if early_stop_patience > 0 and not smoke_test:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stop_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_hf_compute_metrics(label_space),
        callbacks=callbacks,
    )

    # GPU memory baseline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        logger.info("CUDA device: %s", torch.cuda.get_device_name(device))
    else:
        logger.warning("No CUDA device available — training will fall back to CPU (slow).")

    # Train
    logger.info("Starting training: epochs=%s max_steps=%s bs=%s lr=%s bf16=%s",
                epochs, max_steps, ta_kwargs["per_device_train_batch_size"],
                ta_kwargs["learning_rate"], ta_kwargs["bf16"])
    t0 = time.perf_counter()
    train_out = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    train_seconds = time.perf_counter() - t0
    logger.info("Training wall-clock: %.2f s", train_seconds)
    logger.info("train_runtime=%.2fs global_step=%d final_train_loss=%.4f",
                train_out.metrics.get("train_runtime", -1.0),
                trainer.state.global_step,
                train_out.training_loss)

    # GPU memory peak during train
    gpu_mem_peak_train_mb = 0
    if device.type == "cuda":
        gpu_mem_peak_train_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        logger.info("Peak GPU memory during train: %d MB", gpu_mem_peak_train_mb)
        torch.cuda.reset_peak_memory_stats(device)

    # Save Trainer log_history as train_log.jsonl
    train_log_path = metrics_dir / "train_log.jsonl"
    with open(train_log_path, "w", encoding="utf-8") as f:
        for entry in trainer.state.log_history:
            f.write(json.dumps(entry) + "\n")
    logger.info("Saved training log to %s (%d entries)",
                train_log_path, len(trainer.state.log_history))

    # Evaluate on validation
    logger.info("Evaluating on val...")
    val_eval = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="val")
    logger.info("VAL  acc=%.4f macro_f1=%.4f",
                val_eval.get("val_accuracy", float("nan")),
                val_eval.get("val_macro_f1", float("nan")))

    # Evaluate on test — use Trainer.predict so we get logits for per-class metrics
    logger.info("Predicting on test...")
    test_pred_out = trainer.predict(test_ds, metric_key_prefix="test")
    test_logits = test_pred_out.predictions
    if isinstance(test_logits, tuple):
        test_logits = test_logits[0]
    y_test_pred_ids = np.argmax(test_logits, axis=-1).tolist()
    y_test_pred_str = [label_space.id2label[i] for i in y_test_pred_ids]

    # Confidence for error analysis (softmax max prob)
    test_logits_t = torch.from_numpy(test_logits)
    test_probs = torch.softmax(test_logits_t.float(), dim=-1)
    test_conf = test_probs.max(dim=-1).values.numpy().tolist()

    # Test metrics (use the full 27-intent label order so matrix is comparable)
    full_label_order = label_space.labels_sorted
    test_metrics = compute_classification_metrics(
        y_test_str, y_test_pred_str, labels=full_label_order
    )
    val_metrics_for_results = {
        "accuracy": round(float(val_eval.get("val_accuracy", 0.0)), 4),
        "macro_f1": round(float(val_eval.get("val_macro_f1", 0.0)), 4),
        "macro_precision": round(float(val_eval.get("val_macro_precision", 0.0)), 4),
        "macro_recall": round(float(val_eval.get("val_macro_recall", 0.0)), 4),
        "weighted_f1": round(float(val_eval.get("val_weighted_f1", 0.0)), 4),
    }
    logger.info("TEST acc=%.4f macro_f1=%.4f weighted_f1=%.4f",
                test_metrics["accuracy"], test_metrics["macro_f1"], test_metrics["weighted_f1"])

    # Confusion matrix (test)
    cm_path = figures_dir / art_cfg["confusion_matrix_filename"]
    plot_confusion_matrix(
        y_test_str, y_test_pred_str, labels=full_label_order, save_path=cm_path
    )
    logger.info("Saved confusion matrix to %s", cm_path)

    # Error analysis CSV
    errors_df = pd.DataFrame({
        "text": X_test,
        "true_label": y_test_str,
        "predicted_label": y_test_pred_str,
        "confidence": test_conf,
    })
    errors_df = errors_df[errors_df["true_label"] != errors_df["predicted_label"]].copy()
    errors_df["text_length"] = errors_df["text"].str.len()
    errors_csv_path = errors_dir / art_cfg["errors_csv_filename"]
    errors_df.to_csv(errors_csv_path, index=False)
    logger.info("Saved %d errors to %s", len(errors_df), errors_csv_path)

    # Inference latency (GPU, CUDA events)
    n_warmup = int(runtime_cfg.get("n_latency_warmup", 20))
    n_measure = int(runtime_cfg.get("n_latency_samples", 300))
    if smoke_test:
        n_warmup = 5
        n_measure = 20
    # Reset peak mem before latency run so inference peak is measured cleanly
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    latency = measure_latency_gpu(
        trainer.model, tokenizer, X_test,
        max_length=max_length, warmup=n_warmup, n_measure=n_measure, device=device,
    )
    gpu_mem_peak_infer_mb = 0
    if device.type == "cuda":
        gpu_mem_peak_infer_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    if latency["p50_ms"] is not None:
        logger.info(
            "Latency p50=%.3fms p95=%.3fms p99=%.3fms throughput=%.0f/s",
            latency["p50_ms"], latency["p95_ms"], latency["p99_ms"],
            latency["throughput_per_sec"],
        )

    # Save a clean "best" copy (Trainer already keeps best; also save consolidated).
    # Trainer with save_total_limit=2 + load_best_model_at_end keeps best + last.
    final_save_dir = ckpt_dir / "final"
    trainer.save_model(str(final_save_dir))
    tokenizer.save_pretrained(str(final_save_dir))
    # Disk footprint
    total_bytes = 0
    for p in final_save_dir.rglob("*"):
        if p.is_file():
            total_bytes += p.stat().st_size
    disk_size_mb = round(total_bytes / (1024 * 1024), 2)
    logger.info("Saved best model to %s (%.2f MB)", final_save_dir, disk_size_mb)

    # Parameter counts
    param_total = int(sum(p.numel() for p in trainer.model.parameters()))
    param_trainable = int(
        sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    )

    # Assemble results.json (schema: .claude/skills/ml-evaluation/SKILL.md)
    checkpoint_rel = str(final_save_dir.relative_to(art_root))
    test_set_rel = str(test_path.relative_to(repo_root))
    results: dict[str, Any] = {
        "run_id": run_id,
        "model_name": runtime_cfg.get("model_name", "DistilBERT"),
        "model_family": runtime_cfg.get("model_family", "bert-classifier"),
        "checkpoint_path": checkpoint_rel,
        "test_set": test_set_rel,
        "test_set_hash": test_set_hash(test_path),
        "n_samples": len(X_test),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": git_short_hash(repo_root),
        "seed": effective_seed,
        "training_wallclock_seconds": round(train_seconds, 3),
        "framework_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "metrics": {
            "intent": test_metrics,
            "intent_val": val_metrics_for_results,
        },
        "latency": latency,
        "resources": {
            "gpu_memory_mb_peak_train": gpu_mem_peak_train_mb,
            "gpu_memory_mb_peak_inference": gpu_mem_peak_infer_mb,
            "disk_size_mb": disk_size_mb,
            "parameters_trainable": param_trainable,
            "parameters_total": param_total,
        },
        "parse_errors": 0,
        "notes": (
            f"Baseline 2 (DistilBERT). epochs={epochs} max_steps={max_steps} "
            f"bs={ta_kwargs['per_device_train_batch_size']} "
            f"lr={ta_kwargs['learning_rate']} bf16={ta_kwargs['bf16']} "
            f"max_length={max_length}. "
            f"val_accuracy={val_metrics_for_results['accuracy']} "
            f"val_macro_f1={val_metrics_for_results['macro_f1']}."
        ),
    }

    if not smoke_test:
        validate_results(results)

    # Persist results + config snapshot
    (metrics_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with open(metrics_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    logger.info("Saved results to %s/results.json", metrics_dir)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DistilBERT intent classifier (Baseline 2)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/distilbert_config.yaml"),
        help="Path to YAML config (relative to --repo-root or absolute).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root (all data/output paths are resolved against this).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output root (e.g. /tmp/smoke) for smoke tests.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on 100 train / 50 val / 50 test samples, max 50 steps.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to a Trainer checkpoint directory to resume from.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(level=getattr(logging, args.log_level))

    # Quieter HuggingFace logs unless user asked for DEBUG
    if args.log_level != "DEBUG":
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("datasets").setLevel(logging.WARNING)

    repo_root = args.repo_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        return 2

    output_root = args.output_root.resolve() if args.output_root else None
    resume_from = args.resume_from.resolve() if args.resume_from else None

    results = run_pipeline(
        config_path=config_path,
        repo_root=repo_root,
        smoke_test=args.smoke_test,
        seed=args.seed,
        resume_from=resume_from,
        output_root=output_root,
    )
    logger.info(
        "DONE run_id=%s test_acc=%.4f test_macro_f1=%.4f",
        results["run_id"],
        results["metrics"]["intent"]["accuracy"],
        results["metrics"]["intent"]["macro_f1"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
