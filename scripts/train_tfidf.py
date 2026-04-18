"""Train Baseline 1: TF-IDF + LogisticRegression intent classifier.

This is the canonical training script for Baseline 1. It is imported by
notebooks/03_baseline_tfidf.ipynb so the notebook and CLI produce identical
artifacts.

Design (frozen by user — do not change without a new config):
  - TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95,
                    sublinear_tf=True, lowercase=True)
  - LogisticRegression(solver='saga', multi_class='multinomial',
                       C=1.0, max_iter=1000, random_state=42, n_jobs=-1)
  - wrapped in a single sklearn Pipeline so vectorizer + classifier
    save/load together via joblib.

Outputs (written under --output-root, default repo root):
  - checkpoints/tfidf/pipeline.joblib
  - outputs/metrics/tfidf_<YYYYMMDD>_<shorthash>/results.json
  - outputs/metrics/tfidf_<YYYYMMDD>_<shorthash>/config.yaml
  - outputs/figures/03_tfidf_confusion_matrix.png
  - outputs/error_analysis/tfidf_errors.csv

Usage:
    python scripts/train_tfidf.py --config configs/tfidf_config.yaml
    python scripts/train_tfidf.py --config configs/tfidf_config.yaml --smoke-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("train_tfidf")


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


def make_run_id(model_tag: str, repo_root: Path) -> str:
    """<model>_<YYYYMMDD>_<shorthash> per CLAUDE.md conventions."""
    date = datetime.now().strftime("%Y%m%d")
    return f"{model_tag}_{date}_{git_short_hash(repo_root)}"


def test_set_hash(parquet_path: Path) -> str:
    """Hash the test parquet so all 3 models can prove identical eval data.

    Matches the procedure in .claude/skills/ml-evaluation/SKILL.md:
    sort columns for stability, then sha256 of CSV bytes, first 16 hex chars.
    """
    df = pd.read_parquet(parquet_path)
    df = df.reindex(sorted(df.columns), axis=1)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return f"sha256:{hashlib.sha256(csv_bytes).hexdigest()[:16]}"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_pipeline(cfg: dict[str, Any]) -> Pipeline:
    """Build the frozen TF-IDF + LogisticRegression pipeline.

    The vectorizer and classifier share the sklearn Pipeline so that save/load
    is atomic (one joblib file).
    """
    vec_cfg = cfg["vectorizer"]
    clf_cfg = cfg["classifier"]
    vectorizer = TfidfVectorizer(
        ngram_range=tuple(vec_cfg["ngram_range"]),
        min_df=vec_cfg["min_df"],
        max_df=vec_cfg["max_df"],
        sublinear_tf=vec_cfg["sublinear_tf"],
        lowercase=vec_cfg["lowercase"],
    )
    classifier = LogisticRegression(
        solver=clf_cfg["solver"],
        C=clf_cfg["C"],
        max_iter=clf_cfg["max_iter"],
        n_jobs=clf_cfg["n_jobs"],
        random_state=clf_cfg["random_state"],
    )
    return Pipeline(steps=[("tfidf", vectorizer), ("clf", classifier)])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
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


def measure_latency_cpu(
    pipeline: Pipeline,
    inputs: list[str],
    warmup: int,
    n_measure: int,
) -> dict[str, Any]:
    """CPU latency: per-sample predict (batch_size=1), perf_counter in ms."""
    n_measure = min(n_measure, max(0, len(inputs) - warmup))
    if n_measure < 10:
        # Not enough samples for stable percentiles — degrade gracefully.
        logger.warning(
            "Only %d samples available for latency after warmup; results may be noisy.",
            n_measure,
        )

    # Warmup
    for x in inputs[:warmup]:
        _ = pipeline.predict([x])

    times_ms: list[float] = []
    for x in inputs[warmup : warmup + n_measure]:
        t0 = time.perf_counter()
        _ = pipeline.predict([x])
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
            "device": "CPU",
            "precision": "fp64",
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
        "device": "CPU",
        "precision": "fp64",
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
    title: str = "TF-IDF + LR — Intent Confusion Matrix (normalized)",
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

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
# Validation
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
def run_pipeline(
    config_path: Path,
    repo_root: Path,
    smoke_test: bool = False,
    seed: int | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Train, evaluate, and save all artifacts. Returns the results dict.

    Parameters
    ----------
    config_path : Path
        Path to YAML config.
    repo_root : Path
        Repository root (used for output paths and git hash).
    smoke_test : bool
        If True, use 100 train samples, 50 val, 50 test; tiny latency run.
    seed : int | None
        Override random seed in the classifier.
    output_root : Path | None
        If provided, overrides repo_root for artifact output locations
        (useful for smoke tests to /tmp).
    """
    cfg = load_config(config_path)
    runtime = cfg.get("runtime", {})
    if seed is not None:
        cfg["classifier"]["random_state"] = seed
        runtime["seed"] = seed

    art_root = output_root if output_root is not None else repo_root

    # Resolve paths
    data_cfg = cfg["data"]
    train_path = repo_root / data_cfg["train_path"]
    val_path = repo_root / data_cfg["val_path"]
    test_path = repo_root / data_cfg["test_path"]
    text_col = data_cfg["text_column"]
    label_col = data_cfg["label_column"]

    # Run identity
    model_tag = "tfidf_smoke" if smoke_test else "tfidf"
    run_id = make_run_id(model_tag, repo_root)
    logger.info("run_id=%s smoke_test=%s seed=%s", run_id, smoke_test, runtime.get("seed"))

    # Output dirs
    ckpt_dir = art_root / cfg["artifacts"]["checkpoint_dir"]
    metrics_dir = art_root / cfg["artifacts"]["metrics_root"] / run_id
    figures_dir = art_root / cfg["artifacts"]["figures_dir"]
    errors_dir = art_root / cfg["artifacts"]["error_analysis_dir"]
    for d in (ckpt_dir, metrics_dir, figures_dir, errors_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Data
    train_limit = 100 if smoke_test else None
    eval_limit = 50 if smoke_test else None
    logger.info("Loading data (smoke_test=%s)", smoke_test)
    X_train, y_train = load_split(train_path, text_col, label_col, limit=train_limit)
    X_val, y_val = load_split(val_path, text_col, label_col, limit=eval_limit)
    X_test, y_test = load_split(test_path, text_col, label_col, limit=eval_limit)
    logger.info(
        "train=%d val=%d test=%d | n_intents_train=%d",
        len(X_train), len(X_val), len(X_test), len(set(y_train)),
    )

    # Canonical label set = intents seen in training data, sorted.
    # In smoke mode only a subset may appear — that's OK, we only use this
    # list for the confusion matrix axes, not for metric filtering.
    label_order = sorted(set(y_train) | set(y_val) | set(y_test))

    # Build + fit
    pipe = build_pipeline(cfg)
    logger.info("Fitting pipeline: %s", pipe)
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0
    logger.info("Training wall-clock: %.2f s", train_seconds)

    # Feature / model stats
    vectorizer: TfidfVectorizer = pipe.named_steps["tfidf"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    vocab_size = len(vectorizer.vocabulary_)
    n_classes = len(clf.classes_)
    n_params = int(clf.coef_.size + clf.intercept_.size)
    logger.info(
        "Fitted | vocab_size=%d classes=%d coef_params=%d",
        vocab_size, n_classes, n_params,
    )

    # Save checkpoint
    ckpt_path = ckpt_dir / "pipeline.joblib"
    joblib.dump(pipe, ckpt_path, compress=3)
    disk_size_mb = round(ckpt_path.stat().st_size / (1024 * 1024), 2)
    logger.info("Saved pipeline to %s (%.2f MB)", ckpt_path, disk_size_mb)

    # Predict val + test
    logger.info("Predicting on val/test...")
    y_val_pred = pipe.predict(X_val).tolist()
    y_test_pred = pipe.predict(X_test).tolist()

    val_metrics = compute_classification_metrics(y_val, y_val_pred, labels=label_order)
    test_metrics = compute_classification_metrics(y_test, y_test_pred, labels=label_order)
    logger.info(
        "VAL  acc=%.4f macro_f1=%.4f",
        val_metrics["accuracy"], val_metrics["macro_f1"],
    )
    logger.info(
        "TEST acc=%.4f macro_f1=%.4f",
        test_metrics["accuracy"], test_metrics["macro_f1"],
    )

    # Confusion matrix (test)
    cm_path = figures_dir / "03_tfidf_confusion_matrix.png"
    plot_confusion_matrix(y_test, y_test_pred, labels=label_order, save_path=cm_path)
    logger.info("Saved confusion matrix to %s", cm_path)

    # Error analysis (test)
    errors_df = pd.DataFrame(
        {
            "text": X_test,
            "true_label": y_test,
            "predicted_label": y_test_pred,
        }
    )
    errors_df = errors_df[errors_df["true_label"] != errors_df["predicted_label"]].copy()
    errors_df["text_length"] = errors_df["text"].str.len()
    errors_csv_path = errors_dir / "tfidf_errors.csv"
    errors_df.to_csv(errors_csv_path, index=False)
    logger.info("Saved %d errors to %s", len(errors_df), errors_csv_path)

    # Latency (CPU, per-sample)
    n_warmup = int(runtime.get("n_latency_warmup", 50))
    n_measure = int(runtime.get("n_latency_samples", 500))
    if smoke_test:
        n_warmup = 5
        n_measure = 20
    latency = measure_latency_cpu(pipe, X_test, warmup=n_warmup, n_measure=n_measure)
    if latency["p50_ms"] is not None:
        logger.info(
            "Latency p50=%.3fms p95=%.3fms throughput=%.0f/s",
            latency["p50_ms"], latency["p95_ms"], latency["throughput_per_sec"],
        )

    # Assemble results.json (matches ml-evaluation schema)
    import platform
    import sklearn

    checkpoint_rel = str(ckpt_path.relative_to(art_root))
    test_set_rel = str(test_path.relative_to(repo_root))
    results: dict[str, Any] = {
        "run_id": run_id,
        "model_name": runtime.get("model_name", "TF-IDF + LR"),
        "model_family": runtime.get("model_family", "traditional-ml"),
        "checkpoint_path": checkpoint_rel,
        "test_set": test_set_rel,
        "test_set_hash": test_set_hash(test_path),
        "n_samples": len(X_test),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": git_short_hash(repo_root),
        "seed": runtime.get("seed", 42),
        "training_wallclock_seconds": round(train_seconds, 3),
        "framework_versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "metrics": {
            "intent": test_metrics,
            "intent_val": val_metrics,
        },
        "latency": latency,
        "resources": {
            "gpu_memory_mb_peak_train": 0,
            "gpu_memory_mb_peak_inference": 0,
            "disk_size_mb": disk_size_mb,
            "parameters_trainable": n_params,
            "parameters_total": n_params,
            "vocab_size": vocab_size,
        },
        "parse_errors": 0,
        "notes": (
            "Baseline 1. CPU-only. No grid search, frozen config. "
            f"val_accuracy={val_metrics['accuracy']}, "
            f"val_macro_f1={val_metrics['macro_f1']}."
        ),
    }

    if not smoke_test:
        # Full runs should meet schema sanity checks.
        validate_results(results)

    # Persist results + config
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
        description="Train TF-IDF + LogisticRegression baseline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tfidf_config.yaml"),
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
        help="Run on 100 train / 50 val / 50 test samples for a fast end-to-end check.",
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

    repo_root = args.repo_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        return 2

    output_root = args.output_root.resolve() if args.output_root else None

    results = run_pipeline(
        config_path=config_path,
        repo_root=repo_root,
        smoke_test=args.smoke_test,
        seed=args.seed,
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
