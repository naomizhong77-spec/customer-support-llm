"""Train Baseline 1 (BANKING77): TF-IDF + LogisticRegression intent classifier.

This is the BANKING77 counterpart to scripts/train_tfidf.py. The Bitext script
is frozen for reproducibility, so this file is a deliberate minimal-change copy:
only data paths, column names, artifact filenames, and the test-set hash
procedure differ. Hyperparameters (vectorizer + classifier) are IDENTICAL so
that any accuracy delta is attributable to dataset difficulty, not model tuning.

Differences vs train_tfidf.py:
  - Data: data/banking77/processed/{train,val,test}.parquet
  - Columns: text / label_name (vs instruction / intent)
  - 77 fine-grained classes (vs Bitext's 27)
  - test_set_hash is the FULL sha256 of the parquet FILE bytes, not the
    first-16-hex of the CSV bytes. This is the cross-model pinning value the
    DistilBERT and Qwen BANKING77 scripts must also emit.
  - Output locations carry the `_banking77` suffix:
      checkpoints/tfidf_banking77/pipeline.joblib
      outputs/metrics/tfidf_banking77_<YYYYMMDD>_<shorthash>/results.json
      outputs/figures/08_tfidf_banking77_confusion_matrix.png
      outputs/error_analysis/tfidf_banking77_errors.csv

Usage:
    python scripts/train_tfidf_banking77.py \
        --config configs/tfidf_banking77_config.yaml
    python scripts/train_tfidf_banking77.py \
        --config configs/tfidf_banking77_config.yaml --smoke-test
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
logger = logging.getLogger("train_tfidf_banking77")


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
    """Full sha256 of the raw parquet file bytes — BANKING77 cross-model contract.

    CONTRACT (must be copied verbatim into the BANKING77 DistilBERT and Qwen
    QLoRA training/eval scripts, NOT the Bitext ones):

        with open(parquet_path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        return f"sha256:{digest}"

    Why this differs from the Bitext scripts (which use a sorted-CSV truncated
    digest): cross-dataset hash methods do NOT need to match — they only need
    to be internally consistent within a dataset. Picking the file-bytes method
    for BANKING77 means any developer can verify the pin with one shell call:

        sha256sum data/banking77/processed/test.parquet

    Expected pin (config-driven): see configs/tfidf_banking77_config.yaml.
    """
    with open(parquet_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return f"sha256:{digest}"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_pipeline(cfg: dict[str, Any]) -> Pipeline:
    """Build the frozen TF-IDF + LogisticRegression pipeline.

    Identical to train_tfidf.py so cross-dataset comparisons stay clean.
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
    title: str = "TF-IDF + LR — BANKING77 Intent Confusion Matrix (normalized)",
) -> None:
    """Plot a normalized confusion matrix.

    For BANKING77's 77 classes a 77x77 annotated heatmap is unreadable, so we
    skip per-cell annotations and enlarge the figure; labels still rendered.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    annot = len(labels) <= 30  # Bitext (27) gets annotations, BANKING77 (77) does not.
    fig, ax = plt.subplots(figsize=(22, 20))
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt=".2f" if annot else "",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        cbar_kws={"shrink": 0.6},
        ax=ax,
        annot_kws={"size": 6} if annot else None,
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.xticks(rotation=90, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
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
    """Train, evaluate, and save all artifacts. Returns the results dict."""
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

    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing data split: {p}. Regenerate via notebook 08 "
                "(notebooks/08_banking77_cleaning.ipynb)."
            )

    # Run identity
    model_tag = "tfidf_banking77_smoke" if smoke_test else "tfidf_banking77"
    run_id = make_run_id(model_tag, repo_root)
    logger.info("run_id=%s smoke_test=%s seed=%s", run_id, smoke_test, runtime.get("seed"))

    # Output dirs
    ckpt_dir = art_root / cfg["artifacts"]["checkpoint_dir"]
    metrics_dir = art_root / cfg["artifacts"]["metrics_root"] / run_id
    figures_dir = art_root / cfg["artifacts"]["figures_dir"]
    errors_dir = art_root / cfg["artifacts"]["error_analysis_dir"]
    for d in (ckpt_dir, metrics_dir, figures_dir, errors_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Verify test-set hash pin if provided
    observed_hash = test_set_hash(test_path)
    expected_hash = runtime.get("expected_test_set_hash")
    if expected_hash is not None:
        if observed_hash != expected_hash:
            raise RuntimeError(
                "test_set_hash mismatch — cross-model pinning broken.\n"
                f"  expected: {expected_hash}\n"
                f"  observed: {observed_hash}\n"
                f"  path:     {test_path}\n"
                "Regenerate test.parquet via notebook 08 or update the "
                "expected_test_set_hash in the config."
            )
        logger.info("test_set_hash matches expected pin (%s)", observed_hash)
    else:
        logger.warning(
            "No expected_test_set_hash in config; recording %s without verification.",
            observed_hash,
        )

    # Data
    # Smoke-test limits sized so the classifier sees more than one class:
    # 400 train samples covers most of the 77 BANKING77 intents at ~5/class.
    train_limit = 400 if smoke_test else None
    eval_limit = 200 if smoke_test else None
    logger.info("Loading data (smoke_test=%s)", smoke_test)
    X_train, y_train = load_split(train_path, text_col, label_col, limit=train_limit)
    X_val, y_val = load_split(val_path, text_col, label_col, limit=eval_limit)
    X_test, y_test = load_split(test_path, text_col, label_col, limit=eval_limit)
    logger.info(
        "train=%d val=%d test=%d | n_intents_train=%d",
        len(X_train), len(X_val), len(X_test), len(set(y_train)),
    )

    # Canonical label set = union across splits (only used for CM axes).
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
    cm_path = figures_dir / "08_tfidf_banking77_confusion_matrix.png"
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
    errors_csv_path = errors_dir / "tfidf_banking77_errors.csv"
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
        "model_name": runtime.get("model_name", "TF-IDF + LR (BANKING77)"),
        "model_family": runtime.get("model_family", "traditional-ml"),
        "dataset": "banking77",
        "checkpoint_path": checkpoint_rel,
        "test_set": test_set_rel,
        "test_set_hash": observed_hash,
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
            "Baseline 1 — BANKING77. CPU-only. No grid search, frozen config "
            "(identical hyperparameters to Bitext TF-IDF for comparability). "
            f"val_accuracy={val_metrics['accuracy']}, "
            f"val_macro_f1={val_metrics['macro_f1']}."
        ),
    }

    if not smoke_test:
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
        description="Train TF-IDF + LogisticRegression baseline (BANKING77)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tfidf_banking77_config.yaml"),
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
        help="Run on 400 train / 200 val / 200 test samples for a fast end-to-end check.",
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
