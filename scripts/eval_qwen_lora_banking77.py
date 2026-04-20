"""Re-evaluate a trained BANKING77 Qwen QLoRA adapter without retraining.

This companion to scripts/train_qwen_lora_banking77.py exists to recover the
JOBID 21690 run. The original run's generation step produced correct bare-label
outputs (e.g. "card_arrival") that the original parser — which required an
"Intent:" anchor — rejected 100% of the time (3079/3079 parse errors).

This script:
  1. Loads the trained LoRA adapter via Unsloth FastLanguageModel.
  2. Runs greedy generation on BOTH val and test JSONL splits, using the
     SAME prompt formatting as training (format_for_generation).
  3. Uses the fixed parser cascade (parse_qwen_output) in train_qwen_lora_banking77.
  4. Writes artifacts under a new `<run_id>_reval` suffix so the original
     buggy artifacts are preserved for the debugging retro.

Imports as much as possible from the training script to avoid drift. Anything
that is not trivially reusable (e.g. flow orchestration) lives here.

Usage:
    srun -p MGPU-TC2 --gres=gpu:1 --time=00:45:00 --cpus-per-task=8 --mem=64G --pty bash
    conda activate customer-support-llm
    python scripts/eval_qwen_lora_banking77.py \\
        --adapter checkpoints/qwen_lora_banking77/qwen_lora_banking77_20260420_3eb1a21/final_adapter \\
        --config configs/qwen_lora_banking77_config.yaml \\
        --run-id-suffix _reval

The --sanity-only flag runs the parser over the existing buggy
outputs/error_analysis/qwen_lora_banking77_errors.csv rows and reports the
path breakdown without touching the GPU. Useful as a pre-flight before paying
the ~25-min generation cost.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# --- Pull everything reusable from the training script ---------------------
# The training script lives in the same scripts/ dir; add it to sys.path.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from train_qwen_lora_banking77 import (  # noqa: E402
    compute_classification_metrics,
    format_for_generation,
    generate_one,
    git_dirty_files,
    git_short_hash,
    load_config,
    load_jsonl,
    parse_qwen_output,
    plot_confusion_matrix,
    test_set_hash,
    validate_results,
)

logger = logging.getLogger("eval_qwen_lora_banking77")


def _setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


# ---------------------------------------------------------------------------
# Parser sanity check against the existing (buggy-run) pred_raw column
# ---------------------------------------------------------------------------
def sanity_check_parser(
    errors_csv: Path,
    valid_intents: list[str],
    n: int = 10,
) -> dict[str, int]:
    """Run the fixed parser over the first `n` pred_raw rows of a prior errors.csv.

    No GPU needed — uses existing decoded strings. Returns path-count dict.
    """
    df = pd.read_csv(errors_csv)
    logger.info("Parser sanity: %d existing pred_raw rows available, inspecting first %d",
                len(df), n)
    counts: dict[str, int] = {"regex": 0, "bare": 0, "fuzzy": 0, "none": 0}
    logger.info("First %d sanity-check rows:", n)
    for i, raw in enumerate(df["pred_raw"].head(n).tolist()):
        intent, perr, path = parse_qwen_output(str(raw), valid_intents)
        counts[path] += 1
        logger.info(
            "  [%d] path=%-5s err=%-5s pred=%-30s raw=%r",
            i, path, perr, intent, raw,
        )
    logger.info("Sanity-check path breakdown over first %d: %s", n, counts)
    return counts


# ---------------------------------------------------------------------------
# Per-split generative eval (factored out of run_pipeline)
# ---------------------------------------------------------------------------
def run_generation_on_split(  # type: ignore[no-untyped-def]
    model,
    tokenizer,
    rows,
    valid_intents: list[str],
    max_new_tokens: int,
    device,
    n_warmup: int,
    n_latency_samples: int,
    split_name: str,
) -> dict[str, Any]:
    """Greedy-generate on every row, parse, collect metrics.

    Returns a dict with predictions, latencies, parse-path counts.
    """
    import torch

    y_true: list[str] = []
    y_pred: list[str] = []
    generations: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    parse_errors = 0
    parse_path_counts: dict[str, int] = {"regex": 0, "bare": 0, "fuzzy": 0, "none": 0}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    logger.info("[%s] generating n=%d rows (greedy, max_new_tokens=%d)",
                split_name, len(rows), max_new_tokens)
    t0 = time.perf_counter()

    for i, row in enumerate(rows):
        prompt = format_for_generation(row, tokenizer)
        decoded, elapsed_ms = generate_one(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device,
        )
        if i >= n_warmup and len(latencies_ms) < n_latency_samples:
            latencies_ms.append(elapsed_ms)

        gold_int, _, _ = parse_qwen_output(row.output, valid_intents)
        pred_int, perr, ppath = parse_qwen_output(decoded, valid_intents)
        parse_path_counts[ppath] = parse_path_counts.get(ppath, 0) + 1
        if perr:
            parse_errors += 1

        y_true.append(gold_int or "UNKNOWN")
        y_pred.append(pred_int or "PARSE_ERROR")
        generations.append({
            "text": row.input,
            "gold": row.output,
            "pred_raw": decoded,
            "pred_intent": pred_int,
            "parse_error": perr,
            "parse_path": ppath,
        })

        if (i + 1) % max(len(rows) // 10, 1) == 0:
            logger.info("  [%s] %d/%d  parse_errors=%d paths=%s",
                        split_name, i + 1, len(rows), parse_errors, parse_path_counts)

    wall = time.perf_counter() - t0
    gpu_mem_peak_mb = 0
    if device.type == "cuda":
        gpu_mem_peak_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    logger.info("[%s] done in %.1fs, parse_errors=%d/%d paths=%s peak_mem=%dMB",
                split_name, wall, parse_errors, len(rows),
                parse_path_counts, gpu_mem_peak_mb)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "generations": generations,
        "latencies_ms": latencies_ms,
        "parse_errors": parse_errors,
        "parse_path_counts": parse_path_counts,
        "gpu_mem_peak_mb": gpu_mem_peak_mb,
        "wall_seconds": wall,
    }


def latency_summary(
    latencies_ms: list[float],
    batch_size: int,
    device_str: str,
) -> dict[str, Any]:
    if not latencies_ms:
        return {
            "p50_ms": None, "p95_ms": None, "p99_ms": None,
            "mean_ms": None, "std_ms": None, "throughput_per_sec": None,
            "batch_size": batch_size, "device": device_str,
            "precision": "bf16+4bit", "n_samples_timed": 0,
        }
    arr = np.asarray(latencies_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "throughput_per_sec": float(1000.0 / max(arr.mean(), 1e-9)),
        "batch_size": batch_size,
        "device": device_str,
        "precision": "bf16+4bit",
        "n_samples_timed": int(arr.size),
    }


# ---------------------------------------------------------------------------
# Main re-eval flow
# ---------------------------------------------------------------------------
def run_reval(  # noqa: C901, PLR0912, PLR0915
    adapter_path: Path,
    config_path: Path,
    repo_root: Path,
    run_id_suffix: str,
    original_run_id: str | None,
    sanity_only: bool,
) -> dict[str, Any] | None:
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    gen_cfg = cfg["generation"]
    eval_cfg = cfg["evaluation"]
    art_cfg = cfg["artifacts"]
    runtime_cfg = cfg.get("runtime", {})

    # Paths
    val_jsonl = repo_root / data_cfg["val_jsonl"]
    test_jsonl = repo_root / data_cfg["test_jsonl"]
    test_parquet = repo_root / data_cfg["test_parquet"]
    expected_hash = str(data_cfg["expected_test_set_hash"])
    label_column = str(data_cfg.get("label_column", "label_name"))

    # Verify test parquet hash
    actual_hash = test_set_hash(test_parquet)
    logger.info("test_set_hash actual=%s expected=%s", actual_hash, expected_hash)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Test set hash mismatch. Expected {expected_hash}, got {actual_hash}."
        )

    # Label space
    test_df = pd.read_parquet(test_parquet)
    valid_intents: list[str] = sorted(test_df[label_column].astype(str).unique().tolist())
    logger.info("Label space: %d intents from %s", len(valid_intents), test_parquet.name)

    # --- Parser sanity check over existing buggy pred_raw rows -----------
    existing_errors_csv = repo_root / art_cfg["error_analysis_dir"] / art_cfg["errors_csv_filename"]
    if existing_errors_csv.exists():
        sanity_counts = sanity_check_parser(existing_errors_csv, valid_intents, n=10)
        n_fail = sanity_counts.get("none", 0)
        if n_fail > 2:
            raise RuntimeError(
                f"Parser sanity check failed: {n_fail}/10 rows unparseable. "
                f"Fix the cascade before paying the full generation cost."
            )
        logger.info("Parser sanity OK (%d/10 rows unparseable).", n_fail)
    else:
        logger.warning("No prior errors.csv found at %s; skipping sanity check.",
                       existing_errors_csv)

    if sanity_only:
        logger.info("sanity_only=True, exiting before model load.")
        return None

    # --- Resolve run_id and output paths ---------------------------------
    if original_run_id is None:
        # Try to infer from adapter path: .../qwen_lora_banking77/<run_id>/final_adapter
        original_run_id = adapter_path.resolve().parent.name
    reval_run_id = f"{original_run_id}{run_id_suffix}"
    metrics_dir = repo_root / art_cfg["metrics_root"] / reval_run_id
    figures_dir = repo_root / art_cfg["figures_dir"]
    errors_dir = repo_root / art_cfg["error_analysis_dir"]
    for d in (metrics_dir, figures_dir, errors_dir):
        d.mkdir(parents=True, exist_ok=True)

    # git provenance
    git_info_lines = [
        f"commit: {git_short_hash(repo_root)}",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"original_run_id: {original_run_id}",
        f"adapter_path: {adapter_path}",
        "dirty_files:",
        *[f"  {line}" for line in git_dirty_files(repo_root)],
    ]
    (metrics_dir / "git_info.txt").write_text("\n".join(git_info_lines), encoding="utf-8")

    # --- Heavy imports + model load --------------------------------------
    import torch
    import transformers
    from unsloth import FastLanguageModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Re-eval requires CUDA — no GPU available.")

    base_model = model_cfg["base_model"]
    max_seq_length = int(model_cfg["max_seq_length"])
    logger.info("Loading base model via Unsloth: %s (max_seq_length=%d, 4bit=%s)",
                base_model, max_seq_length, model_cfg["load_in_4bit"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=bool(model_cfg["load_in_4bit"]),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Attach LoRA adapter
    logger.info("Loading LoRA adapter from %s", adapter_path)
    from peft import PeftModel  # noqa: WPS433

    model = PeftModel.from_pretrained(model, str(adapter_path))
    FastLanguageModel.for_inference(model)
    logger.info("Adapter loaded; switched to inference mode.")

    # --- Data -----------------------------------------------------------
    val_rows = load_jsonl(val_jsonl)
    test_rows = load_jsonl(test_jsonl)
    logger.info("val=%d test=%d", len(val_rows), len(test_rows))

    max_new_tokens = int(gen_cfg["max_new_tokens"])
    n_warmup = int(eval_cfg["n_latency_warmup"])
    n_latency_samples = int(eval_cfg["n_latency_samples"])

    # --- Val ------------------------------------------------------------
    val_out = run_generation_on_split(
        model, tokenizer, val_rows, valid_intents,
        max_new_tokens=max_new_tokens, device=device,
        n_warmup=n_warmup, n_latency_samples=n_latency_samples,
        split_name="val",
    )
    val_intent_labels = sorted(set(val_out["y_true"]) | set(val_out["y_pred"]))
    val_metrics = compute_classification_metrics(
        val_out["y_true"], val_out["y_pred"], labels=val_intent_labels,
    )
    logger.info("VAL   acc=%.4f macro_f1=%.4f weighted_f1=%.4f parse_errors=%d",
                val_metrics["accuracy"], val_metrics["macro_f1"],
                val_metrics["weighted_f1"], val_out["parse_errors"])

    # --- Test -----------------------------------------------------------
    test_out = run_generation_on_split(
        model, tokenizer, test_rows, valid_intents,
        max_new_tokens=max_new_tokens, device=device,
        n_warmup=n_warmup, n_latency_samples=n_latency_samples,
        split_name="test",
    )
    full_intent_order = valid_intents
    test_intent_labels = (
        full_intent_order + ["PARSE_ERROR"]
        if "PARSE_ERROR" in test_out["y_pred"]
        else full_intent_order
    )
    test_metrics = compute_classification_metrics(
        test_out["y_true"], test_out["y_pred"], labels=test_intent_labels,
    )
    logger.info("TEST  acc=%.4f macro_f1=%.4f weighted_f1=%.4f parse_errors=%d",
                test_metrics["accuracy"], test_metrics["macro_f1"],
                test_metrics["weighted_f1"], test_out["parse_errors"])

    # --- Artifacts: new CM (reval) + new errors.csv (reval) --------------
    reval_cm_filename = art_cfg["confusion_matrix_filename"].replace(
        ".png", "_reval.png",
    )
    cm_path = figures_dir / reval_cm_filename
    plot_confusion_matrix(
        test_out["y_true"], test_out["y_pred"],
        labels=test_intent_labels, save_path=cm_path,
        title="Qwen2.5-7B-QLoRA (BANKING77) — Intent Confusion Matrix (reval, normalized)",
    )
    logger.info("Saved confusion matrix to %s", cm_path)

    reval_errors_csv_filename = art_cfg["errors_csv_filename"].replace(
        ".csv", "_reval.csv",
    )
    errors_df = pd.DataFrame(test_out["generations"])
    errors_df["true_intent"] = test_out["y_true"]
    errors_df["predicted_intent"] = test_out["y_pred"]
    errors_df = errors_df[
        errors_df["true_intent"] != errors_df["predicted_intent"]
    ].copy()
    errors_df["text_length"] = errors_df["text"].str.len()
    errors_csv_path = errors_dir / reval_errors_csv_filename
    errors_df.to_csv(errors_csv_path, index=False)
    logger.info("Saved %d test errors to %s", len(errors_df), errors_csv_path)

    # --- Latency (use val split to avoid skew from already-timed test) ---
    latency = latency_summary(
        val_out["latencies_ms"],
        batch_size=int(eval_cfg["batch_size"]),
        device_str=str(device),
    )
    logger.info(
        "Val latency p50=%.1fms p95=%.1fms p99=%.1fms throughput=%.2f/s n_timed=%d",
        latency["p50_ms"] or -1, latency["p95_ms"] or -1, latency["p99_ms"] or -1,
        latency["throughput_per_sec"] or -1, latency["n_samples_timed"],
    )

    # --- Assemble results.json (ml-evaluation schema) --------------------
    adapter_rel = str(adapter_path.resolve().relative_to(repo_root)) if adapter_path.resolve().is_relative_to(repo_root) else str(adapter_path)  # noqa: E501
    test_set_rel = str(test_parquet.relative_to(repo_root))
    results: dict[str, Any] = {
        "run_id": reval_run_id,
        "model_name": runtime_cfg.get("model_name", "Qwen2.5-7B-QLoRA (BANKING77)"),
        "model_family": runtime_cfg.get("model_family", "llm-generative"),
        "checkpoint_path": adapter_rel,
        "test_set": test_set_rel,
        "test_set_hash": actual_hash,
        "n_samples": len(test_rows),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": git_short_hash(repo_root),
        "seed": int(cfg["training"].get("seed", 42)),
        "training_wallclock_seconds": None,  # no retrain; see original run_id
        "framework_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "metrics": {
            "intent": test_metrics,
            "intent_val": val_metrics,
        },
        "latency": latency,
        "resources": {
            "gpu_memory_mb_peak_train": None,  # re-eval only
            "gpu_memory_mb_peak_inference": max(
                val_out["gpu_mem_peak_mb"], test_out["gpu_mem_peak_mb"]
            ),
            "disk_size_mb": None,
            "parameters_trainable": None,
            "parameters_total": None,
        },
        "parse_errors": int(test_out["parse_errors"]),
        "parse_errors_val": int(val_out["parse_errors"]),
        "parse_path_counts": {k: int(v) for k, v in test_out["parse_path_counts"].items()},
        "parse_path_counts_val": {k: int(v) for k, v in val_out["parse_path_counts"].items()},
        "notes": (
            f"Re-evaluation of adapter {original_run_id} with fixed parser cascade. "
            f"No retrain. Gen settings: greedy, max_new_tokens={max_new_tokens}. "
            f"val_acc={val_metrics['accuracy']} test_acc={test_metrics['accuracy']} "
            f"test_parse_errors={test_out['parse_errors']}/{len(test_rows)}."
        ),
        "reval_of_run_id": original_run_id,
        "reval_generation_wallclock_seconds": round(
            val_out["wall_seconds"] + test_out["wall_seconds"], 3,
        ),
    }

    validate_results(results)

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
        description=(
            "Re-evaluate a trained Qwen BANKING77 LoRA adapter with the fixed "
            "parser cascade. No retrain."
        )
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="Path to the trained LoRA adapter directory (contains adapter_model.safetensors).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qwen_lora_banking77_config.yaml"),
        help="Path to the training config YAML (for model + data paths).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument(
        "--run-id-suffix",
        default="_reval",
        help="Suffix appended to the original run_id to name new artifacts.",
    )
    parser.add_argument(
        "--original-run-id",
        default=None,
        help="Override the original run_id (default: inferred from adapter parent dir).",
    )
    parser.add_argument(
        "--sanity-only",
        action="store_true",
        help="Only run the parser sanity check on existing errors.csv, then exit.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(level=getattr(logging, args.log_level))

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

    adapter_path = args.adapter
    if not adapter_path.is_absolute():
        adapter_path = (repo_root / adapter_path).resolve()
    if not adapter_path.exists():
        logger.error("Adapter path not found: %s", adapter_path)
        return 2

    results = run_reval(
        adapter_path=adapter_path,
        config_path=config_path,
        repo_root=repo_root,
        run_id_suffix=args.run_id_suffix,
        original_run_id=args.original_run_id,
        sanity_only=args.sanity_only,
    )
    if results is not None:
        logger.info(
            "DONE reval_run_id=%s val_acc=%.4f test_acc=%.4f test_parse_errors=%d",
            results["run_id"],
            results["metrics"]["intent_val"]["accuracy"],
            results["metrics"]["intent"]["accuracy"],
            results["parse_errors"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
