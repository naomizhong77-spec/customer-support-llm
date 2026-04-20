"""Train BANKING77 Main Model: Qwen2.5-7B-Instruct + QLoRA for generative intent classification.

BANKING77 variant of scripts/train_qwen_lora.py. Key differences vs. Bitext:
  - Data paths point at data/banking77/{processed,instruction}/*
  - Flat 77-class label space (no category hierarchy). The JSONL `output` field
    is a single line `"Intent: <label_name>"` (no "Category:" line).
  - response_template anchor is `"Intent:"` (was `"Category:"`).
  - Output parser returns (intent, parse_error) — no category logic.
  - test_set_hash uses the FILE-BYTES sha256 method (copied verbatim from
    scripts/train_tfidf_banking77.py) — not the Bitext sorted-CSV method.
  - Output artifacts carry the `_banking77` suffix.

Design (frozen for this config):
  - Base: unsloth/Qwen2.5-7B-Instruct (load_in_4bit=True resolves to the
    -bnb-4bit quantized checkpoint via Unsloth's FLOAT_TO_INT_MAPPER).
  - LoRA: r=16, alpha=32, dropout=0, bias="none", task_type="CAUSAL_LM"
    target_modules = [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  - Quant: 4-bit NF4, double-quant, compute_dtype=bfloat16.
  - Training: 3 epochs, lr=2e-4, per_device_train_batch_size=4, grad_accum=4
    (effective bs=16), warmup_ratio=0.03, cosine, bf16, grad_checkpoint,
    optim="paged_adamw_8bit", max_seq_length=256.
  - Response-only loss: TRL DataCollatorForCompletionOnlyLM,
    response_template="Intent:".
  - Eval: generation-based (greedy); parse "Intent: y" robustly.
  - Save LoRA adapter only to checkpoints/qwen_lora_banking77/<run_id>/.

Outputs (under repo-root, or --output-root for smoke tests):
  - checkpoints/qwen_lora_banking77/<run_id>/       (adapter, save_total_limit=2)
  - outputs/metrics/qwen_lora_banking77_<run_id>/results.json   (ml-evaluation schema)
  - outputs/metrics/qwen_lora_banking77_<run_id>/config.yaml
  - outputs/metrics/qwen_lora_banking77_<run_id>/train_log.jsonl
  - outputs/metrics/qwen_lora_banking77_<run_id>/git_info.txt
  - outputs/figures/10_qwen_lora_banking77_confusion_matrix.png
  - outputs/error_analysis/qwen_lora_banking77_errors.csv

Usage:
    python scripts/train_qwen_lora_banking77.py --config configs/qwen_lora_banking77_config.yaml
    python scripts/train_qwen_lora_banking77.py --config configs/qwen_lora_banking77_config.yaml --smoke-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("train_qwen_lora_banking77")


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
    """Full sha256 of the raw parquet file bytes — BANKING77 cross-model contract.

    Copied VERBATIM from scripts/train_tfidf_banking77.py. Do not change.

        with open(parquet_path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        return f"sha256:{digest}"

    Verify locally with:
        sha256sum data/banking77/processed/test.parquet
    """
    with open(parquet_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return f"sha256:{digest}"


# ---------------------------------------------------------------------------
# Data: JSONL -> chat messages -> chat-template string
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a banking support intent classifier. Given a customer banking "
    "query, output exactly one line: 'Intent: <intent>'. "
    "Do not add any other text."
)


@dataclass
class InstructionRow:
    """One row from the BANKING77 instruction JSONL.

    Input JSONL schema (produced by scripts/prepare_banking77_instruction_data.py):
        {"instruction": <task desc>,
         "input": <customer banking query>,
         "output": "Intent: <label_name>"}
    """
    instruction: str
    input: str
    output: str


def load_jsonl(path: Path, limit: int | None = None) -> list[InstructionRow]:
    rows: list[InstructionRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(
                InstructionRow(
                    instruction=str(obj["instruction"]),
                    input=str(obj.get("input", "")),
                    output=str(obj["output"]),
                )
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_messages(row: InstructionRow, include_answer: bool) -> list[dict[str, str]]:
    """Build Qwen2.5 chat-template messages. BANKING77 prompt scaffold."""
    user_content = (
        "Classify the following customer banking query into its intent.\n"
        f"Query: {row.input}"
    )
    msgs: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if include_answer:
        msgs.append({"role": "assistant", "content": row.output})
    return msgs


def format_for_training(row: InstructionRow, tokenizer) -> str:  # type: ignore[no-untyped-def]
    """Return full chat-template string with the assistant answer appended."""
    return tokenizer.apply_chat_template(
        build_messages(row, include_answer=True),
        tokenize=False,
        add_generation_prompt=False,
    )


def format_for_generation(row: InstructionRow, tokenizer) -> str:  # type: ignore[no-untyped-def]
    """Return chat-template prompt *without* the assistant answer (eval-time)."""
    return tokenizer.apply_chat_template(
        build_messages(row, include_answer=False),
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Generative output parser (BANKING77: single-line "Intent: y")
# ---------------------------------------------------------------------------
# Post-hoc fix (2026-04-20): the original implementation required the model's
# decoded output to contain an "Intent:" anchor, but DataCollatorForCompletionOnlyLM
# with response_template="Intent:" masks -100 on tokens up to and including the
# anchor, meaning the training loss only shapes tokens AFTER "Intent:". As a
# result the adapter learns to emit the bare label (e.g. "card_arrival") and
# the old regex rejected 100% of generations. The cascade below handles both
# shapes: the "Intent: y" form (e.g. the JSONL gold strings, or any verbose
# inference output) and the bare-label form.
_BARE_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def parse_qwen_output(
    text: str,
    valid_intents: list[str],
) -> tuple[str | None, bool, str]:
    """Parse Qwen output -> (intent, parse_error, parse_path).

    Cascade:
      (a) "regex"   — original anchored regex r'[Ii]ntent\\s*[:=]\\s*([A-Za-z0-9_]+)'.
                      Handles gold strings ("Intent: card_arrival") and verbose
                      model outputs that echo the anchor.
      (b) "bare"    — first whitespace-delimited identifier token of the output,
                      exact (case-insensitive) match against valid_intents.
                      Handles bare-label outputs the trained adapter actually
                      emits (the common case with response-only loss masking).
      (c) "fuzzy"   — difflib.get_close_matches cutoff=0.7 against valid_intents
                      on either the regex-extracted or bare-extracted candidate.
                      Handles near-miss typos / minor suffix drift.
      (d) "none"    — all fell through; parse_error=True and caller must treat
                      the row as PARSE_ERROR.

    Signature note: return is now a 3-tuple. Callers that previously unpacked
    the 2-tuple must be updated — grep for `parse_qwen_output(` in this repo.
    """
    # --- (a) anchored regex ------------------------------------------------
    intent_match = re.search(r"[Ii]ntent\s*[:=]\s*([A-Za-z0-9_]+)", text)
    if intent_match:
        candidate = intent_match.group(1).lower()
        if candidate in valid_intents:
            return candidate, False, "regex"
        # regex hit but label not in vocab -> try fuzzy on this candidate
        matches = get_close_matches(candidate, valid_intents, n=1, cutoff=0.7)
        if matches:
            return matches[0], False, "fuzzy"
        # fall through to bare-token pass below, in case the model emitted
        # both an "Intent:" anchor AND a separate bare-label elsewhere in text
        regex_candidate = candidate
    else:
        regex_candidate = None

    # --- (b) bare label ----------------------------------------------------
    # Take the first identifier-shaped token of the decoded text; BANKING77
    # labels are lowercase_snake with optional digits. Strip leading whitespace
    # and any stray punctuation the chat template may have leaked.
    stripped = text.strip()
    tok_match = _BARE_TOKEN_RE.search(stripped)
    bare_candidate: str | None = None
    if tok_match:
        bare_candidate = tok_match.group(0).lower()
        if bare_candidate in valid_intents:
            return bare_candidate, False, "bare"

    # --- (c) fuzzy fallback on whichever candidate we have -----------------
    for cand in (bare_candidate, regex_candidate):
        if cand:
            matches = get_close_matches(cand, valid_intents, n=1, cutoff=0.7)
            if matches:
                return matches[0], False, "fuzzy"

    # --- (d) unparseable ---------------------------------------------------
    return None, True, "none"


# ---------------------------------------------------------------------------
# Metrics (ml-evaluation skill schema)
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


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    save_path: Path,
    title: str = "Qwen2.5-7B-QLoRA (BANKING77) — Intent Confusion Matrix (normalized)",
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    # 77-class matrix is large; scale figure + drop annotations for readability.
    fig, ax = plt.subplots(figsize=(22, 20))
    sns.heatmap(
        cm_norm,
        annot=False,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Greens",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    import matplotlib.pyplot as _plt
    _plt.xticks(rotation=90, ha="center", fontsize=7)
    _plt.yticks(rotation=0, fontsize=7)
    _plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Results schema validation (matches ml-evaluation skill)
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
    # Relaxed 2026-04-20 after the parser-cascade fix. The original 0.05 guard
    # was meant to catch exactly the response-only-mask / parser-anchor divergence
    # that it itself failed to prevent (it aborted post-generation, not pre-train).
    # With the fixed parser, 0.5 is the correct floor for a tuned 7B on BANKING77.
    assert acc > 0.5, "Suspicious low accuracy — expected >0.5 for tuned Qwen on BANKING77"
    f1 = results["metrics"]["intent"]["macro_f1"]
    assert f1 <= acc + 0.05, f"macro_F1 ({f1}) > accuracy ({acc}) shouldn't happen often"
    p50 = results["latency"]["p50_ms"]
    p95 = results["latency"]["p95_ms"]
    if p50 is not None and p95 is not None:
        assert p95 >= p50, "p95 must be >= p50"


# ---------------------------------------------------------------------------
# Generation-based evaluation on the test set
# ---------------------------------------------------------------------------
def generate_one(  # type: ignore[no-untyped-def]
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    device,
) -> tuple[str, float]:
    """Single-sample greedy generation; returns (decoded_response, elapsed_ms)."""
    import torch

    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    input_len = inputs["input_ids"].shape[1]

    use_cuda_events = torch.cuda.is_available() and device.type == "cuda"
    if use_cuda_events:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,  # irrelevant when do_sample=False; avoid warnings
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if use_cuda_events:
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = float(start_event.elapsed_time(end_event))
    else:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    gen_ids = out[0, input_len:]
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return decoded, elapsed_ms


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
    """Train, evaluate (generative), and save all artifacts. Returns results dict."""
    # Heavy deps deferred so the module is importable on a CPU login node.
    import torch
    import transformers
    from datasets import Dataset
    from transformers import set_seed as hf_set_seed
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    gen_cfg = cfg["generation"]
    eval_cfg = cfg["evaluation"]
    art_cfg = cfg["artifacts"]
    runtime_cfg = cfg.get("runtime", {})

    if seed is not None:
        train_cfg["seed"] = seed
    effective_seed = int(train_cfg.get("seed", 42))
    hf_set_seed(effective_seed)

    art_root = output_root if output_root is not None else repo_root

    # Data paths (always relative to repo_root — data lives with the code)
    train_jsonl = repo_root / data_cfg["train_jsonl"]
    val_jsonl = repo_root / data_cfg["val_jsonl"]
    test_jsonl = repo_root / data_cfg["test_jsonl"]
    test_parquet = repo_root / data_cfg["test_parquet"]
    expected_hash = str(data_cfg["expected_test_set_hash"])
    label_column = str(data_cfg.get("label_column", "label_name"))

    # === Run identity ======================================================
    model_tag = "qwen_lora_banking77_smoke" if smoke_test else "qwen_lora_banking77"
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

    # === Verify test-set hash BEFORE we spend GPU cycles ===================
    actual_hash = test_set_hash(test_parquet)
    logger.info("test_set_hash actual=%s expected=%s", actual_hash, expected_hash)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Test set hash mismatch. Expected {expected_hash}, got {actual_hash}. "
            "Refusing to train — every BANKING77 model must eval on the same test.parquet."
        )

    # Canonical label space derived from the test parquet (77 flat intents).
    test_df = pd.read_parquet(test_parquet)
    valid_intents: list[str] = sorted(test_df[label_column].astype(str).unique().tolist())
    logger.info(
        "Label space: %d flat intents (column=%r, from %s)",
        len(valid_intents), label_column, test_parquet.name,
    )

    # === Load data =========================================================
    smoke = runtime_cfg  # alias
    train_limit = int(smoke["smoke_train_samples"]) if smoke_test else None
    eval_limit = int(smoke["smoke_eval_samples"]) if smoke_test else None
    logger.info("Loading JSONL data (smoke_test=%s)", smoke_test)
    train_rows = load_jsonl(train_jsonl, limit=train_limit)
    val_rows = load_jsonl(val_jsonl, limit=eval_limit)
    test_rows = load_jsonl(test_jsonl, limit=eval_limit)
    logger.info("train=%d val=%d test=%d", len(train_rows), len(val_rows), len(test_rows))

    # === Model + LoRA via Unsloth =========================================
    base_model = model_cfg["base_model"]
    max_seq_length = int(model_cfg["max_seq_length"])
    logger.info("Loading base model via Unsloth: %s (max_seq_length=%d)",
                base_model, max_seq_length)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,                               # auto-detect bf16 on A40
        load_in_4bit=bool(model_cfg["load_in_4bit"]),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info("Tokenizer pad_token=%r eos_token=%r vocab_size=%d",
                tokenizer.pad_token, tokenizer.eos_token, tokenizer.vocab_size)

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora_cfg["r"]),
        target_modules=list(lora_cfg["target_modules"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        bias=str(lora_cfg["bias"]),
        use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
        random_state=effective_seed,
    )

    param_total = int(sum(p.numel() for p in model.parameters()))
    param_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    logger.info("Params trainable=%s/%s (%.3f%%)",
                f"{param_trainable:,}", f"{param_total:,}",
                100.0 * param_trainable / max(param_total, 1))

    # === HF Datasets + formatting + response-only collator =================
    if smoke_test and train_rows:
        _r0 = train_rows[0]
        logger.info("===== FIRST TRAIN PROMPT (smoke-only debug) =====")
        logger.info("RAW JSONL instruction=%r", _r0.instruction)
        logger.info("RAW JSONL input=%r", _r0.input)
        logger.info("RAW JSONL output=%r", _r0.output)
        logger.info("RENDERED (format_for_training):\n%s", format_for_training(_r0, tokenizer))
        logger.info("RENDERED (format_for_generation):\n%s", format_for_generation(_r0, tokenizer))
        logger.info("===== END FIRST TRAIN PROMPT =====")

    def _rows_to_texts(rows: list[InstructionRow]) -> list[str]:
        return [format_for_training(r, tokenizer) for r in rows]

    train_texts = _rows_to_texts(train_rows)
    val_texts = _rows_to_texts(val_rows)
    # Sanity: response_template MUST appear in every formatted example,
    # otherwise the collator silently masks out ALL labels -> NaN loss.
    response_template = str(data_cfg["response_template"])
    missing = sum(1 for t in train_texts if response_template not in t)
    if missing > 0:
        raise RuntimeError(
            f"response_template={response_template!r} missing in {missing}/{len(train_texts)} "
            f"train examples. Loss masking would drop these. First bad sample: "
            f"{train_texts[[i for i, t in enumerate(train_texts) if response_template not in t][0]][:200]!r}"
        )
    logger.info("response_template=%r present in 100%% of %d train examples",
                response_template, len(train_texts))

    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds = Dataset.from_dict({"text": val_texts})

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # Smoke-only: dump one tokenized sample with labels so the user can verify
    # the loss mask hugs the response span only. -100 = ignored by loss.
    if smoke_test and train_texts:
        sample_tok = tokenizer(train_texts[0], add_special_tokens=False)
        # DataCollatorForCompletionOnlyLM expects a list of dicts with input_ids
        collated = data_collator([{"input_ids": sample_tok["input_ids"]}])
        input_ids = collated["input_ids"][0].tolist()
        labels = collated["labels"][0].tolist()
        token_rows = []
        for i, (tid, lid) in enumerate(zip(input_ids, labels)):
            tok_str = tokenizer.decode([tid], skip_special_tokens=False)
            mask_marker = "TRAIN" if lid != -100 else "-----"
            token_rows.append(f"  [{i:3d}] {mask_marker} id={tid:>6d} label={lid:>6d} tok={tok_str!r}")
        logger.info("===== TOKENIZED SAMPLE WITH LOSS MASK (smoke-only debug) =====")
        logger.info("response_template=%r", response_template)
        logger.info("Only tokens marked TRAIN contribute to loss (labels != -100):")
        for line in token_rows:
            logger.info(line)
        n_train_tokens = sum(1 for lid in labels if lid != -100)
        logger.info("Tokens contributing to loss: %d/%d (%.1f%%)",
                    n_train_tokens, len(labels),
                    100.0 * n_train_tokens / max(len(labels), 1))
        logger.info("===== END TOKENIZED SAMPLE =====")

    # === SFTConfig (full vs smoke overrides) ==============================
    epochs = 1 if smoke_test else int(train_cfg["num_train_epochs"])
    max_steps = int(smoke["smoke_max_steps"]) if smoke_test else -1
    logging_steps = (
        int(smoke["smoke_logging_steps"]) if smoke_test else int(train_cfg["logging_steps"])
    )
    eval_steps = (
        int(smoke["smoke_eval_steps"]) if smoke_test else int(train_cfg["eval_steps"])
    )
    save_steps = (
        int(smoke["smoke_save_steps"]) if smoke_test else int(train_cfg["save_steps"])
    )

    sft_args = SFTConfig(
        output_dir=str(ckpt_dir),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        lr_scheduler_type=str(train_cfg["lr_scheduler_type"]),
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        bf16=bool(train_cfg["bf16"]),
        fp16=bool(train_cfg["fp16"]),
        gradient_checkpointing=bool(train_cfg["gradient_checkpointing"]),
        optim=str(train_cfg["optim"]),
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=bool(train_cfg["packing"]),
        logging_steps=logging_steps,
        eval_strategy=str(train_cfg["eval_strategy"]),
        eval_steps=eval_steps,
        save_strategy=str(train_cfg["save_strategy"]),
        save_steps=save_steps,
        save_total_limit=int(train_cfg["save_total_limit"]),
        load_best_model_at_end=bool(train_cfg["load_best_model_at_end"]),
        metric_for_best_model=str(train_cfg["metric_for_best_model"]),
        greater_is_better=bool(train_cfg["greater_is_better"]),
        dataloader_num_workers=int(train_cfg["dataloader_num_workers"]),
        report_to=str(train_cfg["report_to"]),
        seed=effective_seed,
        save_safetensors=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        args=sft_args,
    )

    # === GPU memory baseline ==============================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        logger.info("CUDA device: %s (capacity=%.1f GB)",
                    torch.cuda.get_device_name(device),
                    torch.cuda.get_device_properties(device).total_memory / (1024 ** 3))
    else:
        logger.warning("No CUDA device available — QLoRA requires GPU. Will fail shortly.")

    # === Train ============================================================
    logger.info(
        "Starting training: epochs=%s max_steps=%s per_device_bs=%s grad_accum=%s "
        "effective_bs=%s lr=%s bf16=%s grad_checkpoint=%s optim=%s",
        epochs, max_steps,
        sft_args.per_device_train_batch_size,
        sft_args.gradient_accumulation_steps,
        sft_args.per_device_train_batch_size * sft_args.gradient_accumulation_steps,
        sft_args.learning_rate, sft_args.bf16,
        sft_args.gradient_checkpointing, sft_args.optim,
    )
    t0 = time.perf_counter()
    train_out = trainer.train(
        resume_from_checkpoint=str(resume_from) if resume_from else None
    )
    train_seconds = time.perf_counter() - t0
    logger.info("Training wall-clock: %.2f s", train_seconds)
    logger.info(
        "train_runtime=%.2fs global_step=%d final_train_loss=%.4f",
        train_out.metrics.get("train_runtime", -1.0),
        trainer.state.global_step,
        train_out.training_loss,
    )

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

    # Save the (best, since load_best_model_at_end=True) LoRA adapter
    final_save_dir = ckpt_dir / "final_adapter"
    trainer.save_model(str(final_save_dir))
    tokenizer.save_pretrained(str(final_save_dir))
    total_bytes = 0
    for p in final_save_dir.rglob("*"):
        if p.is_file():
            total_bytes += p.stat().st_size
    disk_size_mb = round(total_bytes / (1024 * 1024), 2)
    logger.info("Saved LoRA adapter to %s (%.2f MB)", final_save_dir, disk_size_mb)

    # === Generative evaluation on test set ================================
    logger.info("Switching to inference mode")
    FastLanguageModel.for_inference(model)

    max_new_tokens = int(gen_cfg["max_new_tokens"])
    logger.info("Generating predictions on test set (n=%d, max_new_tokens=%d, greedy)",
                len(test_rows), max_new_tokens)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    y_true_intent: list[str] = []
    y_pred_intent: list[str] = []
    generations: list[dict[str, Any]] = []
    parse_errors = 0
    parse_path_counts: dict[str, int] = {"regex": 0, "bare": 0, "fuzzy": 0, "none": 0}
    latencies_ms: list[float] = []
    n_warmup = int(eval_cfg["n_latency_warmup"])
    n_latency_samples = int(eval_cfg["n_latency_samples"])
    if smoke_test:
        n_warmup = 5
        n_latency_samples = 20

    for i, row in enumerate(test_rows):
        prompt = format_for_generation(row, tokenizer)
        decoded, elapsed_ms = generate_one(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device,
        )
        if i >= n_warmup and len(latencies_ms) < n_latency_samples:
            latencies_ms.append(elapsed_ms)

        # Gold: JSONL output is "Intent: <label>"; parser handles it (regex path)
        gold_int, _, _ = parse_qwen_output(row.output, valid_intents)
        pred_int, perr, ppath = parse_qwen_output(decoded, valid_intents)
        parse_path_counts[ppath] = parse_path_counts.get(ppath, 0) + 1
        if perr:
            parse_errors += 1
        y_true_intent.append(gold_int or "UNKNOWN")
        y_pred_intent.append(pred_int or "PARSE_ERROR")
        generations.append({
            "text": row.input,
            "gold": row.output,
            "pred_raw": decoded,
            "pred_intent": pred_int,
            "parse_error": perr,
            "parse_path": ppath,
        })

        if (i + 1) % max(len(test_rows) // 10, 1) == 0:
            logger.info(
                "  generated %d/%d (parse_errors so far: %d, paths=%s)",
                i + 1, len(test_rows), parse_errors, parse_path_counts,
            )

    gpu_mem_peak_infer_mb = 0
    if device.type == "cuda":
        gpu_mem_peak_infer_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    # Metrics on the full 77-intent label order (+PARSE_ERROR absorbing class)
    full_intent_order = valid_intents
    intent_labels = (
        full_intent_order + ["PARSE_ERROR"]
        if "PARSE_ERROR" in y_pred_intent
        else full_intent_order
    )

    intent_metrics = compute_classification_metrics(
        y_true_intent, y_pred_intent, labels=intent_labels
    )
    logger.info(
        "TEST intent  acc=%.4f macro_f1=%.4f weighted_f1=%.4f parse_errors=%d",
        intent_metrics["accuracy"], intent_metrics["macro_f1"],
        intent_metrics["weighted_f1"], parse_errors,
    )

    # Confusion matrix
    cm_path = figures_dir / art_cfg["confusion_matrix_filename"]
    plot_confusion_matrix(
        y_true_intent, y_pred_intent, labels=intent_labels, save_path=cm_path,
    )
    logger.info("Saved confusion matrix to %s", cm_path)

    # Error analysis CSV
    errors_df = pd.DataFrame(generations)
    errors_df["true_intent"] = y_true_intent
    errors_df["predicted_intent"] = y_pred_intent
    errors_df = errors_df[errors_df["true_intent"] != errors_df["predicted_intent"]].copy()
    errors_df["text_length"] = errors_df["text"].str.len()
    errors_csv_path = errors_dir / art_cfg["errors_csv_filename"]
    errors_df.to_csv(errors_csv_path, index=False)
    logger.info("Saved %d errors to %s", len(errors_df), errors_csv_path)

    # Latency summary
    if latencies_ms:
        arr = np.asarray(latencies_ms)
        latency = {
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std()),
            "throughput_per_sec": float(1000.0 / max(arr.mean(), 1e-9)),
            "batch_size": int(eval_cfg["batch_size"]),
            "device": str(device),
            "precision": "bf16+4bit",
            "n_samples_timed": int(arr.size),
        }
        logger.info(
            "Latency p50=%.1fms p95=%.1fms p99=%.1fms throughput=%.2f/s",
            latency["p50_ms"], latency["p95_ms"], latency["p99_ms"],
            latency["throughput_per_sec"],
        )
    else:
        latency = {
            "p50_ms": None, "p95_ms": None, "p99_ms": None,
            "mean_ms": None, "std_ms": None, "throughput_per_sec": None,
            "batch_size": int(eval_cfg["batch_size"]), "device": str(device),
            "precision": "bf16+4bit", "n_samples_timed": 0,
        }

    # === Assemble results.json =============================================
    checkpoint_rel = str(final_save_dir.relative_to(art_root))
    test_set_rel = str(test_parquet.relative_to(repo_root))
    results: dict[str, Any] = {
        "run_id": run_id,
        "model_name": runtime_cfg.get("model_name", "Qwen2.5-7B-QLoRA (BANKING77)"),
        "model_family": runtime_cfg.get("model_family", "llm-generative"),
        "checkpoint_path": checkpoint_rel,
        "test_set": test_set_rel,
        "test_set_hash": actual_hash,
        "n_samples": len(test_rows),
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
            "intent": intent_metrics,
        },
        "latency": latency,
        "resources": {
            "gpu_memory_mb_peak_train": gpu_mem_peak_train_mb,
            "gpu_memory_mb_peak_inference": gpu_mem_peak_infer_mb,
            "disk_size_mb": disk_size_mb,
            "parameters_trainable": param_trainable,
            "parameters_total": param_total,
        },
        "parse_errors": int(parse_errors),
        "parse_path_counts": {k: int(v) for k, v in parse_path_counts.items()},
        "notes": (
            f"Main model (Qwen2.5-7B QLoRA generative) on BANKING77 (77 flat intents). "
            f"epochs={epochs} max_steps={max_steps} "
            f"per_device_bs={sft_args.per_device_train_batch_size} "
            f"grad_accum={sft_args.gradient_accumulation_steps} "
            f"effective_bs={sft_args.per_device_train_batch_size * sft_args.gradient_accumulation_steps} "
            f"lr={sft_args.learning_rate} bf16={sft_args.bf16} "
            f"max_seq_length={max_seq_length}. "
            f"parse_errors={parse_errors}/{len(test_rows)}."
        ),
    }

    if not smoke_test:
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
            "Train Qwen2.5-7B-Instruct + QLoRA on BANKING77 (77 flat intents, "
            "Main model, 6th cell of the 2x3 experimental grid)."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qwen_lora_banking77_config.yaml"),
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
        "DONE run_id=%s test_acc=%.4f test_macro_f1=%.4f parse_errors=%d",
        results["run_id"],
        results["metrics"]["intent"]["accuracy"],
        results["metrics"]["intent"]["macro_f1"],
        results["parse_errors"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
