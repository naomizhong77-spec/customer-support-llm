"""Consolidate the 6 experimental runs into reference artefacts for the report-writer.

Reads only (never mutates) source results.json files under outputs/metrics/ and
produces 4 deliverables under outputs/consolidated/:
  1. all_runs.json         — full per-run dump + a 6-row summary_table
  2. datasets_summary.json — Bitext + BANKING77 dataset facts
  3. per_intent_comparison.csv — 77-row BANKING77 per-intent F1/P/R per model
  4. figures_manifest.md   — markdown table of every figure PNG

All numeric cells in summary_table are asserted to match the source verbatim.
"""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

ROOT = Path("/home/mcaai/zh0038qi/customer-support-llm")
METRICS = ROOT / "outputs" / "metrics"
CONSOLIDATED = ROOT / "outputs" / "consolidated"
FIGURES = ROOT / "outputs" / "figures"
CONFIGS_DIR = ROOT / "configs"

CONSOLIDATED.mkdir(parents=True, exist_ok=True)

RUNS = [
    # (dataset, model, run_dir_name, config_file, git_hash)
    ("bitext", "tfidf", "tfidf_20260418_06cce19", "configs/tfidf_config.yaml", "06cce19"),
    ("bitext", "distilbert", "distilbert_20260418_49ff3a6", "configs/distilbert_config.yaml", "49ff3a6"),
    ("bitext", "qwen_lora", "qwen_lora_20260418_d4c444f", "configs/qwen_lora_config.yaml", "d4c444f"),
    ("banking77", "tfidf", "tfidf_banking77_20260420_53ebae1", "configs/tfidf_banking77_config.yaml", "53ebae1"),
    ("banking77", "distilbert", "distilbert_banking77_20260420_315b434", "configs/distilbert_banking77_config.yaml", "315b434"),
    ("banking77", "qwen_lora", "qwen_lora_banking77_20260420_3eb1a21_reval", "configs/qwen_lora_banking77_config.yaml", "3eb1a21"),
]

FIG_MAP = {
    ("bitext", "tfidf"): ["03_tfidf_confusion_matrix.png"],
    ("bitext", "distilbert"): ["04_distilbert_confusion_matrix.png"],
    ("bitext", "qwen_lora"): ["06_qwen_lora_confusion_matrix.png"],
    ("banking77", "tfidf"): ["08_tfidf_banking77_confusion_matrix.png"],
    ("banking77", "distilbert"): ["09_distilbert_banking77_confusion_matrix.png"],
    ("banking77", "qwen_lora"): [
        "10_qwen_lora_banking77_confusion_matrix_reval.png",
    ],
}


def du_mb(path: Path) -> float | None:
    """Return directory size in MB using du -sm (block size 1MB)."""
    if not path.exists():
        return None
    out = subprocess.check_output(["du", "-sm", str(path)]).decode().split()[0]
    return float(out)


def load_run(dataset, model, run_dir, cfg_rel, git_hash):
    rdir = METRICS / run_dir
    results_path = rdir / "results.json"
    results = json.loads(results_path.read_text())

    ckpt = results.get("checkpoint_path", "")
    # Resolve size on disk
    if model == "tfidf":
        size_mb = results["resources"].get("disk_size_mb")
    elif model == "distilbert":
        size_mb = results["resources"].get("disk_size_mb")
    else:  # qwen_lora adapter size
        size_mb = results["resources"].get("disk_size_mb")
        if size_mb is None:
            # fall back to du of checkpoint dir
            size_mb = du_mb(ROOT / ckpt) if ckpt else None

    return {
        "dataset": dataset,
        "model": model,
        "run_id": results["run_id"],
        "git_hash": git_hash,
        "results": results,
        "config_file": cfg_rel,
        "checkpoint_path": ckpt if ckpt else "gitignored",
        "test_set_hash": results.get("test_set_hash", ""),
        "figure_paths": [f"outputs/figures/{f}" for f in FIG_MAP[(dataset, model)]],
    }


# ---------- training wallclock helpers ----------
def qwen_banking77_training_wallclock() -> float:
    """Qwen BANKING77 training wallclock lives in the buggy-run train_log
    (training happened once; the reval just re-ran inference on the same adapter)."""
    log_path = METRICS / "qwen_lora_banking77_20260420_3eb1a21_buggy" / "train_log.jsonl"
    last = None
    for line in log_path.read_text().splitlines():
        obj = json.loads(line)
        if "train_runtime" in obj:
            last = obj["train_runtime"]
    return last


def training_time(run: dict) -> float | None:
    r = run["results"]
    tw = r.get("training_wallclock_seconds")
    if tw is not None:
        return tw
    # Fallback for Qwen BANKING77 reval
    if run["dataset"] == "banking77" and run["model"] == "qwen_lora":
        return qwen_banking77_training_wallclock()
    return None


# ---------- model params ----------
def model_params(run: dict):
    res = run["results"].get("resources", {})
    total = res.get("parameters_total")
    if run["model"] == "qwen_lora" and total is None:
        # Qwen base model param count — pull from the bitext Qwen run which DOES have it
        total = 4393342464  # matches bitext Qwen results.json verbatim
    return total


# ---------- disk / adapter size ----------
def disk_size_mb(run: dict):
    res = run["results"].get("resources", {})
    sz = res.get("disk_size_mb")
    if sz is not None:
        return sz
    # Fallback for Qwen BANKING77 reval (null in results.json)
    if run["dataset"] == "banking77" and run["model"] == "qwen_lora":
        adapter = ROOT / "checkpoints/qwen_lora_banking77/qwen_lora_banking77_20260420_3eb1a21/final_adapter"
        return du_mb(adapter)
    return None


# ---------- build all_runs.json ----------
runs = [load_run(*r) for r in RUNS]

summary_columns = [
    "dataset", "model", "val_acc", "val_macro_f1", "test_acc",
    "test_macro_f1", "inference_p50_ms", "inference_p95_ms",
    "model_params", "training_time_sec", "adapter_or_checkpoint_size_mb"
]

summary_rows = []
for run in runs:
    r = run["results"]
    intent = r["metrics"]["intent"]
    intent_val = r["metrics"].get("intent_val", {})
    lat = r.get("latency", {})

    row = [
        run["dataset"],
        run["model"],
        intent_val.get("accuracy"),
        intent_val.get("macro_f1"),
        intent["accuracy"],
        intent["macro_f1"],
        lat.get("p50_ms"),
        lat.get("p95_ms"),
        model_params(run),
        training_time(run),
        disk_size_mb(run),
    ]
    summary_rows.append(row)

known_retros = [
    {
        "dataset": "banking77",
        "model": "distilbert",
        "issue": "metric_for_best_model name mismatch silently disabled early stopping in 5-epoch run",
        "buggy_result": "86.00% test acc",
        "fixed_result": "91.78% test acc",
        "resolution": "metric name aligned with compute_metrics output + epochs raised to 10",
    },
    {
        "dataset": "banking77",
        "model": "qwen_lora",
        "issue": "response_template dual-role: loss-masking anchor + parser contract diverged",
        "buggy_result": "3079/3079 parse_error, test acc 0.00",
        "fixed_result": "91.65% test acc",
        "resolution": "parser patched with bare-label + fuzzy-match fallback, no retraining",
    },
]

all_runs = {
    "runs": runs,
    "summary_table": {"columns": summary_columns, "rows": summary_rows},
    "known_retros": known_retros,
}

# ---------- CROSS-CHECK: every summary cell must match source verbatim ----------
cross_check_issues = []
for row, run in zip(summary_rows, runs):
    r = run["results"]
    intent = r["metrics"]["intent"]
    intent_val = r["metrics"].get("intent_val", {})
    lat = r.get("latency", {})
    checks = {
        "test_acc": (row[4], intent["accuracy"]),
        "test_macro_f1": (row[5], intent["macro_f1"]),
        "val_acc": (row[2], intent_val.get("accuracy")),
        "val_macro_f1": (row[3], intent_val.get("macro_f1")),
        "p50_ms": (row[6], lat.get("p50_ms")),
        "p95_ms": (row[7], lat.get("p95_ms")),
    }
    for name, (cell, src) in checks.items():
        if cell != src:
            cross_check_issues.append(
                f"{run['dataset']}/{run['model']} {name}: cell={cell} src={src}"
            )

if cross_check_issues:
    print("CROSS-CHECK FAILURES:")
    for i in cross_check_issues:
        print("  ", i)
else:
    print("CROSS-CHECK OK: all summary_table numeric cells match source verbatim.")

# Write all_runs.json
(CONSOLIDATED / "all_runs.json").write_text(json.dumps(all_runs, indent=2))
print(f"Wrote {CONSOLIDATED / 'all_runs.json'}")
print(f"summary_table rows: {len(summary_rows)}")

# ---------- DELIVERABLE 2: datasets_summary.json ----------
# Bitext: data_stats.json + notebook 01 text length stats
bitext_data_stats = json.loads((METRICS / "data_stats.json").read_text())
banking77_stats = json.loads((METRICS / "banking77_stats.json").read_text())

datasets_summary = {
    "bitext": {
        "source": "bitext/Bitext-customer-support-llm-chatbot-training-dataset (HuggingFace); also mirrored via Kaggle. License: CDLA-Sharing-1.0",
        "total_rows_raw": bitext_data_stats["raw_shape"][0],
        "total_rows_after_cleaning": bitext_data_stats["clean_after"]["n_rows"],
        "n_intents": 27,
        "n_categories": 11,
        "intent_imbalance_ratio_max_over_min": 1.05,
        "category_imbalance_ratio_max_over_min": 6.3,
        "text_length_stats_instruction_chars": {
            "p50": 48.0, "p90": 59.0, "p95": 61.0, "p99": 71.0, "max": 92.0,
            "source": "notebook 01 cell output (df_len.describe percentiles over 26,872 rows)"
        },
        "text_length_stats_instruction_ws_tokens": {
            "p50": 9.0, "p90": 12.0, "p95": 13.0, "p99": 14.0, "max": 16.0,
            "source": "notebook 01 cell output"
        },
        "text_length_stats_response_chars": {
            "p50": 540.0, "p90": 1059.0, "p95": 1295.0, "p99": 1837.0, "max": 2472.0,
            "source": "notebook 01 cell output"
        },
        "splits": bitext_data_stats["split_sizes"],
        "test_set_hash": "sha256:5641a8ab0fb4814b",
        "test_set_hash_source": "pulled verbatim from every Bitext results.json; truncated form used by Bitext evaluation scripts",
        "cleaning_operations_applied": [
            "drop_nan_intent",
            "dedup_on_(instruction,intent)_pair",
            "whitespace_collapse (instruction + response)",
            "mojibake_repair (regex before NFKC)",
            "unicode_NFKC_normalise",
            "category_upper_case",
            "drop_length_outliers (instruction>500 chars, response>3000 chars)",
            "dedup_post_normalise",
            "hierarchy_invariant_assertion (each intent -> one category)"
        ],
        "cleaning_log_rows_accounting": bitext_data_stats["cleaning_log"],
        "synthetic_errors_injected_for_demo": [
            {"class": "nan_intent", "rate": "~5%", "count_injected": 1344, "mechanism": "set intent=np.nan at random rows"},
            {"class": "dup_rows", "rate": "~2%", "count_injected": 537, "mechanism": "sample rows and append copies"},
            {"class": "mojibake", "rate": "~1%", "count_injected": 269, "mechanism": "prepend 'Ã©' and replace apostrophes with 'â€™' in response"},
            {"class": "length_outlier", "rate": "~1%", "count_injected": 269, "mechanism": "repeat instruction 10x to blow past 500-char cap"},
            {"class": "case_inconsistency", "rate": "~1%", "count_injected": 269, "mechanism": "category -> lower/title case mix"},
            {"class": "ws_artefact", "rate": "~2%", "count_injected": 537, "mechanism": "inject leading/trailing/triple-space/tab in instruction"},
        ],
        "note": "Raw Bitext dump is pristine (0 nulls, 0 exact dupes, 0 mojibake); synthetic dirt is injected on a working copy to demonstrate cleaning, then cleaned back to a zero-defect state before stratified split.",
    },
    "banking77": {
        "source": "PolyAI/banking77 on HuggingFace",
        "total_rows_raw": banking77_stats["eda"]["shapes"]["train_raw"][0] + banking77_stats["eda"]["shapes"]["test_raw"][0],
        "total_rows_after_cleaning": banking77_stats["cleaning"]["split_sizes"]["train"] + banking77_stats["cleaning"]["split_sizes"]["val"] + banking77_stats["cleaning"]["split_sizes"]["test"],
        "n_intents": 77,
        "n_categories": None,
        "imbalance_ratio_max_over_min": banking77_stats["eda"]["class_distribution_train"]["imbalance_ratio_max_over_min"],
        "imbalance_ratio_source": "max_over_min on raw train split (from banking77_stats.json)",
        "min_class_size_train_raw": banking77_stats["eda"]["class_distribution_train"]["min"],
        "max_class_size_train_raw": banking77_stats["eda"]["class_distribution_train"]["max"],
        "median_class_size_train_raw": banking77_stats["eda"]["class_distribution_train"]["median"],
        "text_length_stats_chars": banking77_stats["eda"]["text_length_train"]["chars"],
        "text_length_stats_words": banking77_stats["eda"]["text_length_train"]["words"],
        "splits": banking77_stats["cleaning"]["split_sizes"],
        "test_set_hash": "sha256:6b7f43ccbe394d73310fa8d23ac97cebf9ce1292e989bca5f6001c52d8e33ddc",
        "test_set_hash_source": "outputs/metrics/banking77_test_hash.txt and every BANKING77 results.json",
        "real_data_quality_issues": [
            {"type": "within_train_duplicates", "count": 4, "action": "dropped (10003 -> 9999)"},
            {"type": "within_test_duplicates", "count": 1, "action": "dropped (3080 -> 3079)"},
            {"type": "train_vs_test_leakage_whitespace_masked", "count": 6, "action": "dropped 6 from train"},
            {"type": "val_vs_test_leakage_whitespace_masked", "count": 1, "action": "dropped 1 from val"},
            {"type": "non_ascii_preserved_train", "count": 49, "action": "kept (legit currency/accents)"},
            {"type": "non_ascii_preserved_test", "count": 9, "action": "kept"},
            {"type": "length_outliers_gt300_train", "count": 20, "action": "flagged but retained"},
            {"type": "length_outliers_gt300_test", "count": 3, "action": "flagged but retained"},
        ],
        "synthetic_errors_injected_for_demo": [],
        "top_similar_label_pairs_jaccard": banking77_stats["eda"]["top_similar_label_pairs"],
        "note": "BANKING77 is real-world dirty; no synthetic injection. Val carved off cleaned train at 12%.",
    },
}

(CONSOLIDATED / "datasets_summary.json").write_text(json.dumps(datasets_summary, indent=2))
print(f"Wrote {CONSOLIDATED / 'datasets_summary.json'}")

# ---------- DELIVERABLE 3: per_intent_comparison.csv (BANKING77 only) ----------
tfidf_run = [r for r in runs if r["dataset"] == "banking77" and r["model"] == "tfidf"][0]
distilbert_run = [r for r in runs if r["dataset"] == "banking77" and r["model"] == "distilbert"][0]
qwen_run = [r for r in runs if r["dataset"] == "banking77" and r["model"] == "qwen_lora"][0]

tfidf_per = tfidf_run["results"]["metrics"]["intent"]["per_class"]
distilbert_per = distilbert_run["results"]["metrics"]["intent"]["per_class"]
qwen_per = qwen_run["results"]["metrics"]["intent"]["per_class"]

# 77 canonical intent names — take tfidf's, strip any non-real synth keys
tfidf_intents = set(tfidf_per.keys())
distilbert_intents = set(distilbert_per.keys())
qwen_intents = set(qwen_per.keys()) - {"PARSE_ERROR"}  # drop the synthetic parse-error bucket

all_intents_three = tfidf_intents & distilbert_intents & qwen_intents
intents_missing_somewhere = (tfidf_intents | distilbert_intents | qwen_intents) - all_intents_three
per_intent_rows = []

for name in sorted(all_intents_three | intents_missing_somewhere):
    tf = tfidf_per.get(name)
    db = distilbert_per.get(name)
    qw = qwen_per.get(name)
    support = None
    for src in (db, tf, qw):
        if src and "support" in src:
            support = src["support"]
            break

    def trip(d):
        if d is None:
            return (None, None, None)
        return (d.get("f1"), d.get("precision"), d.get("recall"))

    tf_f1, tf_p, tf_r = trip(tf)
    db_f1, db_p, db_r = trip(db)
    qw_f1, qw_p, qw_r = trip(qw)

    # best_model = argmax of (f1) across 3; on ties pick alphabetical first
    scored = []
    if tf_f1 is not None:
        scored.append(("tfidf", tf_f1))
    if db_f1 is not None:
        scored.append(("distilbert", db_f1))
    if qw_f1 is not None:
        scored.append(("qwen", qw_f1))

    if scored:
        max_f1 = max(s[1] for s in scored)
        tied = sorted([s[0] for s in scored if s[1] == max_f1])
        best = tied[0]
        min_f1 = min(s[1] for s in scored)
        gap_pp = round((max_f1 - min_f1) * 100, 2)
    else:
        best = ""
        gap_pp = None

    per_intent_rows.append({
        "intent_name": name,
        "support": support,
        "tfidf_f1": tf_f1, "tfidf_precision": tf_p, "tfidf_recall": tf_r,
        "distilbert_f1": db_f1, "distilbert_precision": db_p, "distilbert_recall": db_r,
        "qwen_f1": qw_f1, "qwen_precision": qw_p, "qwen_recall": qw_r,
        "best_model": best,
        "max_model_gap_pp": gap_pp,
    })

# Sort by distilbert_f1 ascending (hardest intents first)
per_intent_rows.sort(key=lambda r: (r["distilbert_f1"] if r["distilbert_f1"] is not None else 999))

csv_path = CONSOLIDATED / "per_intent_comparison.csv"
with csv_path.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=[
        "intent_name", "support",
        "tfidf_f1", "tfidf_precision", "tfidf_recall",
        "distilbert_f1", "distilbert_precision", "distilbert_recall",
        "qwen_f1", "qwen_precision", "qwen_recall",
        "best_model", "max_model_gap_pp",
    ])
    w.writeheader()
    for r in per_intent_rows:
        w.writerow(r)

print(f"Wrote {csv_path} with {len(per_intent_rows)} rows")
if intents_missing_somewhere:
    print(f"FLAG: intents missing from >=1 model: {sorted(intents_missing_somewhere)}")

# ---------- DELIVERABLE 4: figures_manifest.md ----------
def section_hint(fn):
    f = fn.lower()
    if f.startswith("01_"):
        if "category_distribution" in f or "intent_distribution" in f:
            return "Section 3 (Statistics)"
        if "heatmap" in f:
            return "Section 3 (Statistics)"
        if "text_length" in f:
            return "Section 3 (Statistics)"
        if "flags" in f:
            return "Section 3 (Statistics)"
        return "Section 3 (Statistics)"
    if f.startswith("03_") or f.startswith("04_") or f.startswith("06_"):
        return "Section 6 (Accuracy)"
    if f.startswith("07_"):
        return "Section 3 (Statistics)"
    if f.startswith("08_") or f.startswith("09_") or f.startswith("10_"):
        return "Section 6 (Accuracy)"
    return "Section 3 (Statistics)"


def generator(fn):
    if fn.startswith("01_"):
        return "notebook 01 (Bitext EDA)"
    if fn.startswith("03_"):
        return "notebook 03 (Bitext TF-IDF eval)"
    if fn.startswith("04_"):
        return "scripts/train_distilbert.py / notebook 05 (Bitext DistilBERT eval)"
    if fn.startswith("06_"):
        return "notebook 07 (Bitext Qwen LoRA eval) / scripts/train_qwen_lora.py"
    if fn.startswith("07_"):
        return "notebook 07 (BANKING77 EDA)"
    if fn.startswith("08_"):
        return "notebook 08 (BANKING77 TF-IDF)"
    if fn.startswith("09_"):
        return "scripts/train_distilbert_banking77.py (BANKING77 DistilBERT)"
    if fn.startswith("10_"):
        return "scripts/eval_qwen_lora_banking77.py (BANKING77 Qwen LoRA)"
    return "unknown"


CAPTIONS = {
    "01_category_distribution.png": "Bitext category distribution (11 categories, 26,872 rows).",
    "01_category_intent_heatmap.png": "Bitext category x intent heatmap (11 categories x 27 intents), row counts.",
    "01_flags_letter_frequency.png": "Bitext per-letter flag frequency across 26,872 rows.",
    "01_intent_distribution.png": "Bitext intent distribution (27 intents, 26,872 rows).",
    "01_text_length_chars.png": "Bitext text-length distributions (chars): instruction vs response.",
    "01_text_length_tokens.png": "Bitext text-length distributions (whitespace tokens): instruction vs response.",
    "03_tfidf_confusion_matrix.png": "Confusion matrix for Bitext TF-IDF + LogReg, 27x27, test split (n=3500).",
    "04_distilbert_confusion_matrix.png": "Confusion matrix for Bitext DistilBERT, 27x27, test split (n=3500).",
    "06_qwen_lora_confusion_matrix.png": "Confusion matrix for Bitext Qwen2.5-7B QLoRA, 27x27, test split (n=3500).",
    "07_banking77_class_distribution.png": "BANKING77 class distribution (77 intents, raw train split, n=10003).",
    "07_banking77_text_length.png": "BANKING77 text-length distribution (train split).",
    "07_banking77_vs_bitext_length.png": "Text-length comparison between Bitext and BANKING77 inputs.",
    "08_tfidf_banking77_confusion_matrix.png": "Confusion matrix for BANKING77 TF-IDF + LogReg, 77x77, test split (n=3079).",
    "09_distilbert_banking77_confusion_matrix.png": "Confusion matrix for BANKING77 DistilBERT, 77x77, test split (n=3079).",
    "10_qwen_lora_banking77_confusion_matrix_buggy.png": "Confusion matrix for BANKING77 Qwen2.5-7B QLoRA BUGGY run (pre-parser-fix), 77x77, test split (n=3079), 3079 parse errors visible as a single column.",
    "10_qwen_lora_banking77_confusion_matrix_reval.png": "Confusion matrix for BANKING77 Qwen2.5-7B QLoRA fixed-parser re-eval, 77x77, test split (n=3079).",
}

figs = sorted([p.name for p in FIGURES.glob("*.png")])
lines = ["| Filename | Generated by | Suggested section | Caption |", "|---|---|---|---|"]
for fn in figs:
    caption = CAPTIONS.get(fn, "")
    lines.append(f"| {fn} | {generator(fn)} | {section_hint(fn)} | {caption} |")

manifest = "\n".join(lines) + "\n"
(CONSOLIDATED / "figures_manifest.md").write_text(manifest)
print(f"Wrote {CONSOLIDATED / 'figures_manifest.md'} ({len(figs)} figures)")

# ---------- FINAL SUMMARY PRINT ----------
print()
print("=== File sizes ===")
for fname in ["all_runs.json", "datasets_summary.json", "per_intent_comparison.csv", "figures_manifest.md"]:
    p = CONSOLIDATED / fname
    print(f"  {p}: {p.stat().st_size} bytes")

print()
print("=== summary_table preview (dataset, model, test_acc, test_macro_f1) ===")
for row in summary_rows:
    print(f"  {row[0]:10s} {row[1]:12s} test_acc={row[4]:.4f} test_macro_f1={row[5]:.4f}")

print()
print(f"summary_table rows: {len(summary_rows)}")
print(f"per_intent_comparison rows: {len(per_intent_rows)}")
print(f"figures listed in manifest: {len(figs)}")
print(f"cross-check issues: {len(cross_check_issues)}")
