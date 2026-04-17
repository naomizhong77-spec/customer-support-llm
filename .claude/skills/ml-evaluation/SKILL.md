---
name: ml-evaluation
description: Use this skill when computing evaluation metrics, comparing models, analyzing errors, or producing visualizations for classification models. Defines the unified results JSON schema that all three models in this project must conform to, plus standard evaluation procedures, visualization conventions, and PM-framed comparison templates.
---

# ML Evaluation Methodology

This skill enforces a consistent evaluation protocol across all three models so comparisons are apples-to-apples.

## The Unified Results Schema

**Every evaluation run MUST produce a JSON file at `outputs/metrics/<run_id>/results.json` with this exact structure:**

```json
{
  "run_id": "string, format: <model>_<YYYYMMDD>_<shorthash>",
  "model_name": "string, human-readable",
  "model_family": "one of: traditional-ml, bert-classifier, llm-generative",
  "checkpoint_path": "string or null",
  "test_set": "data/processed/test.parquet",
  "test_set_hash": "sha256:...",
  "n_samples": 4050,
  "timestamp": "ISO 8601",
  "git_commit": "short hash",
  "framework_versions": {
    "python": "3.10.x",
    "torch": "2.4.1",
    "transformers": "4.46.0"
  },
  "metrics": {
    "intent": {
      "accuracy": 0.921,
      "macro_f1": 0.873,
      "macro_precision": 0.881,
      "macro_recall": 0.869,
      "weighted_f1": 0.920,
      "per_class": {
        "<INTENT_NAME>": {
          "precision": 0.95,
          "recall": 0.91,
          "f1": 0.93,
          "support": 150
        }
      }
    },
    "category": {
      "accuracy": 0.948,
      "macro_f1": 0.921,
      "macro_precision": 0.93,
      "macro_recall": 0.915,
      "weighted_f1": 0.948,
      "per_class": {}
    }
  },
  "latency": {
    "p50_ms": 245.3,
    "p95_ms": 412.8,
    "p99_ms": 580.1,
    "mean_ms": 267.1,
    "std_ms": 58.2,
    "throughput_per_sec": 4.1,
    "batch_size": 1,
    "device": "NVIDIA A40",
    "precision": "bf16+4bit",
    "n_samples_timed": 200
  },
  "resources": {
    "gpu_memory_mb_peak_train": 18432,
    "gpu_memory_mb_peak_inference": 6850,
    "disk_size_mb": 142,
    "parameters_trainable": 41943040,
    "parameters_total": 7616000000
  },
  "parse_errors": 0,
  "notes": "free-form string, include anything relevant"
}
```

## Metric Computation Code

Use `sklearn.metrics.classification_report(..., output_dict=True)` as the basis, then enhance:

```python
import json
import hashlib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

def compute_classification_metrics(y_true, y_pred, labels=None):
    """
    Compute complete metrics for a classification run.
    Returns dict matching the schema above (under 'metrics.intent' or 'metrics.category').
    """
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
    
    per_class = {}
    for label_name, scores in report.items():
        if label_name in {"accuracy", "macro avg", "weighted avg"}:
            continue
        per_class[label_name] = {
            "precision": round(scores["precision"], 4),
            "recall": round(scores["recall"], 4),
            "f1": round(scores["f1-score"], 4),
            "support": int(scores["support"]),
        }
    
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
    }
```

## Test Set Hash Verification

```python
import hashlib
import pandas as pd

def test_set_hash(parquet_path: str) -> str:
    """Compute hash of test set to verify all models eval on same data."""
    df = pd.read_parquet(parquet_path)
    # Use sorted column order for stability
    df = df.reindex(sorted(df.columns), axis=1)
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    return f"sha256:{hashlib.sha256(csv_bytes).hexdigest()[:16]}"

# In every eval run:
expected_hash = test_set_hash("data/processed/test.parquet")
print(f"Evaluating on test set: {expected_hash}")
# Save this in results.json
```

## Latency Measurement

Use CUDA events for accurate GPU timing:

```python
import torch
import numpy as np

def measure_latency(model_callable, inputs_list, warmup=10, n_measure=200):
    """
    Measure per-sample latency.
    model_callable: function taking one input, returning prediction
    inputs_list: list of test inputs
    """
    # Warmup
    for x in inputs_list[:warmup]:
        _ = model_callable(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure
    times_ms = []
    for x in inputs_list[warmup:warmup + n_measure]:
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _ = model_callable(x)
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)
        else:
            import time
            t0 = time.perf_counter()
            _ = model_callable(x)
            elapsed = (time.perf_counter() - t0) * 1000
        times_ms.append(elapsed)
    
    times_ms = np.array(times_ms)
    return {
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "p99_ms": float(np.percentile(times_ms, 99)),
        "mean_ms": float(times_ms.mean()),
        "std_ms": float(times_ms.std()),
        "throughput_per_sec": float(1000 / times_ms.mean()),
        "n_samples_timed": int(n_measure),
    }
```

## Generative Output Parser (for Qwen)

The Qwen model generates text like "Category: ORDER\nIntent: CANCEL_ORDER". Parser must be robust:

```python
import re
from difflib import get_close_matches

def parse_qwen_output(text: str, valid_categories: list, valid_intents: list):
    """
    Parse Qwen's generated output into (category, intent, parse_error).
    Returns None for category/intent if parsing fails.
    """
    # Try strict regex first
    cat_match = re.search(r"Category:\s*([A-Z_]+)", text)
    intent_match = re.search(r"Intent:\s*([A-Z_]+)", text)
    
    category = cat_match.group(1) if cat_match else None
    intent = intent_match.group(1) if intent_match else None
    
    parse_error = False
    
    # Fuzzy match category
    if category and category not in valid_categories:
        matches = get_close_matches(category, valid_categories, n=1, cutoff=0.7)
        if matches:
            category = matches[0]
        else:
            parse_error = True
    elif category is None:
        parse_error = True
    
    # Fuzzy match intent
    if intent and intent not in valid_intents:
        matches = get_close_matches(intent, valid_intents, n=1, cutoff=0.7)
        if matches:
            intent = matches[0]
        else:
            parse_error = True
    elif intent is None:
        parse_error = True
    
    return category, intent, parse_error
```

## Visualization Standards

### Color Palette (use consistently across all figures)

```python
MODEL_COLORS = {
    "TF-IDF + LR": "#4c72b0",           # blue
    "DistilBERT": "#dd8452",             # orange
    "Qwen2.5-7B-QLoRA": "#55a868",       # green
}
```

### Confusion Matrix

```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels, title, save_path, normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt=".2f" if normalize else "d",
        xticklabels=labels, yticklabels=labels,
        cmap="Blues", cbar_kws={"shrink": 0.8}, ax=ax,
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

### Model Comparison Bar Chart

```python
import pandas as pd

def plot_model_comparison(results_dicts, metrics_to_plot, save_path):
    """
    results_dicts: list of results.json dicts, one per model
    metrics_to_plot: list of metric keys, e.g. ["accuracy", "macro_f1", "weighted_f1"]
    """
    data = []
    for r in results_dicts:
        for m in metrics_to_plot:
            data.append({
                "Model": r["model_name"],
                "Metric": m,
                "Value": r["metrics"]["intent"][m],
            })
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="Metric", y="Value", hue="Model", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Intent Classification Performance Comparison", fontsize=14)
    ax.legend(title="Model", loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

## Error Analysis Pipeline

For each model, produce `outputs/error_analysis/<model>_errors.csv`:

```python
def export_errors(texts, y_true, y_pred, confidences, save_path):
    """Export misclassified samples for analysis."""
    errors = []
    for text, yt, yp, conf in zip(texts, y_true, y_pred, confidences):
        if yt != yp:
            errors.append({
                "text": text,
                "true_label": yt,
                "predicted_label": yp,
                "confidence": conf,
                "text_length": len(text),
                "error_category": classify_error_type(text, yt, yp),
            })
    df = pd.DataFrame(errors)
    df.to_csv(save_path, index=False)
    
    # Summary statistics
    print(f"Total errors: {len(errors)}")
    print(df["error_category"].value_counts())
    return df

def classify_error_type(text, true_label, pred_label):
    """Heuristic classification of why a sample failed."""
    if pred_label == "UNKNOWN" or pred_label is None:
        return "parse_error"
    if len(text) > 500:
        return "long_text"
    if len(text) < 20:
        return "short_text"
    # Sibling intents within same category — "confusable_classes"
    # This needs a mapping, simplified here
    return "confusable_classes"
```

## PM Comparison Narrative Template

When writing the final comparison section, answer these:

### 1. Accuracy Winner
- Which model achieved highest macro-F1?
- By how much? Is the margin meaningful or within noise?

### 2. Accuracy per Millisecond
- TF-IDF: ~1 F1 per 1ms ratio (very fast but lower accuracy)
- DistilBERT: ~46 F1 per 20ms
- Qwen: ~91 F1 per 250ms
- Which wins for real-time chatbot? (latency ≤50ms) → likely DistilBERT
- Which wins for batch ticket routing? (latency irrelevant) → likely Qwen

### 3. Cost Profile
- Training cost: GPU-hours × spot price (rough estimate)
- Inference cost per million requests
- Storage cost (model size × deployment replicas)

### 4. Deployment Complexity
- TF-IDF: pickle file, pip install sklearn — trivial
- DistilBERT: ~250MB model + transformers dep — easy
- Qwen QLoRA: base model + adapter + 4-bit inference stack — complex, needs GPU

### 5. Where All Models Fail
- Intersection of error sets across three models
- These are the genuinely hard cases (annotation issues, ambiguous text)
- Recommendation: review taxonomy or escalate these to human review

### 6. PM Decision Framework
Provide a decision tree:
```
If accuracy is top priority AND latency budget > 200ms → Qwen
If latency budget 10-200ms AND accuracy >= 90% required → DistilBERT
If latency < 10ms required OR no GPU budget → TF-IDF
If task changes frequently (need fast retraining) → TF-IDF or small BERT
If bringing in new label taxonomy → LLM (zero-shot or few-shot fine-tune)
```

## Validation Before Saving Results

```python
def validate_results(results: dict) -> None:
    """Raise AssertionError if results don't meet schema or sanity checks."""
    # Schema checks
    required_top = ["run_id", "model_name", "test_set_hash", "metrics", "latency", "resources"]
    for k in required_top:
        assert k in results, f"Missing required field: {k}"
    
    # Sanity checks
    acc = results["metrics"]["intent"]["accuracy"]
    assert 0 <= acc <= 1, f"Accuracy out of range: {acc}"
    assert acc != 1.0, f"Suspicious perfect accuracy — likely data leak"
    assert acc > 0.05, f"Suspicious near-zero accuracy — likely bug"
    
    f1 = results["metrics"]["intent"]["macro_f1"]
    assert f1 <= acc + 0.05, f"macro_F1 ({f1}) > accuracy ({acc}) shouldn't happen often"
    
    # Latency sanity
    p50 = results["latency"]["p50_ms"]
    p95 = results["latency"]["p95_ms"]
    assert p95 >= p50, "p95 must be >= p50"
```

Call `validate_results(results)` before writing the JSON file.
