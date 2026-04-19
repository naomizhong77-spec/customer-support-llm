"""Convert BANKING77 processed parquet splits into JSONL instruction-format files.

Reads `{train,val,test}.parquet` from an input directory (default
`data/banking77/processed/`) and writes `{train,val,test}.jsonl` to an output
directory (default `data/banking77/instruction/`). Each JSONL row is an
Alpaca-style record for a single single-label intent classification example:

    {
      "instruction": "Classify the following customer banking query into its intent.",
      "input": "<customer banking query>",
      "output": "Intent: <intent_name>"
    }

BANKING77 has **no category hierarchy** (flat 77-way classification), so the
output line is a single `Intent: ...` row. The `"Intent:"` anchor is kept so
the Qwen QLoRA `DataCollator` + `response_template="Intent:"` pipeline written
for the Bitext dataset can be reused with minimal changes.

The input parquet is expected to have columns ``text`` (the user message)
and a label column. The label column can be either:

- ``label_name`` (string, e.g. ``"card_arrival"``) — preferred, written by
  notebook 08. Used verbatim.
- ``label`` (string) — also used verbatim.
- ``label`` (int 0-76) — HuggingFace ClassLabel integer. Automatically
  mapped to the intent name using the ``PolyAI/banking77`` ClassLabel names
  pulled from the HuggingFace hub on first use.

If both ``label_name`` and ``label`` are present, ``label_name`` wins.

Usage (standalone)::

    python scripts/prepare_banking77_instruction_data.py \\
        --input-dir data/banking77/processed \\
        --output-dir data/banking77/instruction

Add ``--limit N`` to emit only the first N rows per split (smoke-test mode).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger("prepare_banking77_instruction_data")

# System-level instruction shared across all examples. Kept short and specific
# to the banking domain so the prompt-to-response token ratio is favourable.
INSTRUCTION_TEXT: str = (
    "Classify the following customer banking query into its intent."
)

SPLITS: tuple[str, ...] = ("train", "val", "test")
# One of LABEL_COL_PREFERENCES must be present; the first one found wins.
LABEL_COL_PREFERENCES: tuple[str, ...] = ("label_name", "label")


def _load_banking77_class_names() -> list[str]:
    """Lazily fetch the 77 BANKING77 intent names from HuggingFace.

    Only called when the input parquet's label column is integer-typed. We
    pull the tiny dataset info (no data download beyond the ClassLabel
    metadata) so this stays lightweight.
    """
    from datasets import load_dataset  # local import to keep base import light
    ds = load_dataset("PolyAI/banking77", split="train")
    names = list(ds.features["label"].names)
    if len(names) != 77:
        raise RuntimeError(
            f"Expected 77 BANKING77 class names, got {len(names)} "
            "— is the HuggingFace dataset schema still current?"
        )
    return names


def _resolve_label_column(df: pd.DataFrame) -> pd.Series:
    """Return a string pandas Series of intent names, deriving it from the
    best-available label column in `df`.

    Accepts ``label_name`` (string) or ``label`` (string or int). Raises
    ``ValueError`` if neither is present.
    """
    for col in LABEL_COL_PREFERENCES:
        if col in df.columns:
            series = df[col]
            if pd.api.types.is_integer_dtype(series):
                LOGGER.info(
                    "Label column '%s' is integer-typed — mapping via "
                    "BANKING77 ClassLabel names from HuggingFace.",
                    col,
                )
                names = _load_banking77_class_names()
                return series.map(lambda i: names[int(i)])
            return series.astype(str)
    raise ValueError(
        f"No label column found. Tried: {LABEL_COL_PREFERENCES}. "
        f"Columns present: {list(df.columns)}"
    )


def row_to_example(
    text: str,
    label: str,
    task_prompt: str = INSTRUCTION_TEXT,
) -> dict:
    """Turn a single processed-parquet row into an instruction-format dict.

    The label is used verbatim (no casing change) since BANKING77 intent names
    have mixed casing in the upstream dataset (e.g. ``Refund_not_showing_up``
    and ``reverted_card_payment?``) and preserving that shape simplifies
    post-processing in the evaluator.
    """
    return {
        "instruction": task_prompt,
        "input": text,
        "output": f"Intent: {label}",
    }


def iter_examples(df: pd.DataFrame) -> Iterable[dict]:
    """Yield instruction-format dicts from a processed-parquet dataframe.

    Uses ``_resolve_label_column`` so the caller can pass a parquet with
    either a string label column (``label_name`` or ``label``) or an
    integer-typed ``label`` column.
    """
    if df.empty:
        return
    texts = df["text"].astype(str)
    labels = _resolve_label_column(df)
    for text, label in zip(texts, labels):
        yield row_to_example(text=text, label=str(label))


def convert_split(
    input_path: Path,
    output_path: Path,
    limit: int | None = None,
) -> int:
    """Read one processed parquet file and write its JSONL counterpart.

    Returns the number of rows written.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    df = pd.read_parquet(input_path)
    if "text" not in df.columns:
        raise ValueError(
            f"{input_path} is missing required column 'text'. "
            f"Found: {list(df.columns)}"
        )
    if not any(c in df.columns for c in LABEL_COL_PREFERENCES):
        raise ValueError(
            f"{input_path} is missing a label column. "
            f"Expected one of {LABEL_COL_PREFERENCES}. "
            f"Found: {list(df.columns)}"
        )

    if limit is not None:
        df = df.head(limit)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for example in iter_examples(df):
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")
            n_written += 1

    LOGGER.info(
        "Wrote %d rows -> %s (from %s, limit=%s)",
        n_written,
        output_path,
        input_path,
        limit,
    )
    return n_written


def convert_all(
    input_dir: Path,
    output_dir: Path,
    splits: Iterable[str] = SPLITS,
    limit: int | None = None,
) -> dict[str, int]:
    """Convert all requested splits. Returns a split_name -> row_count map."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in splits:
        in_path = input_dir / f"{split}.parquet"
        out_path = output_dir / f"{split}.jsonl"
        counts[split] = convert_split(in_path, out_path, limit=limit)
    return counts


def _build_argparser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert BANKING77 processed parquet splits to JSONL instruction "
            "format (Alpaca-style, flat intent classification)."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/banking77/processed"),
        help=(
            "Directory containing train/val/test.parquet "
            "(default: data/banking77/processed)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/banking77/instruction"),
        help=(
            "Directory to write train/val/test.jsonl "
            "(default: data/banking77/instruction)."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        help="Which splits to convert (default: train val test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row cap per split (for smoke tests).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level (default: INFO).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a POSIX-style exit code."""
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    LOGGER.info(
        "input_dir=%s output_dir=%s splits=%s limit=%s",
        args.input_dir, args.output_dir, args.splits, args.limit,
    )
    counts = convert_all(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        splits=args.splits,
        limit=args.limit,
    )
    LOGGER.info("Done. Row counts per split: %s", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
