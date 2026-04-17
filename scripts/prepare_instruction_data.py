"""Convert processed parquet splits into JSONL instruction-format files for Qwen SFT.

Reads `{train,val,test}.parquet` from an input directory (default
`data/processed/`) and writes `{train,val,test}.jsonl` to an output directory
(default `data/instruction/`). Each JSONL row encodes a single classification
example:

    {
      "instruction": "Classify this customer support message into one of 27 intents.",
      "input": "<customer message>",
      "output": "Category: <CATEGORY>\\nIntent: <intent>"
    }

This format keeps the prompt decoupled from any specific chat template — the
training script (notebook 06 / `scripts/train_qwen_lora.py`) is responsible
for wrapping each row in the Qwen2.5 chat template.

Usage (standalone)::

    python scripts/prepare_instruction_data.py \\
        --input-dir data/processed \\
        --output-dir data/instruction

Add ``--limit N`` to emit only the first N rows per split (smoke-test mode).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger("prepare_instruction_data")

# System-level instruction shared across all examples. Kept short so the
# prompt-to-response token ratio is favourable during SFT.
INSTRUCTION_TEXT: str = (
    "Classify this customer support message into one of 27 intents."
)

SPLITS: tuple[str, ...] = ("train", "val", "test")
REQUIRED_COLS: tuple[str, ...] = ("instruction", "category", "intent")


def row_to_example(
    instruction_text: str,
    category: str,
    intent: str,
    task_prompt: str = INSTRUCTION_TEXT,
) -> dict:
    """Turn a single processed-parquet row into an instruction-format dict.

    The category is upper-cased and the intent is lower-cased so the output
    shape matches the training labels seen during fine-tuning.
    """
    return {
        "instruction": task_prompt,
        "input": instruction_text,
        "output": f"Category: {category.upper()}\nIntent: {intent.lower()}",
    }


def iter_examples(df: pd.DataFrame) -> Iterable[dict]:
    """Yield instruction-format dicts from a processed-parquet dataframe."""
    if df.empty:
        return
    for _, row in df.iterrows():
        yield row_to_example(
            instruction_text=str(row["instruction"]),
            category=str(row["category"]),
            intent=str(row["intent"]),
        )


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
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{input_path} is missing required columns: {missing}. "
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
            "Convert processed parquet splits to JSONL instruction format "
            "for Qwen SFT."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing train/val/test.parquet (default: data/processed).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/instruction"),
        help="Directory to write train/val/test.jsonl (default: data/instruction).",
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
    LOGGER.info("input_dir=%s output_dir=%s splits=%s limit=%s",
                args.input_dir, args.output_dir, args.splits, args.limit)
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
