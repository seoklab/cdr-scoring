#!/usr/bin/env python
"""Merge precomputed metric parquet shards for one source.

This is intended for recovery runs where an initial parquet contains a subset
of targets and a second output directory contains only missing targets.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


TARGET_KEY = ["target_id", "source"]
DECOY_KEY = ["target_id", "source", "seed", "sample"]


def _read_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _dedupe(frame: pd.DataFrame, key: Iterable[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    usable_key = [col for col in key if col in frame.columns]
    if not usable_key:
        return frame.drop_duplicates()
    return frame.drop_duplicates(subset=usable_key, keep="last").reset_index(drop=True)


def _merge_table(paths: list[Path], output_path: Path, key: Iterable[str]) -> tuple[int, int]:
    frames = [_read_if_exists(path) for path in paths]
    frames = [frame for frame in frames if not frame.empty]
    if frames:
        merged = pd.concat(frames, ignore_index=True, sort=False)
        merged = _dedupe(merged, key)
    else:
        merged = pd.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    return sum(len(frame) for frame in frames), len(merged)


def merge_source(inputs: list[Path], output_dir: Path, source: str) -> None:
    tables = [
        ("targets/target_metrics.parquet", TARGET_KEY),
        ("metrics/loop_metrics.parquet", DECOY_KEY),
        ("metrics/interface_metrics.parquet", DECOY_KEY),
        ("metrics/dockq_metrics.parquet", DECOY_KEY),
    ]
    for rel_path, key in tables:
        source_paths = [root / source / rel_path for root in inputs]
        output_path = output_dir / source / rel_path
        raw_rows, merged_rows = _merge_table(source_paths, output_path, key)
        print(f"{rel_path}: raw_rows={raw_rows} merged_rows={merged_rows} out={output_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Source directory name, e.g. Boltz2_s10n10_valid198.")
    parser.add_argument(
        "--input-root",
        action="append",
        required=True,
        help="Input metrics root. Pass multiple times; later roots win on duplicate keys.",
    )
    parser.add_argument("--output-root", required=True, help="Output merged metrics root.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    merge_source(
        inputs=[Path(path) for path in args.input_root],
        output_dir=Path(args.output_root),
        source=args.source,
    )


if __name__ == "__main__":
    main()
