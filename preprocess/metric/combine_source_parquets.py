#!/usr/bin/env python
"""Concatenate per-source metric parquets into a single combined parquet set.

precompute_source_metrics_fast.py writes one sub-directory per source:

    {output_dir}/{source}/targets/target_metrics.parquet
    {output_dir}/{source}/metrics/loop_metrics.parquet
    {output_dir}/{source}/metrics/interface_metrics.parquet
    {output_dir}/{source}/metrics/dockq_metrics.parquet

This tool stacks the matching files across the requested sources (rows already
carry a ``source`` column) and writes them under a single combined directory:

    {combined_dir}/targets/target_metrics.parquet
    {combined_dir}/metrics/{loop,interface,dockq}_metrics.parquet

Example
-------
python preprocess/metric/combine_source_parquets.py \
    --output-dir /.../metrics_precomputed_pertmd_43_44 \
    --sources PertMD_cdr PertMD_all \
    --combined-name combined
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RELATIVE_FILES = [
    ("targets", "target_metrics.parquet"),
    ("metrics", "loop_metrics.parquet"),
    ("metrics", "interface_metrics.parquet"),
    ("metrics", "dockq_metrics.parquet"),
]


def _discover_sources(output_dir: Path) -> list[str]:
    sources = []
    for child in sorted(output_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "targets").exists() or (child / "metrics").exists():
            sources.append(child.name)
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="Directory that holds the per-source sub-directories.")
    parser.add_argument("--sources", nargs="+", default=None, help="Source sub-directories to combine. Default: auto-detect.")
    parser.add_argument("--combined-name", default="combined", help="Sub-directory name for the combined output. Default: combined.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise SystemExit(f"output dir not found: {output_dir}")

    sources = args.sources or _discover_sources(output_dir)
    sources = [s for s in sources if s != args.combined_name]
    if not sources:
        raise SystemExit(f"No source sub-directories found under {output_dir}")
    combined_dir = output_dir / args.combined_name

    print(f"[combine] output_dir = {output_dir}")
    print(f"[combine] sources    = {sources}")
    print(f"[combine] combined   = {combined_dir}")

    for sub, filename in RELATIVE_FILES:
        frames = []
        for source in sources:
            path = output_dir / source / sub / filename
            if path.exists():
                frames.append(pd.read_parquet(path))
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        out_path = combined_dir / sub / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out_path, index=False)
        print(f"[combine] {sub}/{filename}: {sum(len(f) for f in frames)} rows from {len(frames)} source(s) -> {out_path}")

    print("[combine] done.")


if __name__ == "__main__":
    main()
