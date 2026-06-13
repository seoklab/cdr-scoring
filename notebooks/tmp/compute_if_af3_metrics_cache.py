#!/usr/bin/env python3
"""Compute and cache IF AF3 benchmark structural metrics.

This script uses the same cache format as analyze_if_test_af3.ipynb, so it can
be run in the background and resumed safely. Existing cache rows are skipped.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path("/home/sujin/projects/cdr-scoring/cdr-code")
DEFAULT_DATA_PATH = Path("/home/sujin/projects/cdr-scoring/cdr-data/inference/IF-af3-benchmark.csv")
DEFAULT_CACHE_PATH = REPO_ROOT / "notebooks" / "tmp" / "IF-af3-benchmark.metrics_cache.csv"
METRIC_COLS = ["loop_rmsd", "loop_lddt", "irmsd", "lrmsd"]
CACHE_FIELDS = ["metric_key", "target_id", "sample_id", "native_path", "path", *METRIC_COLS, "metric_error"]


def as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def metric_key(row: dict) -> str:
    return "||".join(str(row.get(k, "")) for k in ["target_id", "sample_id", "native_path", "path"])


def group_rows(rows: Iterable[dict], key: str = "target_id") -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return grouped


def load_rows(data_path: Path) -> List[dict]:
    with data_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    numeric_cols = [
        "rank_by_model",
        "pred_score",
        "seed",
        "sample",
        "af3_rank",
        "af3_ranking_score",
        "loop_rmsd",
        "loop_lddt",
        "irmsd",
        "lrmsd",
        "n_nodes",
        "n_edges",
    ]
    for row in rows:
        for col in numeric_cols:
            if col in row:
                row[col] = as_float(row[col])
        row["metric_key"] = metric_key(row)
    return rows


def top_n_per_target(rows: List[dict], score_col: str, n: int, *, reverse: bool) -> List[dict]:
    selected: List[dict] = []
    for target_rows in group_rows(rows).values():
        selected.extend(sorted(target_rows, key=lambda r: as_float(r[score_col]), reverse=reverse)[:n])
    return selected


def select_rows(rows: List[dict], *, mode: str, top_n: int) -> List[dict]:
    if mode == "all":
        selected = rows
    elif mode == "top-n":
        selected = [
            *top_n_per_target(rows, "pred_score", top_n, reverse=False),
            *top_n_per_target(rows, "af3_ranking_score", top_n, reverse=True),
        ]
    else:
        raise ValueError(f"unknown mode: {mode}")

    dedup = {}
    for row in selected:
        dedup[row["metric_key"]] = row
    return list(dedup.values())


def read_cache(cache_path: Path) -> Dict[str, dict]:
    if not cache_path.exists():
        return {}

    with cache_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    cache = {}
    for row in rows:
        if not row.get("metric_key"):
            row["metric_key"] = metric_key(row)
        for col in METRIC_COLS:
            row[col] = as_float(row.get(col))
        cache[row["metric_key"]] = row
    return cache


def write_cache(cache_path: Path, cache: Dict[str, dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CACHE_FIELDS)
        writer.writeheader()
        for row in cache.values():
            writer.writerow({field: row.get(field, "") for field in CACHE_FIELDS})
    tmp_path.replace(cache_path)


def compute_metrics_for_target(payload: Tuple[str, List[dict], str]) -> List[dict]:
    target_id, target_rows, metrics = payload

    sys.path.insert(0, str(REPO_ROOT / "libs"))
    from evaluation.loop_metrics import compute_loop_metrics_from_structures, load_structure

    compute_dockq_style_metrics_from_structures = None
    if metrics == "all":
        from evaluation.docking_metrics import compute_dockq_style_metrics_from_structures

    native_cache = {}
    out_rows = []
    for rec in target_rows:
        out = {
            "metric_key": rec["metric_key"],
            "target_id": rec["target_id"],
            "sample_id": rec["sample_id"],
            "native_path": rec["native_path"],
            "path": rec["path"],
            "loop_rmsd": math.nan,
            "loop_lddt": math.nan,
            "irmsd": math.nan,
            "lrmsd": math.nan,
            "metric_error": "",
        }
        try:
            native_path = rec["native_path"]
            if native_path not in native_cache:
                native_cache[native_path] = load_structure(native_path)
            native_structure = native_cache[native_path]
            model_structure = load_structure(rec["path"])

            loop = compute_loop_metrics_from_structures(
                native_structure,
                model_structure,
                native_path=native_path,
                model_path=rec["path"],
                rmsd_atom_type="backbone",
            )
            out["loop_rmsd"] = loop.loop_rmsd
            out["loop_lddt"] = loop.loop_lddt

            if compute_dockq_style_metrics_from_structures is not None:
                dock = compute_dockq_style_metrics_from_structures(
                    native_structure,
                    model_structure,
                    native_path=native_path,
                    model_path=rec["path"],
                )
                out["irmsd"] = dock.irmsd
                out["lrmsd"] = dock.lrmsd
        except Exception as exc:  # Keep the run resumable across bad structures.
            out["metric_error"] = f"{type(exc).__name__}: {exc}"[:500]
        out_rows.append(out)
    return out_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--mode", choices=["all", "top-n"], default="all")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--metrics", choices=["all", "loop"], default="all")
    parser.add_argument("--max-workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--flush-every-targets", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()

    rows = load_rows(args.data)
    selected = select_rows(rows, mode=args.mode, top_n=args.top_n)
    cache = read_cache(args.cache)
    missing = [row for row in selected if row["metric_key"] not in cache]
    jobs = [(target_id, target_rows, args.metrics) for target_id, target_rows in group_rows(missing).items()]

    print(f"data rows: {len(rows):,}", flush=True)
    print(f"selected rows: {len(selected):,} mode={args.mode}", flush=True)
    print(f"cached rows: {len(cache):,}", flush=True)
    print(f"missing rows: {len(missing):,} across {len(jobs):,} targets", flush=True)
    print(f"metrics: {args.metrics}; max_workers: {args.max_workers}", flush=True)

    if args.dry_run or not missing:
        return 0

    completed_targets = 0
    completed_rows = 0

    if args.max_workers <= 1:
        for job in jobs:
            result_rows = compute_metrics_for_target(job)
            for row in result_rows:
                cache[row["metric_key"]] = row
            completed_targets += 1
            completed_rows += len(result_rows)
            if completed_targets % args.flush_every_targets == 0 or completed_targets == len(jobs):
                write_cache(args.cache, cache)
                print(
                    f"completed {completed_targets:,}/{len(jobs):,} targets; "
                    f"{completed_rows:,}/{len(missing):,} new rows; elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(compute_metrics_for_target, job) for job in jobs]
            for future in as_completed(futures):
                result_rows = future.result()
                for row in result_rows:
                    cache[row["metric_key"]] = row
                completed_targets += 1
                completed_rows += len(result_rows)
                if completed_targets % args.flush_every_targets == 0 or completed_targets == len(jobs):
                    write_cache(args.cache, cache)
                    print(
                        f"completed {completed_targets:,}/{len(jobs):,} targets; "
                        f"{completed_rows:,}/{len(missing):,} new rows; elapsed={time.time() - start:.1f}s",
                        flush=True,
                    )

    errors = [row for row in cache.values() if row.get("metric_error")]
    print(f"cache written: {args.cache} ({len(cache):,} rows)", flush=True)
    print(f"metric errors in cache: {len(errors):,}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
