#!/usr/bin/env python
"""Inspect precomputed metrics written by precompute_source_metrics_fast.py.

Reads any of the source-level parquet files under a source directory:

    {source_dir}/targets/target_metrics.parquet
    {source_dir}/metrics/loop_metrics.parquet
    {source_dir}/metrics/interface_metrics.parquet
    {source_dir}/metrics/dockq_metrics.parquet

and prints, for each requested metric group, a per-target aggregation plus an
overall summary. Works on a single source (e.g. ``PertMD_all``) or on the
merged ``combined`` directory produced by combine_source_parquets.py.

Examples
--------
# All metric groups for the combined PertMD output
python preprocess/metric/inspect_dockq_metrics.py \
    --root /home/sujin/projects/cdr-scoring/cdr-data/metrics_precomputed_pertmd_43_44 \
    --source combined

# Only loop + interface for a single source
python preprocess/metric/inspect_dockq_metrics.py \
    --root .../metrics_precomputed_boltz2_s10n10_combined --source Boltz2_s10n10 \
    --metric loop interface

# Everything for one target, per-decoy rows, save CSVs
python preprocess/metric/inspect_dockq_metrics.py --root ... --source combined \
    --target 1a14_H_L_A --show-decoys --save-dir /tmp/inspect_1a14
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Per-group: parquet relative path, value columns to summarize, and the column
# whose non-NaN value marks a "valid" (scored) decoy.
GROUP_SPECS = {
    "loop": {
        "rel_path": ("metrics", "loop_metrics.parquet"),
        "value_cols": [
            "cdr_rmsd", "cdr_lddt",
            "H1_loop_rmsd", "H2_loop_rmsd", "H3_loop_rmsd",
            "L1_loop_rmsd", "L2_loop_rmsd", "L3_loop_rmsd",
            "H1_loop_lddt", "H2_loop_lddt", "H3_loop_lddt",
            "L1_loop_lddt", "L2_loop_lddt", "L3_loop_lddt",
            "missing_backbone_atom_count",
        ],
        "valid_col": "cdr_rmsd",
    },
    "interface": {
        "rel_path": ("metrics", "interface_metrics.parquet"),
        "value_cols": [
            "interface_bb_lddt", "interface_rmsd", "irmsd", "lrmsd",
            "fnat", "cdr_antigen_contact_recovery", "cdr_antigen_contact_count",
            "native_contact_count", "decoy_contact_count",
        ],
        "valid_col": "irmsd",
    },
    "dockq": {
        "rel_path": ("metrics", "dockq_metrics.parquet"),
        "value_cols": ["dockq", "fnat", "irmsd", "lrmsd"],
        "valid_col": "dockq",
    },
}

DECOY_ID_COLS = ["target_id", "source", "seed", "sample", "decoy_id", "ranking", "ranking_score"]


def _source_dir(args) -> Path:
    if args.source_dir:
        return Path(args.source_dir)
    if args.root and args.source:
        return Path(args.root) / args.source
    raise SystemExit("Provide --source-dir, or both --root and --source.")


def _load_group(source_dir: Path, group: str) -> pd.DataFrame | None:
    sub, filename = GROUP_SPECS[group]["rel_path"]
    path = source_dir / sub / filename
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _per_target_summary(df: pd.DataFrame, value_cols: list[str], valid_col: str) -> pd.DataFrame:
    value_cols = [c for c in value_cols if c in df.columns]
    rows = []
    for target_id, grp in df.groupby("target_id", sort=True):
        valid = grp.dropna(subset=[valid_col]) if valid_col in grp.columns else grp
        row = {"target_id": target_id, "n_decoys": len(grp), "n_valid": len(valid)}
        for col in value_cols:
            row[f"{col}.mean"] = valid[col].mean() if len(valid) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _print_per_target_tall(summary: pd.DataFrame, value_cols: list[str]):
    """Transposed view: counts table + a metric(rows) x target(cols) table."""
    counts = summary[["target_id", "n_decoys", "n_valid"]]
    print("\nDecoy counts:")
    print(counts.to_string(index=False))

    mean_cols = [f"{c}.mean" for c in value_cols if f"{c}.mean" in summary.columns]
    if not mean_cols:
        return
    tall = summary.set_index("target_id")[mean_cols].T
    tall.index = [c[: -len(".mean")] for c in tall.index]  # strip ".mean"
    tall.index.name = "metric"
    print("\nPer-target mean (metric x target):")
    print(tall.round(4).to_string())


def _print_group(group: str, df: pd.DataFrame, args):
    spec = GROUP_SPECS[group]
    value_cols = [c for c in spec["value_cols"] if c in df.columns]
    valid_col = spec["valid_col"]

    print("\n" + "=" * 100)
    print(f"[{group.upper()}]  rows={len(df)}  targets={df['target_id'].nunique()}")
    print("=" * 100)

    summary = _per_target_summary(df, value_cols, valid_col)
    if args.antigen_only and group in ("interface", "dockq"):
        summary = summary[summary["n_valid"] > 0]

    if args.layout == "wide":
        print("\nPer-target mean:")
        print(summary.round(4).to_string(index=False))
    else:
        _print_per_target_tall(summary, value_cols)

    valid_summary = summary[summary["n_valid"] > 0]
    if len(valid_summary):
        print("\nOverall mean across valid decoys (per column):")
        valid_df = df.dropna(subset=[valid_col]) if valid_col in df.columns else df
        overall = valid_df[value_cols].mean().round(4)
        print(overall.to_string())

    if args.show_decoys:
        view = df.copy()
        sort_col = args.sort_by if args.sort_by in df.columns else valid_col
        ascending = sort_col not in {"dockq", "cdr_lddt", "interface_bb_lddt",
                                     "fnat", "cdr_antigen_contact_recovery"}
        view = view.sort_values(["target_id", sort_col], ascending=[True, ascending])
        show_cols = [c for c in DECOY_ID_COLS if c in view.columns] + value_cols
        print("\nPer-decoy:")
        print(view[show_cols].round(4).to_string(index=False))

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(save_dir / f"{group}_per_target.csv", index=False)
        df.to_csv(save_dir / f"{group}_per_decoy.csv", index=False)
        print(f"\n[saved] {save_dir}/{group}_per_target.csv and {group}_per_decoy.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", help="Output root dir (contains the source / combined sub-directories).")
    parser.add_argument("--source", help="Source sub-directory (e.g. PertMD_all, Boltz2_s10n10, or combined).")
    parser.add_argument("--source-dir", help="Direct path to a source directory (overrides --root/--source).")
    parser.add_argument("--metric", nargs="+", default=["all"],
                        choices=["loop", "interface", "dockq", "all"],
                        help="Metric groups to show. Default: all.")
    parser.add_argument("--target", help="Only show this target_id.")
    parser.add_argument("--antigen-only", action="store_true", help="Drop targets with no valid interface/dockq (no antigen).")
    parser.add_argument("--layout", choices=["tall", "wide"], default="tall",
                        help="Per-target summary layout. 'tall' (default) transposes metrics to rows; 'wide' is one row per target.")
    parser.add_argument("--show-decoys", action="store_true", help="Print per-decoy rows for each shown group.")
    parser.add_argument("--sort-by", default="", help="Column to sort --show-decoys by (default: the group's primary metric).")
    parser.add_argument("--save-dir", help="Directory to write per-target and per-decoy CSVs for each group.")
    parser.add_argument("--max-rows", type=int, default=80, help="Max rows to print per table.")
    args = parser.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", args.max_rows)

    source_dir = _source_dir(args)
    if not source_dir.exists():
        raise SystemExit(f"source dir not found: {source_dir}")

    groups = ["loop", "interface", "dockq"] if "all" in args.metric else list(dict.fromkeys(args.metric))

    # Target overview (if present).
    target_df = None
    target_path = source_dir / "targets" / "target_metrics.parquet"
    if target_path.exists():
        target_df = pd.read_parquet(target_path)
        if args.target:
            target_df = target_df[target_df["target_id"] == args.target]

    print("=" * 100)
    print(f"Source dir : {source_dir}")
    print(f"Metrics    : {groups}")
    if target_df is not None and not target_df.empty:
        cols = [c for c in ["target_id", "source", "has_holo_antigen", "antibody_chains",
                            "antigen_chains", "num_total_residues_original", "H3_count"]
                if c in target_df.columns]
        print("\n[Target overview]")
        print(target_df[cols].to_string(index=False))

    any_loaded = False
    for group in groups:
        df = _load_group(source_dir, group)
        if df is None:
            print(f"\n[{group.upper()}] parquet not found under {source_dir} -- skipped")
            continue
        if args.target:
            df = df[df["target_id"] == args.target]
            if df.empty:
                print(f"\n[{group.upper()}] no rows for target_id={args.target}")
                continue
        any_loaded = True
        _print_group(group, df, args)

    if not any_loaded:
        raise SystemExit("No metric parquet files were found/loaded.")


if __name__ == "__main__":
    main()
