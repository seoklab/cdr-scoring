#!/usr/bin/env python
"""Run Galaxy PPDock_eval for best-ranking Boltz2 decoys from incremental CSVs.

This script treats the live best-ranking metric CSVs as read-only. It writes a
separate PPDock result CSV keyed by target/source/seed/sample, and can also
write a merged copy of dockq_metrics.csv with PPDock columns after the producing
job has finished.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_CSV_DIR = Path(
    "/home/sujin/projects/cdr-scoring/cdr-data/"
    "metrics_precomputed_boltz2_s10n10_chothia_valid198_best_ranking/incremental_csv"
)
DEFAULT_PPDOCK = Path("/opt/conda/envs/gp/bin/PPDock_eval.py")
KEY_COLUMNS = ["target_id", "source", "seed", "sample"]


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _chain_string(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return "".join(part.strip() for part in str(value).split(",") if part.strip())


def _calc_dockq(lrmsd: float, irmsd: float, fnat: float) -> float:
    return (
        1 / (1 + (lrmsd / 8.5) ** 2)
        + 1 / (1 + (irmsd / 1.5) ** 2)
        + fnat
    ) / 3


def _parse_eval_dat(path: Path) -> tuple[float, float, float] | None:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            return float(parts[0]), float(parts[1]), float(parts[2])
    return None


def _write_single_model_ensemble(input_pdb: Path, output_pdb: Path, chain_order: str = ""):
    with open(input_pdb, encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    atom_by_chain: dict[str, list[str]] = {}
    other_lines: list[str] = []
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")) and len(line) > 21:
            atom_by_chain.setdefault(line[21], []).append(line)
        elif line.startswith(("END", "MODEL", "ENDMDL", "TER")):
            continue
        else:
            other_lines.append(line)

    ordered_chains = []
    for chain_id in chain_order:
        if chain_id in atom_by_chain and chain_id not in ordered_chains:
            ordered_chains.append(chain_id)
    for chain_id in atom_by_chain:
        if chain_id not in ordered_chains:
            ordered_chains.append(chain_id)

    with open(output_pdb, "w", encoding="utf-8") as handle:
        handle.write("MODEL        1\n")
        handle.writelines(other_lines)
        for chain_id in ordered_chains:
            handle.writelines(atom_by_chain[chain_id])
            handle.write("TER\n")
        handle.write("ENDMDL\nEND\n")


def _run_ppdock(ppdock: Path, row: pd.Series, work_root: Path) -> dict:
    target_id = str(row["target_id"])
    native_path = Path(str(row["native_path"]))
    decoy_path = Path(str(row["decoy_path"]))
    rec_chains = _chain_string(row["antigen_chains"])
    lig_chains = _chain_string(row["antibody_chains"])

    common = {
        "target_id": row["target_id"],
        "source": row["source"],
        "seed": row["seed"],
        "sample": row["sample"],
        "decoy_id": row.get("decoy_id", ""),
        "ranking": row.get("ranking", ""),
        "ranking_score": row.get("ranking_score", ""),
        "native_path": str(native_path),
        "decoy_path": str(decoy_path),
        "ppdock_receptor_chains": rec_chains,
        "ppdock_ligand_chains": lig_chains,
    }

    if not native_path.exists() or not decoy_path.exists() or not rec_chains or not lig_chains:
        return {
            **common,
            "ppdock_dockq": float("nan"),
            "ppdock_lrmsd": float("nan"),
            "ppdock_irmsd": float("nan"),
            "ppdock_fnat": float("nan"),
            "ppdock_status": "missing_input",
        }

    with tempfile.TemporaryDirectory(dir=work_root) as tmp:
        tmpdir = Path(tmp)
        title = target_id.replace("/", "_")
        decoy_ensemble = tmpdir / f"{title}_decoy_model_1.pdb"
        _write_single_model_ensemble(decoy_path, decoy_ensemble, chain_order=rec_chains + lig_chains)
        cmd = [
            str(ppdock), "-t", title, "-cwd", "-n_proc", "1",
            "-r", str(native_path), "-m", str(decoy_ensemble),
            "-rc", rec_chains, "-lc", lig_chains,
            "-lrmsd", "-irmsd", "-fnat", "--write",
        ]
        proc = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True)
        dat = tmpdir / f"{title}.eval.dat"
        parsed = _parse_eval_dat(dat) if dat.exists() else None
        if parsed is None:
            return {
                **common,
                "ppdock_dockq": float("nan"),
                "ppdock_lrmsd": float("nan"),
                "ppdock_irmsd": float("nan"),
                "ppdock_fnat": float("nan"),
                "ppdock_status": f"failed rc={proc.returncode}",
                "ppdock_stdout_tail": proc.stdout[-500:],
                "ppdock_stderr_tail": proc.stderr[-500:],
            }

    lrmsd, irmsd, fnat = parsed
    return {
        **common,
        "ppdock_dockq": _calc_dockq(lrmsd, irmsd, fnat),
        "ppdock_lrmsd": lrmsd,
        "ppdock_irmsd": irmsd,
        "ppdock_fnat": fnat,
        "ppdock_status": "ok",
    }


def _append_rows(rows: list[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def _key_tuples(frame: pd.DataFrame) -> set[tuple]:
    if frame.empty:
        return set()
    return set(tuple(row) for row in frame[KEY_COLUMNS].itertuples(index=False, name=None))


def _iter_batches(rows: Iterable[pd.Series], size: int):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def run(args):
    csv_dir = Path(args.csv_dir)
    dockq_csv = Path(args.dockq_csv) if args.dockq_csv else csv_dir / "dockq_metrics.csv"
    targets_csv = Path(args.targets_csv) if args.targets_csv else csv_dir / "targets.csv"
    output_csv = Path(args.output_csv) if args.output_csv else csv_dir / "ppdock_dockq_metrics.csv"
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    if args.overwrite_output and output_csv.exists():
        output_csv.unlink()

    dockq = pd.read_csv(dockq_csv)
    targets = pd.read_csv(targets_csv)
    targets = targets[["target_id", "source", "has_holo_antigen", "antibody_chains", "antigen_chains"]]
    rows = dockq.merge(targets, on=["target_id", "source"], how="left")
    rows = rows[rows["has_holo_antigen"].map(_truthy)].copy()
    rows = rows.drop_duplicates(KEY_COLUMNS, keep="last")

    if args.max_targets is not None:
        rows = rows.head(args.max_targets)

    done_keys = set()
    if args.resume and output_csv.exists():
        done_keys = _key_tuples(pd.read_csv(output_csv))
        rows = rows[~rows[KEY_COLUMNS].apply(tuple, axis=1).isin(done_keys)]

    print(
        f"[INFO] dockq_csv={dockq_csv} targets_csv={targets_csv} output_csv={output_csv} "
        f"holo_rows_to_run={len(rows)} skipped_existing={len(done_keys)}",
        flush=True,
    )

    n_done = 0
    t0 = time.perf_counter()
    for batch in _iter_batches((row for _idx, row in rows.iterrows()), args.flush_interval):
        out_rows = [_run_ppdock(Path(args.ppdock), row, work_root) for row in batch]
        _append_rows(out_rows, output_csv)
        n_done += len(out_rows)
        print(f"[PROGRESS] ppdock rows={n_done}/{len(rows)} elapsed_sec={time.perf_counter() - t0:.1f}", flush=True)

    if args.merged_output:
        ppdock = pd.read_csv(output_csv)
        merged = dockq.merge(
            ppdock[KEY_COLUMNS + ["ppdock_dockq", "ppdock_lrmsd", "ppdock_irmsd", "ppdock_fnat", "ppdock_status"]],
            on=KEY_COLUMNS,
            how="left",
        )
        merged_path = Path(args.merged_output)
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = merged_path.with_suffix(merged_path.suffix + ".tmp")
        merged.to_csv(tmp_path, index=False)
        tmp_path.replace(merged_path)
        print(f"[INFO] wrote merged_output={merged_path} rows={len(merged)}", flush=True)

    print(f"[DONE] ppdock rows={n_done} elapsed_sec={time.perf_counter() - t0:.1f}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", default=str(DEFAULT_CSV_DIR), help="Directory containing incremental CSVs.")
    parser.add_argument("--dockq-csv", default="", help="Override dockq_metrics.csv path.")
    parser.add_argument("--targets-csv", default="", help="Override targets.csv path.")
    parser.add_argument("--output-csv", default="", help="PPDock sidecar CSV path. Default: {csv-dir}/ppdock_dockq_metrics.csv.")
    parser.add_argument("--merged-output", default="", help="Optional merged CSV copy with PPDock columns.")
    parser.add_argument("--overwrite-output", action="store_true", help="Replace the PPDock sidecar CSV before running.")
    parser.add_argument("--ppdock", default=str(DEFAULT_PPDOCK), help="Path to PPDock_eval.py.")
    parser.add_argument("--work-root", default="/tmp/cdr_ppdock_eval", help="Temporary working directory root.")
    parser.add_argument("--flush-interval", type=int, default=1, help="Append output after this many rows.")
    parser.add_argument("--max-targets", type=int, default=None, help="Limit holo rows for smoke tests.")
    parser.add_argument("--resume", action="store_true", help="Skip key rows already present in output CSV.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
