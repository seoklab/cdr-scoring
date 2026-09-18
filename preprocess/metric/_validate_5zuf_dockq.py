#!/usr/bin/env python
"""Standalone validation of our DockQ computation against PPDock .eval.dat (5ZUF).

This does NOT modify the production computation code. It imports the helpers from
``precompute_source_metrics_fast.py`` and drives them on raw PDB files, using the
same chain partition PPDock used for this case: receptor = A,B,C ; ligand = D,E.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from Bio.PDB import PDBParser

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import precompute_source_metrics_fast as P  # noqa: E402

# PPDock command for this case: -rc ABC -lc ED  => receptor = A,B,C ; ligand = D,E.
# lRMSD superimposes on the receptor (antigen A,B,C) and is measured on the ligand
# (D,E). In our helpers ANTIBODY_CHAINS names the side lRMSD is measured on (the
# antigen is treated as the superposition frame per the Galaxy DockQ convention),
# so point it at the PPDock ligand chains D,E.
P.ANTIBODY_CHAINS = ("D", "E")

# We only validate fnat/irmsd/lrmsd here; skip the expensive interface lDDT
# (O(N^2) over the whole complex) which is irrelevant to this comparison.
P._interface_lddt_cached = lambda *a, **k: float("nan")

BASE = Path(
    "/home/dnduq97/projects/abag_scoring/benchmark/abag_docking_set/"
    "galaxytongdock/bound/5ZUF/5ZUF"
)
NATIVE = BASE / "5ZUF_native_matched_0.pdb"
DAT = BASE / "5ZUF_model_0.eval.dat"
MODEL_FILES = {
    "model_for_eval.pdb": BASE / "model" / "model_for_eval.pdb",
    "model.pdb": BASE / "model" / "model.pdb",
}


class _OneModel:
    """Wrap a single Bio.PDB Model so it quacks like a Structure for our helpers."""

    def __init__(self, model):
        self._m = model

    def get_models(self):
        return iter([self._m])


def load_models(path: Path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    return [_OneModel(m) for m in structure.get_models()]


def parse_dat(path: Path):
    """Return dict: 1-based model index -> (lrmsd, irmsd, fnat, fnon)."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        lrmsd, irmsd, fnat, fnon = (float(x) for x in parts[:4])
        midx = None
        for tok in parts:
            if tok.startswith("model="):
                midx = int(tok.split("=", 1)[1])
        if midx is None:
            continue
        out[midx] = (lrmsd, irmsd, fnat, fnon)
    return out


def main():
    device = torch.device("cpu")
    native = load_models(NATIVE)[0]
    ref = parse_dat(DAT)
    n_ref = len(ref)
    print(f"Reference .dat entries: {n_ref}")

    results = {}
    for label, path in MODEL_FILES.items():
        decoys = load_models(path)
        rows = P.compute_interface_rows_for_models_fast(
            native,
            decoys,
            contact_cutoff_a=5.0,
            interface_cutoff_a=10.0,
            device=device,
        )
        results[label] = rows
        print(f"\n[{label}] decoys parsed: {len(decoys)}  rows: {len(rows)}")

    # ---- Compare each model file to the .dat reference ----
    for label, rows in results.items():
        print("\n" + "=" * 78)
        print(f"COMPARE  {label}  vs  PPDock .dat")
        print("=" * 78)
        print(f"{'mdl':>3} | {'lrmsd_our':>10} {'lrmsd_dat':>10} {'dL':>8} | "
              f"{'irmsd_our':>10} {'irmsd_dat':>10} {'dI':>8} | "
              f"{'fnat_our':>9} {'fnat_dat':>9} {'dF':>8}")
        max_dl = max_di = max_df = 0.0
        n_cmp = 0
        worst_lines = []
        for i, (interface_row, dockq_row) in enumerate(rows):
            midx = i + 1
            if midx not in ref:
                continue
            dl_, di_, df_, _ = ref[midx]
            our_l = dockq_row["lrmsd"]
            our_i = dockq_row["irmsd"]
            our_f = dockq_row["fnat"]
            ddl = abs(our_l - dl_) if np.isfinite(our_l) else float("nan")
            ddi = abs(our_i - di_) if np.isfinite(our_i) else float("nan")
            ddf = abs(our_f - df_) if np.isfinite(our_f) else float("nan")
            n_cmp += 1
            for v, store in ((ddl, "l"), (ddi, "i"), (ddf, "f")):
                if np.isfinite(v):
                    if store == "l":
                        max_dl = max(max_dl, v)
                    elif store == "i":
                        max_di = max(max_di, v)
                    else:
                        max_df = max(max_df, v)
            line = (f"{midx:>3} | {our_l:>10.3f} {dl_:>10.3f} {ddl:>8.3f} | "
                    f"{our_i:>10.3f} {di_:>10.3f} {ddi:>8.3f} | "
                    f"{our_f:>9.4f} {df_:>9.4f} {ddf:>8.4f}")
            worst_lines.append((max(ddl if np.isfinite(ddl) else 1e9,
                                    ddi if np.isfinite(ddi) else 1e9), line))
            if midx <= 10:
                print(line)
        print(f"... ({n_cmp} models compared)")
        print(f"MAX |Δlrmsd| = {max_dl:.4f}   MAX |Δirmsd| = {max_di:.4f}   "
              f"MAX |Δfnat| = {max_df:.4f}")
        worst_lines.sort(reverse=True)
        print("Worst 5 rows by RMSD diff:")
        for _, ln in worst_lines[:5]:
            print("  " + ln)

    # ---- Numbering invariance: model.pdb vs model_for_eval.pdb ----
    if "model.pdb" in results and "model_for_eval.pdb" in results:
        print("\n" + "=" * 78)
        print("NUMBERING INVARIANCE  model.pdb  vs  model_for_eval.pdb (our code)")
        print("=" * 78)
        a = results["model_for_eval.pdb"]
        b = results["model.pdb"]
        max_dl = max_di = max_df = 0.0
        for i in range(min(len(a), len(b))):
            la, ia, fa = a[i][1]["lrmsd"], a[i][1]["irmsd"], a[i][1]["fnat"]
            lb, ib, fb = b[i][1]["lrmsd"], b[i][1]["irmsd"], b[i][1]["fnat"]
            if np.isfinite(la) and np.isfinite(lb):
                max_dl = max(max_dl, abs(la - lb))
            if np.isfinite(ia) and np.isfinite(ib):
                max_di = max(max_di, abs(ia - ib))
            if np.isfinite(fa) and np.isfinite(fb):
                max_df = max(max_df, abs(fa - fb))
            if i < 5:
                print(f"mdl {i+1}: eval(l={la:.3f} i={ia:.3f} f={fa:.4f})  "
                      f"raw(l={lb!r} i={ib!r} f={fb!r})")
        print(f"MAX |Δlrmsd| = {max_dl:.4f}  MAX |Δirmsd| = {max_di:.4f}  "
              f"MAX |Δfnat| = {max_df:.4f}")


if __name__ == "__main__":
    main()
