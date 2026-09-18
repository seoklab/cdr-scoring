#!/usr/bin/env python
"""Validate our DockQ computation against the AF3 (after210930) reference.

Reference values come from
``org_yubeen/capri/merged_capri_results.csv`` which was produced by
``rfepi.common.capri.get_capri`` on PDB inputs with
``ligand_chain = antibody (H+L)`` and ``receptor_chain = antigen``.

Our production pipeline reads *target pickles*, but the DockQ math itself lives
in ``precompute_decoy_metrics.compute_interface_rows`` which operates on Bio.PDB
structures. To isolate the algorithm (not the data-loading path) we feed the very
same reference PDBs (native crystal + AF3 ``model_rechain_final.pdb``) into that
function and compare fnat / iRMSD / lRMSD / DockQ against the CSV.

Chain convention: our code names the lRMSD-measured side ``ANTIBODY_CHAINS`` and
superposes on the antigen (Galaxy/DockQ convention), matching the reference's
receptor=antigen / ligand=antibody choice. Antibody chains are read per target
from the directory name and injected via ``P.ANTIBODY_CHAINS``.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import precompute_decoy_metrics as P  # noqa: E402
from evaluation.loop_metrics import load_structure  # noqa: E402

REF_DIR = Path(
    "/home/dnduq97/projects/abag_scoring/benchmark/after210930/alphafold3"
)
CSV = REF_DIR / "org_yubeen" / "capri" / "merged_capri_results.csv"
NATIVE_DIR = Path(
    "/home/yubeen/projects/rfepi/benchmark_after210930/pdb_real_final_new"
)

# Targets to validate and how many decoys to sample per CAPRI class.
TARGETS = ["7q6c_h_l_a", "7vgr_d_c_ab"]
PER_CLASS = 3

# ---- CAPRI thresholds, copied verbatim from reference get_capri_final.py ----
_CLS = {3: "high", 2: "medium", 1: "acceptable", 0: "incorrect"}


def _eval_fnat(f):
    return 3 if f >= 0.5 else 2 if f >= 0.3 else 1 if f >= 0.1 else 0


def _eval_lrmsd(l):
    return 3 if l <= 1.0 else 2 if l <= 5.0 else 1 if l <= 10.0 else 0


def _eval_irmsd(i):
    return 3 if i <= 1.0 else 2 if i <= 2.0 else 1 if i <= 4.0 else 0


def capri_class(lrmsd, irmsd, fnat):
    return _CLS[min(_eval_fnat(fnat), max(_eval_lrmsd(lrmsd), _eval_irmsd(irmsd)))]


def calc_dockq(lrmsd, irmsd, fnat):
    return (1 / (1 + (lrmsd / 8.5) ** 2) + 1 / (1 + (irmsd / 1.5) ** 2) + fnat) / 3


def native_name(name: str) -> str:
    pdb, h, l, ag = name.split("_")
    up = lambda x: "#" if x == "" else x.upper()
    return f"{pdb}_{up(h)}_{up(l)}_{up(ag)}"


def antibody_chains(name: str):
    _pdb, h, l, _ag = name.split("_")
    return tuple(c.upper() for c in (h, l) if c not in ("", "#"))


def main():
    # DockQ needs only fnat/irmsd/lrmsd; skip the O(N^2) interface lDDT.
    P._interface_lddt = lambda *a, **k: float("nan")

    rows_by_target: dict[str, list[dict]] = {}
    with open(CSV) as f:
        for row in csv.DictReader(f):
            if row["name"] in TARGETS:
                rows_by_target.setdefault(row["name"], []).append(row)

    selected: list[dict] = []
    for t in TARGETS:
        per: dict[str, list[dict]] = {}
        for row in rows_by_target.get(t, []):
            per.setdefault(row["capri"], [])
            if len(per[row["capri"]]) < PER_CLASS:
                per[row["capri"]].append(row)
        for cls in ("high", "medium", "acceptable", "incorrect"):
            selected.extend(per.get(cls, []))

    header = (
        f"{'name':<12}{'sd':>3}{'sp':>3} | {'refCLS':>10}{'myCLS':>10} | "
        f"{'refDQ':>7}{'myDQ':>7}{'dDQ':>7} | "
        f"{'refL':>7}{'myL':>7} | {'refI':>6}{'myI':>6} | {'refF':>6}{'myF':>6}"
    )
    print(header)
    print("-" * len(header))

    max_ddq = max_dl = max_di = max_df = 0.0
    n_cmp = 0
    n_cls_match = 0
    for row in selected:
        name, seed, sample = row["name"], row["seed"], row["sample"]
        native = NATIVE_DIR / native_name(name) / f"{native_name(name)}.pdb"
        decoy = (
            REF_DIR / "org_yubeen" / name / f"seed-{seed}_sample-{sample}"
            / "model_rechain_final.pdb"
        )
        if not native.exists() or not decoy.exists():
            print(f"{name:<12}{seed:>3}{sample:>3} | MISSING native={native.exists()} decoy={decoy.exists()}")
            continue

        P.ANTIBODY_CHAINS = antibody_chains(name)
        ns = load_structure(str(native))
        ms = load_structure(str(decoy))
        _iface, dq = P.compute_interface_rows(
            ns, ms, contact_cutoff_a=5.0, interface_cutoff_a=10.0
        )
        myL, myI, myF, myDQ = dq["lrmsd"], dq["irmsd"], dq["fnat"], dq["dockq"]
        refL = float(row["lrmsd"])
        refI = float(row["irmsd"])
        refF = float(row["fnat"])
        refDQ = float(row["dockq"])
        refCLS = row["capri"]
        myCLS = capri_class(myL, myI, myF)

        ddq = abs(myDQ - refDQ)
        max_ddq = max(max_ddq, ddq)
        max_dl = max(max_dl, abs(myL - refL))
        max_di = max(max_di, abs(myI - refI))
        max_df = max(max_df, abs(myF - refF))
        n_cmp += 1
        n_cls_match += int(myCLS == refCLS)

        print(
            f"{name:<12}{seed:>3}{sample:>3} | {refCLS:>10}{myCLS:>10} | "
            f"{refDQ:>7.3f}{myDQ:>7.3f}{ddq:>7.3f} | "
            f"{refL:>7.2f}{myL:>7.2f} | {refI:>6.2f}{myI:>6.2f} | "
            f"{refF:>6.3f}{myF:>6.3f}"
        )

    print("-" * len(header))
    print(
        f"compared={n_cmp}  class_match={n_cls_match}/{n_cmp}  "
        f"MAX|dDockQ|={max_ddq:.4f}  MAX|dlRMSD|={max_dl:.3f}  "
        f"MAX|diRMSD|={max_di:.3f}  MAX|dfnat|={max_df:.4f}"
    )


if __name__ == "__main__":
    main()
