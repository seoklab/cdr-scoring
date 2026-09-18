#!/usr/bin/env python
"""Cross-check our DockQ against Galaxy PPDock_eval on pertMD target pickles.

For a few pertMD targets we take the native (crystal) + a handful of decoys
straight out of the target pickle, then:

  * compute fnat / iRMSD / lRMSD / DockQ with our
    ``precompute_decoy_metrics.compute_interface_rows`` (same code path the
    build pipeline uses), and
  * write the identical structures to PDB and score them with the reference
    ``PPDock_eval.py`` (Galaxy) using receptor=antigen, ligand=antibody.

Both operate on the *same* coordinates and numbering, so any difference is the
scoring algorithm, not the inputs.
"""
from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

CDR_ROOT = Path("/home/sujin/projects/cdr-scoring/cdr-code")
sys.path.insert(0, str(CDR_ROOT / "libs"))
sys.path.insert(0, str(CDR_ROOT / "preprocess" / "metric"))

from Bio.PDB import PDBIO, Structure as BioStructure  # noqa: E402

import precompute_decoy_metrics as P  # noqa: E402

PICKLE_DIR = Path("/home/sujin/DB/h3-loop-modeling/ab_ag/1_tr_abag/43_pertmd_cdr_pdb2dict")
PPDOCK = "/opt/conda/envs/gp/bin/PPDock_eval.py"

TARGETS = ["1a14_H_L_A", "1acy_H_L_A", "1afv_H_L_A"]
N_DECOYS = 6

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


def chains_from_name(name):
    pdb, h, l, ag = name.split("_")
    ab = tuple(c.upper() for c in (h, l) if c not in ("", "#"))
    return ab, ag.upper()


def write_single(structure, path):
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path))


def write_multimodel(decoy_structures, path):
    combined = BioStructure.Structure("decoys")
    for i, dec in enumerate(decoy_structures, start=1):
        src_model = list(dec.get_models())[0]
        m = src_model.copy()
        m.id = i
        m.serial_num = i
        combined.add(m)
    io = PDBIO()
    io.set_structure(combined)
    io.save(str(path))


def parse_eval_dat(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return rows


def run_ppdock(workdir, title, native_pdb, decoys_pdb, rec, lig):
    cmd = [
        PPDOCK, "-t", title, "-cwd", "-n_proc", "1",
        "-r", str(native_pdb), "-m", str(decoys_pdb),
        "-rc", rec, "-lc", lig,
        "-lrmsd", "-irmsd", "-fnat", "--write",
    ]
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    dat = Path(workdir) / f"{title}.eval.dat"
    if not dat.exists():
        print(f"  PPDock produced no eval.dat.\n  stdout:{proc.stdout[-500:]}\n  stderr:{proc.stderr[-500:]}")
        return []
    return parse_eval_dat(dat)


def main():
    P._interface_lddt = lambda *a, **k: float("nan")

    header = (
        f"{'target':<12}{'m':>3} | {'myCLS':>10}{'ppCLS':>10} | "
        f"{'myDQ':>7}{'ppDQ':>7}{'dDQ':>7} | {'myL':>7}{'ppL':>7} | "
        f"{'myI':>6}{'ppI':>6} | {'myF':>6}{'ppF':>6}"
    )

    for name in TARGETS:
        pkl = PICKLE_DIR / f"{name}.pkl"
        if not pkl.exists():
            print(f"[skip] missing pickle {pkl}")
            continue
        with open(pkl, "rb") as f:
            target = pickle.load(f)
        ab, ag = chains_from_name(name)
        P.ANTIBODY_CHAINS = ab

        decoys = target.models[:N_DECOYS]
        print(f"\n### {name}  antibody={ab} antigen={ag}  decoys={len(decoys)}")
        print(header)
        print("-" * len(header))

        with tempfile.TemporaryDirectory() as wd:
            native_pdb = Path(wd) / f"{name}_native.pdb"
            decoys_pdb = Path(wd) / f"{name}_decoys.pdb"
            write_single(target.gt_structure, native_pdb)
            write_multimodel([d.md_structure for d in decoys], decoys_pdb)

            pp_rows = run_ppdock(wd, name, native_pdb, decoys_pdb, rec=ag, lig="".join(ab))

            for i, dec in enumerate(decoys):
                _iface, dq = P.compute_interface_rows(
                    target.gt_structure, dec.md_structure,
                    contact_cutoff_a=5.0, interface_cutoff_a=10.0,
                )
                myL, myI, myF, myDQ = dq["lrmsd"], dq["irmsd"], dq["fnat"], dq["dockq"]
                myCLS = capri_class(myL, myI, myF)

                if i < len(pp_rows):
                    ppL, ppI, ppF = pp_rows[i]
                    ppDQ = calc_dockq(ppL, ppI, ppF)
                    ppCLS = capri_class(ppL, ppI, ppF)
                    ddq = myDQ - ppDQ
                    print(
                        f"{name:<12}{i + 1:>3} | {myCLS:>10}{ppCLS:>10} | "
                        f"{myDQ:>7.3f}{ppDQ:>7.3f}{ddq:>+7.3f} | "
                        f"{myL:>7.2f}{ppL:>7.2f} | {myI:>6.2f}{ppI:>6.2f} | "
                        f"{myF:>6.3f}{ppF:>6.3f}"
                    )
                else:
                    print(
                        f"{name:<12}{i + 1:>3} | {myCLS:>10}{'NA':>10} | "
                        f"{myDQ:>7.3f}{'NA':>7}{'':>7} | {myL:>7.2f}{'':>7} | "
                        f"{myI:>6.2f}{'':>6} | {myF:>6.3f}"
                    )


if __name__ == "__main__":
    main()
