#!/usr/bin/env python
"""Precompute ULR loop lDDT for general-protein (GP) decoys.

GP targets have a single remodeled loop = the ULR (from {pdb}.ulr). We compute a
superposition-free backbone lDDT of the ULR vs the native, with neighbours = the
whole protein backbone within LDDT_CUTOFF_A (no antigen). The value is stored in
the `cdr_lddt` column so it is drop-in compatible with the v1 precomputed-metric
store / label_metric=loop_lddt. (GP has one loop, so ULR lDDT == "H3" lDDT; we do
NOT write a separate H3 column, per request.)

Decoy order (matches the legacy {pdb}_fp.rmsd / graph .dat index):
    out_falc.pdb models (1000) then out_pertMD.pdb models (32)  -> sample = 0..N-1
    seed = None (graph-pickle / multi-MODEL identity).
"""
from __future__ import annotations
import sys, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "libs"))
from evaluation.loop_metrics import _backbone_lddt, BACKBONE_ATOMS  # noqa: E402
from Bio.PDB import PDBParser  # noqa: E402
warnings.filterwarnings("ignore")

GP_ROOT = Path("/home/sujin/DB/h3-loop-modeling/general_protein/decoy")
_PARSER = PDBParser(QUIET=True)


def parse_ulr(path: Path):
    """Return list of (chain, start, end) ULR segments."""
    segs = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        rng = next((p for p in parts if "-" in p and p.replace("-", "").isdigit()), None)
        if rng is None:
            continue
        start, end = (int(x) for x in rng.split("-"))
        chain = parts[-1]
        segs.append((chain, start, end))
    return segs


def _residues(model):
    out = {}
    for chain in model:
        for res in chain:
            if res.id[0] != " ":
                continue
            out[(chain.id, res.id)] = res
    return out


def _bb(res_n, res_m):
    nat, mod = [], []
    for a in BACKBONE_ATOMS:
        an = res_n.child_dict.get(a); am = res_m.child_dict.get(a)
        if an is None or am is None:
            continue
        nat.append(an.get_coord()); mod.append(am.get_coord())
    return nat, mod


def ulr_loop_lddt(native_model, decoy_model, segs):
    nat_map = _residues(native_model); mod_map = _residues(decoy_model)
    seg_set = segs
    def in_ulr(chain, resseq):
        return any(c == chain and s <= resseq <= e for c, s, e in seg_set)
    nat_pts, mod_pts, atom_to_res, mask = [], [], [], []
    ridx = 0
    for key in nat_map:
        mr = mod_map.get(key)
        if mr is None:
            continue
        nb, mb = _bb(nat_map[key], mr)
        if not nb:
            continue
        is_ulr = in_ulr(key[0], int(key[1][1]))
        for na, ma in zip(nb, mb):
            nat_pts.append(na); mod_pts.append(ma); atom_to_res.append(ridx); mask.append(is_ulr)
        ridx += 1
    if not any(mask):
        return float("nan")
    return _backbone_lddt(np.asarray(nat_pts, float), np.asarray(mod_pts, float),
                          np.asarray(atom_to_res, np.int32), np.asarray(mask, bool))


def _models(pdb_path):
    if not pdb_path.exists():
        return []
    st = _PARSER.get_structure(pdb_path.stem, str(pdb_path))
    return list(st.get_models())


def compute_target(pdb):
    d = GP_ROOT / pdb
    nat_path = d / f"{pdb}.pdb"; ulr_path = d / f"{pdb}.ulr"
    if not nat_path.exists() or not ulr_path.exists():
        return None
    segs = parse_ulr(ulr_path)
    if not segs:
        return None
    native = next(_PARSER.get_structure(pdb, str(nat_path)).get_models())
    decoys = _models(d / "out_falc.pdb") + _models(d / "out_pertMD.pdb")
    rows = []
    for i, dm in enumerate(decoys):
        rows.append({"target_id": pdb, "source": "GP", "seed": np.nan, "sample": i,
                     "decoy_id": f"sample={i}", "cdr_lddt": ulr_loop_lddt(native, dm, segs)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="file with one GP pdb id per line")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    ids = [l.strip() for l in open(args.targets) if l.strip()]
    if args.limit:
        ids = ids[: args.limit]
    all_rows, done, skipped = [], 0, 0
    for n, pdb in enumerate(ids, 1):
        try:
            r = compute_target(pdb)
        except Exception as exc:
            print(f"[WARN] {pdb}: {exc}", flush=True); r = None
        if r:
            all_rows.extend(r); done += 1
        else:
            skipped += 1
        if n % 50 == 0:
            print(f"[{n}/{len(ids)}] done={done} skipped={skipped} rows={len(all_rows)}", flush=True)
    df = pd.DataFrame(all_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[DONE] targets={done} skipped={skipped} rows={len(df)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
