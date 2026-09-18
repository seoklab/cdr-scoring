#!/usr/bin/env python
"""Build a FIXED validation candidate manifest for v2 (and all later models).

Freezes, once, the exact decoys used to evaluate each validation target — stored
by stable identity key (target_id / source / seed / sample), never by index.
IMMUTABLE after creation: do not regenerate for v2/v3/... model comparisons.

Selection rule (fixed at build time):
  * gate (--gate_metric):
      dockq: holo decoys need DockQ >= --gate; apo decoys (NaN DockQ) kept.
      fnat:  holo decoys need fnat  >  --gate; apo decoys (NaN fnat)  kept.
    (fnat is strict > to match the project-wide "fnat>0.5" filter definition)
  * drop decoys with non-finite cdr_lddt.
  * multisource pool: per-target cap of --cap (default 64) via DETERMINISTIC
    stratified sampling over (source x cdr_lddt tier) buckets — even quota per
    bucket, round-robin water-filling so deficits are redistributed to remaining
    buckets deterministically. Within a bucket, take the first-k by (seed,sample).
  * boltz2 pool: NO cap — keep every gated candidate (deployment-like Boltz2
    distribution; top1 is target-wise equal-weight so uneven counts are fine).

Usage:
  python preprocess/build_val_manifest.py --pool multisource --out .../val_multisource.json
  python preprocess/build_val_manifest.py --pool boltz2      --out .../val_boltz2.json
"""
import argparse
import hashlib
import json
import os
import pickle
from collections import OrderedDict, Counter

import pandas as pd

DATA_ROOT = "/home/sujin/projects/cdr-scoring/cdr-data"
SOURCE_DIRS = {
    "Boltz2_s10n10": "metrics_precomputed_boltz2_s10n10_chothia_cdr/Boltz2_s10n10",
    "Boltz2":        "metrics_precomputed_boltz2_cdr/Boltz2",
    "ComMat":        "metrics_precomputed_commat_cdr/ComMat",
    "PertMD_cdr":    "metrics_precomputed_pertmd_cdr_cdr/PertMD_cdr",
    "PertMD_all":    "metrics_precomputed_pertmd_all_cdr/PertMD_all",
}
POOL_SOURCES = {
    "multisource": list(SOURCE_DIRS.keys()),
    "boltz2": ["Boltz2_s10n10"],
}


def _norm_seed(seed):
    if seed is None or (isinstance(seed, float) and seed != seed):
        return None
    return int(seed)


def _tier(v):
    if v >= 0.99:
        return "X"
    if v >= 0.90:
        return "A"
    if v >= 0.80:
        return "B"
    if v >= 0.70:
        return "C"
    return "D"


def _load_valid_targets(valid_pkl, valid_key, info_pkl):
    if not valid_pkl:
        return None
    with open(valid_pkl, "rb") as f:
        info = pickle.load(f)
    lst = info.get(valid_key)
    if isinstance(lst, dict):
        lst = list(lst.keys())
    ids = [str(x) for x in lst]
    if info_pkl and os.path.exists(info_pkl):
        with open(info_pkl, "rb") as f:
            m = pickle.load(f)
        new_to_old = m.get("list_old", {})
        old_to_new = {v: k for k, v in new_to_old.items()}
        ids = [old_to_new.get(x, x) for x in ids]
    return set(ids)


def _stratified_select(rows, cap):
    """rows: list of (source, norm_seed, sample, cdr_lddt). Return list of
    (source, norm_seed, sample) — deterministic stratified pick over (source,tier).

    Even round-robin quota per bucket; exhausted buckets are skipped so their
    unfilled quota is redistributed to remaining buckets, deterministically."""
    if cap is None or len(rows) <= cap:
        return [(s, sd, sm) for (s, sd, sm, _l) in rows]
    buckets = OrderedDict()
    # deterministic bucket + within-bucket order
    for (s, sd, sm, l) in sorted(rows, key=lambda r: (r[0], _tier(r[3]),
                                                       -r[3], (r[1] if r[1] is not None else -1), r[2])):
        buckets.setdefault((s, _tier(l)), []).append((s, sd, sm))
    bkeys = sorted(buckets.keys())
    alloc = {k: 0 for k in bkeys}
    remaining = cap
    progressed = True
    while remaining > 0 and progressed:
        progressed = False
        for k in bkeys:
            if remaining == 0:
                break
            if alloc[k] < len(buckets[k]):
                alloc[k] += 1
                remaining -= 1
                progressed = True
    sel = []
    for k in bkeys:
        sel.extend(buckets[k][:alloc[k]])
    return sel


def build(pool, out_path, gate, valid_pkl, valid_key, info_pkl, cap,
          gate_metric="dockq"):
    sources = POOL_SOURCES[pool]
    valid_targets = _load_valid_targets(valid_pkl, valid_key, info_pkl)

    frames = []
    for s in sources:
        base = os.path.join(DATA_ROOT, SOURCE_DIRS[s], "metrics")
        lo = pd.read_parquet(os.path.join(base, "loop_metrics.parquet"),
                             columns=["target_id", "seed", "sample", "cdr_lddt"])
        if gate_metric == "dockq":
            gt = pd.read_parquet(os.path.join(base, "dockq_metrics.parquet"),
                                 columns=["target_id", "seed", "sample", "dockq"])
        else:
            gt = pd.read_parquet(os.path.join(base, "interface_metrics.parquet"),
                                 columns=["target_id", "seed", "sample", "fnat"])
        m = lo.merge(gt, on=["target_id", "seed", "sample"], how="left")
        m["source"] = s
        frames.append(m)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["cdr_lddt"].notna()]
    if valid_targets is not None:
        df = df[df["target_id"].isin(valid_targets)]
    if gate_metric == "dockq":
        df = df[df["dockq"].isna() | (df["dockq"] >= gate)].copy()
    else:  # fnat: strict >, apo (NaN fnat) kept
        df = df[df["fnat"].isna() | (df["fnat"] > gate)].copy()

    targets = {}
    src_dist = Counter()
    tier_dist = Counter()
    for tid, g in df.groupby("target_id", sort=True):
        rows = [(str(r.source), _norm_seed(r.seed), int(r.sample), float(r.cdr_lddt))
                for r in g.itertuples(index=False)]
        sel = _stratified_select(rows, cap)
        # record source/tier distribution over the SELECTED decoys
        lut = {(s, sd, sm): l for (s, sd, sm, l) in rows}
        for (s, sd, sm) in sel:
            src_dist[s] += 1
            tier_dist[_tier(lut[(s, sd, sm)])] += 1
        targets[str(tid)] = [[s, sd, sm] for (s, sd, sm) in sel]

    counts = [len(v) for v in targets.values()]
    n_decoys = sum(counts)
    flat = []
    for tid in sorted(targets):
        for src, seed, samp in targets[tid]:
            flat.append(f"{tid}|{src}|{seed}|{samp}")
    content_hash = hashlib.sha256("\n".join(flat).encode()).hexdigest()[:16]

    stats = {
        "n_targets": len(targets),
        "n_decoys": n_decoys,
        "candidates_per_target": {
            "mean": round(n_decoys / max(1, len(targets)), 2),
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
        },
        "source_distribution": dict(src_dist),
        "tier_distribution": {t: tier_dist.get(t, 0) for t in "XABCD"},
    }
    gate_desc = ("DockQ>=gate(holo)/apo-kept" if gate_metric == "dockq"
                 else "fnat>gate(holo)/apo-kept")
    manifest = {
        "version": f"v2-val-{pool}-{'1' if gate_metric == 'dockq' else '2-fnat'}",
        "hash": content_hash,
        "pool": pool,
        "gate": gate,
        "gate_metric": gate_metric,
        "cap": cap,
        "rule": (f"{gate_desc}; finite cdr_lddt; "
                 + ("deterministic stratified (source x tier) cap"
                    if cap is not None else "no cap (all gated)")),
        "sources": sources,
        "stats": stats,
        "targets": targets,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f)

    print(f"\n[manifest:{pool}] -> {out_path}")
    print(f"  hash            = {content_hash}")
    print(f"  cap             = {cap}")
    print(f"  n_targets       = {stats['n_targets']}")
    print(f"  candidates/tgt  = mean {stats['candidates_per_target']['mean']} "
          f"min {stats['candidates_per_target']['min']} max {stats['candidates_per_target']['max']}")
    print(f"  n_decoys        = {n_decoys}")
    print(f"  source dist     = {stats['source_distribution']}")
    print(f"  tier   dist     = {stats['tier_distribution']}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, choices=["multisource", "boltz2"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate", type=float, default=0.49)
    ap.add_argument("--gate_metric", choices=["dockq", "fnat"], default="dockq",
                    help="dockq: DockQ>=gate. fnat: fnat>gate (strict). apo kept either way.")
    ap.add_argument("--valid_pkl", default="/home/sujin/DB/h3-loop-modeling/ab_ag/1_tr_abag/0_info/valid.pkl")
    ap.add_argument("--valid_key", default="list")
    ap.add_argument("--info_pkl", default="/home/sujin/DB/h3-loop-modeling/ab_ag/1_tr_abag/0_info/info.pkl")
    ap.add_argument("--cap", type=int, default=None,
                    help="per-target cap. Omit for no cap. Recommended: 64 for multisource.")
    a = ap.parse_args()
    # pool defaults: multisource -> stratified cap 64; boltz2 -> no cap
    cap = a.cap
    if cap is None and a.pool == "multisource":
        cap = 64
    build(a.pool, a.out, a.gate, a.valid_pkl, a.valid_key, a.info_pkl, cap,
          gate_metric=a.gate_metric)
