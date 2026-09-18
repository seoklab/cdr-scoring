#!/usr/bin/env python
"""Build a MULTISOURCE test manifest: the four test decoy sources in one pool.

Why this exists: exp9a was selected on the multisource VALIDATION pool, so the
matching question is what it does on a multisource pool of TEST targets. The four
test sources have only ever been scored separately.

Selection rule mirrors val_multisource_fnat05.json so the two pools are directly
comparable:
  * fnat filter: keep only decoys with finite fnat > --gate (strict). Apo decoys
    have no fnat, so the filter is undefined for them and they are DROPPED here --
    the request is "evaluate only on what survives the filter".
  * drop non-finite cdr_lddt
  * per-target cap of --cap via the SAME deterministic stratified (source x tier)
    water-filling used for the validation manifest. Without a cap PertMD_test
    alone is 256 candidates/target (63 % of the pool) and 69 % of it sits above
    0.90, which would make top-1 meaningless.

  python preprocess/build_test_multisource_manifest.py \
      --out .../test_manifests/test_multisource_fnat05.json
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.build_val_manifest import _norm_seed, _stratified_select, _tier

DATA_ROOT = '/home/sujin/projects/cdr-scoring/cdr-data'
TEST_ROOT = os.path.join(DATA_ROOT, 'metrics_test_h3_v2')
SOURCES = ['AF3', 'Boltz2_test', 'ComMat_test', 'PertMD_test']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--gate', type=float, default=0.5, help='fnat gate, strict >')
    ap.add_argument('--cap', type=int, default=64)
    ap.add_argument('--min-cands', type=int, default=2,
                    help='drop targets left with fewer than this many candidates')
    a = ap.parse_args()

    frames = []
    for s in SOURCES:
        base = os.path.join(TEST_ROOT, s, 'metrics')
        lo = pd.read_parquet(os.path.join(base, 'loop_metrics.parquet'),
                             columns=['target_id', 'seed', 'sample', 'cdr_lddt'])
        fi = pd.read_parquet(os.path.join(base, 'interface_metrics.parquet'),
                             columns=['target_id', 'seed', 'sample', 'fnat'])
        m = lo.merge(fi, on=['target_id', 'seed', 'sample'], how='left')
        m['source'] = s
        print(f'{s:14s} {len(m):7,} decoys  {m.target_id.nunique():4d} targets  '
              f'fnat>{a.gate}: {100*(m.fnat > a.gate).mean():5.1f}%')
        frames.append(m)
    df = pd.concat(frames, ignore_index=True)
    df = df[np.isfinite(df.cdr_lddt)]
    df = df[np.isfinite(df.fnat) & (df.fnat > a.gate)].copy()
    print(f'\nafter fnat>{a.gate} (apo dropped): {len(df):,} decoys, '
          f'{df.target_id.nunique()} targets')

    targets, src_dist, tier_dist = {}, Counter(), Counter()
    for tid, g in df.groupby('target_id', sort=True):
        rows = [(str(r.source), _norm_seed(r.seed), int(r.sample), float(r.cdr_lddt))
                for r in g.itertuples(index=False)]
        sel = _stratified_select(rows, a.cap)
        if len(sel) < a.min_cands:
            continue
        lut = {(s, sd, sm): l for (s, sd, sm, l) in rows}
        for (s, sd, sm) in sel:
            src_dist[s] += 1
            tier_dist[_tier(lut[(s, sd, sm)])] += 1
        targets[str(tid)] = [[s, sd, sm] for (s, sd, sm) in sel]

    counts = [len(v) for v in targets.values()]
    n_dec = sum(counts)
    flat = [f'{t}|{s}|{sd}|{sm}' for t in sorted(targets) for s, sd, sm in targets[t]]
    payload = {
        'version': 1,
        'pool': os.path.basename(a.out).replace('.json', ''),
        'gate': a.gate,
        'gate_metric': 'fnat',
        'cap': a.cap,
        'rule': ('fnat>gate strict, apo DROPPED; finite cdr_lddt; deterministic '
                 'stratified (source x tier) cap -- same rule as val_multisource'),
        'sources': SOURCES,
        'hash': hashlib.sha256('\n'.join(flat).encode()).hexdigest()[:16],
        'stats': {
            'n_targets': len(targets),
            'n_decoys': n_dec,
            'candidates_per_target': {'mean': round(n_dec / max(1, len(targets)), 2),
                                      'min': int(min(counts)), 'max': int(max(counts))},
            'source_distribution': dict(src_dist),
            'tier_distribution': {t: tier_dist.get(t, 0) for t in 'XABCD'},
        },
        'targets': {k: targets[k] for k in sorted(targets)},
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(payload, f, indent=1)
    print(f'\nwrote {a.out}')
    print(json.dumps(payload['stats'], indent=1))


if __name__ == '__main__':
    main()
