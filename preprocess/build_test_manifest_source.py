#!/usr/bin/env python
"""Build a fixed TEST manifest for one decoy source, matching test_AF3.json's format.

Used to ask whether the valid->test performance drop is caused by the time split
(post-2021 targets) or by the decoy SOURCE distribution: the same test targets
are scored once per source, so the target set is held constant and only the
generator changes.

  python preprocess/build_test_manifest_source.py --source ComMat_test \
      --metrics-root .../metrics_test_h3_v2/ComMat_test --out .../test_ComMat.json
"""
import argparse, hashlib, json, os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--metrics-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cap', type=int, default=None,
                    help='max candidates per target (deterministic: first-k by (seed,sample))')
    ap.add_argument('--targets-from', default=None,
                    help='optional manifest json; restrict to its target list so pools are comparable')
    a = ap.parse_args()

    lm = pd.read_parquet(os.path.join(a.metrics_root, 'metrics', 'loop_metrics.parquet'))
    need = {'target_id', 'seed', 'sample', 'cdr_lddt'}
    missing = need - set(lm.columns)
    if missing:
        raise SystemExit(f'missing columns {missing} in {a.metrics_root}')
    lm = lm[np.isfinite(lm.cdr_lddt)]

    keep = None
    if a.targets_from:
        keep = set(json.load(open(a.targets_from))['targets'].keys())
        lm = lm[lm.target_id.isin(keep)]

    targets = {}
    for tid, g in lm.groupby('target_id'):
        g = g.sort_values(['seed', 'sample'], kind='mergesort')
        if a.cap:
            g = g.head(a.cap)
        cands = []
        for _, r in g.iterrows():
            s = r['seed']
            s = None if (s is None or s != s) else int(s)
            cands.append([a.source, s, int(r['sample'])])
        if cands:
            targets[str(tid)] = cands

    n_dec = sum(len(v) for v in targets.values())
    per = [len(v) for v in targets.values()]
    payload = {
        'version': 1,
        'pool': os.path.basename(a.out).replace('.json', ''),
        'gate': None,
        'cap': a.cap,
        'rule': 'all candidates (finite cdr_lddt), (seed,sample) order preserved',
        'sources': [a.source],
    }
    h = hashlib.sha1(json.dumps(
        {k: targets[k] for k in sorted(targets)}, sort_keys=True).encode()).hexdigest()[:16]
    payload['hash'] = h
    payload['stats'] = {'n_targets': len(targets), 'n_decoys': n_dec,
                        'candidates_per_target': {'mean': float(np.mean(per)),
                                                  'min': int(np.min(per)),
                                                  'max': int(np.max(per))}}
    payload['targets'] = {k: targets[k] for k in sorted(targets)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(payload, f)
    print(f'[manifest] {a.out}  targets={len(targets)} decoys={n_dec} '
          f'per_target={payload["stats"]["candidates_per_target"]} hash={h}')


if __name__ == '__main__':
    main()
