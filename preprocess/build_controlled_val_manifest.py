#!/usr/bin/env python
"""Fixed validation manifest for the controlled validation set (Step 3).

Decoys = the selected decoys of analyze/reports/controlled_benchmark/valid/decoys.csv
(Boltz-2 partial diffusion, target-wise low / intermediate / high, <= 12 per target) of the
PRIMARY targets, mapped to the training source 'Boltz2_pdiff' keys (seed, sample) through
cdr-data/metrics_precomputed_boltz2_pdiff/Boltz2_pdiff/metrics/decoy_map.csv.
Same JSON layout as preprocess/build_val_manifest.py (consumed by MyDataset.load_val_manifest).
Do not rebuild once training runs use it (the hash identifies the pool).

  python preprocess/build_controlled_val_manifest.py
  -> cdr-data/val_manifests/val_controlled.json
"""
import hashlib
import json
import os

import pandas as pd

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = '/home/sujin/projects/cdr-scoring/cdr-data'
CB = os.path.join(CODE, 'analyze', 'reports', 'controlled_benchmark', 'valid')
MAP = f'{DATA}/metrics_precomputed_boltz2_pdiff/Boltz2_pdiff/metrics/decoy_map.csv'
OUT = f'{DATA}/val_manifests/val_controlled.json'


def main():
    T = pd.read_csv(os.path.join(CB, 'targets.csv'))
    primary = set(T[T.primary].target)
    S = pd.read_csv(os.path.join(CB, 'decoys.csv'))
    S = S[S.target.isin(primary)]
    m = pd.read_csv(MAP, usecols=['target_id', 'seed', 'sample', 'decoy_key'])
    S = S.merge(m, left_on=['target', 'decoy_key'], right_on=['target_id', 'decoy_key'], how='left')
    assert S.seed.notna().all(), f'{S.seed.isna().sum()} selected decoys missing from the Boltz2_pdiff source'
    S = S.sort_values(['target', 'seed', 'sample'])
    targets = {t: [['Boltz2_pdiff', int(r.seed), int(r['sample'])] for _, r in g.iterrows()] for t, g in S.groupby('target')}
    h = hashlib.sha256(json.dumps(targets, sort_keys=True).encode()).hexdigest()[:16]
    n = S.groupby('target').size()
    man = {'version': 'v3-val-controlled-1', 'hash': h, 'pool': 'controlled', 'gate': 0.5, 'gate_metric': 'fnat',
           'cap': 12, 'rule': 'controlled validation (Step 3): primary targets, target-wise thirds low/intermediate/high, '
           '4 per tier at evenly spaced Fv-context values, <= 1 per (start structure, strength) per tier; '
           'holo fnat >= 0.5, apo no filter',
           'sources': ['Boltz2_pdiff'],
           'stats': {'n_targets': len(targets), 'n_decoys': int(len(S)),
                     'candidates_per_target': {'mean': round(float(n.mean()), 2), 'min': int(n.min()), 'max': int(n.max())},
                     'source_distribution': {'Boltz2_pdiff': int(len(S))},
                     'tier_distribution': S.tier.value_counts().to_dict(),
                     'holo_targets': int(T[T.primary].holo.sum()), 'apo_targets': int((~T[T.primary].holo).sum())},
           'targets': targets}
    with open(OUT, 'w') as fh:
        json.dump(man, fh)
    print(f'{OUT}: {len(targets)} targets, {len(S)} decoys, hash {h}')


if __name__ == '__main__':
    main()
