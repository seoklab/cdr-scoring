#!/usr/bin/env python
"""Pick targets and ComMat seed models for the ComMat-seeded weak-pdiff pilot.

Targets: from the native-start Boltz-2 pilot (1_tr_abag/52_pdiff_pilot), those with a
narrow Fv-context range (max - min < --narrow, fnat >= 0.5 decoys) whose ComMat decoys go
clearly lower than the native-start minimum (by >= --margin), ~--n_targets of them,
balanced valid/train and antibody/nanobody.
Seeds: every ComMat model of a picked target is scored against the native here
(Fv-context CDR-lDDT, fnat); among models with fnat >= --fnat_min, 3 with different
Fv-context: the lowest, the lower third and the median.

Writes <out_dir>/seeds.csv (target, origin, path, model, seed_fv_context, seed_fnat, ...)
for prepare_pdiff_inputs.py --seeds_csv, and <out_dir>/seed_candidates.csv (all models).
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'metric'))
warnings.filterwarnings('ignore')
from fv_context_lddt import compute_cdr_lddt_variants  # noqa: E402
from precompute_source_metrics_fast import compute_interface_rows  # noqa: E402

TRVAL = '/home/sujin/DB/h3-loop-modeling/ab_ag/1_tr_abag'


def score_commat(t):
    path = f'{TRVAL}/21_commat_xtal/{t}/{t}.pdb'
    native = PDBParser(QUIET=True).get_structure('n', f'{TRVAL}/04_xtal_chothia_ag_renum/{t}.pdb')
    rows = []
    # MODEL records read "MODEL 1" (short), so Bio.PDB's serial_num is 0: number models by file
    # order (1..n), the same numbering prepare_pdiff_inputs.read_seed_model uses
    for k, model in enumerate(PDBParser(QUIET=True).get_structure('c', path), start=1):
        s = type(native)('m')
        s.add(model.copy())
        v = compute_cdr_lddt_variants(native, s)
        _, dq = compute_interface_rows(native, s, contact_cutoff_a=5.0, interface_cutoff_a=10.0)
        rows.append({'target': t, 'origin': 'ComMat', 'path': path, 'model': k,
                     'seed_fv_context': v['fv_context_cdr_lddt'], 'seed_cdr_context': v['cdr_context_lddt'],
                     'seed_H3_fv_context': v['H3_fv_context_lddt'], 'seed_fnat': dq['fnat']})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pilot_eval', default=f'{TRVAL}/52_pdiff_pilot/eval/decoys.csv')
    ap.add_argument('--out_dir', default=f'{TRVAL}/53_pdiff_seeded_pilot')
    ap.add_argument('--narrow', type=float, default=0.10)
    ap.add_argument('--margin', type=float, default=0.05)
    ap.add_argument('--n_targets', type=int, default=10)
    ap.add_argument('--fnat_min', type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    valid = set(open(os.path.join(HERE, 'pilot_valid_targets.txt')).read().split())

    p = pd.read_csv(args.pilot_eval)
    p = p[p.fnat >= 0.5]
    nat = p.groupby('target').fv_context_cdr_lddt.agg(['min', 'max'])
    nat['range'] = nat['max'] - nat['min']
    narrow = nat[nat.range < args.narrow].index.tolist()
    print(f'native-start pilot: {len(nat)} targets, narrow (range < {args.narrow}): {len(narrow)}')

    cands = pd.concat([score_commat(t) for t in narrow], ignore_index=True)
    cands.to_csv(os.path.join(args.out_dir, 'seed_candidates.csv'), index=False)
    ok = cands[cands.seed_fnat >= args.fnat_min]
    lowest = ok.groupby('target').seed_fv_context.min()
    gain = (nat.loc[lowest.index, 'min'] - lowest).rename('commat_gain')
    info = pd.DataFrame({'commat_gain': gain, 'native_min': nat.loc[lowest.index, 'min'], 'commat_min': lowest})
    info['split'] = np.where(info.index.isin(valid), 'valid', 'train')
    info['type'] = np.where(info.index.str.split('_').str[2] == 'X', 'nanobody', 'antibody')
    elig = info[info.commat_gain >= args.margin].sort_values('commat_gain', ascending=False)
    print(f'narrow targets whose ComMat seeds go >= {args.margin} lower: {len(elig)}')
    # balanced pick: round-robin over (split, type) by gain
    groups = [g for _, g in elig.groupby(['split', 'type'])]
    picked, i = [], 0
    while len(picked) < min(args.n_targets, len(elig)):
        for g in groups:
            if i < len(g) and len(picked) < args.n_targets:
                picked.append(g.index[i])
        i += 1
    seeds = []
    for t in picked:
        g = ok[ok.target == t].drop_duplicates('model').sort_values('seed_fv_context').reset_index(drop=True)
        idx = sorted({0, int(round((len(g) - 1) / 3)), int(round((len(g) - 1) / 2))})
        seeds.append(g.loc[idx])
    S = pd.concat(seeds, ignore_index=True)
    S.to_csv(os.path.join(args.out_dir, 'seeds.csv'), index=False)
    with pd.option_context('display.width', 200):
        print(info.loc[picked].round(3).to_string())
        print(S[['target', 'model', 'seed_fv_context', 'seed_fnat']].round(3).to_string(index=False))
    print('->', os.path.join(args.out_dir, 'seeds.csv'))


if __name__ == '__main__':
    main()
