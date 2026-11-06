#!/usr/bin/env python
"""Plan ComMat-seeded weak partial diffusion to fill empty Fv-context CDR-lDDT ranges,
per target and tool, for the controlled TEST set (AF3 and Boltz-2) and the controlled
VALIDATION set (Boltz-2; valid holo targets).

Empty range: a CDR-lDDT range with fewer than --per_range pose-correct (fnat >= 0.5)
partial-diffusion decoys of that tool for the target.
  test : existing decoys = native-start pdiff 52/53/54 of the same tool
  valid: existing decoys = native-start pilot (1_tr_abag/52) + ComMat-seeded pilot (53);
         targets without a native-start run are assumed to have only the >= 0.95 range
         (pilot: valid native-start collapses to ~0.97) and get a native-start ladder
         (--native_ladder) in addition.
Seeds: ComMat models scored here against the native (Fv-context, fnat). For an empty range
[lo, hi) the window is [lo - 0.03, hi - 0.01) (1-2 A outputs move +0.01..+0.04 above the seed),
fnat >= 0.5, at most --seeds_per_range models chosen greedily for CDR diversity (each next
seed = largest minimum CDR backbone RMSD to the seeds already chosen for this target;
ComMat shares the native frame, so no superposition). A model is used once per target and tool.
Outputs per seed (about 6-9 per range): 3 seeds -> 1 A x2 + 2 A x1 each; 2 seeds -> 1 A x3 + 2 A x1;
1 seed -> 1 A x4 + 2 A x2. The >= 0.95 range is not seeded (native start covers it).

Writes analyze/reports/commat_seed_plan/
  commat_scores_<dataset>.csv   every ComMat model: Fv-context, fnat (+ pairwise CDR RMSD in .npz)
  seeds.csv                     dataset, tool, target, range, model, seed metrics, outputs per strength
  ranges.csv                    per dataset x tool x target x range: existing, empty?, seeds, planned outputs
  plan_rows_<dataset>_<tool>.csv  rows for make_pdiff_manifest.py plan (target = seeded input name)
  native_ladder_valid.txt       valid targets that still need the native-start ladder
"""
import argparse
import functools
import multiprocessing as mp
import os
import sys
import warnings

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'metric'))
warnings.filterwarnings('ignore')
from fv_context_lddt import compute_cdr_lddt_variants  # noqa: E402
from precompute_source_metrics_fast import compute_interface_rows  # noqa: E402
from evaluation.loop_metrics import DEFAULT_NAMED_LOOP_RANGES  # noqa: E402

BENCH = '/home/sujin/DB/h3-loop-modeling/ab_ag/3_h3_benchmark_after210930'
TRVAL = '/home/sujin/DB/h3-loop-modeling/ab_ag/1_tr_abag'
OUT = os.path.join(CODE, 'analyze', 'reports', 'commat_seed_plan')
RANGES = [('< 0.75', 0.0, 0.75), ('0.75-0.85', 0.75, 0.85), ('0.85-0.90', 0.85, 0.90),
          ('0.90-0.95', 0.90, 0.95), ('>= 0.95', 0.95, 1.0001)]
PATHS = {'test': {'native': f'{BENCH}/02_xtal_chothia_w_ag_renum/{{t}}.pdb',
                  'commat': f'{BENCH}/11_commat_xtal/{{t}}/{{t}}.pdb'},
         'valid': {'native': f'{TRVAL}/04_xtal_chothia_ag_renum/{{t}}.pdb',
                   'commat': f'{TRVAL}/21_commat_xtal/{{t}}/{{t}}.pdb'}}
OUTPUTS = {3: (2, 1), 2: (3, 1), 1: (4, 2)}   # seeds in range -> (1 A samples, 2 A samples) per seed


def is_cdr(chain, num):
    return any(c == chain and s <= num <= e for _, c, s, e in DEFAULT_NAMED_LOOP_RANGES)


def score_target(t, dataset):
    p = PDBParser(QUIET=True)
    native = p.get_structure('n', PATHS[dataset]['native'].format(t=t))
    path = PATHS[dataset]['commat'].format(t=t)
    rows, coords = [], []
    for k, model in enumerate(p.get_structure('c', path), start=1):   # file order (MODEL lines are short)
        s = type(native)('m')
        s.add(model.copy())
        v = compute_cdr_lddt_variants(native, s)
        _, dq = compute_interface_rows(native, s, contact_cutoff_a=5.0, interface_cutoff_a=10.0)
        rows.append({'dataset': dataset, 'target': t, 'path': path, 'model': k,
                     'seed_fv_context': v['fv_context_cdr_lddt'], 'seed_H3_fv_context': v['H3_fv_context_lddt'],
                     'seed_fnat': dq['fnat']})
        c = {}
        for ch in model:
            if ch.id in ('H', 'L'):
                for r in ch:
                    if r.id[0] == ' ' and is_cdr(ch.id, r.id[1]):
                        for a in ('N', 'CA', 'C', 'O'):
                            if a in r:
                                c[(ch.id, r.id[1], r.id[2], a)] = r[a].coord.astype(float)
        coords.append(c)
    keys = sorted(set.intersection(*[set(c) for c in coords]))
    X = np.array([[c[k] for k in keys] for c in coords])               # models x atoms x 3
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1).mean(-1))       # CDR backbone RMSD
    return rows, t, D


def existing_decoys():
    """Pose-correct pdiff decoys already made, per dataset, tool, target."""
    out = []
    for run in ('52_pdiff_sweep', '53_pdiff_calibration', '54_pdiff_decoys'):
        d = pd.read_csv(f'{BENCH}/{run}/eval/decoys.csv')
        out.append(d.assign(dataset='test'))
    for run in ('52_pdiff_pilot', '53_pdiff_seeded_pilot'):
        p = f'{TRVAL}/{run}/eval/decoys.csv'
        if os.path.exists(p):
            out.append(pd.read_csv(p).assign(dataset='valid'))
    d = pd.concat(out, ignore_index=True)
    return d[d.fnat >= 0.5]


def pick_seeds(cand, D, models_index, k, used):
    """Greedy CDR-diverse pick of up to k models from cand (DataFrame), skipping used models."""
    cand = cand[~cand.model.isin(used)].sort_values(['seed_fnat'], ascending=False)
    chosen = []
    while len(chosen) < k and len(cand):
        if not chosen:
            pick = cand.iloc[0]
        else:
            idx = [models_index[m] for m in chosen]
            dmin = cand.model.map(lambda m: D[models_index[m], idx].min())
            pick = cand.loc[dmin.idxmax()]
            if dmin.max() < 0.5:   # nothing structurally distinct left
                break
        chosen.append(int(pick.model))
        cand = cand[cand.model != pick.model]
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--per_range', type=int, default=5)
    ap.add_argument('--seeds_per_range', type=int, default=3)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    ap.add_argument('--native_ladder', default='1 2 4 8 16')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    holo_test = sorted(pd.read_csv(os.path.join(CODE, 'analyze/reports/test_partial_diffusion_needs_per_target.csv')).target)
    import pickle
    valid = [t for t in pickle.load(open(f'{TRVAL}/0_info/valid.pkl', 'rb'))['list'] if t.count('_') == 3]
    valid = [t for t in valid if os.path.exists(PATHS['valid']['commat'].format(t=t))]
    targets = {'test': holo_test, 'valid': valid}

    scores, Dmats = {}, {}
    for ds, ts in targets.items():
        cache = os.path.join(OUT, f'commat_scores_{ds}.csv')
        npz = os.path.join(OUT, f'commat_cdr_rmsd_{ds}.npz')
        if os.path.exists(cache) and os.path.exists(npz):
            scores[ds] = pd.read_csv(cache)
            z = np.load(npz)
            Dmats[ds] = {t: z[t] for t in z.files}
            continue
        with mp.Pool(args.workers) as pool:
            res = pool.map(functools.partial(score_target, dataset=ds), ts)
        scores[ds] = pd.DataFrame([r for rows, _, _ in res for r in rows])
        Dmats[ds] = {t: D for _, t, D in res}
        scores[ds].to_csv(cache, index=False)
        np.savez_compressed(npz, **Dmats[ds])
        print(f'[{ds}] scored {len(scores[ds])} ComMat models over {len(ts)} targets', flush=True)

    E = existing_decoys()
    seeds, ranges = [], []
    tools = {'test': ['AF3', 'Boltz2'], 'valid': ['Boltz2']}
    ladder_valid = []
    for ds, ts in targets.items():
        S = scores[ds]
        for tool in tools[ds]:
            for t in ts:
                e = E[(E.dataset == ds) & (E.tool == tool) & (E.target == t)].fv_context_cdr_lddt
                has_native_run = len(e) > 0
                if ds == 'valid' and not has_native_run:
                    ladder_valid.append(t)
                c = S[(S.target == t) & (S.seed_fnat >= 0.5)]
                mi = {m: i for i, m in enumerate(S[S.target == t].model)}
                used = []
                for lab, lo, hi in RANGES:
                    n_have = int(((e >= lo) & (e < hi)).sum())
                    if ds == 'valid' and not has_native_run and lab == '>= 0.95':
                        n_have = args.per_range   # the native-start ladder will cover it
                    empty = n_have < args.per_range
                    chosen = []
                    if empty and lab != '>= 0.95':
                        win = c[(c.seed_fv_context >= lo - 0.03) & (c.seed_fv_context < hi - 0.01)]
                        chosen = pick_seeds(win, Dmats[ds][t], mi, args.seeds_per_range, used)
                        used += chosen
                    n1, n2 = OUTPUTS.get(len(chosen), (0, 0))
                    for m in chosen:
                        r = c[c.model == m].iloc[0]
                        seeds.append({'dataset': ds, 'tool': tool, 'target': t, 'range': lab, 'model': m,
                                      'path': r.path, 'origin': 'ComMat', 'seed_fv_context': r.seed_fv_context,
                                      'seed_fnat': r.seed_fnat, 'n_seeds_in_range': len(chosen),
                                      'samples_1A': n1, 'samples_2A': n2, 'input_name': f'{t}__co{m}'})
                    ranges.append({'dataset': ds, 'tool': tool, 'target': t, 'range': lab, 'existing': n_have,
                                   'empty': empty, 'seeds': len(chosen), 'planned_outputs': len(chosen) * (n1 + n2),
                                   'commat_candidates_in_window': int(((c.seed_fv_context >= lo - 0.03) &
                                                                       (c.seed_fv_context < hi - 0.01)).sum())})
    SE, RA = pd.DataFrame(seeds), pd.DataFrame(ranges)
    SE.to_csv(os.path.join(OUT, 'seeds.csv'), index=False)
    RA.to_csv(os.path.join(OUT, 'ranges.csv'), index=False)
    with open(os.path.join(OUT, 'native_ladder_valid.txt'), 'w') as fh:
        fh.write(''.join(f'{t}\n' for t in sorted(set(ladder_valid))))
    for (ds, tool), g in SE.groupby(['dataset', 'tool']):
        rows = []
        for r in g.itertuples():
            rows += [{'target': r.input_name, 'tool': tool, 'range': r.range, 'strength': 1.0, 'n_generate': r.samples_1A,
                      'basis': 'commat_seed'},
                     {'target': r.input_name, 'tool': tool, 'range': r.range, 'strength': 2.0, 'n_generate': r.samples_2A,
                      'basis': 'commat_seed'}]
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f'plan_rows_{ds}_{tool}.csv'), index=False)

    with pd.option_context('display.width', 220):
        for (ds, tool), g in RA.groupby(['dataset', 'tool']):
            print(f'\n[{ds} {tool}] targets {g.target.nunique()}')
            print(g.groupby('range', sort=False).agg(targets_empty=('empty', 'sum'),
                                                     filled_by_seeds=('seeds', lambda x: int((x > 0).sum())),
                                                     seeds=('seeds', 'sum'), outputs=('planned_outputs', 'sum'),
                                                     one_seed_only=('seeds', lambda x: int((x == 1).sum()))).to_string())
            sg = SE[(SE.dataset == ds) & (SE.tool == tool)]
            print(f'  seeds {len(sg)} on {sg.target.nunique()} targets; seeds per target median '
                  f'{sg.groupby("target").size().median() if len(sg) else 0}; outputs {int(g.planned_outputs.sum())}')
        print(f'\nvalid targets needing the native-start ladder ({args.native_ladder} A x 2): {len(set(ladder_valid))}')


if __name__ == '__main__':
    main()
