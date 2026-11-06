#!/usr/bin/env python
"""Round-2 partial-diffusion plan (no generation): training targets, and apo targets of the
test and validation sets. Same rules as plan_commat_seeds.py (the executed holo test/valid plan).

Datasets
  train      all training targets (holo + apo), Boltz-2. Existing structures = the generative
             Boltz-2 decoys already in the training data (Boltz2_s10n10 + Boltz2 full), pose-valid
             (holo fnat >= 0.5; apo all, no antigen). Goal: Boltz-2 structural diversity per target.
  test_apo   43 apo test targets, AF3 and Boltz-2. No partial diffusion exists yet.
  valid_apo  72 apo validation targets, Boltz-2. No partial diffusion exists yet.
Per target x tool x Fv-context CDR-lDDT range (< 0.75 | 0.75-0.85 | 0.85-0.90 | 0.90-0.95 | >= 0.95):
  existing count; a range is empty if it has fewer than --per_range structures.
  >= 0.95 empty  -> native-start ladder (--native_ladder A x 2 samples)
  other empty    -> up to --seeds_per_range CDR-diverse ComMat seeds in [lo - 0.03, hi - 0.01)
                    (holo: seed fnat >= 0.5; apo: no fnat), outputs per seed as before
                    (3 seeds: 1 A x2 + 2 A x1; 2: 1 A x3 + 2 A x1; 1: 1 A x4 + 2 A x2).

Writes analyze/reports/commat_seed_plan/round2/
  targets.csv  one row per dataset x tool x target: existing per range, empty ranges, seeds,
               planned outputs, native ladder yes/no   <- the per-target overview
  ranges.csv, seeds.csv, plan_rows_<ds>_<tool>.csv, native_ladder_<ds>_<tool>.txt,
  commat_scores_<ds>.csv / commat_cdr_rmsd_<ds>.npz (cache)
"""
import argparse
import functools
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plan_commat_seeds import OUTPUTS, RANGES, TRVAL, BENCH, CODE, pick_seeds, score_target  # noqa: E402

OUT = os.path.join(CODE, 'analyze', 'reports', 'commat_seed_plan', 'round2')
PATH_KEY = {'train': 'valid', 'valid_apo': 'valid', 'test_apo': 'test'}   # which native/ComMat roots
TOOLS = {'train': ['Boltz2'], 'test_apo': ['AF3', 'Boltz2'], 'valid_apo': ['Boltz2']}
VANILLA_TRAIN = {'Boltz2_s10n10': 'metrics_precomputed_boltz2_s10n10_chothia_cdr', 'Boltz2': 'metrics_precomputed_boltz2_cdr'}
DATA = '/home/sujin/projects/cdr-scoring/cdr-data'


def target_lists():
    tr = pickle.load(open(f'{TRVAL}/0_info/train.pkl', 'rb'))['list']
    va = pickle.load(open(f'{TRVAL}/0_info/valid.pkl', 'rb'))['list']
    ok_trval = lambda t: (os.path.exists(f'{TRVAL}/141_boltz2_s10n10_yaml/{t}.yaml')
                          and os.path.exists(f'{TRVAL}/14_boltz2_s10n10/{t}/seed_42/boltz_results_{t}/predictions/{t}/{t}_model_0.cif')
                          and os.path.exists(f'{TRVAL}/21_commat_xtal/{t}/{t}.pdb'))
    test_all = [d for d in os.listdir(f'{BENCH}/22_af3') if os.path.isdir(f'{BENCH}/22_af3/{d}') and d != d.lower()]
    return {'train': sorted(t for t in tr if ok_trval(t)),
            'test_apo': sorted(t for t in test_all if t.count('_') == 2 and os.path.exists(f'{BENCH}/11_commat_xtal/{t}/{t}.pdb')),
            'valid_apo': sorted(t for t in va if t.count('_') == 2 and ok_trval(t))}


def existing(ds):
    """Fv-context of the structures already available for planning (pose-valid)."""
    if ds != 'train':
        return pd.DataFrame(columns=['tool', 'target', 'fv_context_cdr_lddt'])
    out = []
    for src, d in VANILLA_TRAIN.items():
        base = f'{DATA}/{d}/{src}/metrics'
        f = pd.read_parquet(f'{base}/fv_context_metrics.parquet', columns=['target_id', 'decoy_id', 'fv_context_cdr_lddt'])
        fi = pd.read_parquet(f'{base}/interface_metrics.parquet', columns=['target_id', 'decoy_id', 'fnat'])
        m = f.merge(fi, on=['target_id', 'decoy_id'], how='left')
        holo = m.target_id.str.count('_') == 3
        m = m[~holo | (m.fnat >= 0.5)]
        out.append(m.rename(columns={'target_id': 'target'}).assign(tool='Boltz2', vanilla_source=src))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='*', default=['train', 'test_apo', 'valid_apo'])
    ap.add_argument('--per_range', type=int, default=5)
    ap.add_argument('--seeds_per_range', type=int, default=3)
    ap.add_argument('--native_ladder', default='1 2 4 8 16')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    lists = target_lists()
    n_ladder = 2 * len(args.native_ladder.split())

    seeds, ranges, trows = [], [], []
    for ds in args.datasets:
        ts = lists[ds]
        cache, npz = os.path.join(OUT, f'commat_scores_{ds}.csv'), os.path.join(OUT, f'commat_cdr_rmsd_{ds}.npz')
        if os.path.exists(cache) and os.path.exists(npz):
            S = pd.read_csv(cache)
            z = np.load(npz)
            Dm = {t: z[t] for t in z.files}
        else:
            with mp.Pool(args.workers) as pool:
                res = pool.map(functools.partial(score_target, dataset=PATH_KEY[ds]), ts, chunksize=4)
            S = pd.DataFrame([r for rows, _, _ in res for r in rows]).assign(dataset=ds)
            Dm = {t: D for _, t, D in res}
            S.to_csv(cache, index=False)
            np.savez_compressed(npz, **Dm)
            print(f'[{ds}] scored {len(S)} ComMat models over {len(ts)} targets', flush=True)
        E = existing(ds)
        for tool in TOOLS[ds]:
            for t in ts:
                holo = t.count('_') == 3
                e = E[(E.tool == tool) & (E.target == t)].fv_context_cdr_lddt
                st = S[S.target == t]
                c = st[st.seed_fnat >= 0.5] if holo else st
                mi = {m: i for i, m in enumerate(st.model)}
                used, tr = [], {'dataset': ds, 'tool': tool, 'target': t, 'holo': holo, 'existing_total': len(e)}
                ladder = False
                for lab, lo, hi in RANGES:
                    n_have = int(((e >= lo) & (e < hi)).sum())
                    empty = n_have < args.per_range
                    chosen = []
                    if empty and lab == '>= 0.95':
                        ladder = True
                    elif empty:
                        win = c[(c.seed_fv_context >= lo - 0.03) & (c.seed_fv_context < hi - 0.01)]
                        chosen = pick_seeds(win, Dm[t], mi, args.seeds_per_range, used)
                        used += chosen
                    n1, n2 = OUTPUTS.get(len(chosen), (0, 0))
                    for m in chosen:
                        r = c[c.model == m].iloc[0]
                        seeds.append({'dataset': ds, 'tool': tool, 'target': t, 'range': lab, 'model': m, 'path': r.path,
                                      'origin': 'ComMat', 'seed_fv_context': r.seed_fv_context, 'seed_fnat': r.seed_fnat,
                                      'n_seeds_in_range': len(chosen), 'samples_1A': n1, 'samples_2A': n2,
                                      'input_name': f'{t}__co{m}'})
                    n_cand = int(((c.seed_fv_context >= lo - 0.03) & (c.seed_fv_context < hi - 0.01)).sum())
                    ranges.append({'dataset': ds, 'tool': tool, 'target': t, 'holo': holo, 'range': lab, 'existing': n_have,
                                   'empty': empty, 'seeds': len(chosen), 'planned_outputs': len(chosen) * (n1 + n2),
                                   'commat_candidates_in_window': n_cand})
                    tr[f'existing[{lab}]'] = n_have
                    tr[f'seeds[{lab}]'] = len(chosen)
                tr['empty_ranges'] = ' | '.join(r['range'] for r in ranges[-len(RANGES):] if r['empty'])
                tr['unfilled_ranges'] = ' | '.join(r['range'] for r in ranges[-len(RANGES):]
                                                   if r['empty'] and r['range'] != '>= 0.95' and r['seeds'] == 0)
                tr['native_ladder'] = ladder
                tr['planned_seeded_outputs'] = sum(r['planned_outputs'] for r in ranges[-len(RANGES):])
                tr['planned_native_outputs'] = n_ladder if ladder else 0
                trows.append(tr)
    SE, RA, TT = pd.DataFrame(seeds), pd.DataFrame(ranges), pd.DataFrame(trows)
    SE.to_csv(os.path.join(OUT, 'seeds.csv'), index=False)
    RA.to_csv(os.path.join(OUT, 'ranges.csv'), index=False)
    TT.to_csv(os.path.join(OUT, 'targets.csv'), index=False)
    for (ds, tool), g in TT.groupby(['dataset', 'tool']):
        with open(os.path.join(OUT, f'native_ladder_{ds}_{tool}.txt'), 'w') as fh:
            fh.write(''.join(f'{t}\n' for t in g[g.native_ladder].target))
    for (ds, tool), g in SE.groupby(['dataset', 'tool']):
        rows = []
        for r in g.itertuples():
            rows += [{'target': r.input_name, 'tool': tool, 'range': r.range, 'strength': 1.0, 'n_generate': r.samples_1A, 'basis': 'commat_seed'},
                     {'target': r.input_name, 'tool': tool, 'range': r.range, 'strength': 2.0, 'n_generate': r.samples_2A, 'basis': 'commat_seed'}]
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f'plan_rows_{ds}_{tool}.csv'), index=False)

    with pd.option_context('display.width', 220):
        for (ds, tool), g in RA.groupby(['dataset', 'tool']):
            tt = TT[(TT.dataset == ds) & (TT.tool == tool)]
            print(f'\n[{ds} {tool}] targets {tt.target.nunique()} (holo {int(tt.holo.sum())}, apo {int((~tt.holo).sum())})')
            print(g.groupby('range', sort=False).agg(targets_empty=('empty', 'sum'),
                                                     filled_by_seeds=('seeds', lambda x: int((x > 0).sum())),
                                                     seeds=('seeds', 'sum'), outputs=('planned_outputs', 'sum')).to_string())
            print(f'  native ladder: {int(tt.native_ladder.sum())} targets ({int(tt.planned_native_outputs.sum())} outputs); '
                  f'seeded outputs {int(tt.planned_seeded_outputs.sum())}; targets needing nothing: '
                  f'{int(((tt.planned_seeded_outputs == 0) & ~tt.native_ladder).sum())}')


if __name__ == '__main__':
    main()
