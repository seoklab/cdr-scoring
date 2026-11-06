#!/usr/bin/env python
"""Turn Boltz-2 partial-diffusion decoys into the training source 'Boltz2_pdiff'.

Train targets : every decoy of 58_pdiff_train_native + 59_pdiff_train_commat_seeded
                (pose gate is left to the data loader, like every other source)
Valid targets : the controlled-validation candidates
                (analyze/reports/controlled_benchmark/valid/candidates.csv)
Per decoy: renumber to native Chothia (same renumber() as the metric evaluation), write
    61_boltz2_pdiff_chothia/<target>/<target>_sd-<seed>_sp-<sample>.pdb
where (seed, sample) is unique inside a target: seed = 1000 + index of the
(run, input structure, strength, round) group, sample = Boltz-2 model index.
Per target: a Target pickle (data_loading.pdb2dict) in 62_boltz2_pdiff_pdb2dict/<target>.pkl.
Labels (already computed by eval_pdiff_sweep.py) are written as the usual parquets in
    cdr-data/metrics_precomputed_boltz2_pdiff/Boltz2_pdiff/metrics/
      loop_metrics, fv_context_metrics, interface_metrics, dockq_metrics (.parquet)
      decoy_map.csv  (seed, sample) <-> run, input structure, seed_origin, strength, decoy_key
seed_origin (native / ComMat) is metadata only.

  python preprocess/pdiff/build_pdiff_source.py --workers 32
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import warnings
from pathlib import Path

import pandas as pd
from Bio.PDB import MMCIFParser, PDBIO

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(CODE, 'libs'))
warnings.filterwarnings('ignore')
from eval_pdiff_sweep import renumber  # noqa: E402
from data_loading.pdb2dict import Model, Target  # noqa: E402

TRVAL = '/home/sujin/DB/h3-loop-modeling/ab_ag/1_tr_abag'
PDB_DIR = f'{TRVAL}/61_boltz2_pdiff_chothia'
PKL_DIR = f'{TRVAL}/62_boltz2_pdiff_pdb2dict'
METRICS = '/home/sujin/projects/cdr-scoring/cdr-data/metrics_precomputed_boltz2_pdiff/Boltz2_pdiff/metrics'
SOURCE = 'Boltz2_pdiff'
TRAIN_RUNS = {'r58': f'{TRVAL}/58_pdiff_train_native', 'r59': f'{TRVAL}/59_pdiff_train_commat_seeded'}
VALID_IN = {'v52': '51_pdiff_inputs', 'v53': '51_pdiff_inputs', 'v54': '51_pdiff_inputs', 'v55': '51_pdiff_inputs',
            'v56': '51_pdiff_inputs_apo', 'v57': '51_pdiff_inputs_apo'}
LOOPS = ['H1', 'H2', 'H3', 'L1', 'L2', 'L3']


def load_decoys():
    parts = []
    for run, d in TRAIN_RUNS.items():
        e = pd.read_csv(f'{d}/eval/decoys.csv')
        parts.append(e.assign(run=run, in_dir=f'{TRVAL}/51_pdiff_inputs_train', split='train'))
    v = pd.read_csv(os.path.join(CODE, 'analyze', 'reports', 'controlled_benchmark', 'valid', 'candidates.csv'))
    keys = set(v.decoy_key)
    for run, sub in VALID_IN.items():
        p = {'v52': '52_pdiff_pilot', 'v53': '53_pdiff_seeded_pilot', 'v54': '54_pdiff_commat_seeded',
             'v55': '55_pdiff_native_ladder', 'v56': '56_pdiff_apo_native', 'v57': '57_pdiff_apo_commat_seeded'}[run]
        e = pd.read_csv(f'{TRVAL}/{p}/eval/decoys.csv').assign(run=run, in_dir=f'{TRVAL}/{sub}', split='valid')
        parts.append(e)
    D = pd.concat(parts, ignore_index=True)
    D['input_name'] = D.input_name.fillna(D.target)
    D['seed_origin'] = D.seed_origin.fillna('native')
    D['round_dir'] = D.path.str.extract(r'boltz_results_([^/]+)/', expand=False)
    D['decoy_key'] = (D.run + ':' + D.input_name + ':s' + D.strength.map(lambda s: f'{s:g}') + ':'
                      + D.round_dir.astype(str) + ':' + D['sample'].astype(str))
    D = D[(D.split == 'train') | D.decoy_key.isin(keys)].copy()
    # unique (seed, sample) inside each target
    grp = D[['target', 'run', 'input_name', 'strength', 'round_dir']].drop_duplicates() \
        .sort_values(['target', 'run', 'input_name', 'strength', 'round_dir'])
    grp['seed'] = 1000 + grp.groupby('target').cumcount()
    D = D.merge(grp, on=['target', 'run', 'input_name', 'strength', 'round_dir'])
    D['sample'] = D['sample'].astype(int)
    assert not D.duplicated(['target', 'seed', 'sample']).any()
    D['decoy_path'] = [f'{PDB_DIR}/{t}/{t}_sd-{s}_sp-{k}.pdb' for t, s, k in zip(D.target, D.seed, D['sample'])]
    D['target_pickle_path'] = [f'{PKL_DIR}/{t}.pkl' for t in D.target]
    return D


def native_of(in_dir, input_name, target):
    rm = json.load(open(os.path.join(in_dir, 'resmap', f'{input_name}.json')))
    return rm['resmap'], rm.get('native', f'{TRVAL}/04_xtal_chothia_ag_renum/{target}.pdb')


def build_target(job):
    t, rows = job
    out = Path(PKL_DIR) / f'{t}.pkl'
    try:
        native = None
        models = []
        for r in rows:
            resmap, native = native_of(r['in_dir'], r['input_name'], t)
            dst = Path(r['decoy_path'])
            if not dst.exists():
                dec = renumber(MMCIFParser(QUIET=True).get_structure('decoy', r['path']), resmap)
                dst.parent.mkdir(parents=True, exist_ok=True)
                io = PDBIO()
                io.set_structure(dec)
                io.save(str(dst) + '.tmp')
                os.replace(str(dst) + '.tmp', dst)
            from Bio.PDB import PDBParser
            st = PDBParser(QUIET=True).get_structure(dst.stem, str(dst))
            m = Model(method=SOURCE, md_structure=st, pdb_path=dst, ranking_score=float(r['ranking_score']),
                      confidence_score=float(r['ranking_score']), iptm=float(r['iptm']))
            setattr(m, 'seed', int(r['seed']))
            setattr(m, 'model_idx', int(r['sample']))
            models.append(m)
        target = Target(pdb_id=t, pdb_path=Path(native), db_dir=Path(PDB_DIR) / t)
        order = sorted(range(len(models)), key=lambda i: (-models[i].ranking_score if models[i].ranking_score == models[i].ranking_score else 0))
        for rank, i in enumerate(order, start=1):
            models[i].ranking = rank
        target.models = models
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(str(out) + '.tmp', 'wb') as fh:
            pickle.dump(target, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(str(out) + '.tmp', out)
        return t, len(models), native, ''
    except Exception as e:  # noqa: BLE001
        return t, 0, native, f'{type(e).__name__}: {e}'[:300]


def write_parquets(D, natives):
    os.makedirs(METRICS, exist_ok=True)
    D = D.assign(source=SOURCE, decoy_id='seed=' + D.seed.astype(str) + ':sample=' + D['sample'].astype(str),
                 native_path=D.target.map(natives), ranking=D.groupby('target').ranking_score.rank(ascending=False, method='first'))
    base = ['target_id', 'source', 'seed', 'sample', 'decoy_id', 'native_path', 'decoy_path', 'target_pickle_path',
            'ranking', 'ranking_score']
    D = D.rename(columns={'target': 'target_id'})
    loop = D[base + ['cdr_rmsd', 'cdr_lddt'] + [f'{c}_loop_rmsd' for c in LOOPS] + [f'{c}_loop_lddt' for c in LOOPS]] \
        .assign(missing_backbone_atom_count=0, missing_backbone_report='')
    loop.to_parquet(f'{METRICS}/loop_metrics.parquet', index=False)
    fv = ['cdr_context_lddt', 'fv_context_cdr_lddt', 'cdr_internal_lddt'] + [f'{c}_fv_context_lddt' for c in LOOPS] + \
        ['n_pairs_cdr_context', 'n_pairs_fv_context', 'n_pairs_cdr_internal']
    D[['target_id', 'seed', 'sample', 'decoy_id'] + fv].to_parquet(f'{METRICS}/fv_context_metrics.parquet', index=False)
    D[base + ['fnat', 'irmsd']].assign(lrmsd=float('nan')).to_parquet(f'{METRICS}/interface_metrics.parquet', index=False)
    D[base + ['dockq', 'fnat', 'irmsd']].assign(lrmsd=float('nan')).to_parquet(f'{METRICS}/dockq_metrics.parquet', index=False)
    D[['target_id', 'seed', 'sample', 'split', 'run', 'input_name', 'seed_origin', 'seed_model', 'strength', 'round_dir',
       'decoy_key', 'path', 'decoy_path', 'held_rmsd']].to_csv(f'{METRICS}/decoy_map.csv', index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    ap.add_argument('--targets', nargs='*', help='only these targets (smoke test)')
    args = ap.parse_args()
    D = load_decoys()
    if args.targets:
        D = D[D.target.isin(args.targets)]
    print(f'{len(D)} decoys, {D.target.nunique()} targets (train {D[D.split == "train"].target.nunique()}, '
          f'valid {D[D.split == "valid"].target.nunique()})', flush=True)
    cols = ['in_dir', 'input_name', 'path', 'decoy_path', 'seed', 'sample', 'ranking_score', 'iptm']
    jobs = [(t, g[cols].to_dict('records')) for t, g in D.groupby('target')]
    with mp.Pool(args.workers) as pool:
        res = pool.map(build_target, jobs, chunksize=2)
    bad = [r for r in res if r[3]]
    for t, _, _, err in bad[:20]:
        print(f'[error] {t}: {err}')
    ok = {t for t, n, _, err in res if not err}
    natives = {t: nat for t, _, nat, _ in res}
    D = D[D.target.isin(ok)]
    write_parquets(D, natives)
    print(f'targets ok {len(ok)}, failed {len(bad)}; decoys {len(D)} -> {PKL_DIR}, {METRICS}')


if __name__ == '__main__':
    main()
