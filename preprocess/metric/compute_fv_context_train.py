#!/usr/bin/env python
"""Fv-context CDR-lDDT (+ CDR-context, CDR-internal, per-CDR) for the train/valid
multi-source decoys, read from the same Target pickles as precompute_source_metrics_fast
and keyed the same way (target_id, seed, sample, decoy_id), so the result joins with
the existing loop/interface/dockq parquets.

Per source the input is <metrics_dir>/<Source>/metrics/loop_metrics.parquet
(target_pickle_path per target). Shards write
<metrics_dir>/<Source>/fv_context_shards/shard_<k>.parquet; --merge concatenates them
into <metrics_dir>/<Source>/metrics/fv_context_metrics.parquet and checks
cdr_context_lddt == stored cdr_lddt.

  python preprocess/metric/compute_fv_context_train.py --source ComMat --shard 0 --n_shards 40
  python preprocess/metric/compute_fv_context_train.py --source ComMat --merge
"""
import argparse
import glob
import multiprocessing as mp
import os
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from precompute_source_metrics_fast import _decoy_identity, _load_target_pickle  # noqa: E402
from fv_context_lddt import compute_cdr_lddt_variants  # noqa: E402

DATA = '/home/sujin/projects/cdr-scoring/cdr-data'
SOURCES = {  # source -> metrics dir holding <Source>/metrics/loop_metrics.parquet
    'Boltz2_s10n10': 'metrics_precomputed_boltz2_s10n10_chothia_cdr',
    'Boltz2': 'metrics_precomputed_boltz2_cdr',
    'ComMat': 'metrics_precomputed_commat_cdr',
    'PertMD_cdr': 'metrics_precomputed_pertmd_cdr_cdr',
    'PertMD_all': 'metrics_precomputed_pertmd_all_cdr',
}


def do_target(args):
    target_id, pkl = args
    rows = []
    try:
        target = _load_target_pickle(pkl)
        native = target.gt_structure
        for pos, model in enumerate(target.models):
            ident = _decoy_identity(model, pos)
            try:
                v = compute_cdr_lddt_variants(native, model.md_structure)
            except Exception as e:  # noqa: BLE001
                v = {'error': repr(e)}
            rows.append({'target_id': target_id, 'seed': ident.seed, 'sample': ident.sample,
                         'decoy_id': ident.decoy_id, **v})
    except Exception as e:  # noqa: BLE001
        rows.append({'target_id': target_id, 'error': f'target: {e!r}'})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', required=True, choices=list(SOURCES))
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--n_shards', type=int, default=1)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    base = os.path.join(DATA, SOURCES[args.source], args.source)
    shard_dir = os.path.join(base, 'fv_context_shards')
    loop = pd.read_parquet(os.path.join(base, 'metrics', 'loop_metrics.parquet'),
                           columns=['target_id', 'seed', 'sample', 'decoy_id', 'target_pickle_path', 'cdr_lddt'])

    if args.merge:
        parts = sorted(glob.glob(os.path.join(shard_dir, 'shard_*.parquet')))
        d = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        err = d[d['error'].notna()] if 'error' in d else d.iloc[:0]
        d = d[d['error'].isna()].drop(columns='error') if 'error' in d else d
        m = d.merge(loop[['target_id', 'decoy_id', 'cdr_lddt']], on=['target_id', 'decoy_id'], how='left')
        dev = (m.cdr_context_lddt - m.cdr_lddt).abs()
        print(f'[{args.source}] shards {len(parts)}, rows {len(d)} (loop parquet {len(loop)}), errors {len(err)}, '
              f'unmatched {int(m.cdr_lddt.isna().sum())}, max |cdr_context - stored cdr_lddt| = {dev.max():.2e}')
        if len(err):
            print(err.head().to_string())
        out = os.path.join(base, 'metrics', 'fv_context_metrics.parquet')
        d.to_parquet(out, index=False)
        print('->', out)
        return

    targets = loop.drop_duplicates('target_id').sort_values('target_id')[['target_id', 'target_pickle_path']]
    mine = targets.iloc[args.shard::args.n_shards]
    os.makedirs(shard_dir, exist_ok=True)
    print(f'[{args.source}] shard {args.shard}/{args.n_shards}: {len(mine)} targets', flush=True)
    rows = []
    with mp.Pool(args.workers) as pool:
        for r in pool.imap_unordered(do_target, list(mine.itertuples(index=False, name=None))):
            rows.extend(r)
    pd.DataFrame(rows).to_parquet(os.path.join(shard_dir, f'shard_{args.shard:03d}.parquet'), index=False)
    print(f'done: {len(rows)} rows', flush=True)


if __name__ == '__main__':
    main()
