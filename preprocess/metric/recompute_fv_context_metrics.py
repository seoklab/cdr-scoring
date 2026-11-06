#!/usr/bin/env python
"""Fv-context CDR-lDDT (+ CDR-context and CDR-internal) for the vanilla test decoys.

For every decoy in metrics_test_h3_v2/<source>/metrics/loop_metrics.parquet
(AF3 and Boltz2_test, holo and apo targets, no fnat filter) computes
preprocess/metric/fv_context_lddt.compute_cdr_lddt_variants and writes
  metrics_test_h3_v2/<source>/metrics/fv_context_metrics.parquet
keyed like loop_metrics.parquet (target_id, source, seed, sample, decoy_id).
cdr_context_lddt must equal the stored cdr_lddt (checked).

  python preprocess/metric/recompute_fv_context_metrics.py --workers 32
"""
import argparse
import functools
import multiprocessing as mp
import os
import sys
import warnings

import pandas as pd
from Bio.PDB import PDBParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from fv_context_lddt import compute_cdr_lddt_variants  # noqa: E402

ROOT = '/home/sujin/projects/cdr-scoring/cdr-data/metrics_test_h3_v2'
KEYS = ['target_id', 'source', 'seed', 'sample', 'decoy_id']


@functools.lru_cache(maxsize=4)
def _native(path):
    return PDBParser(QUIET=True).get_structure('n', path)


def work(item):
    try:
        model = PDBParser(QUIET=True).get_structure('d', item['decoy_path'])
        return {**{k: item[k] for k in KEYS}, 'cdr_lddt_stored': item['cdr_lddt'],
                **compute_cdr_lddt_variants(_native(item['native_path']), model)}
    except Exception as e:  # noqa: BLE001
        return {**{k: item[k] for k in KEYS}, 'error': repr(e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sources', nargs='*', default=['AF3', 'Boltz2_test'])
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    args = ap.parse_args()
    for src in args.sources:
        lo = pd.read_parquet(f'{ROOT}/{src}/metrics/loop_metrics.parquet',
                             columns=KEYS + ['native_path', 'decoy_path', 'cdr_lddt'])
        items = lo.sort_values(['target_id', 'seed', 'sample']).to_dict('records')
        print(f'[{src}] {len(items)} decoys, {lo.target_id.nunique()} targets', flush=True)
        with mp.Pool(args.workers) as pool:
            rows = list(pool.imap(work, items, chunksize=8))
        d = pd.DataFrame(rows)
        err = d[d['error'].notna()] if 'error' in d else d.iloc[:0]
        if len(err):
            print(f'  errors: {len(err)}', err.head(3).to_string())
            d = d[d['error'].isna()].drop(columns='error')
        dev = (d.cdr_context_lddt - d.cdr_lddt_stored).abs()
        print(f'  check: cdr_context_lddt vs stored cdr_lddt, max |diff| = {dev.max():.2e}')
        out = f'{ROOT}/{src}/metrics/fv_context_metrics.parquet'
        d.drop(columns='cdr_lddt_stored').to_parquet(out, index=False)
        print('  ->', out, flush=True)


if __name__ == '__main__':
    main()
