#!/usr/bin/env python
"""Write partial-diffusion decoys in native Chothia numbering, for scoring with a trained model.

AF3 / Boltz-2 outputs number residues 1..N; the scoring model finds CDRs by Chothia number on
chains H/L, so every decoy is renumbered with its input's resmap (same renumber() as the metric
evaluation) and saved as PDB:
    <out>/<tool>/<target>/<decoy key, ':' -> '__'>/<target>_model.pdb
Decoys come from analyze/reports/controlled_benchmark/<dataset>/candidates.csv (every candidate,
which includes the selected ones). Existing files are skipped.

  python preprocess/pdiff/export_chothia_decoys.py --dataset test
  -> score: python infer_structures.py --input-dir <out>/<tool> --checkpoint ... --score-only
     (path column -> target, decoy key)
"""
import argparse
import functools
import json
import multiprocessing as mp
import os
import sys
import warnings

import pandas as pd
from Bio.PDB import MMCIFParser, PDBIO

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
warnings.filterwarnings('ignore')
from eval_pdiff_sweep import renumber  # noqa: E402

BENCH = '/home/sujin/DB/h3-loop-modeling/ab_ag/3_h3_benchmark_after210930'
TRVAL = '/home/sujin/DB/h3-loop-modeling/ab_ag/1_tr_abag'
IN_DIR = {'t52': f'{BENCH}/51_pdiff_inputs', 't53': f'{BENCH}/51_pdiff_inputs', 't54': f'{BENCH}/51_pdiff_inputs',
          't55': f'{BENCH}/51_pdiff_inputs', 't56': f'{BENCH}/51_pdiff_inputs_apo', 't57': f'{BENCH}/51_pdiff_inputs_apo',
          'v52': f'{TRVAL}/51_pdiff_inputs', 'v53': f'{TRVAL}/51_pdiff_inputs', 'v54': f'{TRVAL}/51_pdiff_inputs',
          'v55': f'{TRVAL}/51_pdiff_inputs', 'v56': f'{TRVAL}/51_pdiff_inputs_apo', 'v57': f'{TRVAL}/51_pdiff_inputs_apo'}
OUT = {'test': f'{BENCH}/59_pdiff_scoring_pdb', 'valid': f'{TRVAL}/60_pdiff_scoring_pdb'}


def key_dir(key):
    return key.replace(':', '__')


@functools.lru_cache(maxsize=256)
def resmap_of(in_dir, input_name):
    return json.load(open(os.path.join(in_dir, 'resmap', f'{input_name}.json')))['resmap']


def export(rows, out):
    done = err = 0
    for r in rows:
        dst = os.path.join(out, r['tool'], r['target'], key_dir(r['decoy_key']), f"{r['target']}_model.pdb")
        if os.path.exists(dst):
            done += 1
            continue
        try:
            raw = MMCIFParser(QUIET=True).get_structure('decoy', r['path'])
            dec = renumber(raw, resmap_of(IN_DIR[r['run']], r['input_name']))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            io = PDBIO()
            io.set_structure(dec)
            io.save(dst + '.tmp')
            os.replace(dst + '.tmp', dst)
            done += 1
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {r['decoy_key']}: {e!r}", flush=True)
            err += 1
    return done, err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=['test', 'valid'], required=True)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    args = ap.parse_args()
    C = pd.read_csv(os.path.join(CODE_ROOT, 'analyze', 'reports', 'controlled_benchmark', args.dataset, 'candidates.csv'))
    out = OUT[args.dataset]
    # one chunk per input so the resmap cache is reused
    chunks = [g[['tool', 'target', 'decoy_key', 'run', 'input_name', 'path']].to_dict('records')
              for _, g in C.groupby('input_name')]
    with mp.Pool(args.workers) as pool:
        res = pool.map(functools.partial(export, out=out), chunks, chunksize=4)
    print(f'{args.dataset}: {sum(d for d, _ in res)} written or present, {sum(e for _, e in res)} failed -> {out}')
    # map file -> decoy for reading score dumps back
    C.assign(pdb=[os.path.join(out, t, g, key_dir(k), f'{g}_model.pdb')
                  for t, g, k in zip(C.tool, C.target, C.decoy_key)])[['tool', 'target', 'decoy_key', 'pdb']] \
        .to_csv(os.path.join(out, 'index.csv'), index=False)


if __name__ == '__main__':
    main()
