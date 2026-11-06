#!/usr/bin/env python
"""Score the region-restricted partial-diffusion sweep against the native.

Per decoy (tool x strength x target x sample):
  cdr_lddt, cdr_rmsd, <CDR>_loop_*  - the stored project metric (compute_loop_row), now called
                                      CDR-context lDDT (= cdr_context_lddt): partners = all H/L
                                      + antigen backbone within 15 A; + per CDR
  fv_context_cdr_lddt, <CDR>_fv_context_lddt, cdr_internal_lddt
                                    - same scored CDR atoms, partners = CDR + non-CDR Fv only
                                      (no antigen / constant domain), or CDR only
                                      (preprocess/metric/fv_context_lddt.py)
  fnat, irmsd                       - compute_interface_rows (5 A contact, 10 A interface)
  held_rmsd                         - heavy atoms outside the flexible region vs the
                                      init structure, after superposition on them
                                      (~0 when fix_mode=replace works). Symmetry-
                                      equivalent atom pairs (Arg NH1/NH2, Asp OD1/OD2,
                                      ...) are skipped: Boltz-2 may write them swapped.
  iptm, ranking_score               - generator confidence
Decoys are renumbered to the native Chothia numbering with resmap/<T>.json
before scoring, so the numbers are comparable to the existing test decoys.

  python preprocess/pdiff/eval_pdiff_sweep.py --sweep_dir .../52_pdiff_sweep
Writes <sweep_dir>/eval/decoys.csv and summary_by_strength.csv. No fnat filter
is applied here; summary columns state it where they use one.
"""
import argparse
import functools
import glob
import json
import multiprocessing as mp
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODE_ROOT)
sys.path.insert(0, os.path.join(CODE_ROOT, 'preprocess', 'metric'))
warnings.filterwarnings('ignore')
from precompute_source_metrics_fast import compute_interface_rows, compute_loop_row  # noqa: E402
from fv_context_lddt import compute_cdr_lddt_variants  # noqa: E402

BENCH = '/home/sujin/DB/h3-loop-modeling/ab_ag/3_h3_benchmark_after210930'
BINS = [(-np.inf, 0.75, '< 0.75'), (0.75, 0.85, '0.75-0.85'), (0.85, 0.90, '0.85-0.90'),
        (0.90, 0.95, '0.90-0.95'), (0.95, np.inf, '>= 0.95')]


def renumber(structure, resmap):
    """Model with sequential numbering -> new structure in native numbering."""
    out = Structure('decoy')
    model = Model(0)
    out.add(model)
    for chain in next(structure.get_models()):
        labels = resmap.get(chain.id)
        if labels is None:
            continue
        new_chain = Chain(chain.id)
        for i, res in enumerate(r for r in chain if r.id[0] == ' '):
            lab = labels[i] if i < len(labels) else None
            if lab is None:
                continue
            m = re.match(r'(-?\d+)([A-Za-z]?)$', lab)
            res = res.copy()
            res.id = (' ', int(m.group(1)), m.group(2) or ' ')
            new_chain.add(res)
        model.add(new_chain)
    return out


def read_init_held(init_cif):
    """(chain, seq, atom) -> xyz for held atoms (B-factor 0) of the init cif."""
    held, cols = {}, []
    for line in open(init_cif):
        if line.startswith('_atom_site.'):
            cols.append(line.split('.', 1)[1].strip())
        elif line.startswith('ATOM'):
            f = line.split()
            c = {k: j for j, k in enumerate(cols)}
            if float(f[c['B_iso_or_equiv']]) == 0.0:
                held[(f[c['label_asym_id']], int(f[c['label_seq_id']]), f[c['label_atom_id']].strip('"'))] = \
                    np.array([float(f[c['Cartn_x']]), float(f[c['Cartn_y']]), float(f[c['Cartn_z']])])
    return held


SYMMETRIC_ATOMS = {'ARG': {'NH1', 'NH2'}, 'ASP': {'OD1', 'OD2'}, 'GLU': {'OE1', 'OE2'},
                   'PHE': {'CD1', 'CD2', 'CE1', 'CE2'}, 'TYR': {'CD1', 'CD2', 'CE1', 'CE2'}}


def held_rmsd(structure, held):
    a, b = [], []
    for chain in next(structure.get_models()):
        for i, res in enumerate(r for r in chain if r.id[0] == ' '):
            skip = SYMMETRIC_ATOMS.get(res.get_resname(), ())
            for atom in res:
                key = (chain.id, i + 1, atom.get_id())
                if key in held and atom.get_id() not in skip:
                    a.append(atom.coord)
                    b.append(held[key])
    a, b = np.array(a, float), np.array(b, float)
    if len(a) < 3:
        return float('nan'), len(a)
    ac, bc = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(ac.T @ bc)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return float(np.sqrt((((ac @ r.T) - bc) ** 2).sum(1).mean())), len(a)


def find_decoys(sweep_dir):
    """Yield (tool, strength, target, sample, model_path, confidence_path)."""
    for sdir in sorted(glob.glob(os.path.join(sweep_dir, 'af3', 's*'))):
        s = float(os.path.basename(sdir)[1:])
        for p in sorted(glob.glob(os.path.join(sdir, '*', 'seed-*_sample-*', '*_model.cif'))):
            m = re.search(r'seed-(\d+)_sample-(\d+)', p)
            t = os.path.basename(os.path.dirname(os.path.dirname(p)))
            conf = p.replace('_model.cif', '_summary_confidences.json')
            yield 'AF3', s, t, f'{m.group(1)}-{m.group(2)}', p, conf
    for sdir in sorted(glob.glob(os.path.join(sweep_dir, 'boltz2', 's*'))):
        s = float(os.path.basename(sdir)[1:])
        for p in sorted(glob.glob(os.path.join(sdir, 'boltz_results_*', 'predictions', '*', '*_model_*.cif'))):
            t = os.path.basename(os.path.dirname(p))
            k = re.search(r'_model_(\d+)\.cif$', p).group(1)
            conf = os.path.join(os.path.dirname(p), f'confidence_{t}_model_{k}.json')
            yield 'Boltz2', s, t, k, p, conf


@functools.lru_cache(maxsize=None)
def _target_refs(t, in_dir):
    rm = json.load(open(os.path.join(in_dir, 'resmap', f'{t}.json')))
    resmap = rm['resmap']
    # inputs written since the train/valid support store the native path; older test inputs do not
    native_path = rm.get('native', f'{BENCH}/02_xtal_chothia_w_ag_renum/{t}.pdb')
    native = PDBParser(QUIET=True).get_structure('native', native_path)
    held = read_init_held(os.path.join(in_dir, 'init', f'{t}.cif'))
    meta = {'target': rm.get('target', t), 'input_name': t, 'seed_origin': rm.get('seed_origin', 'native'),
            'seed_model': rm.get('seed_model')}
    return native, resmap, held, meta


def score_decoy(item, in_dir):
    try:
        return _score_decoy(item, in_dir)
    except Exception as e:  # noqa: BLE001  (e.g. a cif still being written by a running job)
        return {'error': f'{item[4]}: {e!r}'}


def _score_decoy(item, in_dir):
    tool, s, t, sample, path, conf = item
    native, resmap, held, meta = _target_refs(t, in_dir)
    raw = MMCIFParser(QUIET=True).get_structure('decoy', path)
    dec = renumber(raw, resmap)
    row = {'tool': tool, 'strength': s, 'sample': sample, 'path': path, **meta}
    loop = compute_loop_row(native, dec)
    row.update({k: v for k, v in loop.items() if k.endswith(('_lddt', '_rmsd'))})  # cdr + per-CDR
    # cdr_lddt above = CDR-context lDDT; add Fv-context CDR-lDDT (+ per CDR) and CDR-internal
    row.update(compute_cdr_lddt_variants(native, dec))
    _iface, dockq = compute_interface_rows(native, dec, contact_cutoff_a=5.0, interface_cutoff_a=10.0)
    row.update({'fnat': dockq['fnat'], 'irmsd': dockq['irmsd'], 'dockq': dockq['dockq']})
    row['held_rmsd'], row['n_held_atoms'] = held_rmsd(raw, held)
    if os.path.exists(conf):
        c = json.load(open(conf))
        row['iptm'] = c.get('iptm')
        row['ranking_score'] = c.get('ranking_score', c.get('confidence_score'))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sweep_dir', default=f'{BENCH}/52_pdiff_sweep')
    ap.add_argument('--in_dir', default=f'{BENCH}/51_pdiff_inputs')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    args = ap.parse_args()

    # target-major order so each worker reuses its parsed native
    items = sorted(find_decoys(args.sweep_dir), key=lambda x: (x[2], x[0], x[1], x[3]))
    print(f'{len(items)} decoys, {args.workers} workers', flush=True)
    rows = []
    with mp.Pool(args.workers) as pool:
        for row in pool.imap_unordered(functools.partial(score_decoy, in_dir=args.in_dir), items, chunksize=4):
            if 'error' in row:
                print('[skip]', row['error'], flush=True)
                continue
            rows.append(row)
            print(f"{row['tool']:6s} s={row['strength']:<6g} {row['target']:14s} {row['sample']:5s} "
                  f"cdr_lddt={row['cdr_lddt']:.3f} fnat={row['fnat']:.2f} held_rmsd={row['held_rmsd']:.3f}",
                  flush=True)
    rows.sort(key=lambda r: (r['tool'], r['strength'], r['target'], r['sample']))

    out = os.path.join(args.sweep_dir, 'eval')
    os.makedirs(out, exist_ok=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(out, 'decoys.csv'), index=False)

    def summarize(g):
        r = {'n_decoys': len(g), 'n_targets': g.target.nunique(),
             'cdr_lddt_median': g.cdr_lddt.median(),
             'cdr_lddt_within_target_range_median': g.groupby('target').cdr_lddt.agg(lambda x: x.max() - x.min()).median(),
             'fnat_median': g.fnat.median(), 'frac_fnat_gt_0.5': (g.fnat > 0.5).mean(),
             'held_rmsd_max': g.held_rmsd.max()}
        for lo, hi, lab in BINS:
            r[f'frac[{lab}] (fnat filter off)'] = ((g.cdr_lddt >= lo) & (g.cdr_lddt < hi)).mean()
        return pd.Series(r)

    summ = d.groupby(['tool', 'strength']).apply(summarize).reset_index()
    summ.to_csv(os.path.join(out, 'summary_by_strength.csv'), index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(summ.round(3).to_string(index=False))
    print(f'-> {out}/decoys.csv, summary_by_strength.csv')


if __name__ == '__main__':
    main()
