#!/usr/bin/env python
"""Residue map (Protenix sequence index -> native Chothia label) for every Protenix target.

Protenix numbers residues 1..N per chain. The scoring model finds CDRs by Chothia number on
H/L, and the metric code matches decoy to native residues by (chain, residue number). Each
chain of the Protenix input JSON is globally aligned to the native chain
(04_xtal_chothia_ag_renum) and every sequence position takes the native residue label,
antigen chains included, so decoy and native residue keys agree across native numbering gaps.
Positions with no aligned native residue get null and are dropped on renumbering.

Differs from 15_boltz2_s10n10_chothia_w_ag_renum, which numbered antigen chains 1..N (out of
step with the native after the first numbering gap) and fell back to 1..N on a few H/L chains;
the hl_eq_15 / ag_eq_15 columns record where.

Writes <root>/resmap/<target>.json and <root>/resmap_summary.csv.

  python preprocess/protenix/build_protenix_resmap.py
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pdiff'))
from prepare_pdiff_inputs import THREE_TO_ONE, align_chain, read_native  # noqa: E402

DB = '/data/galaxy4/user/sujin/DB/h3-loop-modeling/ab_ag/1_tr_abag'
ROOT = '/data/galaxy4/user/sujin/projects/cdr-scoring/cdr-data/protenix/train_valid'


def json_chains(path):
    """Protenix input JSON -> ordered {chain id: sequence}."""
    data = json.load(open(path))
    data = data[0] if isinstance(data, list) else data
    seqs = {}
    for entity in data['sequences']:
        (kind, v), = entity.items()
        if kind != 'proteinChain':
            raise ValueError(f'{path}: non-protein entity {kind}')
        for cid in v['id']:
            seqs[cid] = v['sequence']
    return seqs


def pdb_labels(path):
    """PDB -> {chain: [residue label]} in file order."""
    out = {}
    for line in open(path):
        if line.startswith(('ATOM', 'HETATM')):
            lab = line[22:26].strip() + line[26].strip()
            lst = out.setdefault(line[21], [])
            if not lst or lst[-1] != lab:
                lst.append(lab)
    return out


def build(t, args):
    seqs = json_chains(f'{args.root}/json/{t}.json')
    native = f'{args.native_dir}/{t}.pdb'
    nat = read_native(native)
    antibody = set(t.split('_')[1:3])
    resmap, row = {}, {'target': t, 'chains': ''.join(seqs), 'n_res': 0, 'n_unmapped': 0,
                       'n_type_mismatch': 0, 'ag_native_gap': 0}
    for c, seq in seqs.items():
        if c not in nat:
            raise ValueError(f'{t}: chain {c} not in native ({sorted(nat)})')
        mapping = align_chain(nat[c], seq)
        labels = []
        for i, aa in enumerate(seq):
            j = mapping.get(i)
            if j is None:
                labels.append(None)
                row['n_unmapped'] += 1
                continue
            r = nat[c][j]
            labels.append(f"{r['num']}{r['icode']}")
            row['n_type_mismatch'] += THREE_TO_ONE[r['name']] != aa
        resmap[c] = labels
        row['n_res'] += len(seq)
        if c not in antibody:
            row['ag_native_gap'] |= [r['num'] for r in nat[c]] != list(range(nat[c][0]['num'], nat[c][0]['num'] + len(nat[c])))
    ref = f'{DB}/15_boltz2_s10n10_chothia_w_ag_renum/{t}/{t}_seed_42_sample_0.pdb'
    if os.path.isfile(ref):
        got = pdb_labels(ref)
        kept = {c: [lab for lab in v if lab is not None] for c, v in resmap.items()}
        row['hl_eq_15'] = int(all(got.get(c) == kept[c] for c in resmap if c in antibody))
        row['ag_eq_15'] = int(all(got.get(c) == kept[c] for c in resmap if c not in antibody))
    os.makedirs(f'{args.root}/resmap', exist_ok=True)
    with open(f'{args.root}/resmap/{t}.json', 'w') as fh:
        json.dump({'resmap': resmap, 'target': t, 'native': native,
                   'input_json': f'{args.root}/json/{t}.json'}, fh)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=ROOT, help='Protenix run root (json/, targets.csv)')
    ap.add_argument('--native_dir', default=f'{DB}/04_xtal_chothia_ag_renum')
    ap.add_argument('--targets', nargs='*', help='default: every target in <root>/targets.csv')
    args = ap.parse_args()
    targets = args.targets or [r['target'] for r in csv.DictReader(open(f'{args.root}/targets.csv'))]
    rows, errors = [], []
    for k, t in enumerate(targets, 1):
        try:
            rows.append(build(t, args))
        except Exception as exc:  # noqa: BLE001
            errors.append((t, f'{type(exc).__name__}: {exc}'))
        if k % 200 == 0:
            print(f'{k}/{len(targets)}', flush=True)
    cols = ['target', 'chains', 'n_res', 'n_unmapped', 'n_type_mismatch', 'ag_native_gap', 'hl_eq_15', 'ag_eq_15']
    with open(f'{args.root}/resmap_summary.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})
    print(f'ok {len(rows)}  error {len(errors)}')
    for t, e in errors:
        print('ERROR', t, e)


if __name__ == '__main__':
    main()
