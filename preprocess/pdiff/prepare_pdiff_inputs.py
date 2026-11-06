#!/usr/bin/env python
"""Build region-restricted partial-diffusion inputs (AF3 + Boltz-2) from natives.

Controlled test benchmark: keep the native binding pose (framework + antigen
held at the crystal coordinates) and regenerate only the 6 Chothia CDRs, at a
range of starting noise levels, to fill the within-target CDR-lDDT range.

Init structure = a complete-atom model cif of the same target with its coordinates
replaced by the native. --dataset test: an AF3 model cif (token/atom layout matches the
AF3 fold input exactly). --dataset trval (train/valid, 1_tr_abag): a Boltz-2 s10n10 output
cif; Boltz-2 inputs only (no AF3 fold input exists for these targets).
  * residue present in the native with all heavy atoms  -> native coordinates
  * residue present but some heavy atoms missing         -> native atoms + AF3
    atoms superposed on the residue's native atoms; held (pose stays fixed),
    flexible only with --flex_missing_atoms
  * residue missing in the native, or a different type   -> AF3 residue
    superposed per chain onto the native (CA); residue made flexible
Flexible = 6 CDRs (Chothia, libs/evaluation/loop_metrics DEFAULT_NAMED_LOOP_RANGES,
integer residue number, insertion codes included) + the repaired residues above.

Outputs under --out_dir (default <benchmark>/51_pdiff_inputs):
  init/<T>.cif          init structure (B-factor column: 1 = flexible, 0 = held)
  flex/<T>.json         {"flexible": [{"chain_id", "res_idx" (1-based)}]}  (AF3)
  af3/<T>.json          AF3 fold input (= 22_af3 <T>_data.json, MSA/templates kept)
  boltz2/<T>.yaml       Boltz-2 partial_diffusion input (cif, per-chain msa, flexible)
  resmap/<T>.json       chain -> [native residue label or null] per sequence index,
                        and the reason each flexible residue is flexible
  summary.csv           per-target counts and sanity checks

  python preprocess/pdiff/prepare_pdiff_inputs.py                # all 142 holo
  python preprocess/pdiff/prepare_pdiff_inputs.py --targets 7b5g_H_X_A 7q3q_H_X_A
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from Bio import Align

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from libs.evaluation.loop_metrics import DEFAULT_NAMED_LOOP_RANGES

BENCH = '/home/sujin/DB/h3-loop-modeling/ab_ag/3_h3_benchmark_after210930'
TRVAL = '/home/sujin/DB/h3-loop-modeling/ab_ag/1_tr_abag'


def read_yaml_chains(path):
    """Boltz YAML -> (ordered {chain id: sequence}, {chain id: msa path}) (simple line parser)."""
    seqs, msa, cur = {}, {}, None
    for line in open(path):
        s = line.strip()
        if s.startswith('- protein') or s.startswith('protein:'):
            cur = None
        elif s.startswith('id:'):
            cur = s.split(':', 1)[1].strip().strip('"\'')
        elif s.startswith('sequence:') and cur is not None:
            seqs[cur] = s.split(':', 1)[1].strip().strip('"\'')
        elif s.startswith('msa:') and cur is not None:
            msa[cur] = s.split(':', 1)[1].strip().strip('"\'')
        elif s.startswith('partial_diffusion'):
            break
    return seqs, msa


def target_sources(t, args):
    """Where sequences, the complete-atom template cif, the native and the MSAs come from."""
    if args.dataset == 'test':
        data = json.load(open(f'{BENCH}/22_af3/{t}/{t}_data.json'))
        seqs = {}
        for s in data['sequences']:
            (kind, v), = s.items()
            if kind != 'protein':
                raise ValueError(f'{t}: non-protein entity {kind}')
            for cid in (v['id'] if isinstance(v['id'], list) else [v['id']]):
                seqs[cid] = v['sequence']
        _, msa = read_yaml_chains(f'{BENCH}/32_boltz2_s5n10_yaml/{t}.yaml')
        return {'seqs': seqs, 'af3_data': data, 'msa': msa,
                'template_cif': f'{BENCH}/22_af3/{t}/{args.af3_sample}/{t}_{args.af3_sample}_model.cif',
                'native': f'{BENCH}/02_xtal_chothia_w_ag_renum/{t}.pdb'}
    # train/valid: Boltz-2 s10n10 output (seed 42, model 0) as the template; no AF3 input exists
    seqs, msa = read_yaml_chains(f'{TRVAL}/141_boltz2_s10n10_yaml/{t}.yaml')
    return {'seqs': seqs, 'af3_data': None, 'msa': msa,
            'template_cif': (f'{TRVAL}/14_boltz2_s10n10/{t}/seed_42/boltz_results_{t}/predictions/{t}/'
                             f'{t}_model_0.cif'),
            'native': f'{TRVAL}/04_xtal_chothia_ag_renum/{t}.pdb'}
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOLO_CSV = os.path.join(CODE_ROOT, 'analyze/reports/test_partial_diffusion_needs_per_target.csv')

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
    'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}
HETATM_AS = {'MSE': 'MET'}  # modified residues read as their parent
SKIP_ATOMS = {'OXT'}         # AF3 writes it, crystals usually don't


# ── structure I/O ────────────────────────────────────────────────────────────

def read_native(path):
    """Native PDB -> {chain: [{'num', 'icode', 'name', 'atoms': {atom: xyz}}]}.

    First altloc wins; hydrogens and non-protein HETATM are dropped.
    """
    chains = {}
    for line in open(path):
        rec = line[:6].strip()
        if rec not in ('ATOM', 'HETATM'):
            continue
        resname = line[17:20].strip()
        if rec == 'HETATM':
            if resname not in HETATM_AS:
                continue
            resname = HETATM_AS[resname]
        if resname not in THREE_TO_ONE:
            continue
        atom = line[12:16].strip()
        element = line[76:78].strip() or atom[0]
        if element == 'H' or (atom.startswith('H') and element not in ('HG',)):
            continue
        if resname == 'MET' and atom == 'SE':
            atom = 'SD'
        chain, num, icode = line[21], int(line[22:26]), line[26].strip()
        residues = chains.setdefault(chain, [])
        if not residues or (residues[-1]['num'], residues[-1]['icode']) != (num, icode):
            residues.append({'num': num, 'icode': icode, 'name': resname, 'atoms': {}})
        atoms = residues[-1]['atoms']
        if atom not in atoms:  # first altloc
            atoms[atom] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return chains


def read_af3_cif(path):
    """AF3 model cif -> (lines, atom_site column names, rows).

    rows: list of dicts {line_idx, fields, chain, seq, comp, atom}.
    Only the _atom_site loop is parsed; everything else is copied verbatim.
    """
    lines = open(path).read().split('\n')
    cols, rows, i = [], [], 0
    while i < len(lines):
        if lines[i].startswith('_atom_site.'):
            while lines[i].startswith('_atom_site.'):
                cols.append(lines[i].split('.', 1)[1].strip())
                i += 1
            c = {k: j for j, k in enumerate(cols)}
            while i < len(lines) and lines[i].startswith(('ATOM', 'HETATM')):
                f = lines[i].split()
                start, cont = i, []
                # some writers wrap one row over two lines (valid CIF): join until complete
                while len(f) < len(cols) and i + 1 < len(lines) and \
                        not lines[i + 1].startswith(('ATOM', 'HETATM', '#', 'loop_', '_')):
                    i += 1
                    f += lines[i].split()
                    cont.append(i)
                assert len(f) == len(cols), (path, start, lines[start])
                rows.append({
                    'line_idx': start, 'cont': cont, 'fields': f, 'chain': f[c['label_asym_id']],
                    'seq': int(f[c['label_seq_id']]), 'comp': f[c['label_comp_id']],
                    'atom': f[c['label_atom_id']].strip('"'),
                })
                i += 1
            break
        i += 1
    if not rows:
        raise ValueError(f'{path}: no _atom_site loop')
    return lines, cols, rows


def kabsch(mobile, target):
    """Rotation R and translation t with mobile @ R.T + t ~= target."""
    mc, tc = mobile.mean(0), target.mean(0)
    h = (mobile - mc).T @ (target - tc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, tc - mc @ r.T


# ── per target ───────────────────────────────────────────────────────────────

def align_chain(native_res, seq):
    """Global alignment native residues -> fold-input sequence (0-based)."""
    nat_seq = ''.join(THREE_TO_ONE[r['name']] for r in native_res)
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score, aligner.mismatch_score = 2, -1
    aligner.open_gap_score, aligner.extend_gap_score = -5, -0.5
    aln = aligner.align(nat_seq, seq)[0]
    mapping = {}  # seq index -> native residue index
    for (ns, ne), (ss, _) in zip(*aln.aligned):
        for k in range(ne - ns):
            mapping[ss + k] = ns + k
    return mapping


def cdr_ranges():
    out = {}
    for name, chain, start, end in DEFAULT_NAMED_LOOP_RANGES:
        out.setdefault(chain, []).append((name, start, end))
    return out


def read_seed_model(path, model):
    """One model of a multi-model PDB (e.g. ComMat) -> {(chain, num, icode): {atom: xyz}}."""
    out, cur = {}, None
    for line in open(path):
        if line.startswith('MODEL'):
            cur = int(line.split()[1])
        elif line.startswith('ENDMDL') and cur == model:
            break
        elif line.startswith('ATOM') and cur == model:
            atom = line[12:16].strip()
            if atom.startswith('H'):
                continue
            key = (line[21], int(line[22:26]), line[26].strip())
            out.setdefault(key, {}).setdefault(
                atom, np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]))
    if not out:
        raise ValueError(f'{path}: model {model} not found')
    return out


def prepare_target(t, args, out, seed=None):
    """seed: None (native start) or {'origin', 'path', 'model'} -- the CDR (flexible) residues
    then start from that model's coordinates (same numbering/frame as the native, e.g. ComMat);
    everything else, which is held, stays native. Output name <t>__<origin><model>."""
    out_name = t if seed is None else f"{t}__{seed['origin'].lower()[:2]}{seed['model']}"
    src = target_sources(t, args)
    seqs, data = src['seqs'], src['af3_data']
    model_cif = src['template_cif']
    lines, cols, rows = read_af3_cif(model_cif)
    col = {k: j for j, k in enumerate(cols)}
    native = read_native(src['native'])

    # AF3 atoms per (chain, seq)
    af3 = {}
    for r in rows:
        af3.setdefault((r['chain'], r['seq']), []).append(r)
    for (c, s), rs in af3.items():
        if rs[0]['comp'] != {v: k for k, v in THREE_TO_ONE.items()}.get(seqs[c][s - 1], rs[0]['comp']):
            raise ValueError(f'{t}: model cif residue {c}{s} {rs[0]["comp"]} != fold input')

    cdrs = cdr_ranges()
    new_xyz = {}     # row line_idx -> xyz
    flex = {}        # chain -> set of 1-based seq idx
    reason = {}      # (chain, seq) -> reason
    resmap = {}
    stats = {'target': t, 'n_res': 0, 'n_atoms': len(rows), 'n_cdr_res': 0,
             'n_missing_res': 0, 'n_mismatch_res': 0, 'n_missing_atom_res': 0,
             'cdr_found': ''}
    found_cdrs = set()
    for c, seq in seqs.items():
        if c not in native:
            raise ValueError(f'{t}: chain {c} not in native ({sorted(native)})')
        nres = native[c]
        mapping = align_chain(nres, seq)
        # per-chain superposition of the AF3 model on the native (CA of same-type residues)
        pairs = [(i, j) for i, j in mapping.items()
                 if THREE_TO_ONE[nres[j]['name']] == seq[i]
                 and 'CA' in nres[j]['atoms']]
        af3_ca = np.array([next(r for r in af3[(c, i + 1)] if r['atom'] == 'CA')['fields'][col['Cartn_x']:col['Cartn_z'] + 1]
                           for i, _ in pairs], dtype=float)
        nat_ca = np.array([nres[j]['atoms']['CA'] for _, j in pairs])
        rot, tr = kabsch(af3_ca, nat_ca)
        stats[f'af3_vs_native_ca_rmsd_{c}'] = round(float(np.sqrt(((af3_ca @ rot.T + tr - nat_ca) ** 2).sum(1).mean())), 2)

        resmap[c] = []
        for i in range(len(seq)):
            s = i + 1
            stats['n_res'] += 1
            ratoms = af3[(c, s)]
            xyz_af3 = {r['atom']: np.array(r['fields'][col['Cartn_x']:col['Cartn_z'] + 1], dtype=float) for r in ratoms}
            j = mapping.get(i)
            nat = nres[j] if j is not None else None
            label = None if nat is None else f"{nat['num']}{nat['icode']}"
            resmap[c].append(label)
            if nat is None or THREE_TO_ONE[nat['name']] != seq[i]:
                why = 'missing_res' if nat is None else 'mismatch_res'
                stats['n_missing_res' if nat is None else 'n_mismatch_res'] += 1
                for r in ratoms:
                    new_xyz[r['line_idx']] = xyz_af3[r['atom']] @ rot.T + tr
                flex.setdefault(c, set()).add(s)
                reason[(c, s)] = why
                continue
            have = [a for a in xyz_af3 if a in nat['atoms']]
            missing = [a for a in xyz_af3 if a not in nat['atoms']]
            r_rot, r_tr = rot, tr
            if missing:
                # place missing atoms by superposing the AF3 residue on the native one
                if len(have) >= 3:
                    r_rot, r_tr = kabsch(np.array([xyz_af3[a] for a in have]),
                                         np.array([nat['atoms'][a] for a in have]))
            for r in ratoms:
                a = r['atom']
                new_xyz[r['line_idx']] = nat['atoms'][a] if a in nat['atoms'] else xyz_af3[a] @ r_rot.T + r_tr
            if [a for a in missing if a not in SKIP_ATOMS]:
                stats['n_missing_atom_res'] += 1
                if args.flex_missing_atoms:
                    flex.setdefault(c, set()).add(s)
                    reason.setdefault((c, s), 'missing_atoms')
            for name, start, end in cdrs.get(c, []):
                if start <= nat['num'] <= end:
                    flex.setdefault(c, set()).add(s)
                    reason[(c, s)] = name
                    found_cdrs.add(name)
        # CDR residues that are missing in the native are already flexible

    stats['n_cdr_res'] = sum(1 for v in reason.values() if v[0] in 'HL' and v[1:].isdigit())
    stats['cdr_found'] = ' '.join(sorted(found_cdrs))
    n_flex_res = sum(len(v) for v in flex.values())
    stats['n_flex_res'] = n_flex_res
    stats['n_flex_atoms'] = sum(len(af3[(c, s)]) for c, ss in flex.items() for s in ss)
    stats['flex_atom_frac'] = round(stats['n_flex_atoms'] / len(rows), 3)

    # sanity: C(i)-N(i+1) > 2 A = residues missing in the crystal that the
    # sequence joins (the AF3/Boltz-2 test decoys used the same sequence).
    # cdr_chain_breaks counts those inside or next to a CDR: the CDR is shorter
    # than the real loop there, consider excluding the target.
    breaks, cdr_breaks = [], []
    for c, seq in seqs.items():
        for s in range(1, len(seq)):
            cs = [r for r in af3[(c, s)] if r['atom'] == 'C']
            ns = [r for r in af3[(c, s + 1)] if r['atom'] == 'N']
            d = np.linalg.norm(new_xyz[cs[0]['line_idx']] - new_xyz[ns[0]['line_idx']])
            if d > 2.0:
                lab = f'{c}{resmap[c][s - 1]}-{resmap[c][s]}:{d:.1f}A'
                breaks.append(lab)
                if any(reason.get((c, k), '')[:1] in ('H', 'L') for k in (s, s + 1)):
                    cdr_breaks.append(lab)
    stats['chain_breaks'] = len(breaks)
    stats['cdr_chain_breaks'] = len(cdr_breaks)
    stats['cdr_chain_break_sites'] = ' '.join(cdr_breaks)
    stats['max_abs_coord'] = round(float(max(np.abs(v).max() for v in new_xyz.values())), 1)

    if seed is not None:
        # start the flexible CDR residues from the seed model (held atoms stay native)
        seed_xyz = read_seed_model(seed['path'], seed['model'])
        n_set, n_cdr_atoms = 0, 0
        for (c, s_), why in reason.items():
            if not (why[:1] in ('H', 'L') and why[1:].isdigit()):
                continue
            lab = resmap[c][s_ - 1]
            num, icode = int(lab.rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')), lab.lstrip('-0123456789')
            sx = seed_xyz.get((c, num, icode), {})
            for r in af3[(c, s_)]:
                n_cdr_atoms += 1
                if r['atom'] in sx:
                    new_xyz[r['line_idx']] = sx[r['atom']]
                    n_set += 1
        stats['seed_atoms_set'] = n_set
        stats['seed_cdr_atoms'] = n_cdr_atoms
        stats['target'], stats['name'] = t, out_name

    # write the init cif: coordinates + B-factor flag
    out_lines = list(lines)
    for r in rows:
        f = list(r['fields'])
        x, y, z = new_xyz[r['line_idx']]
        f[col['Cartn_x']], f[col['Cartn_y']], f[col['Cartn_z']] = f'{x:.3f}', f'{y:.3f}', f'{z:.3f}'
        f[col['B_iso_or_equiv']] = '1.00' if r['seq'] in flex.get(r['chain'], ()) else '0.00'
        out_lines[r['line_idx']] = ' '.join(f)
        for k in r.get('cont', ()):   # continuation lines of a wrapped row are now in the joined line
            out_lines[k] = None
    for sub in ('init', 'flex', 'af3', 'boltz2', 'resmap'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    init_cif = os.path.join(out, 'init', f'{out_name}.cif')
    with open(init_cif, 'w') as fh:
        fh.write('\n'.join(line for line in out_lines if line is not None))

    flexible = [{'chain_id': c, 'res_idx': sorted(v)} for c, v in sorted(flex.items())]
    with open(os.path.join(out, 'flex', f'{out_name}.json'), 'w') as fh:
        json.dump({'flexible': flexible}, fh)

    if data is not None:
        fold = dict(data)
        fold['name'] = out_name
        fold['modelSeeds'] = [1]
        with open(os.path.join(out, 'af3', f'{out_name}.json'), 'w') as fh:
            json.dump(fold, fh)

    # Boltz-2: per-chain MSA from the existing YAML of that dataset
    msa = src['msa']
    missing_msa = sorted(set(seqs) - set(msa))
    if missing_msa:
        raise ValueError(f'{t}: no Boltz-2 MSA for chains {missing_msa}')
    with open(os.path.join(out, 'boltz2', f'{out_name}.yaml'), 'w') as fh:
        fh.write('version: 1\n')
        fh.write(f'name: {out_name}\n')
        fh.write('sequences:\n')  # ignored in partial-diffusion mode (taken from the cif)
        for c, seq in seqs.items():
            fh.write(f'  - protein:\n      id: {c}\n      sequence: {seq}\n      msa: {msa[c]}\n')
        fh.write('partial_diffusion:\n')
        fh.write(f'  cif: {os.path.abspath(init_cif)}\n')
        fh.write('  msa:\n')
        for c in seqs:
            fh.write(f'    {c}: {msa[c]}\n')
        fh.write('  flexible:\n')
        for item in flexible:
            fh.write(f"    - chain_id: {item['chain_id']}\n      res_idx: {item['res_idx']}\n")

    with open(os.path.join(out, 'resmap', f'{out_name}.json'), 'w') as fh:
        json.dump({'resmap': resmap, 'target': t,
                   'flexible_reason': {f'{c}:{s}': v for (c, s), v in sorted(reason.items())},
                   'template_cif': model_cif, 'native': src['native'],
                   'seed_origin': 'native' if seed is None else seed['origin'],
                   'seed_path': None if seed is None else seed['path'],
                   'seed_model': None if seed is None else seed['model']}, fh)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=['test', 'trval'], default='test',
                    help='test: AF3 model cif template + AF3 json (3_h3_benchmark_after210930); '
                         'trval: Boltz-2 s10n10 output template, Boltz-2 only (1_tr_abag)')
    ap.add_argument('--targets', nargs='*', help='default (test): the 142 holo test targets')
    ap.add_argument('--targets_file', help='one target per line')
    ap.add_argument('--shard', type=int, default=0, help='with --n_shards: process every n-th job only')
    ap.add_argument('--n_shards', type=int, default=1)
    ap.add_argument('--seeds_csv', help='seeded starts instead of the native: CSV with target, origin, '
                                         'path (multi-model PDB in native numbering), model')
    ap.add_argument('--out_dir', default=None,
                    help='default: <dataset root>/51_pdiff_inputs')
    ap.add_argument('--flex_missing_atoms', action='store_true',
                    help='also regenerate residues with missing heavy atoms (default: hold them)')
    ap.add_argument('--af3_sample', default='seed-1_sample-0',
                    help='AF3 model whose layout/atoms are reused (coordinates replaced)')
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = f'{BENCH if args.dataset == "test" else TRVAL}/51_pdiff_inputs'
    if args.targets_file:
        args.targets = [l.strip() for l in open(args.targets_file) if l.strip()]
    targets = args.targets or [r['target'] for r in csv.DictReader(open(HOLO_CSV))]
    os.makedirs(args.out_dir, exist_ok=True)
    rows, failed = [], []
    jobs = [(t, None) for t in targets]
    if args.seeds_csv:
        jobs = [(r['target'], {'origin': r['origin'], 'path': r['path'], 'model': int(r['model'])})
                for r in csv.DictReader(open(args.seeds_csv))]
    jobs = jobs[args.shard::args.n_shards]
    for t, seed in jobs:
        try:
            rows.append(prepare_target(t, args, args.out_dir, seed=seed))
        except Exception as e:  # noqa: BLE001
            failed.append((t, repr(e)))
            print(f'[fail] {t} {seed}: {e!r}', flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path = os.path.join(args.out_dir, ('summary_seeded' if args.seeds_csv else 'summary')
                        + (f'_part{args.shard:02d}' if args.n_shards > 1 else '') + '.csv')
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} targets written to {args.out_dir}, {len(failed)} failed -> {path}')
    for t, e in failed:
        print('  ', t, e)


if __name__ == '__main__':
    main()
