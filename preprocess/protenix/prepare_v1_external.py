#!/usr/bin/env python
"""Put the epitope_benchmark Protenix-v1 predictions into our own target layout.

yubeen's run folded one benchmark row at a time: antigen component chains first, then the
antibody chains, and Protenix relabels the output A, B, C... in that input order. The chains
are also whole deposited constructs, so an antibody is usually a Fab where ours is an Fv and
the antigen may carry copies we never folded.

Per target: the folded chain whose sequence CONTAINS ours takes our chain id, chains we do
not use are dropped, and the result is written as

    <out>/prep/<target>/<target>/seed_<N>/predictions/<target>_sample_<K>.cif

which is the layout preprocess/target_pickle/1_build_pertmd_target_pickles.py already reads
for --source-type protenix. <out>/json/<target>.json repeats the folded sequences under our
chain ids so build_protenix_resmap.py can map each sequence position onto a native Chothia
label; the extra constant residues align to nothing and are dropped there, which is where the
Fab becomes an Fv.

A target is prepared only when EVERY chain of ours is found, so nothing is scored against a
native it does not match; the others are listed in prep_summary.csv with the reason.

  python preprocess/protenix/prepare_v1_external.py [--jobs 8] [--limit N] [--dry_run]
"""
import argparse
import csv
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

BENCH = '/home/yubeen/projects/foundation_model/epitope_benchmark'
OUT = '/home/sujin/DB/h3-loop-modeling/ab_ag/1_tr_abag/70_protenix_v1'
RAW = OUT
PTX = '/data/galaxy4/user/sujin/projects/cdr-scoring/cdr-data/protenix'
SEEDS = (101, 102, 103, 104, 105)
SAMPLES = (0, 1, 2, 3, 4)
RANK = {'ab+pdb': 0, 'ab': 1, 'obs': 2, 'pdb': 3}


def our_chains(target):
    """Our own Protenix input JSON -> {chain id: sequence}."""
    for sub in ('train_valid', 'vanilla_test'):
        path = f'{PTX}/{sub}/json/{target}.json'
        if os.path.exists(path):
            data = json.load(open(path))[0]
            out = {}
            for entity in data['sequences']:
                (_, v), = entity.items()
                for cid in v['id']:
                    out[cid] = v['sequence']
            return out
    return None


def folded_chains(unit, inputs):
    """Their folded chains in output order: antigen entries first, then antibody."""
    row = inputs[unit]
    return [c['seq'] for c in row['antigen']] + [c['seq'] for c in row['antibody']]


def letter(i):
    return chr(ord('A') + i)


def rewrite_cif(src, dst, rename):
    """Copy one prediction, renaming the chains we keep and dropping the rest."""
    cols, out = [], []
    for line in open(src):
        s = line.rstrip('\n')
        t = s.strip()
        if t.startswith('_atom_site.'):
            cols.append(t.split('.')[1])
            out.append(s)
        elif cols and t.startswith(('ATOM', 'HETATM')):
            f = t.split()
            keep = rename.get(f[cols.index('label_asym_id')])
            if keep is None:
                continue
            for key in ('label_asym_id', 'auth_asym_id'):
                if key in cols:
                    f[cols.index(key)] = keep
            out.append(' '.join(f))
        else:
            out.append(s)
    with open(dst, 'w') as fh:
        fh.write('\n'.join(out) + '\n')


def plan_one(target, units, inputs):
    """Pick the unit to use and work out which output letter carries which of our chains."""
    ours = our_chains(target)
    if ours is None:
        return dict(target=target, status='no_input_json')
    best = None
    for u in sorted(units, key=lambda u: RANK.get(u['match'], 9)):
        uid = u['fold_unit_id']
        if uid not in inputs:
            continue
        seqs = folded_chains(uid, inputs)
        rename, missing = {}, []
        for cid, seq in ours.items():
            hit = [i for i, s in enumerate(seqs) if seq in s]
            if not hit:
                missing.append(cid)
                continue
            # the shortest containing chain is the closest construct to ours
            rename[letter(min(hit, key=lambda i: len(seqs[i])))] = cid
        if len(set(rename.values())) != len(ours) - len(missing):
            missing.append('ambiguous')
        cand = dict(target=target, unit=uid, match=u['match'], rename=rename,
                    missing=','.join(missing), n_our=len(ours), n_folded=len(seqs),
                    extra_dropped=len(seqs) - len(rename),
                    seqs={rename[letter(i)]: s for i, s in enumerate(seqs) if letter(i) in rename})
        if not missing:
            return dict(cand, status='ok')
        if best is None:
            best = dict(cand, status='missing_chain')
    return best or dict(target=target, status='no_unit')


def write_one(plan, dry_run=False):
    t, uid, rename = plan['target'], plan['unit'], plan['rename']
    n = 0
    for seed in SEEDS:
        src_dir = f'{RAW}/{uid}/seed_{seed}/predictions'
        dst_dir = f'{OUT}/prep/{t}/{t}/seed_{seed}/predictions'
        if not dry_run:
            os.makedirs(dst_dir, exist_ok=True)
        for k in SAMPLES:
            src = f'{src_dir}/{uid}_sample_{k}.cif'
            js = f'{src_dir}/{uid}_summary_confidence_sample_{k}.json'
            if not os.path.isfile(src) or not os.path.isfile(js):
                continue
            if not dry_run:
                rewrite_cif(src, f'{dst_dir}/{t}_sample_{k}.cif', rename)
                shutil.copyfile(js, f'{dst_dir}/{t}_summary_confidence_sample_{k}.json')
            n += 1
    if not dry_run:
        entities = [{'proteinChain': {'sequence': s, 'count': 1, 'id': [c]}}
                    for c, s in plan['seqs'].items()]
        with open(f'{OUT}/json/{t}.json', 'w') as fh:
            json.dump([{'name': t, 'sequences': entities}], fh)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    req = json.load(open(f'{OUT}/request_unit_ids.json'))
    inputs = {json.loads(l)['unit_id']: json.loads(l) for l in open(f'{BENCH}/inputs.jsonl')}
    targets = req['targets'][:args.limit] if args.limit else req['targets']

    with ProcessPoolExecutor(args.jobs) as ex:
        plans = list(ex.map(plan_one, [t['target'] for t in targets],
                            [t['units'] for t in targets], [inputs] * len(targets),
                            chunksize=8))
    ok = [p for p in plans if p['status'] == 'ok']
    print(f'planned {len(plans)}  ok {len(ok)}  '
          + '  '.join(f'{s} {sum(p["status"] == s for p in plans)}'
                      for s in sorted({p['status'] for p in plans} - {'ok'})), flush=True)

    if not args.dry_run:
        for sub in ('json', 'prep'):
            os.makedirs(f'{OUT}/{sub}', exist_ok=True)
    with ProcessPoolExecutor(args.jobs) as ex:
        counts = list(ex.map(write_one, ok, [args.dry_run] * len(ok), chunksize=4))

    sets = {t['target']: t['set'] for t in targets}
    cols = ['target', 'set', 'status', 'unit', 'match', 'n_our', 'n_folded',
            'extra_dropped', 'n_structures', 'missing', 'chain_map']
    n_by = dict(zip([p['target'] for p in ok], counts))
    with open(f'{OUT}/prep_summary.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for p in plans:
            w.writerow({c: p.get(c, '') for c in cols}
                       | dict(set=sets.get(p['target'], ''), n_structures=n_by.get(p['target'], 0),
                              chain_map=';'.join(f'{k}->{v}' for k, v in sorted(p.get('rename', {}).items()))))
    with open(f'{OUT}/targets.csv', 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(['target', 'split', 'n_chains', 'n_tokens'])
        for p in ok:
            w.writerow([p['target'], sets.get(p['target'], ''), len(p['seqs']),
                        sum(len(s) for s in p['seqs'].values())])
    print(f'structures written {sum(counts)}  targets.csv {len(ok)}')


if __name__ == '__main__':
    sys.exit(main())
