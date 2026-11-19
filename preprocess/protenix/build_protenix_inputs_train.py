#!/usr/bin/env python
"""Protenix input JSONs for the AbAg train+valid targets, reusing Boltz2's cached MSAs.

The 2,013 targets (train.pkl 1,812 + valid.pkl 201) were each run through Boltz2, which left
its per-chain MSA at ``11_boltz2/boltz_results_<t>/msa/<t>_<i>.csv`` (``key,sequence``). The
key is Boltz's pairing key: a non-negative integer means the row is paired across chains (the
same key is the same taxon in every chain), ``-1`` means unpaired. Those two groups map
directly onto Protenix's ``pairedMsaPath`` / ``unpairedMsaPath``, so no MSA has to be rebuilt
and no compute node has to reach an MSA server.

Paired rows are emitted in key order over the UNION of keys, with a gap row where a chain has
no sequence for that key -- truncating to the intersection instead would cut H/L pairing depth
from ~980 rows to ~4 whenever a shallow antigen chain is present.

  python preprocess/protenix/build_protenix_inputs_train.py [--jobs N] [--limit N]

Writes <out>/msa/<target>/<id>_{pairing,non_pairing}.a3m, <out>/json/<target>.json and
targets.csv (target, split, n_chains, n_tokens).
"""
import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

DB = '/data/galaxy4/user/sujin/DB/h3-loop-modeling/ab_ag/1_tr_abag'
OUT = '/data/galaxy4/user/sujin/projects/cdr-scoring/cdr-data/protenix/train_valid'
YAML = f'{DB}/111_boltz2_yaml'
MSA = f'{DB}/11_boltz2'
INFO = f'{DB}/0_info'

_ENTITY = re.compile(r'-\s*protein:\s*\n\s*id:\s*(\[[^\]]*\]|\S+)\s*\n\s*sequence:\s*(\S+)')


def target_list():
    """train.pkl + valid.pkl, tagged with which split each target came from."""
    import pickle
    out = []
    for split in ('train', 'valid'):
        d = pickle.load(open(f'{INFO}/{split}.pkl', 'rb'))
        lst = d['list'] if isinstance(d, dict) and 'list' in d else d
        out += [(t, split) for t in lst]
    return sorted(out)


def entities(target):
    """[(ids, sequence)] in the order Boltz wrote the per-chain MSA csvs.

    Boltz writes ONE csv per distinct sequence, so a target whose antigen appears twice
    (``id: A`` and ``id: B`` with identical sequence) has fewer csvs than yaml entries. Group
    by sequence in order of first appearance to match, and hand Protenix one chain with
    ``count: 2`` rather than two chains sharing an MSA.
    """
    text = open(f'{YAML}/{target}.yaml').read()
    order, by_seq = [], {}
    for ids, seq in _ENTITY.findall(text):
        ids = [s.strip() for s in ids.strip('[]').split(',')] if ids.startswith('[') else [ids]
        if seq not in by_seq:
            by_seq[seq] = []
            order.append(seq)
        by_seq[seq] += ids
    if not order:
        raise ValueError(f'{target}: no protein entity in yaml')
    return [(by_seq[s], s) for s in order]


def read_msa(path):
    """(paired {key: seq}, unpaired [seq]) from one Boltz MSA csv."""
    paired, unpaired = {}, []
    with open(path) as f:
        for row in csv.DictReader(f):
            k = row['key']
            if k == '-1':
                unpaired.append(row['sequence'])
            else:
                paired.setdefault(int(k), row['sequence'])
    return paired, unpaired


def write_a3m(path, names_seqs):
    """Write only when the content changes, so re-runs stay cheap."""
    body = ''.join(f'>{n}\n{s}\n' for n, s in names_seqs)
    if os.path.exists(path) and os.path.getsize(path) == len(body.encode()):
        return path
    with open(path, 'w') as f:
        f.write(body)
    return path


def convert(args):
    target, split, out = args
    ents = entities(target)
    msa_dir = os.path.join(out, 'msa', target)
    os.makedirs(msa_dir, exist_ok=True)

    per_chain = []
    for i, (ids, seq) in enumerate(ents):
        p = f'{MSA}/boltz_results_{target}/msa/{target}_{i}.csv'
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        paired, unpaired = read_msa(p)
        query = paired.get(0) or (unpaired[0] if unpaired else None)
        if query is None or query.replace('-', '').upper() != seq.upper():
            raise ValueError(f'{target} chain {i}: csv query does not match the yaml sequence')
        per_chain.append((ids, seq, paired, unpaired))

    # paired rows over the union of keys; a chain with no row for a key gets an all-gap row
    keys = sorted(set().union(*[set(c[2]) for c in per_chain])) if len(per_chain) > 1 else []
    seqs = []
    for i, (ids, seq, paired, unpaired) in enumerate(per_chain):
        chain = {'sequence': seq, 'count': len(ids), 'id': list(ids)}
        gap = '-' * len(seq)
        rows = [(f'{k}', paired.get(k, gap)) for k in keys]
        if rows:
            chain['pairedMsaPath'] = write_a3m(os.path.join(msa_dir, f'{ids[0]}_pairing.a3m'), rows)
        # the query heads the unpaired file; Protenix expects it as the first record
        rows = [('101', seq)] + [(f'{j}', s) for j, s in enumerate(unpaired) if s != seq]
        chain['unpairedMsaPath'] = write_a3m(
            os.path.join(msa_dir, f'{ids[0]}_non_pairing.a3m'), rows)
        seqs.append({'proteinChain': chain})

    jdir = os.path.join(out, 'json')
    os.makedirs(jdir, exist_ok=True)
    with open(os.path.join(jdir, f'{target}.json'), 'w') as f:
        json.dump([{'name': target, 'sequences': seqs}], f, indent=2)
    ntok = sum(len(seq) * len(ids) for ids, seq, _, _ in per_chain)
    return target, split, len(seqs), ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    ts = target_list()
    if a.limit:
        ts = ts[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    rows, bad = [], []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(convert, (t, s, a.out)): t for t, s in ts}
        for i, (fu, t) in enumerate(futs.items(), 1):
            try:
                rows.append(fu.result())
            except Exception as e:                      # keep going; report the stragglers
                bad.append((t, f'{type(e).__name__}: {e}'))
            if i % 200 == 0:
                print(f'  {i}/{len(ts)}  ok {len(rows)}  fail {len(bad)}', flush=True)
    rows.sort()
    with open(os.path.join(a.out, 'targets.csv'), 'w') as f:
        f.write('target,split,n_chains,n_tokens\n')
        for t, s, n, k in rows:
            f.write(f'{t},{s},{n},{k}\n')
    with open(os.path.join(a.out, 'targets_by_size.csv'), 'w') as f:
        f.write('target,split,n_chains,n_tokens\n')
        for t, s, n, k in sorted(rows, key=lambda r: -r[3]):
            f.write(f'{t},{s},{n},{k}\n')
    if bad:
        with open(os.path.join(a.out, 'failed.txt'), 'w') as f:
            for t, e in bad:
                f.write(f'{t}\t{e}\n')
    tok = sorted(r[3] for r in rows)
    print(f'{len(rows)} targets -> {a.out}  (실패 {len(bad)})')
    if tok:
        print(f'tokens: min {tok[0]} / median {tok[len(tok)//2]} / p90 {tok[int(len(tok)*0.9)]} '
              f'/ max {tok[-1]}')
    for t, e in bad[:10]:
        print(f'  FAIL {t}: {e}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
