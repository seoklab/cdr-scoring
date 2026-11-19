#!/usr/bin/env python
"""Protenix input JSONs for the vanilla test set, reusing the MSAs AF3 already ran.

The 185 vanilla test targets each have an AF3 ``<target>_data.json`` holding the full
unpaired/paired MSA per chain. Writing those out as a3m and pointing Protenix at them
keeps the MSA identical between the two generators, so a quality difference between the
AF3 and Protenix candidate pools is the model, not the alignment -- and no compute node
needs to reach the Protenix MSA server.

  python preprocess/protenix/build_protenix_inputs.py [--limit N] [--targets a,b,c]

Writes <out>/msa/<target>/<chain>_{paired,non_pairing}.a3m and <out>/json/<target>.json,
plus targets.csv (target, n_chains, n_tokens) for the per-target resource planning.
"""
import argparse
import json
import os

DB = '/data/galaxy4/user/sujin/DB/h3-loop-modeling/ab_ag/3_h3_benchmark_after210930/22_af3'
PARQ = ('/home/sujin/projects/cdr-scoring/cdr-data/metrics_test_h3_v2/AF3/metrics/'
        'fv_context_metrics.parquet')
OUT = '/home/sujin/projects/cdr-scoring/cdr-data/protenix/vanilla_test'


def target_list():
    import pandas as pd
    q = pd.read_parquet(PARQ, columns=['target_id'])
    return sorted(q.target_id.astype(str).unique())


def write_a3m(path, text):
    """Write an MSA only when its content changes (re-runs stay cheap)."""
    if os.path.exists(path) and os.path.getsize(path) == len(text.encode('utf-8')):
        return
    with open(path, 'w') as f:
        f.write(text)


def convert(target, out):
    src = f'{DB}/{target}/{target}_data.json'
    d = json.load(open(src))
    msa_dir = os.path.join(out, 'msa', target)
    os.makedirs(msa_dir, exist_ok=True)
    seqs, ntok = [], 0
    for entry in d['sequences']:
        kind = list(entry)[0]
        e = entry[kind]
        if kind != 'protein':
            raise ValueError(f'{target}: unsupported entity {kind}')
        ids = e['id'] if isinstance(e['id'], list) else [e['id']]
        chain = {'sequence': e['sequence'], 'count': len(ids), 'id': list(ids)}
        if e.get('modifications'):
            chain['modifications'] = [{'ptmType': m['ptmType'], 'ptmPosition': m['ptmPosition']}
                                      for m in e['modifications']]
        for field, suffix in (('unpairedMsa', 'non_pairing'), ('pairedMsa', 'pairing')):
            text = e.get(field)
            if not text:
                continue
            p = os.path.join(msa_dir, f'{ids[0]}_{suffix}.a3m')
            write_a3m(p, text)
            chain['unpairedMsaPath' if field == 'unpairedMsa' else 'pairedMsaPath'] = p
        seqs.append({'proteinChain': chain})
        ntok += len(e['sequence']) * len(ids)
    job = [{'name': target, 'sequences': seqs}]
    jdir = os.path.join(out, 'json')
    os.makedirs(jdir, exist_ok=True)
    with open(os.path.join(jdir, f'{target}.json'), 'w') as f:
        json.dump(job, f, indent=2)
    return len(seqs), ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--targets', default='')
    a = ap.parse_args()
    ts = a.targets.split(',') if a.targets else target_list()
    if a.limit:
        ts = ts[:a.limit]
    rows = []
    for i, t in enumerate(ts, 1):
        n, ntok = convert(t, a.out)
        rows.append((t, n, ntok))
        if i % 25 == 0 or i == len(ts):
            print(f'  {i}/{len(ts)} {t} chains={n} tokens={ntok}', flush=True)
    with open(os.path.join(a.out, 'targets.csv'), 'w') as f:
        f.write('target,n_chains,n_tokens\n')
        for t, n, k in rows:
            f.write(f'{t},{n},{k}\n')
    tok = sorted(r[2] for r in rows)
    print(f'{len(rows)} targets -> {a.out}')
    print(f'tokens: min {tok[0]} / median {tok[len(tok)//2]} / p90 {tok[int(len(tok)*0.9)]} / max {tok[-1]}')


if __name__ == '__main__':
    main()
