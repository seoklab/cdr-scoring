"""Which decoys of a validation manifest are NOT in the graph cache?

A cache miss is not an error — data_module falls back to building that decoy's
graph on the fly — but it is the difference between a validation pass that reads
mmap'd arrays and one that runs the graph builder, so it is worth knowing exactly
what is missing before committing to a long run.

Two kinds of gap are reported separately because they need different fixes:
  no pack        the (source, target) .gpk file does not exist at all
  key missing    the pack exists but this (seed, sample) is not among its keys
                 -- i.e. the pack was built from an older candidate list

Usage:
  python preprocess/report_graph_cache_gaps.py \
      --manifest .../val_boltz2_fnat05.json --cache .../graph_cache
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'libs'))
from data_loading.graph_pack import GraphPack, _key_str        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--manifest', required=True)
ap.add_argument('--cache', required=True)
ap.add_argument('--out', default=None, help='write the missing list to this txt')
a = ap.parse_args()

man = json.load(open(a.manifest))
tgts = man['targets']
print(f'manifest {os.path.basename(a.manifest)}  hash={man.get("hash")}  '
      f'{len(tgts)} targets, {man["stats"]["n_decoys"]:,} decoys')

no_pack = defaultdict(list)        # source -> [target]
key_gap = defaultdict(int)         # source -> n decoys
key_gap_t = defaultdict(set)       # source -> {target}
present = Counter()
missing_rows = []
_cache = {}
for tid, cands in tgts.items():
    for src, seed, samp in cands:
        p = os.path.join(a.cache, src, f'{tid}.gpk')
        if p not in _cache:
            _cache[p] = None
            if os.path.exists(p):
                try:
                    pk = GraphPack(p, mode='pread')
                    _cache[p] = set(pk.header['key_strs'])
                    pk.close()
                except Exception as e:
                    print(f'  ! unreadable {p}: {e}')
        ks = _cache[p]
        if ks is None:
            if tid not in no_pack[src]:
                no_pack[src].append(tid)
            missing_rows.append((tid, src, seed, samp, 'no_pack'))
            continue
        k = _key_str((src, seed, samp))
        if k in ks:
            present[src] += 1
        else:
            key_gap[src] += 1
            key_gap_t[src].add(tid)
            missing_rows.append((tid, src, seed, samp, 'key_missing'))

print('\n' + '=' * 84)
print(f'{"source":18s} {"in cache":>10s} {"key missing":>12s} {"no pack (decoys)":>17s} '
      f'{"targets w/ no pack":>19s}')
print('-' * 84)
tot_p = tot_k = tot_n = 0
for src in sorted(set(list(present) + list(key_gap) + list(no_pack))):
    npk = sum(1 for r in missing_rows if r[1] == src and r[4] == 'no_pack')
    print(f'{src:18s} {present[src]:10,} {key_gap[src]:12,} {npk:17,} '
          f'{len(no_pack[src]):19,}')
    tot_p += present[src]; tot_k += key_gap[src]; tot_n += npk
print('-' * 84)
print(f'{"TOTAL":18s} {tot_p:10,} {tot_k:12,} {tot_n:17,}')
tot = tot_p + tot_k + tot_n
print(f'\ncache hit rate: {100*tot_p/max(1,tot):.2f}%  '
      f'({tot - tot_p:,} of {tot:,} decoys fall back to on-the-fly)')
for src in sorted(key_gap_t):
    print(f'  {src}: {key_gap[src]:,} key-missing decoys over '
          f'{len(key_gap_t[src])} targets')
for src in sorted(no_pack):
    if no_pack[src]:
        print(f'  {src}: NO pack for {len(no_pack[src])} targets, '
              f'e.g. {sorted(no_pack[src])[:5]}')

if a.out:
    with open(a.out, 'w') as f:
        f.write('target\tsource\tseed\tsample\treason\n')
        for r in missing_rows:
            f.write('\t'.join(str(x) for x in r) + '\n')
    print(f'\nmissing list -> {a.out}  ({len(missing_rows):,} rows)')
