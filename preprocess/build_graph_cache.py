#!/usr/bin/env python
"""Build the Layer-B graph cache (indexed containers; see libs/data_loading/graph_pack.py).

For each (target, target_model_pickle source) this loads the pdb2dict Target
pickle ONCE, builds every decoy graph at a WIDE distance envelope (so the
training cutoff + random_range jitter can be reproduced at read time by
edge-masking), and writes one indexed ``.gpk`` container. GP (graph_pickle) and
Xtal sources are skipped — GP already has a legacy per-target cache and Xtal is
handled specially by the loader.

Layout::

    {out_dir}/_cache_meta.json           # build_sig, envelope, params, code_sig
    {out_dir}/{source}/{target}.gpk

The loader (MyDataset with CDR_GRAPH_CACHE_MODE=read / CDR_GRAPH_CACHE_DIR=out_dir)
recomputes build_sig from the live params + graph-code hash and only reads a pack
whose sig matches; otherwise it falls back to on-the-fly. So a param/code change
is *detected*, never silently served stale.

Scoping:
  --targets FILE     one target id per line (training list); or
  --manifest FILE    build exactly the targets in a fixed val manifest
  --sources a,b,c    restrict to these source names (default: all on-the-fly)
Resumable: an existing pack with a matching build_sig is skipped unless --force.

Example::
  python preprocess/build_graph_cache.py \
      --config configs/dataset_sources-v2.yaml \
      --targets train_targets.txt \
      --out /home/sujin/DB/h3-loop-modeling/graph_cache/v2 \
      --envelope-cutoff 11.5
"""
import argparse
import json
import os
import sys
import time

sys.argv_backup = list(sys.argv)
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBS = os.path.join(os.path.dirname(_HERE), 'libs')
if _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)


def _load_targets(args):
    if args.manifest:
        with open(args.manifest) as f:
            man = json.load(f)
        return list(man['targets'].keys())
    if args.targets:
        with open(args.targets) as f:
            return [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
    raise SystemExit('need --targets or --manifest')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='dataset YAML (same as training)')
    ap.add_argument('--out', required=True, help='cache generation output dir')
    ap.add_argument('--targets', help='file: one target id per line')
    ap.add_argument('--manifest', help='fixed val manifest json (build its targets)')
    ap.add_argument('--sources', help='comma-separated source names (default: all on-the-fly)')
    ap.add_argument('--envelope-cutoff', type=float, default=None,
                    help='wide CA-CA cutoff to build at (default: dist_cutoff_center+random_range)')
    ap.add_argument('--shard', type=int, default=0, help='this shard index')
    ap.add_argument('--num-shards', type=int, default=1, help='total shards (round-robin over targets)')
    ap.add_argument('--force', action='store_true', help='rebuild even if a matching pack exists')
    ap.add_argument('--limit', type=int, default=0, help='stop after N packs (debug)')
    args = ap.parse_args()

    # importing data_module parses sys.argv; neutralize it
    sys.argv = ['build_graph_cache']
    import numpy as np  # noqa
    import torch
    torch.set_num_threads(1)

    from dataset.config import load_dataset_spec
    from dataset.source_registry import SourceRegistry
    from data_loading.data_module import _load_target_pickle, graph_build_param_dict
    from data_loading.graph_generation_from_target import generate_graphs_from_target
    from data_loading import graph_pack as GP
    from dataset.precomputed_metrics import decoy_identity_for_lookup

    spec = load_dataset_spec(args.config)
    gb = spec.graph_build
    registry = SourceRegistry(spec)

    params = graph_build_param_dict(spec, gb)
    center = float(gb.dist_cutoff_center)
    rng = float(gb.random_range)
    env_cut = args.envelope_cutoff if args.envelope_cutoff is not None else (center + rng)
    if env_cut < center + rng:
        raise SystemExit(f'envelope-cutoff {env_cut} < center+range {center+rng}: '
                         f'would not cover training jitter')
    params_env = dict(params)
    params_env['dist_cutoff'] = env_cut   # record the wide envelope in the header
    code_sig = GP.compute_code_sig()
    build_sig = GP.compute_build_sig(params_env, code_sig)

    want_sources = set(args.sources.split(',')) if args.sources else None
    # Only cache store-backed sources: the loader reads labels from the parquet by
    # decoy key, so a non-store source (e.g. Xtal, whose label is a constant) can't
    # be served from cache and must stay on-the-fly.
    _pm = getattr(spec, 'precomputed_metrics', None)
    store_sources = set(_pm.sources.keys()) if (_pm is not None and _pm.sources) else None
    targets = _load_targets(args)
    targets = [t for i, t in enumerate(targets) if i % args.num_shards == args.shard]

    os.makedirs(args.out, exist_ok=True)
    meta_path = os.path.join(args.out, '_cache_meta.json')
    if not os.path.exists(meta_path) and args.shard == 0:
        with open(meta_path, 'w') as f:
            json.dump({'build_sig': build_sig, 'code_sig': code_sig,
                       'envelope': params_env, 'params': params,
                       'max_neighbors': params['max_neighbors']}, f, indent=2)

    print(f'[build_graph_cache] out={args.out}')
    print(f'  build_sig={build_sig} code_sig={code_sig} envelope_cutoff={env_cut} '
          f'max_neighbors={params["max_neighbors"]}')
    print(f'  targets={len(targets)} (shard {args.shard}/{args.num_shards}) '
          f'sources={want_sources or "all on-the-fly"}')

    n_written = n_skipped = n_missing = n_failed = 0
    t_start = time.perf_counter()
    for ti, target in enumerate(targets):
        cands = registry.get_candidates(target, epoch=1)
        for cand in cands:
            if cand.file_type != 'target_model_pickle':
                continue
            sname = cand.source_name
            if want_sources is not None and sname not in want_sources:
                continue
            if store_sources is not None and sname not in store_sources:
                continue   # non-store source (e.g. Xtal): loader keeps it on-the-fly
            out_pack = os.path.join(args.out, sname, f'{cand.pdb_id}.gpk')
            if (not args.force) and os.path.exists(out_pack):
                try:
                    pk = GP.GraphPack(out_pack, mode='pread')
                    good = (pk.build_sig == build_sig)
                    pk.close()
                    if good:
                        n_skipped += 1
                        continue
                except Exception:
                    pass
            src_pkl = cand.target_model_path
            if not src_pkl or not os.path.exists(src_pkl):
                n_missing += 1
                continue
            try:
                tobj = _load_target_pickle(src_pkl)
                graphs, rmsds, ranks, dmeta = generate_graphs_from_target(
                    tobj,
                    dist_cutoff_center=env_cut,
                    random_range=0.0,
                    max_neighbors=params['max_neighbors'],
                    use_all_atom=params['use_all_atom'],
                    h3_range=tuple(params['h3_range']),
                    cdr_ranges=params['cdr_ranges'] if not isinstance(params['cdr_ranges'], dict)
                               else {k: tuple(v) for k, v in params['cdr_ranges'].items()},
                    task_scope=params['task_scope'],
                    label_metric=spec.label_metric,
                    cdr_context_cutoff=params['cdr_context_cutoff'],
                    max_context_residues=params['max_context_residues'],
                    graph_crop_debug=False,
                    target_id=cand.pdb_id,
                )
                if not graphs:
                    n_failed += 1
                    continue
                keys = []
                for pos, m in enumerate(tobj.models):
                    sd, sm = decoy_identity_for_lookup(m, pos)
                    keys.append((sname, sd, int(sm)))
                keys = keys[:len(graphs)]
                ssig = GP.compute_struct_sig([src_pkl], keys)
                GP.write_pack(out_pack, graphs, keys, build_sig=build_sig,
                              struct_sig=ssig, envelope=params_env,
                              meta={'target': cand.pdb_id, 'source': sname,
                                    'source_files': [src_pkl]})
                n_written += 1
                del tobj, graphs
            except Exception as e:
                n_failed += 1
                print(f'  FAIL {sname}/{cand.pdb_id}: {type(e).__name__}: {e}')
            if args.limit and n_written >= args.limit:
                print('  --limit reached'); break
        if args.limit and n_written >= args.limit:
            break
        if (ti + 1) % 25 == 0:
            el = time.perf_counter() - t_start
            print(f'  [{ti+1}/{len(targets)}] written={n_written} skipped={n_skipped} '
                  f'missing={n_missing} failed={n_failed} '
                  f'({el/max(1,n_written):.1f}s/pack, {el/60:.1f}min elapsed)')

    print(f'\n[done] written={n_written} skipped={n_skipped} missing={n_missing} '
          f'failed={n_failed} in {(time.perf_counter()-t_start)/60:.1f} min')


if __name__ == '__main__':
    main()
