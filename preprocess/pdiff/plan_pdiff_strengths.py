#!/usr/bin/env python
"""Per-target, per-tool plan: which CDR-lDDT ranges still need decoys, and at
which partial diffusion strength (= starting noise level sigma, A) to make them.

AF3 and Boltz-2 are planned SEPARATELY (they are evaluated as separate pools):
  AF3 pool    = vanilla AF3 test decoys    + AF3 partial-diffusion decoys
  Boltz2 pool = vanilla Boltz2 test decoys + Boltz2 partial-diffusion decoys
Only pose-correct decoys count (fnat > --fnat_cut). Per tool:
  n_needed = max(0, --per_range - existing in that tool's pool)
and the strength comes from that tool's own measured curve for the target.

Curve = median CDR-lDDT over samples at each measured strength, sigma = 0 is the
native (CDR-lDDT 1.0), made non-increasing. A range below the value the curve
settles at (sigma >= 16 A) cannot be reached by regenerating the CDRs with the
pose held -> 'unreachable'.
  basis = measured      sigma within the measured strengths
          extrapolated  sigma below the smallest measured strength (linear to the native)
          provisional   no measured curve for this target and tool: median strength of the
                        measured targets for that range and tool; reachability unknown
Targets without a measured curve should first get a short calibration sweep
(--calibration_strengths, 2 samples); lists per tool are written for the sweep
scripts (TARGETS=...). Re-run this script afterwards.

Outputs (--out_dir, default analyze/reports/pdiff_plan):
  pdiff_plan.json                 target -> tool -> counts, needs, plan rows, unreachable ranges
  pdiff_plan_rows.csv             one row per target x tool x needed range
  pdiff_plan_targets.csv          one row per target x tool: counts and needs per range
  pdiff_strength_intervals.csv    every target x tool x range with a measured curve:
                                  reachable, strength_min / strength / strength_max
  pdiff_strength_curves.csv       measured curves (target, tool, strength, CDR-lDDT, frac fnat > cut)
  calibration_targets_<tool>.txt  targets with a needed range and no measured curve for that tool
"""
import argparse
import datetime
import glob
import json
import math
import os

import numpy as np
import pandas as pd

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = '/home/sujin/DB/h3-loop-modeling/ab_ag/3_h3_benchmark_after210930'
VANILLA = '/home/sujin/projects/cdr-scoring/cdr-data/metrics_test_h3_v2'
TARGET_INFO = os.path.join(CODE_ROOT, 'analyze/reports/test_partial_diffusion_needs_per_target.csv')
RANGES = [('< 0.75', 0.0, 0.75), ('0.75-0.85', 0.75, 0.85), ('0.85-0.90', 0.85, 0.90),
          ('0.90-0.95', 0.90, 0.95), ('>= 0.95', 0.95, 1.0001)]
TOOLS = ('AF3', 'Boltz2')
SAT_FROM = 16.0   # A; at and above this the curve no longer moves (pilot sweep)


def range_of(x):
    for lab, lo, hi in RANGES:
        if lo <= x < hi:
            return lab
    return None


def load_vanilla():
    out = []
    for src, tool in (('AF3', 'AF3'), ('Boltz2_test', 'Boltz2')):
        lo = pd.read_parquet(f'{VANILLA}/{src}/metrics/loop_metrics.parquet',
                             columns=['target_id', 'seed', 'sample', 'cdr_lddt'])
        fi = pd.read_parquet(f'{VANILLA}/{src}/metrics/interface_metrics.parquet',
                             columns=['target_id', 'seed', 'sample', 'fnat'])
        m = lo.merge(fi, on=['target_id', 'seed', 'sample'], how='left')
        m['tool'] = tool
        out.append(m.rename(columns={'target_id': 'target'}))
    return pd.concat(out, ignore_index=True)


def measured_curve(g):
    """[(sigma, median CDR-lDDT)] with (0, 1.0) prepended, made non-increasing."""
    med = g.groupby('strength').cdr_lddt.median().sort_index()
    out, cur = [], 1.0
    for s, v in [(0.0, 1.0)] + [(float(s), float(v)) for s, v in med.items()]:
        cur = min(cur, v)
        out.append((s, cur))
    return out


def sigma_at(curve, v):
    """Smallest sigma where the curve reaches value v (None if it never does)."""
    for (s0, v0), (s1, v1) in zip(curve, curve[1:]):
        if v1 <= v <= v0:
            return s0 if v0 == v1 else s0 + (v0 - v) / (v0 - v1) * (s1 - s0)
    return None


def plan_from_curve(curve, lo, hi):
    """(reachable, sigma, sigma_min, sigma_max) for range [lo, hi) on a measured curve."""
    curve = [(s, v) for s, v in curve if s <= SAT_FROM] or curve
    sat = curve[-1][1]
    if sat >= hi:
        return False, None, None, None
    top = min(hi, 1.0)
    s_min = 0.0 if top >= 1.0 else sigma_at(curve, top)
    s_max = sigma_at(curve, lo) if sat < lo else curve[-1][0]
    s = sigma_at(curve, (max(lo, sat) + top) / 2)
    if s is None or s <= 0:
        s = math.sqrt(max(s_min or 0.5, 0.5) * max(s_max or 0.5, 0.5))
    return True, s, s_min, s_max


def round_sigma(s):
    if s is None:
        return None
    return float(round(s * 4) / 4) if s < 4 else float(round(s))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--per_range', type=int, default=5, help='target decoys per CDR-lDDT range, per tool')
    ap.add_argument('--fnat_cut', type=float, default=0.5)
    ap.add_argument('--oversample', type=float, default=1.5,
                    help='n_generate = ceil(n_needed * oversample): samples scatter and some fail the fnat cut')
    ap.add_argument('--pdiff_eval', nargs='*',
                    default=sorted(glob.glob(f'{BENCH}/5[2-9]_pdiff_*/eval/decoys.csv')),
                    help='eval_pdiff_sweep.py decoys.csv files (curves + existing pdiff decoys); default: '
                         'pilot sweep (52), calibration (53), planned decoys (54)')
    ap.add_argument('--no_count_pdiff', action='store_true',
                    help='count only vanilla decoys as existing (curves are still used)')
    ap.add_argument('--calibration_strengths', default='0.5 1 2 4 8 16')
    ap.add_argument('--out_dir', default=os.path.join(CODE_ROOT, 'analyze/reports/pdiff_plan'))
    args = ap.parse_args()

    info = pd.read_csv(TARGET_INFO).set_index('target')[['type', 'h3_length']]
    targets = list(info.index)

    van = load_vanilla()
    van = van[van.target.isin(targets) & np.isfinite(van.cdr_lddt)]
    van_ok = van[np.isfinite(van.fnat) & (van.fnat > args.fnat_cut)].copy()
    van_ok['range'] = van_ok.cdr_lddt.map(range_of)

    pd_dec = pd.concat([pd.read_csv(p) for p in args.pdiff_eval], ignore_index=True) if args.pdiff_eval else \
        pd.DataFrame(columns=['tool', 'strength', 'target', 'cdr_lddt', 'fnat'])
    pd_ok = pd_dec[pd_dec.fnat > args.fnat_cut].copy()
    pd_ok['range'] = pd_ok.cdr_lddt.map(range_of)

    curves, curve_rows, interval_rows = {}, [], []
    for (t, tool), g in pd_dec.groupby(['target', 'tool']):
        curves[(t, tool)] = measured_curve(g)
        for s, gs in g.groupby('strength'):
            curve_rows.append({'target': t, 'tool': tool, 'strength': s, 'n': len(gs),
                               'cdr_lddt_median': gs.cdr_lddt.median(), 'cdr_lddt_min': gs.cdr_lddt.min(),
                               'cdr_lddt_max': gs.cdr_lddt.max(),
                               'frac_fnat_gt_cut': (gs.fnat > args.fnat_cut).mean()})
        for lab, lo, hi in RANGES:
            ok, s, s0, s1 = plan_from_curve(curves[(t, tool)], lo, hi)
            interval_rows.append({'target': t, 'tool': tool, 'range': lab, 'reachable': ok,
                                  'strength_min': round_sigma(s0), 'strength': round_sigma(s),
                                  'strength_max': round_sigma(min(s1, SAT_FROM)) if s1 is not None else None,
                                  'settles_at': round(curves[(t, tool)][-1][1], 3)})

    provisional = {}
    for tool in TOOLS:
        for lab, lo, hi in RANGES:
            ss = [plan_from_curve(c, lo, hi)[1] for (_, tl), c in curves.items() if tl == tool]
            ss = [x for x in ss if x is not None]
            provisional[(tool, lab)] = (round_sigma(float(np.median(ss))) if ss else None, len(ss))

    def frac_fnat_at(t, tool, s):
        g = pd_dec[(pd_dec.target == t) & (pd_dec.tool == tool)]
        if g.empty or s is None:
            return None
        near = g.strength.iloc[(g.strength - s).abs().argsort()].iloc[0]
        return float((g[g.strength == near].fnat > args.fnat_cut).mean())

    plan_json, rows, trows = {}, [], []
    calib = {tool: [] for tool in TOOLS}
    for t in targets:
        plan_json[t] = {'type': info.loc[t, 'type'], 'h3_length': int(info.loc[t, 'h3_length'])}
        for tool in TOOLS:
            cnt_v = van_ok[(van_ok.target == t) & (van_ok.tool == tool)].range.value_counts()
            cnt_p = pd_ok[(pd_ok.target == t) & (pd_ok.tool == tool)].range.value_counts() \
                if not args.no_count_pdiff else pd.Series(dtype=int)
            curve = curves.get((t, tool))
            tj = {'curve': 'measured' if curve else 'not measured (provisional strengths)',
                  'settles_at': round(curve[-1][1], 3) if curve else None,
                  'ranges': {}, 'plan': [], 'unreachable': []}
            trow = {'target': t, 'tool': tool, 'type': plan_json[t]['type'],
                    'h3_length': plan_json[t]['h3_length'], 'curve': tj['curve'], 'settles_at': tj['settles_at']}
            for lab, lo, hi in RANGES:
                n_v, n_p = int(cnt_v.get(lab, 0)), int(cnt_p.get(lab, 0))
                need = max(0, args.per_range - n_v - n_p)
                tj['ranges'][lab] = {'vanilla': n_v, 'pdiff_so_far': n_p, 'needed': need}
                trow.update({f'n_vanilla[{lab}]': n_v, f'n_pdiff[{lab}]': n_p, f'needed[{lab}]': need})
                if need == 0:
                    continue
                if curve:
                    ok, s, s0, s1 = plan_from_curve(curve, lo, hi)
                    if not ok:
                        tj['unreachable'].append(lab)
                        continue
                    smallest = min(x for x, _ in curve if x > 0)
                    basis = 'extrapolated' if s < smallest else 'measured'
                else:
                    s, n_curves = provisional[(tool, lab)]
                    s0 = s1 = None
                    basis = f'provisional (median of {n_curves} measured)'
                    if s is None:
                        tj['unreachable'].append(lab)
                        continue
                r = {'range': lab, 'n_needed': need, 'strength': round_sigma(s),
                     'strength_min': round_sigma(s0),
                     'strength_max': round_sigma(min(s1, SAT_FROM)) if s1 is not None else None,
                     'n_generate': math.ceil(need * args.oversample), 'basis': basis,
                     'frac_fnat_gt_cut_measured': frac_fnat_at(t, tool, s) if curve else None}
                tj['plan'].append(r)
                rows.append({'target': t, 'tool': tool, **r,
                             'type': plan_json[t]['type'], 'h3_length': plan_json[t]['h3_length']})
            trow['unreachable'] = ' | '.join(tj['unreachable'])
            trows.append(trow)
            plan_json[t][tool] = tj
            if not curve and any(v['needed'] for v in tj['ranges'].values()):
                calib[tool].append(t)

    os.makedirs(args.out_dir, exist_ok=True)
    meta = {
        'created': datetime.date.today().isoformat(),
        'term': 'strength = --partial_diffusion_strength = starting noise level sigma of the '
                'diffusion schedule, in A (AF3 and Boltz-2 use the same unit)',
        'setup': 'CDRs (Chothia, 6 or 3) regenerated; framework + antigen held at native; fix_mode replace',
        'pools': 'AF3 and Boltz2 planned separately; each counts only its own vanilla'
                 + ('' if args.no_count_pdiff else ' + partial-diffusion') + ' decoys',
        'ranges': [lab for lab, *_ in RANGES], 'per_range_per_tool': args.per_range,
        'fnat_filter': f'fnat > {args.fnat_cut} on existing decoys',
        'oversample': args.oversample, 'strength_cap': SAT_FROM,
        'pdiff_eval_files': args.pdiff_eval,
        'calibration': {'strengths': [float(x) for x in args.calibration_strengths.split()],
                        'samples': 2, 'targets_files': [f'calibration_targets_{t}.txt' for t in TOOLS],
                        'why': 'the value each target settles at cannot be predicted from vanilla decoys '
                               '(pilot: mean |error| 0.05-0.15), so provisional strengths are only a start'},
    }
    with open(os.path.join(args.out_dir, 'pdiff_plan.json'), 'w') as fh:
        json.dump({'meta': meta, 'targets': plan_json}, fh, indent=1)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, 'pdiff_plan_rows.csv'), index=False)
    pd.DataFrame(trows).to_csv(os.path.join(args.out_dir, 'pdiff_plan_targets.csv'), index=False)
    pd.DataFrame(interval_rows).to_csv(os.path.join(args.out_dir, 'pdiff_strength_intervals.csv'), index=False)
    pd.DataFrame(curve_rows).to_csv(os.path.join(args.out_dir, 'pdiff_strength_curves.csv'), index=False)
    for tool in TOOLS:
        with open(os.path.join(args.out_dir, f'calibration_targets_{tool}.txt'), 'w') as fh:
            fh.write(''.join(f'{t}\n' for t in calib[tool]))
    stale = os.path.join(args.out_dir, 'calibration_targets.txt')  # pre-split file name
    if os.path.exists(stale):
        os.remove(stale)

    T = pd.DataFrame(trows)
    print(f'{len(targets)} targets, per_range={args.per_range} per tool, {meta["pools"]} ({meta["fnat_filter"]})')
    for tool in TOOLS:
        tt = T[T.tool == tool]
        print(f'[{tool}]')
        for lab, *_ in RANGES:
            col = tt[f'needed[{lab}]']
            unre = int(tt.unreachable.str.contains(lab, regex=False).sum())
            print(f'  {lab:10s} targets needing it: {int((col > 0).sum()):3d}  decoys needed: {int(col.sum()):4d}'
                  f'  (unreachable on measured curves: {unre})')
        print('  provisional strength (A):', {lab: provisional[(tool, lab)][0] for lab, *_ in RANGES},
              ' calibration targets:', len(calib[tool]))
    print('->', args.out_dir)


if __name__ == '__main__':
    main()
