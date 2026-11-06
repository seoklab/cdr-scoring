#!/usr/bin/env python
"""Write a task manifest for sbatch_pdiff_{af3,boltz2}_sweep.sh (MANIFEST=...).

Each line = one array task: strength <TAB> targets file <TAB> samples <TAB> tag.

calibration  every target in --targets at every --strengths, --nsamples each,
             targets split into --chunks lists per strength
plan         from plan rows (plan_pdiff_strengths.py / plan_commat_seeds.py) for one --tool:
             grouped by (strength, n_generate), at most --max_targets per task

  python preprocess/pdiff/make_pdiff_manifest.py calibration --tool AF3 \\
      --targets analyze/reports/pdiff_plan/calibration_targets_AF3.txt --out_dir .../53_pdiff_calibration
  python preprocess/pdiff/make_pdiff_manifest.py plan --tool AF3 --out_dir .../54_pdiff_decoys
Prints the matching sbatch command.
"""
import argparse
import os

import pandas as pd

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = {'AF3': 'preprocess/pdiff/sbatch_pdiff_af3_sweep.sh',
          'Boltz2': 'preprocess/pdiff/sbatch_pdiff_boltz2_sweep.sh'}


def chunks(xs, n_chunks=1, size=None):
    if size is not None:
        n_chunks = max(1, -(-len(xs) // size))
    return [xs[i::n_chunks] for i in range(n_chunks) if xs[i::n_chunks]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['calibration', 'plan'])
    ap.add_argument('--tool', required=True, choices=['AF3', 'Boltz2'])
    ap.add_argument('--out_dir', required=True, help='run output dir (OUT=); manifests go in <out_dir>/_manifests')
    ap.add_argument('--targets', help='calibration: target list file')
    ap.add_argument('--strengths', default='0.5 1 1.5 2 4 8 16')
    ap.add_argument('--nsamples', type=int, default=2)
    ap.add_argument('--chunks', type=int, default=6)
    ap.add_argument('--plan_rows', default=os.path.join(CODE_ROOT, 'analyze/reports/pdiff_plan/pdiff_plan_rows.csv'))
    ap.add_argument('--max_targets', type=int, default=20)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--throttle', type=int, default=8)
    ap.add_argument('--round', type=int, default=1,
                    help='top-up round into the same out_dir: tags/manifests get r<k>_ so nothing is overwritten '
                         '(use a new --seed too)')
    args = ap.parse_args()

    prefix = '' if args.round == 1 else f'r{args.round}_'
    mdir = os.path.join(args.out_dir, '_manifests',
                        f'{args.mode}_{args.tool}' + ('' if args.round == 1 else f'_r{args.round}'))
    os.makedirs(mdir, exist_ok=True)
    tasks = []  # (strength, targets, nsamples)
    if args.mode == 'calibration':
        targets = [l.strip() for l in open(args.targets) if l.strip()]
        for s in [float(x) for x in args.strengths.split()]:
            for c in chunks(targets, n_chunks=args.chunks):
                tasks.append((s, c, args.nsamples))
    else:
        r = pd.read_csv(args.plan_rows)
        r = r[(r.tool == args.tool) & r.strength.notna()]
        if r.basis.str.startswith('provisional').any():
            print(f'WARNING: {r.basis.str.startswith("provisional").sum()} provisional rows (no measured curve)')
        # one task = one strength and one sample count, so every input gets exactly its n_generate
        n_by = r.groupby(['strength', 'target']).n_generate.max().reset_index()
        for (s, n), g in n_by.groupby(['strength', 'n_generate']):
            for c in chunks(sorted(g.target), size=args.max_targets):
                tasks.append((float(s), c, int(n)))

    lines, n_dec = [], 0
    for i, (s, ts, n) in enumerate(tasks):
        # no '.' in the tag: Boltz-2 names its result dir after Path(input_dir).stem,
        # so 's0.5_c0' would become 's0' and every chunk of that strength would share it
        tag = prefix + f's{s:g}'.replace('.', 'p') + f'_c{i}'
        tf = os.path.join(mdir, f'{tag}.txt')
        with open(tf, 'w') as fh:
            fh.write('\n'.join(ts) + '\n')
        lines.append(f'{s:g}\t{tf}\t{n}\t{tag}')
        n_dec += n * len(ts)
    man = os.path.join(mdir, f'manifest_{args.tool}.tsv')
    with open(man, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'{len(tasks)} tasks, {n_dec} decoys, {len({t for _, ts, _ in tasks for t in ts})} targets -> {man}')
    print(f'MANIFEST={man} OUT={args.out_dir} SEED={args.seed} '
          f'sbatch --array=0-{len(tasks) - 1}%{args.throttle} {SCRIPT[args.tool]}')


if __name__ == '__main__':
    main()
