#!/usr/bin/env python
"""Share one Protenix target queue between SLURM arrays that are already queued.

scripts/sbatch_protenix_vanilla_test.sh reads its target from row (task index + 2) of
its TARGETS csv when the task starts, so rewriting rows that no task has read yet moves
work between arrays without cancelling one (and losing its place in the queue).

Every tick:
  1. squeue gives each array its running and pending task indices. Tasks start in
     index order, so the pending ones are next..max.
  2. A target is taken if out/<t>/ exists, or it sits on a running row, or on one of
     the next `conc + 2` pending rows of an array that is not held (those can start
     while the file is being replaced, so they are never rewritten).
  3. The free targets are dealt out, largest first, in proportion to
     running tasks x relative GPU speed. Every array keeps `conc + 2` of them so a
     stalled array has work the moment its node frees up.
  4. Rows past the locked ones get the new assignment, then skip rows that point at
     a target whose .done already exists, and the file is swapped in atomically.

  python rebalance_protenix_queue.py --once --dry_run      # print the plan only
  python rebalance_protenix_queue.py --interval 120         # loop until all done
Touch <root>/rebalance.stop to end the loop.
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = '/data/galaxy4/user/sujin/projects/cdr-scoring/cdr-data/protenix/train_valid'
# job id, its TARGETS csv, array throttle, GPU speed relative to an RTX A5000
ARRAYS = [
    dict(name='nova004', job='5496779', csv='targets_by_size.csv', conc=8, speed=1.0),
    dict(name='nova003', job='5596174', csv='targets_nova003.csv', conc=6, speed=0.65),
]
SKIP = '../_skip/none'          # DEST = out/../_skip/none, whose .done exists
HEADER = ['target', 'split', 'n_chains', 'n_tokens']


def log(msg):
    print(f'{datetime.now():%m-%d %H:%M:%S} {msg}', flush=True)


def read_rows(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == HEADER, f'{path}: unexpected header {rows[0]}'
    return rows[1:]


def write_rows(path, rows):
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(HEADER)
        w.writerows(rows)
    os.replace(tmp, path)


def array_state(job):
    """Running and pending task indices of one array, and whether it is held."""
    out = subprocess.run(['squeue', '-j', job, '-h', '-r', '-o', '%i|%T|%r'],
                         capture_output=True, text=True)
    if out.returncode != 0 and 'Invalid job id' not in out.stderr:
        raise RuntimeError(f'squeue {job}: {out.stderr.strip()}')
    running, pending, held = [], [], True
    for line in out.stdout.splitlines():   # reasons can contain spaces
        if not line.strip():
            continue
        tid, state, reason = line.split('|', 2)
        if '_' not in tid or '[' in tid:
            continue
        idx = int(tid.split('_')[1])
        if state == 'RUNNING':
            running.append(idx)
        elif state == 'PENDING':
            pending.append(idx)
            held &= reason == 'JobHeldUser'
    return sorted(running), sorted(pending), held and bool(pending)


def plan(master, sizes, arrays):
    out_dir = os.path.join(ROOT, 'out')
    started = {t for t in master if os.path.isdir(os.path.join(out_dir, t))}
    owner = {}                  # target -> array name, for running/locked rows
    for a in arrays:
        a['rows'] = read_rows(os.path.join(ROOT, a['csv']))
        a['running'], a['pending'], a['held'] = array_state(a['job'])
        if a['pending']:
            nxt = a['pending'][0]
            a['lock_end'] = nxt + (0 if a['held'] else a['conc'] + 2)
            a['last'] = a['pending'][-1]
        else:
            a['lock_end'] = a['last'] = None
        keep = list(a['running'])
        if a['pending']:
            keep += range(a['pending'][0], a['lock_end'])
        for i in keep:
            if i >= len(a['rows']):
                continue
            t = a['rows'][i][0]
            if t == SKIP:
                continue
            if t in owner and owner[t] != a['name']:
                log(f'WARNING {t} is on a running/locked row of both {owner[t]} and {a["name"]}')
            owner[t] = a['name']
    free = [t for t in master if t not in started and t not in owner]

    live = [a for a in arrays if a['pending']]
    for a in live:
        a['cap'] = a['last'] - a['lock_end'] + 1
        a['w'] = len(a['running']) * a['speed']
    if live and sum(a['w'] for a in live) == 0:
        for a in live:
            a['w'] = a['conc'] * a['speed']

    # quotas: conc + 2 each, the rest by weight, capped by the rows each array has left.
    # A pending array that is neither held nor running already has conc + 2 targets on
    # its locked rows, and the next tick refills it once it starts, so it gets no more.
    quota = {a['name']: 0 if (not a['held'] and not a['running']) else min(a['conc'] + 2, a['cap'])
             for a in live}
    rest = len(free) - sum(quota.values())
    while rest > 0:
        open_ = [a for a in live if quota[a['name']] < a['cap']]
        if not open_:
            break
        # an array with no running task gets nothing extra, unless no open array is running
        w = {a['name']: a['w'] for a in open_}
        if sum(w.values()) == 0:
            w = {a['name']: 1.0 for a in open_}
        wsum = sum(w.values())
        given = 0
        for a in open_:
            add = min(a['cap'] - quota[a['name']], int(rest * w[a['name']] / wsum))
            quota[a['name']] += add
            given += add
        if given == 0:          # rounding left a remainder: hand it out one by one
            for a in sorted(open_, key=lambda a: -w[a['name']]):
                if rest - given > 0 and quota[a['name']] < a['cap'] and w[a['name']] > 0:
                    quota[a['name']] += 1
                    given += 1
        rest -= given
    if rest > 0:
        log(f'WARNING {rest} free targets do not fit in any array')

    # interleave, largest first, so both arrays get large and small targets
    assigned = {a['name']: [] for a in live}
    for t in free:
        cand = [a for a in live if len(assigned[a['name']]) < quota[a['name']]]
        if not cand:
            break
        a = min(cand, key=lambda a: (len(assigned[a['name']]) + 1) / max(quota[a['name']], 1))
        assigned[a['name']].append(t)

    for a in live:
        new = [list(r) for r in a['rows']]
        lst = assigned[a['name']]
        for k, i in enumerate(range(a['lock_end'], a['last'] + 1)):
            new[i] = ([lst[k], *sizes[lst[k]]]
                      if k < len(lst) else [SKIP, 'skip', '0', '0'])
        a['new_rows'] = new
    done = sum(os.path.exists(os.path.join(out_dir, t, '.done')) for t in master)
    return dict(started=len(started), done=done, free=len(free), owner=len(owner),
                assigned={k: len(v) for k, v in assigned.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--master', default=os.path.join(ROOT, 'targets_queue_master.csv'))
    p.add_argument('--interval', type=int, default=120)
    p.add_argument('--once', action='store_true')
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()

    os.makedirs(os.path.join(ROOT, '_skip', 'none'), exist_ok=True)
    open(os.path.join(ROOT, '_skip', 'none', '.done'), 'a').close()
    master_rows = read_rows(args.master)
    master = [r[0] for r in master_rows]
    sizes = {r[0]: r[1:] for r in master_rows}          # split, n_chains, n_tokens

    while True:
        arrays = [dict(a) for a in ARRAYS]
        try:
            s = plan(master, sizes, arrays)
        except Exception as e:  # squeue hiccup: try again next tick
            log(f'ERROR {e!r}')
            s = None
        if s is not None:
            parts = []
            for a in arrays:
                nxt = a['pending'][0] if a['pending'] else '-'
                parts.append(f"{a['name']} run {len(a['running'])} next {nxt}"
                             f"{' HELD' if a['held'] else ''} +{s['assigned'].get(a['name'], 0)}")
                if not args.dry_run and 'new_rows' in a and a['new_rows'] != a['rows']:
                    write_rows(os.path.join(ROOT, a['csv']), a['new_rows'])
            log(f"done {s['done']}/{len(master)} started {s['started']} locked {s['owner']} "
                f"free {s['free']} | " + ' | '.join(parts) + (' [dry run]' if args.dry_run else ''))
            if s['done'] == len(master):
                log('all targets done')
                return
            if not any(a['pending'] or a['running'] for a in arrays):
                log('no array has tasks left')
                return
        if args.once or os.path.exists(os.path.join(ROOT, 'rebalance.stop')):
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    sys.exit(main())
