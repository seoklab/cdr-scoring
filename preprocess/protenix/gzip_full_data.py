#!/usr/bin/env python
"""Shrink Protenix per-sample confidence dumps in place (lossless gzip).

``full_data_sample_*.json`` holds three token x token float64 matrices written as JSON
text, so a 970-token target costs 39 MB per sample and 4 GB per target -- 143 GB for the
185 vanilla test targets, against ~113 GB of quota headroom. gzip gives 9.4x at ~1 s per
file and stays plain JSON (``gzip.open`` + ``json.load``), so nothing downstream has to
know a new format.

Only targets that already have ``.done`` are touched, so a file still being written is
never compressed. Each original is removed only after its .gz passes ``gzip -t``.

  python preprocess/protenix/gzip_full_data.py [--root DIR] [--watch] [--jobs N]
"""
import argparse
import gzip
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = '/data/galaxy4/user/sujin/projects/cdr-scoring/cdr-data/protenix/vanilla_test/out'
PATTERN = '_full_data_sample_'


def pending(root):
    for target in sorted(os.listdir(root)):
        tdir = os.path.join(root, target)
        if not os.path.isdir(tdir) or not os.path.exists(os.path.join(tdir, '.done')):
            continue
        for dirpath, _, names in os.walk(tdir):
            for n in names:
                if PATTERN in n and n.endswith('.json'):
                    yield os.path.join(dirpath, n)


def compress(path):
    """gzip one file, verify, then drop the original. Returns (bytes_before, bytes_after)."""
    dst = path + '.gz'
    tmp = dst + '.part'
    before = os.path.getsize(path)
    with open(path, 'rb') as fi, gzip.open(tmp, 'wb', compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
    if subprocess.run(['gzip', '-t', tmp], capture_output=True).returncode != 0:
        os.remove(tmp)
        raise RuntimeError(f'gzip verify failed: {path}')
    os.replace(tmp, dst)
    os.remove(path)
    return before, os.path.getsize(dst)


def sweep(root, jobs):
    files = list(pending(root))
    if not files:
        return 0, 0, 0
    tot_b = tot_a = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for b, a in ex.map(compress, files):
            tot_b += b
            tot_a += a
    return len(files), tot_b, tot_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--watch', action='store_true', help='keep sweeping while the array runs')
    ap.add_argument('--interval', type=int, default=300)
    ap.add_argument('--job', default='', help='stop watching once this slurm job is gone')
    a = ap.parse_args()
    saved = 0
    while True:
        n, b, c = sweep(a.root, a.jobs)
        if n:
            saved += b - c
            print(f'{time.strftime("%H:%M:%S")} {n} files {b/2**30:.1f}G -> {c/2**30:.1f}G '
                  f'(cumulative saved {saved/2**30:.1f}G)', flush=True)
        if not a.watch:
            break
        if a.job:
            q = subprocess.run(['squeue', '-j', a.job, '-h'], capture_output=True, text=True)
            if not q.stdout.strip():
                n, b, c = sweep(a.root, a.jobs)  # final pass after the last target lands
                saved += b - c
                print(f'array {a.job} finished; last pass {n} files, saved {saved/2**30:.1f}G total',
                      flush=True)
                return 0
        time.sleep(a.interval)
    return 0


if __name__ == '__main__':
    sys.exit(main())
