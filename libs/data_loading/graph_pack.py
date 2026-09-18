"""Indexed random-access container for per-(target, source) decoy graphs.

Motivation
----------
The on-the-fly path spends ~55% of its per-target time inside
``_load_target_pickle`` (deserializing pdb2dict Target pickles: ~1M
``numpy.frombuffer`` + ~860k namedtuple ``_asdict`` calls) and another ~26% in
``build_graph``. Caching graphs removes both, but a naive cache repeats the same
mistake in a new place: one pickle per (target, source) holding all N decoys
forces a full ``pickle.load`` of every decoy even when the tier sampler only
wants 40 of them.

This module stores one file per (target, source) with:
  * a JSON header holding the decoy identity keys and node_ptr / edge_ptr,
  * each node/edge field as a single contiguous, 4096-aligned blob.

Reading K of N decoys touches only ``ptr[i]:ptr[i+1]`` of each blob. There is no
Python-object deserialization at all — slices are viewed straight out of an
``np.memmap`` (or ``os.pread``) and handed to ``torch.from_numpy``.

Because the on-disk layout is *already* the concatenated form that
``dgl.batch`` produces, ``read_batched`` builds the batched graph directly and
skips both per-decoy ``DGLGraph`` construction and ``dgl.batch``.

Layout
------
    0                       8   magic  b'CDRGPK01'
    8                      16   uint64 header_len (little endian)
    16        16+header_len     JSON header (utf-8)
    ...aligned to 4096...
    blob[field_0]                (4096-aligned)
    blob[field_1]
    ...

Topology (``src``/``dst``) is stored as node ids *local to each decoy*, so a
slice is valid on its own; ``read_batched`` adds the batch node offset.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import dgl

MAGIC = b'CDRGPK01'
ALIGN = 4096
_PREFIX_NODE = 'n:'
_PREFIX_EDGE = 'e:'

_TORCH_DTYPE = {
    'float32': torch.float32, 'float16': torch.float16,
    'int64': torch.int64, 'int32': torch.int32, 'int16': torch.int16,
    'uint8': torch.uint8, 'bool': torch.bool,
}


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def _writable(arr: np.ndarray) -> np.ndarray:
    """Ensure a writable array (mmap/pread slices are read-only views).

    ``torch.from_numpy`` on a read-only view warns and forbids in-place ops; a
    copy here matches the on-the-fly path's writable tensors. Arrays produced by
    masking/fancy-indexing are already writable and pass through free.
    """
    return arr if arr.flags.writeable else arr.copy()


def _key_str(key) -> str:
    """Canonical string form of a decoy identity key (source, seed, sample)."""
    src, seed, samp = key
    return f'{src}|{"" if seed is None else int(seed)}|{int(samp)}'


# Column of ``e:e_ij`` that holds the CA-CA distance used for the edge mask.
# build_graph (both heavy and all-atom modes) puts the CA-CA distance first, so
# a read-time subgraph to a narrower cutoff is ``e_ij[:, 0] <= cutoff``.
CA_DIST_COL = 0


# ──────────────────────────────────────────────────────────────────────────
# cache-generation / staleness signatures
# ──────────────────────────────────────────────────────────────────────────

def compute_code_sig(extra_paths: Sequence[str] = ()) -> str:
    """Hash the graph-construction source so a logic change invalidates caches.

    Hashes the *content* (not the path) of the modules that actually build a
    graph — ``graph_generation_from_target`` (holds create_dictionary_from_model,
    build_edge_mask, build_graph) and its coord helper ``coords_rosetta6d`` —
    keyed by basename so the signature is identical regardless of whether the
    repo is reached via /home/... or /data/... (they are the same tree). This
    module (graph_pack) is deliberately excluded: it only stores/reads graphs and
    does not affect graph *content*, so editing the container must not invalidate
    a cache. Ordered by basename for determinism.
    """
    import hashlib
    here = os.path.dirname(os.path.abspath(__file__))
    files = [os.path.join(here, 'graph_generation_from_target.py'),
             os.path.join(here, 'coords_rosetta6d.py')]
    files += list(extra_paths)
    h = hashlib.sha256()
    for p in sorted(files, key=lambda x: os.path.basename(x)):
        h.update(os.path.basename(p).encode('utf-8'))
        try:
            with open(p, 'rb') as f:
                h.update(f.read())
        except OSError:
            h.update(b'<missing>')
    return h.hexdigest()[:16]


# Graph-build params split into two roles for staleness checking:
#   * EXACT-match params go into build_sig — they change the node set, features,
#     or edge candidates and the read path does NOT re-apply them, so a cache
#     built with different values is simply wrong and must be rebuilt.
#   * The COVERAGE param (dist_cutoff) is NOT in build_sig; instead the loader
#     checks the cache envelope covers the training range
#     (envelope_dist_cutoff >= dist_cutoff_center + random_range), because the
#     read narrows edges by distance from the wide envelope.
BUILD_SIG_EXACT_KEYS = (
    'max_neighbors', 'use_all_atom', 'cdr_context_cutoff', 'max_context_residues',
    'h3_range', 'cdr_ranges', 'task_scope',
)
COVERAGE_KEY = 'dist_cutoff'


def compute_build_sig(build_params: dict, code_sig: Optional[str] = None) -> str:
    """Signature naming a cache *generation* = (exact-match params, code).

    ``build_params`` is the full graph-build param dict; only the
    ``BUILD_SIG_EXACT_KEYS`` subset feeds the hash (``dist_cutoff`` is a coverage
    dimension checked separately). ``code_sig`` defaults to ``compute_code_sig``.
    A change in any exact-match param or in the graph code yields a new
    signature → a new cache directory rather than an in-place overwrite.
    """
    import hashlib
    if code_sig is None:
        code_sig = compute_code_sig()

    def _norm(v):
        return list(v) if isinstance(v, (tuple, list)) else v
    payload = {
        'exact': {k: _norm(build_params.get(k)) for k in BUILD_SIG_EXACT_KEYS},
        'code_sig': code_sig,
    }
    blob = json.dumps(payload, separators=(',', ':'), sort_keys=True).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()[:16]


def compute_struct_sig(source_files: Sequence[str], keys: Sequence) -> str:
    """Per-pack signature detecting upstream structure/decoy-set changes.

    Hashes the (path, size, mtime) of the source structure file(s) the pack was
    built from, together with the decoy identity keys. A changed pdb2dict pickle
    or a changed decoy list flips this, so the loader rebuilds *only* that
    (target, source) pack.
    """
    import hashlib
    h = hashlib.sha256()
    for p in source_files:
        try:
            st = os.stat(p)
            h.update(f'{os.path.abspath(p)}|{st.st_size}|{int(st.st_mtime)}'.encode('utf-8'))
        except OSError:
            h.update(f'{p}|<missing>'.encode('utf-8'))
    for k in keys:
        h.update(_key_str(k).encode('utf-8'))
    return h.hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────────────
# writer
# ──────────────────────────────────────────────────────────────────────────

def write_pack(path: str, graphs: Sequence[dgl.DGLGraph], keys: Sequence,
               dtype_overrides: Optional[Dict[str, str]] = None,
               build_sig: Optional[str] = None,
               struct_sig: Optional[str] = None,
               envelope: Optional[dict] = None,
               meta: Optional[dict] = None) -> dict:
    """Write ``graphs`` (single, unbatched) to an indexed container.

    ``keys[i]`` is the decoy identity key ``(source, seed, sample)`` of
    ``graphs[i]``. ``dtype_overrides`` maps a field name (e.g. ``'e:e_ij'``) to a
    numpy dtype string to downcast on write; all fields keep their original
    dtype by default. ``build_sig``/``struct_sig``/``envelope`` are recorded in
    the header for staleness checks (see the signature helpers above); ``meta``
    is free-form (e.g. target_id, source, source_files).
    """
    assert len(graphs) == len(keys), 'graphs/keys length mismatch'
    if not graphs:
        raise ValueError('write_pack: no graphs')
    dtype_overrides = dtype_overrides or {}

    n_nodes = np.array([int(g.num_nodes()) for g in graphs], dtype=np.int64)
    n_edges = np.array([int(g.num_edges()) for g in graphs], dtype=np.int64)
    node_ptr = np.concatenate([[0], np.cumsum(n_nodes)]).astype(np.int64)
    edge_ptr = np.concatenate([[0], np.cumsum(n_edges)]).astype(np.int64)

    # Discover the schema from graph 0 and require every graph to match it, so a
    # slice-and-concatenate read can never silently mix schemas.
    schema: List[Tuple[str, str, tuple]] = []
    g0 = graphs[0]
    for k, v in g0.ndata.items():
        schema.append((_PREFIX_NODE + k, str(v.dtype).replace('torch.', ''), tuple(v.shape[1:])))
    for k, v in g0.edata.items():
        schema.append((_PREFIX_EDGE + k, str(v.dtype).replace('torch.', ''), tuple(v.shape[1:])))
    schema.append(('t:src', 'int64', ()))
    schema.append(('t:dst', 'int64', ()))

    for gi, g in enumerate(graphs):
        if set(g.ndata.keys()) != set(g0.ndata.keys()) or set(g.edata.keys()) != set(g0.edata.keys()):
            raise ValueError(f'write_pack: graph {gi} schema differs from graph 0')

    fields = {}
    off = 0  # relative to data section start; fixed up after header size is known
    for name, dt, row_shape in schema:
        dt_out = dtype_overrides.get(name, dt)
        row_items = int(np.prod(row_shape)) if row_shape else 1
        row_bytes = row_items * np.dtype(dt_out).itemsize
        rows = int(node_ptr[-1]) if name.startswith(_PREFIX_NODE) else int(edge_ptr[-1])
        fields[name] = {
            'kind': 'node' if name.startswith(_PREFIX_NODE) else 'edge',
            'dtype': dt_out, 'row_shape': list(row_shape),
            'row_bytes': row_bytes, 'rows': rows,
            'rel_offset': off, 'nbytes': rows * row_bytes,
        }
        off = _align(off + rows * row_bytes)
    data_bytes = off

    header = {
        'version': 1,
        'n_graphs': len(graphs),
        'keys': [list(k) if not isinstance(k, str) else k for k in keys],
        'key_strs': [_key_str(k) for k in keys],
        'node_ptr': node_ptr.tolist(),
        'edge_ptr': edge_ptr.tolist(),
        'fields': fields,
        'idtype': 'int64',
        'build_sig': build_sig,
        'struct_sig': struct_sig,
        'envelope': envelope or {},
        'ca_dist_col': CA_DIST_COL,
        'meta': meta or {},
    }
    hdr = json.dumps(header, separators=(',', ':')).encode('utf-8')
    data_start = _align(16 + len(hdr))
    for f in fields.values():
        f['offset'] = data_start + f['rel_offset']
    # re-serialize with absolute offsets (length may change; pad keeps it aligned)
    header['fields'] = fields
    hdr2 = json.dumps(header, separators=(',', ':')).encode('utf-8')
    if _align(16 + len(hdr2)) != data_start:
        # absolute offsets grew the header past its alignment slot: recompute once
        data_start = _align(16 + len(hdr2))
        for f in fields.values():
            f['offset'] = data_start + f['rel_offset']
        header['fields'] = fields
        hdr2 = json.dumps(header, separators=(',', ':')).encode('utf-8')
        assert _align(16 + len(hdr2)) == data_start, 'header size did not converge'

    tmp = path + '.tmp'
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<Q', len(hdr2)))
        f.write(hdr2)
        f.write(b'\0' * (data_start - (16 + len(hdr2))))
        f.truncate(data_start + data_bytes)
        for name, meta in fields.items():
            f.seek(meta['offset'])
            np_dt = np.dtype(meta['dtype'])
            for gi, g in enumerate(graphs):
                if name == 't:src':
                    arr = g.edges()[0].numpy()
                elif name == 't:dst':
                    arr = g.edges()[1].numpy()
                elif meta['kind'] == 'node':
                    arr = g.ndata[name[len(_PREFIX_NODE):]].numpy()
                else:
                    arr = g.edata[name[len(_PREFIX_EDGE):]].numpy()
                f.write(np.ascontiguousarray(arr, dtype=np_dt).tobytes())
    os.replace(tmp, path)
    return {'path': path, 'n_graphs': len(graphs), 'bytes': data_start + data_bytes,
            'header_bytes': len(hdr2), 'total_nodes': int(node_ptr[-1]),
            'total_edges': int(edge_ptr[-1])}


# ──────────────────────────────────────────────────────────────────────────
# reader
# ──────────────────────────────────────────────────────────────────────────

class GraphPack:
    """Random-access reader. Open once per DataLoader worker and reuse.

    ``mode='mmap'`` (default) maps the file once and lets the page cache serve
    slices; ``mode='pread'`` issues one positional read per (field, run) and is
    the better choice when the file is far larger than RAM or lives on a
    latency-bound mount where mmap readahead misbehaves.
    """

    __slots__ = ('path', 'mode', '_fd', '_mm', 'header', 'n_graphs', 'node_ptr',
                 'edge_ptr', 'fields', '_key_to_idx', 'keys', 'stat_bytes',
                 'stat_reads', 'build_sig', 'struct_sig', 'envelope', 'ca_dist_col')

    def __init__(self, path: str, mode: str = 'mmap'):
        self.path = str(path)
        self.mode = mode
        self.stat_bytes = 0
        self.stat_reads = 0
        self._fd = os.open(self.path, os.O_RDONLY)
        head = os.pread(self._fd, 16, 0)
        if head[:8] != MAGIC:
            os.close(self._fd)
            raise ValueError(f'{path}: bad magic {head[:8]!r}')
        hlen = struct.unpack('<Q', head[8:16])[0]
        self.header = json.loads(os.pread(self._fd, hlen, 16).decode('utf-8'))
        self.n_graphs = int(self.header['n_graphs'])
        self.node_ptr = np.asarray(self.header['node_ptr'], dtype=np.int64)
        self.edge_ptr = np.asarray(self.header['edge_ptr'], dtype=np.int64)
        self.fields = self.header['fields']
        self.keys = [tuple(k) if isinstance(k, list) else k for k in self.header['keys']]
        self._key_to_idx = {s: i for i, s in enumerate(self.header['key_strs'])}
        self.build_sig = self.header.get('build_sig')
        self.struct_sig = self.header.get('struct_sig')
        self.envelope = self.header.get('envelope', {})
        self.ca_dist_col = int(self.header.get('ca_dist_col', CA_DIST_COL))
        self._mm = None
        if mode == 'mmap':
            self._mm = np.memmap(self.path, dtype=np.uint8, mode='r')

    # -- lifecycle ------------------------------------------------------
    def close(self):
        self._mm = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def drop_cache(self):
        """Best-effort page-cache eviction for this file (cold-read testing)."""
        try:
            os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass

    # -- lookup ---------------------------------------------------------
    def index_of(self, key) -> int:
        return self._key_to_idx[_key_str(key)]

    def indices_of(self, keys: Iterable) -> List[int]:
        """Indices for the keys present in this pack (missing keys skipped)."""
        out = []
        for k in keys:
            i = self._key_to_idx.get(_key_str(k))
            if i is not None:
                out.append(i)
        return out

    # -- raw slice reads ------------------------------------------------
    @staticmethod
    def _runs(idx: Sequence[int]) -> List[Tuple[int, int]]:
        """Collapse a sorted index list into [start, stop) contiguous runs."""
        runs = []
        for i in idx:
            if runs and i == runs[-1][1]:
                runs[-1][1] = i + 1
            else:
                runs.append([i, i + 1])
        return [(a, b) for a, b in runs]

    def _read_field(self, name: str, idx: Sequence[int]) -> np.ndarray:
        meta = self.fields[name]
        ptr = self.node_ptr if meta['kind'] == 'node' else self.edge_ptr
        np_dt = np.dtype(meta['dtype'])
        row_bytes = meta['row_bytes']
        base = meta['offset']
        row_shape = tuple(meta['row_shape'])
        parts = []
        for a, b in self._runs(idx):
            r0, r1 = int(ptr[a]), int(ptr[b])
            nbytes = (r1 - r0) * row_bytes
            if nbytes == 0:
                continue
            off = base + r0 * row_bytes
            if self._mm is not None:
                buf = self._mm[off:off + nbytes]
                arr = np.frombuffer(buf, dtype=np_dt)
            else:
                raw = os.pread(self._fd, nbytes, off)
                arr = np.frombuffer(raw, dtype=np_dt)
            self.stat_bytes += nbytes
            self.stat_reads += 1
            parts.append(arr.reshape((r1 - r0,) + row_shape))
        if not parts:
            return np.empty((0,) + row_shape, dtype=np_dt)
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

    # -- graph reconstruction -------------------------------------------
    def read_batched(self, idx: Sequence[int], edge_cutoff=None) -> dgl.DGLGraph:
        """Build the batched graph for ``idx`` directly from concatenated slices.

        Equivalent to ``dgl.batch([graphs[i] for i in idx])`` but with no
        per-decoy DGLGraph construction and no dgl.batch pass.

        ``edge_cutoff`` implements the wide-envelope trick: the pack is built at
        a wide CA-CA cutoff (e.g. 11.5 A) and this narrows it at read time by
        keeping only edges with ``e_ij[:, ca_dist_col] <= cutoff``. It may be

          * ``None``   — use every cached edge (the built envelope),
          * a float    — one cutoff for all decoys, or
          * a sequence of length ``len(idx)`` — a per-decoy cutoff (used to
            reproduce ``random_range`` jitter, one draw per decoy).

        Because the pack keeps the nearest ``max_neighbors`` within the wide
        radius, distance-narrowing reproduces any ``(cutoff <= envelope,
        max_neighbors <= envelope)`` edge set exactly.
        """
        idx = sorted(int(i) for i in idx)
        if not idx:
            raise ValueError('read_batched: empty index list')
        n_nodes = (self.node_ptr[1:] - self.node_ptr[:-1])[idx]
        n_edges = (self.edge_ptr[1:] - self.edge_ptr[:-1])[idx]

        src = np.ascontiguousarray(self._read_field('t:src', idx))
        dst = np.ascontiguousarray(self._read_field('t:dst', idx))

        # Read edge features first; if narrowing, build the keep-mask from e_ij.
        edge_arrays = {}
        for name, meta in self.fields.items():
            if meta['kind'] == 'edge':
                edge_arrays[name] = np.ascontiguousarray(self._read_field(name, idx))

        # decoy id per (cached) edge — needed both for the node shift and for
        # recomputing per-decoy edge counts after masking.
        decoy_of_edge = np.repeat(np.arange(len(idx), dtype=np.int64), n_edges)

        if edge_cutoff is not None:
            eij = edge_arrays.get(_PREFIX_EDGE + 'e_ij')
            if eij is None:
                raise ValueError('read_batched: edge_cutoff set but e_ij absent')
            ca = eij[:, self.ca_dist_col]
            if np.isscalar(edge_cutoff):
                cut_per_edge = float(edge_cutoff)
            else:
                cut = np.asarray(edge_cutoff, dtype=np.float64)
                if cut.shape[0] != len(idx):
                    raise ValueError('edge_cutoff length must equal len(idx)')
                cut_per_edge = cut[decoy_of_edge]
            keep = ca <= cut_per_edge
            src = src[keep]
            dst = dst[keep]
            decoy_of_edge = decoy_of_edge[keep]
            for name in edge_arrays:
                edge_arrays[name] = edge_arrays[name][keep]
            n_edges = np.bincount(decoy_of_edge, minlength=len(idx)).astype(np.int64)

        # local -> batched node ids (node offsets are unchanged by edge masking)
        node_off = np.concatenate([[0], np.cumsum(n_nodes)[:-1]]).astype(np.int64)
        shift = node_off[decoy_of_edge]
        src_t = torch.from_numpy(_writable(src + shift))
        dst_t = torch.from_numpy(_writable(dst + shift))

        g = dgl.graph((src_t, dst_t), num_nodes=int(n_nodes.sum()))
        for name, meta in self.fields.items():
            if name.startswith('t:'):
                continue
            if meta['kind'] == 'edge':
                g.edata[name[len(_PREFIX_EDGE):]] = torch.from_numpy(_writable(edge_arrays[name]))
            else:
                arr = np.ascontiguousarray(self._read_field(name, idx))
                g.ndata[name[len(_PREFIX_NODE):]] = torch.from_numpy(_writable(arr))
        # readout_nodes / batch_num_nodes() in the model need this
        g.set_batch_num_nodes(torch.from_numpy(np.ascontiguousarray(n_nodes)))
        g.set_batch_num_edges(torch.from_numpy(np.ascontiguousarray(n_edges)))
        return g

    def read_graphs(self, idx: Sequence[int], edge_cutoff=None) -> List[dgl.DGLGraph]:
        """Individual graphs (use ``read_batched`` in the training path)."""
        return dgl.unbatch(self.read_batched(idx, edge_cutoff=edge_cutoff))
