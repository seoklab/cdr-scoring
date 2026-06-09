"""
dataset/conversion.py
=====================
Convert Target/Model pickle → list[DGLGraph] + RMSD list, then cache.

* Atomic write (tmp → os.replace) to avoid partial files.
* ``filelock`` for multi-worker / multi-GPU race protection.
"""
from __future__ import annotations

import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Try to import filelock; fall back to a no-op context manager
try:
    from filelock import FileLock
except ImportError:
    class FileLock:  # type: ignore[no-redef]
        """Dummy lock when filelock is not installed."""
        def __init__(self, path: str, timeout: int = -1):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass


def convert_and_cache(
    target_pickle_path: str,
    cache_dir: str,
    source_name: str,
    pdb_id: str,
    *,
    dist_cutoff_center: float = 10.0,
    random_range: float = 0.0,
    max_neighbors: int = 0,
    use_all_atom: bool = False,
    h3_range: Tuple[int, int] = (95, 102),
) -> Tuple[Optional[str], Optional[str]]:
    """Load a Target pickle, generate graphs, and write cache.

    Returns (graph_cache_path, rmsd_cache_path) or (None, None) on failure.
    """
    cache_source_dir = os.path.join(cache_dir, source_name)
    os.makedirs(cache_source_dir, exist_ok=True)

    graph_cache = os.path.join(cache_source_dir, f"{pdb_id}.dat")
    rmsd_cache = os.path.join(cache_source_dir, f"{pdb_id}.rmsd")
    lock_path = graph_cache + ".lock"

    # Fast path: already cached
    if os.path.exists(graph_cache) and os.path.exists(rmsd_cache):
        return graph_cache, rmsd_cache

    with FileLock(lock_path, timeout=300):
        # Double-check after acquiring lock (another worker may have finished)
        if os.path.exists(graph_cache) and os.path.exists(rmsd_cache):
            return graph_cache, rmsd_cache

        try:
            # Lazy import to avoid circular deps / heavy startup cost
            from data_loading.graph_generation_from_target import generate_graphs_from_target
            from data_loading.data_module import _load_target_pickle

            target = _load_target_pickle(target_pickle_path)

            graphs, rmsds, ranks = generate_graphs_from_target(
                target,
                dist_cutoff_center=dist_cutoff_center,
                random_range=random_range,
                max_neighbors=max_neighbors,
                use_all_atom=use_all_atom,
                h3_range=tuple(h3_range),
            )

            if not graphs:
                logger.warning("No graphs generated for %s from %s", pdb_id, target_pickle_path)
                return None, None

            # Atomic write – graph list
            _atomic_pickle(graph_cache, graphs)
            # Atomic write – RMSD tensor
            rmsd_tensor = torch.tensor(rmsds, dtype=torch.float32)
            _atomic_pickle(rmsd_cache, rmsd_tensor)

            logger.info("Cached %d graphs for %s/%s", len(graphs), source_name, pdb_id)
            return graph_cache, rmsd_cache

        except Exception:
            logger.exception("Failed to convert %s/%s", source_name, pdb_id)
            return None, None


def _atomic_pickle(dest: str, obj) -> None:
    """Write *obj* via pickle to *dest* atomically."""
    d = os.path.dirname(dest)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, dest)
    except BaseException:
        # Clean up partial file
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
