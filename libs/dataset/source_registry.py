"""
dataset/source_registry.py
==========================
For a given (pdb_id, epoch), resolve which decoy candidates are *available*
from each configured source.  Filesystem probing uses template-based path
expansion so only the needed paths are touched (no full directory scan).
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .config import DatasetSpec, SourceSpec

logger = logging.getLogger(__name__)


@dataclass
class DecoyCandidate:
    """One candidate decoy (or a batch of decoys from a single pickle file)."""
    source_name: str
    file_type: str                       # "graph_pickle" | "target_model_pickle"
    graph_path: Optional[str] = None     # set for graph_pickle + cached target_model_pickle
    rmsd_path: Optional[str] = None      # set for graph_pickle
    target_model_path: Optional[str] = None  # set for target_model_pickle
    metrics_path: Optional[str] = None   # Boltz2 etc.
    metrics_format: str = "csv"
    pdb_id: str = ""


# ──────────────────────────────────────────────────────────────
# Deterministic xtal gating
# ──────────────────────────────────────────────────────────────
def _stable_hash_01(pdb_id: str, seed: int) -> float:
    """Return a deterministic value in [0, 1) for (pdb_id, seed)."""
    h = hashlib.md5(f"{pdb_id}:{seed}".encode()).hexdigest()
    return int(h[:8], 16) / 0x1_0000_0000


def should_use_xtal(pdb_id: str, seed: int, xtal_gate_prob: float) -> bool:
    """Deterministic per-target xtal gate."""
    return _stable_hash_01(pdb_id, seed) < xtal_gate_prob


# ──────────────────────────────────────────────────────────────
# Template expansion
# ──────────────────────────────────────────────────────────────
def _expand(template: str, pdb: str, db_root: str) -> str:
    return template.format(pdb=pdb, db_root=db_root)


def _path_exists(path: str) -> bool:
    """Fast existence check (no glob)."""
    return os.path.exists(path)


def _glob_fallback(path: str) -> Optional[str]:
    """If exact path missing, try a glob with wildcard after the stem."""
    import glob as _glob
    d, fn = os.path.split(path)
    stem, ext = os.path.splitext(fn)
    matches = _glob.glob(os.path.join(d, f"{stem}*{ext}"))
    if matches:
        return sorted(matches)[0]
    return None


# ──────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────
class SourceRegistry:
    """Stateless helper – instantiated once with a ``DatasetSpec``."""

    def __init__(self, spec: DatasetSpec):
        self.spec = spec

    # ── public API ──
    def get_candidates(
        self,
        pdb_id: str,
        epoch: int,
    ) -> List[DecoyCandidate]:
        """Return **all available** candidates for *pdb_id*, already filtered
        by xtal gate.  Boltz2 quality filter is *not* applied here (done in
        mixing stage after metrics are loaded).

        *pdb_id* is the ID as it appears in the training list (typically old-format).
        Each source's ``pdb_id_type`` determines which ID namespace is used for
        template expansion (old vs new).  The ``spec.resolve_pdb_id`` helper
        converts between old ↔ new as needed.
        """
        candidates: List[DecoyCandidate] = []
        for src_name, src in self.spec.enabled_sources().items():
            resolved = self.spec.resolve_pdb_id(pdb_id, src.pdb_id_type)
            cands = self._resolve_source(resolved, src_name, src, epoch)
            candidates.extend(cands)
        return candidates

    # ── private ──
    def _resolve_source(
        self, pdb_id: str, name: str, src: SourceSpec, epoch: int
    ) -> List[DecoyCandidate]:
        db_root = self.spec.db_root

        # Xtal: always include when available (no per-target gate)
        # so that every target with xtal data gets at least one xtal decoy in the mix.

        if src.file_type == "graph_pickle":
            return self._resolve_graph_pickle(pdb_id, name, src, db_root)
        elif src.file_type == "target_model_pickle":
            return self._resolve_target_model_pickle(pdb_id, name, src, db_root)
        else:
            logger.warning("Unknown file_type %s for source %s", src.file_type, name)
            return []

    def _resolve_graph_pickle(
        self, pdb_id: str, name: str, src: SourceSpec, db_root: str
    ) -> List[DecoyCandidate]:
        graph_path = _expand(src.graph_template, pdb_id, db_root)
        if not _path_exists(graph_path):
            fb = _glob_fallback(graph_path)
            if fb is None:
                return []
            graph_path = fb

        rmsd_path = _expand(src.rmsd_template, pdb_id, db_root) if src.rmsd_template else None
        if rmsd_path and not _path_exists(rmsd_path):
            fb = _glob_fallback(rmsd_path)
            rmsd_path = fb  # None if still not found

        metrics_path = None
        if src.metrics_template:
            mp = _expand(src.metrics_template, pdb_id, db_root)
            if _path_exists(mp):
                metrics_path = mp

        return [DecoyCandidate(
            source_name=name,
            file_type="graph_pickle",
            graph_path=graph_path,
            rmsd_path=rmsd_path,
            metrics_path=metrics_path,
            metrics_format=src.metrics_format,
            pdb_id=pdb_id,
        )]

    def _resolve_target_model_pickle(
        self, pdb_id: str, name: str, src: SourceSpec, db_root: str
    ) -> List[DecoyCandidate]:
        # Check for cached graph pickle first
        cache_root = self.spec.cache_root or os.path.join(db_root, "_graph_cache")
        cache_path = os.path.join(cache_root, name, f"{pdb_id}.dat")
        if _path_exists(cache_path):
            # Cached version exists → treat as graph_pickle
            rmsd_cache = os.path.join(cache_root, name, f"{pdb_id}.rmsd")
            return [DecoyCandidate(
                source_name=name,
                file_type="graph_pickle",       # already converted
                graph_path=cache_path,
                rmsd_path=rmsd_cache if _path_exists(rmsd_cache) else None,
                pdb_id=pdb_id,
            )]

        # Not cached → need original Target/Model pickle
        pkl_path = _expand(src.target_pickle_template, pdb_id, db_root)
        if not _path_exists(pkl_path):
            fb = _glob_fallback(pkl_path)
            if fb is None:
                return []
            pkl_path = fb

        metrics_path = None
        if src.metrics_template:
            mp = _expand(src.metrics_template, pdb_id, db_root)
            if _path_exists(mp):
                metrics_path = mp

        return [DecoyCandidate(
            source_name=name,
            file_type="target_model_pickle",
            target_model_path=pkl_path,
            metrics_path=metrics_path,
            metrics_format=src.metrics_format,
            pdb_id=pdb_id,
        )]
