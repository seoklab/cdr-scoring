"""
dataset/precomputed_metrics.py
==============================
Load precomputed per-decoy loop metrics (parquet) and expose a fast lookup
keyed by ``(target_id, seed, sample)`` so that training does **not** have to
recompute loop RMSD / lDDT from structures.

The parquet files are produced by ``preprocess/precompute_decoy_metrics.py`` and
live under one directory per source, e.g.::

    {root}/Boltz2/metrics/loop_metrics.parquet
    {root}/ComMat/metrics/loop_metrics.parquet

Each row carries the decoy identity columns ``target_id, source, seed, sample``
and the loop metric columns ``global_loop_rmsd, global_loop_lddt,
H3_loop_rmsd, H3_loop_lddt, ...``.

The decoy identity used as the join key is derived **exactly** the same way the
precompute script derives it (see ``decoy_identity_for_lookup`` below), so that
``(seed, sample)`` produced at training time matches the parquet rows.
"""
from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Module-level cache of parsed parquet tables, keyed by absolute parquet path.
# train.py rebuilds MyDataset (and thus the store) every epoch; caching here
# avoids re-reading/re-parsing the (large) parquet files on each epoch.
_TABLE_CACHE: Dict[str, Dict[Tuple[str, Optional[int], int], Dict[str, float]]] = {}


# ──────────────────────────────────────────────────────────────
# label_metric (training name) → parquet column
# ──────────────────────────────────────────────────────────────
_METRIC_COLUMN = {
    "loop_rmsd": "global_loop_rmsd",
    "loop_lddt": "global_loop_lddt",
    "full_cdr_loop_rmsd": "global_loop_rmsd",
    "full_cdr_loop_lddt": "global_loop_lddt",
    "h3_rmsd": "H3_loop_rmsd",
    "h3_lddt": "H3_loop_lddt",
}


def metric_column(metric_name: str, task_scope: str = "full_cdr") -> str:
    """Translate a training ``label_metric`` into a parquet column name."""
    if task_scope == "h3" and metric_name in ("loop_rmsd", "loop_lddt"):
        metric_name = "h3_rmsd" if metric_name == "loop_rmsd" else "h3_lddt"
    return _METRIC_COLUMN.get(metric_name, metric_name)


# ──────────────────────────────────────────────────────────────
# Decoy identity (must mirror preprocess/precompute_decoy_metrics.py)
# ──────────────────────────────────────────────────────────────
def _parse_seed_sample_from_filename(filename: str) -> Tuple[Optional[int], Optional[int]]:
    patterns = (
        r"sd-(\d+)_sp-(\d+)",
        r"seed-(\d+)_sample-(\d+)",
        r"seed_(\d+)_model_(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1)), int(match.group(2))
    match = re.search(r"_model_(\d+)\.pdb$", filename)
    if match:
        return None, int(match.group(1))
    return None, None


def decoy_identity_for_lookup(model, model_position: int) -> Tuple[Optional[int], int]:
    """Return ``(seed, sample)`` for *model* matching the precompute script.

    Priority:
      1. Parse seed/sample from the decoy filename (AF3 / Boltz2 patterns and the
         ``_model_{N}.pdb`` fallback).
      2. Else use ``model.model_idx`` when set.
      3. Else ``model_position + 1`` (multi-MODEL PDB order; ComMat / PertMD).

    ComMat / PertMD always have ``seed = None``.
    """
    filename = ""
    pdb_path = getattr(model, "pdb_path", None)
    if pdb_path is not None:
        filename = pdb_path.name if hasattr(pdb_path, "name") else str(pdb_path).split("/")[-1]

    seed, sample = _parse_seed_sample_from_filename(filename)

    model_seed = getattr(model, "seed", None)
    if model_seed == -1:
        model_seed = None
    if seed is None and model_seed is not None:
        seed = int(model_seed)

    if sample is None:
        model_idx = getattr(model, "model_idx", None)
        sample = int(model_idx) if model_idx is not None else int(model_position + 1)

    method = (getattr(model, "method", "") or "").lower()
    if method in ("commat", "pertmd"):
        seed = None

    return seed, int(sample)


def _norm_seed(value) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ──────────────────────────────────────────────────────────────
# Store
# ──────────────────────────────────────────────────────────────
class PrecomputedMetricStore:
    """Lazy per-source loader for precomputed loop-metric parquet files.

    Parameters
    ----------
    root : str
        Base directory containing one subdirectory per source.
    source_subdirs : dict
        ``{source_name: subdir}`` mapping.  When a source is absent, its
        ``subdir`` defaults to the source name itself.
    metrics_filename : str
        Relative path of the parquet under ``{root}/{subdir}``.
    """

    def __init__(
        self,
        root: str,
        source_subdirs: Optional[Dict[str, str]] = None,
        metrics_filename: str = "metrics/loop_metrics.parquet",
    ):
        self.root = root
        self.source_subdirs = dict(source_subdirs or {})
        self.metrics_filename = metrics_filename
        # source_name -> {(target_id, seed, sample): {column: value}}
        self._tables: Dict[str, Dict[Tuple[str, Optional[int], int], Dict[str, float]]] = {}
        self._loaded: Dict[str, bool] = {}

    # ── path resolution ──
    def _parquet_path(self, source_name: str) -> str:
        subdir = self.source_subdirs.get(source_name, source_name)
        return os.path.join(self.root, subdir, self.metrics_filename)

    def has_source(self, source_name: str) -> bool:
        """True when a precomputed parquet exists (and loads) for *source_name*."""
        self._ensure_loaded(source_name)
        return bool(self._tables.get(source_name))

    # ── loading ──
    def _ensure_loaded(self, source_name: str) -> None:
        if self._loaded.get(source_name):
            return
        self._loaded[source_name] = True
        path = self._parquet_path(source_name)

        # Reuse a previously parsed table for this parquet (across epochs).
        cached = _TABLE_CACHE.get(path)
        if cached is not None:
            self._tables[source_name] = cached
            return

        if not os.path.exists(path):
            logger.warning("PrecomputedMetricStore: parquet not found for source=%s (%s)", source_name, path)
            return
        try:
            import pandas as pd
        except ModuleNotFoundError:
            logger.error("PrecomputedMetricStore: pandas is required to read %s", path)
            return
        try:
            df = pd.read_parquet(path)
        except Exception:
            logger.exception("PrecomputedMetricStore: failed to read parquet %s", path)
            return

        metric_cols = [c for c in df.columns if c.endswith("_loop_rmsd") or c.endswith("_loop_lddt")]
        table: Dict[Tuple[str, Optional[int], int], Dict[str, float]] = {}
        for row in df.itertuples(index=False):
            d = row._asdict()
            try:
                target_id = str(d["target_id"])
                sample = int(d["sample"])
            except (KeyError, TypeError, ValueError):
                continue
            seed = _norm_seed(d.get("seed"))
            values = {}
            for col in metric_cols:
                try:
                    values[col] = float(d[col])
                except (TypeError, ValueError):
                    values[col] = float("nan")
            table[(target_id, seed, sample)] = values
        self._tables[source_name] = table
        _TABLE_CACHE[path] = table
        logger.debug(
            "PrecomputedMetricStore: loaded source=%s rows=%d targets=%d from %s",
            source_name, len(table), df["target_id"].nunique() if "target_id" in df else -1, path,
        )

    # ── lookup ──
    def lookup(
        self,
        source_name: str,
        target_id: str,
        seed: Optional[int],
        sample: int,
        column: str,
    ) -> float:
        """Return the metric *column* for one decoy, or NaN when not found."""
        self._ensure_loaded(source_name)
        table = self._tables.get(source_name)
        if not table:
            return float("nan")
        key = (str(target_id), _norm_seed(seed), int(sample))
        values = table.get(key)
        if values is None:
            return float("nan")
        return float(values.get(column, float("nan")))
