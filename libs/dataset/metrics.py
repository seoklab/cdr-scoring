"""
dataset/metrics.py
==================
Parse per-decoy metrics (e.g. Boltz2 ag_local_rmsd) from CSV / JSON files.
Returns a mapping  decoy_index → metric_value.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import pickle
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def load_ag_local_rmsd(
    metrics_path: str,
    fmt: str = "csv",
) -> Dict[int, float]:
    """Return {decoy_index: ag_local_rmsd} from *metrics_path*.

    Supported formats
    -----------------
    csv : expects header row; columns ``index`` (or ``rank``) and ``ag_local_rmsd``.
    json: expects a list of dicts or a dict  ``{decoy_index: ag_local_rmsd, ...}``.
    pkl : expects a Target object with ``.models[i].ag_local_rmsd``.
    """
    if not os.path.exists(metrics_path):
        logger.warning("Metrics file not found: %s", metrics_path)
        return {}

    fmt = fmt.lower()
    try:
        if fmt == "csv":
            return _parse_csv(metrics_path)
        elif fmt == "json":
            return _parse_json(metrics_path)
        elif fmt == "pkl":
            return _parse_target_pickle(metrics_path)
        else:
            logger.warning("Unknown metrics format: %s", fmt)
            return {}
    except Exception:
        logger.exception("Failed to parse metrics from %s", metrics_path)
        return {}


def _parse_csv(path: str) -> Dict[int, float]:
    result: Dict[int, float] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        idx_col = "index" if "index" in fields else ("rank" if "rank" in fields else None)
        rmsd_col = "ag_local_rmsd" if "ag_local_rmsd" in fields else None
        if idx_col is None or rmsd_col is None:
            logger.warning("CSV %s missing expected columns (found %s)", path, fields)
            return {}
        for row in reader:
            try:
                result[int(row[idx_col])] = float(row[rmsd_col])
            except (ValueError, KeyError):
                continue
    return result


def _parse_json(path: str) -> Dict[int, float]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {int(k): float(v) for k, v in data.items()}
    if isinstance(data, list):
        return {i: float(d.get("ag_local_rmsd", float("nan"))) for i, d in enumerate(data)}
    return {}


def _parse_target_pickle(path: str) -> Dict[int, float]:
    """Extract ag_local_rmsd from a Target pickle (each model has an ag_local_rmsd attribute)."""
    from data_loading.data_module import _load_target_pickle
    target = _load_target_pickle(path)
    result: Dict[int, float] = {}
    for idx, model in enumerate(target.models):
        rmsd = getattr(model, "ag_local_rmsd", float("nan"))
        result[idx] = float(rmsd)
    return result


def filter_by_ag_local_rmsd(
    rmsd_list: list,
    graph_list: list,
    cutoff: float,
    ag_local_rmsds: Dict[int, float],
) -> tuple:
    """Keep decoys whose ag_local_rmsd is missing/NaN or ≤ cutoff.

    Parameters
    ----------
    rmsd_list : list of float  (h3_rmsd per decoy)
    graph_list: list of DGLGraph
    cutoff    : ag_local_rmsd threshold
    ag_local_rmsds: {decoy_idx: ag_local_rmsd}

    Returns
    -------
    (filtered_graphs, filtered_rmsds)
    """
    if not ag_local_rmsds:
        return graph_list, rmsd_list

    keep_g, keep_r = [], []
    for idx, (g, r) in enumerate(zip(graph_list, rmsd_list)):
        alr = ag_local_rmsds.get(idx, float("nan"))
        # Keep decoys with missing/NaN ag_local_rmsd (e.g., no antigen present).
        if alr != alr or alr <= cutoff:
            keep_g.append(g)
            keep_r.append(r)
    if not keep_g:
        return keep_g, keep_r
    return keep_g, keep_r
