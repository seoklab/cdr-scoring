"""Filter benchmark target lists to those with loadable decoys for YAML inference."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Sequence, Tuple

logger = logging.getLogger(__name__)


def filter_pdb_list_for_yaml_inference(
    pdb_list: Sequence[str],
    dataset_config: str,
    epoch: int = 1,
    verify_models: bool = False,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Keep only targets that have at least one YAML source with decoys.

    Uses SourceRegistry path resolution (same as MyDataset). By default only
    checks that source paths exist (fast). Set ``verify_models=True`` to load
    each target pickle and require ``len(models) > 0`` (slow on large lists).

    Returns:
        (kept_ids, skipped) where skipped is [(pdb_id, reason), ...].
    """
    from dataset.config import load_dataset_spec
    from dataset.source_registry import SourceRegistry

    spec = load_dataset_spec(dataset_config)
    registry = SourceRegistry(spec)
    kept: List[str] = []
    skipped: List[Tuple[str, str]] = []

    for pdb_id in pdb_list:
        candidates = registry.get_candidates(pdb_id, epoch)
        if not candidates:
            skipped.append((pdb_id, "no_yaml_source"))
            continue

        if verify_models:
            ok = False
            for cand in candidates:
                if cand.file_type != "target_model_pickle":
                    ok = True
                    break
                pkl_path = cand.target_model_path
                if pkl_path and _target_pickle_has_models(pkl_path):
                    ok = True
                    break
            if not ok:
                skipped.append((pdb_id, "no_models_in_pickle"))
                continue

        kept.append(pdb_id)

    return kept, skipped


def _target_pickle_has_models(pkl_path: str) -> bool:
    if not pkl_path or not os.path.exists(pkl_path):
        return False
    try:
        from data_loading.data_module import _load_target_pickle

        target = _load_target_pickle(pkl_path)
        return bool(getattr(target, "models", None)) and len(target.models) > 0
    except Exception as exc:
        logger.debug("Could not load target pickle %s: %s", pkl_path, exc)
        return False


def write_skipped_log(
    skipped: Sequence[Tuple[str, str]],
    log_path: str | Path,
    *,
    dataset_config: str = "",
    decoytype: str = "",
) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fp:
        fp.write(f"# Skipped {len(skipped)} targets during inference filtering\n")
        if dataset_config:
            fp.write(f"# dataset_config: {dataset_config}\n")
        if decoytype:
            fp.write(f"# decoytype: {decoytype}\n")
        fp.write("\n")
        for pdb_id, reason in skipped:
            fp.write(f"{pdb_id}\t{reason}\n")
    logger.info("Skipped-target log: %s (%d entries)", log_path, len(skipped))
