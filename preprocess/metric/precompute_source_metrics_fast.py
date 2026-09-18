#!/usr/bin/env python
"""Precompute source-level decoy metrics into compact parquet files.

This script is intentionally independent from training. It reads the same
dataset YAML/source registry used by training, but only materializes metrics
from Target/Model pickles and writes source-level raw parquet files:

    {output_dir}/{source}/targets/target_metrics.parquet
    {output_dir}/{source}/metrics/loop_metrics.parquet
    {output_dir}/{source}/metrics/interface_metrics.parquet
    {output_dir}/{source}/metrics/dockq_metrics.parquet

The decoy identity key is target_id + source + seed + sample. For methods
without seeds (for example ComMat), seed is null and sample identifies the
decoy.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT.parent / "cdr-data"
LIBS_ROOT = REPO_ROOT / "libs"
if str(LIBS_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBS_ROOT))

from data_loading.pdb2dict import Model, Target  # noqa: E402
from dataset.config import DatasetSpec, load_dataset_spec  # noqa: E402
from dataset.source_registry import SourceRegistry  # noqa: E402
from evaluation.loop_metrics import (  # noqa: E402
    BACKBONE_ATOMS,
    DEFAULT_NAMED_LOOP_RANGES,
    DEFAULT_LOOP_RANGES,
    LDDT_CUTOFF_A,
    LDDT_THRESHOLDS_A,
)

try:
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover - user environment issue
    raise SystemExit("pandas is required to write parquet files") from exc

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


ANTIBODY_CHAINS = ("H", "L")
LOOP_COLUMN_PREFIXES = ("H1", "H2", "H3", "L1", "L2", "L3")


class _TargetUnpickler(pickle.Unpickler):
    """Load historical Target pickles whose classes were stored under __main__."""

    _REDIRECT_MODULES = frozenset({
        "__main__", "pdb2dict_combined", "pdb2dict_ag", "pdb2dict_pipeline",
    })
    _CLASS_MAP = {"Target": Target, "Model": Model}

    def find_class(self, module: str, name: str):
        if module in self._REDIRECT_MODULES and name in self._CLASS_MAP:
            return self._CLASS_MAP[name]
        return super().find_class(module, name)


@dataclass(frozen=True)
class DecoyIdentity:
    seed: Optional[int]
    sample: int
    decoy_id: str
    file: str


@dataclass(frozen=True)
class LoopAtomPairs:
    native_coords: np.ndarray
    model_coords: np.ndarray
    atom_to_res: np.ndarray
    loop_ids: np.ndarray
    # Neighbour-only reference atoms (antigen / non-antibody chains). They are
    # never scored and never enter the loop RMSD alignment, but they DO count
    # as lDDT reference neighbours within LDDT_CUTOFF_A of a loop atom, so the
    # loop lDDT becomes sensitive to how the antigen is placed.
    ref_native_coords: np.ndarray
    ref_model_coords: np.ndarray
    ref_atom_to_res: np.ndarray
    missing_backbone_atom_count: int


@dataclass(frozen=True)
class AtomContactInput:
    xyz: np.ndarray
    residue_keys: Tuple[Tuple[str, tuple], ...]
    atom_to_residue_index: np.ndarray


@dataclass(frozen=True)
class NativeInterfaceCache:
    native_map: Dict[str, Dict[tuple, object]]
    receptor_chains: Tuple[str, ...]
    ligand_chains: Tuple[str, ...]
    native_contacts: set[Tuple[str, tuple, str, tuple]]
    native_cdr_contacts: set[Tuple[str, tuple, str, tuple]]
    iface_keys: set[Tuple[str, tuple]]
    receptor_keys: List[Tuple[str, tuple]]
    ligand_keys: List[Tuple[str, tuple]]
    backbone_by_key: Dict[Tuple[str, tuple], Dict[str, np.ndarray]]
    sorted_native_keys: Tuple[Tuple[str, tuple], ...]


TARGET_COLUMNS = [
    "target_id", "source", "native_path", "target_pickle_path", "has_holo_antigen",
    "antibody_chains", "antigen_chains", "num_total_residues_original",
    "H1_count", "H2_count", "H3_count", "L1_count", "L2_count", "L3_count",
]

COMMON_DECOY_COLUMNS = [
    "target_id", "source", "seed", "sample", "decoy_id", "native_path",
    "decoy_path", "target_pickle_path", "ranking", "ranking_score",
]

LOOP_COLUMNS = COMMON_DECOY_COLUMNS + [
    "cdr_rmsd", "cdr_lddt",
    "H1_loop_rmsd", "H2_loop_rmsd", "H3_loop_rmsd",
    "L1_loop_rmsd", "L2_loop_rmsd", "L3_loop_rmsd",
    "H1_loop_lddt", "H2_loop_lddt", "H3_loop_lddt",
    "L1_loop_lddt", "L2_loop_lddt", "L3_loop_lddt",
    "missing_backbone_atom_count", "missing_backbone_report",
]

INTERFACE_COLUMNS = COMMON_DECOY_COLUMNS + [
    "interface_bb_lddt", "interface_rmsd", "irmsd", "lrmsd",
    "cdr_antigen_contact_count", "cdr_antigen_contact_recovery", "fnat",
    "native_contact_count", "decoy_contact_count", "interface_metric_report",
]

DOCKQ_COLUMNS = COMMON_DECOY_COLUMNS + ["dockq", "fnat", "irmsd", "lrmsd"]


def _load_target_pickle(path: str | Path) -> Target:
    with open(path, "rb") as handle:
        return _TargetUnpickler(handle).load()


def _parse_seed_sample_from_filename(filename: str) -> Tuple[Optional[int], Optional[int]]:
    patterns = (
        r"sd-(\d+)_sp-(\d+)",
        r"seed-(\d+)_sample-(\d+)",
        r"seed_(\d+)_sample_(\d+)",
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


def _decoy_identity(model, model_position: int) -> DecoyIdentity:
    filename = ""
    pdb_path = getattr(model, "pdb_path", None)
    if pdb_path is not None:
        filename = Path(pdb_path).name

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
    if method == "commat" or method.startswith("pertmd"):
        seed = None

    decoy_id = f"seed={seed}:sample={sample}" if seed is not None else f"sample={sample}"
    return DecoyIdentity(seed=seed, sample=int(sample), decoy_id=decoy_id, file=filename)


def _load_target_ids(spec: DatasetSpec, split: str, explicit_list: str = "") -> List[str]:
    if explicit_list:
        with open(explicit_list, encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    if split == "valid":
        pkl_path = spec.valid_list_from_pkl
        pkl_key = spec.valid_list_pkl_key
        txt_path = spec.valid_list
    else:
        pkl_path = spec.train_list_from_pkl
        pkl_key = spec.train_list_pkl_key
        txt_path = spec.train_list

    if pkl_path:
        with open(pkl_path, "rb") as handle:
            payload = pickle.load(handle)
        values = payload[pkl_key] if isinstance(payload, dict) and pkl_key in payload else payload
        return [str(item) for item in values]

    if txt_path:
        with open(txt_path, encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    raise ValueError("No target list was provided and the dataset config has no list for split=%s" % split)


def _structure_model(structure):
    return next(structure.get_models())


def _residue_map(structure) -> Dict[str, Dict[tuple, object]]:
    out: Dict[str, Dict[tuple, object]] = {}
    for chain in _structure_model(structure):
        chain_map = {}
        for residue in chain:
            if residue.id[0] != " ":
                continue
            chain_map[residue.id] = residue
        out[chain.id] = chain_map
    return out


def _is_loop_residue(chain_id: str, resseq: int, loop_ranges=DEFAULT_LOOP_RANGES) -> bool:
    return any(start <= resseq <= end for start, end in loop_ranges.get(chain_id, ()))


def _loop_name(chain_id: str, resseq: int) -> Optional[str]:
    for name, c_id, start, end in DEFAULT_NAMED_LOOP_RANGES:
        if chain_id == c_id and start <= resseq <= end:
            return name
    return None


def _target_loop_counts(native_structure) -> Dict[str, int]:
    counts = {f"{name}_count": 0 for name in LOOP_COLUMN_PREFIXES}
    total_residues = 0
    for chain_id, chain_residues in _residue_map(native_structure).items():
        total_residues += len(chain_residues)
        for residue_id in chain_residues:
            name = _loop_name(chain_id, int(residue_id[1]))
            if name is not None:
                counts[f"{name}_count"] += 1
    counts["num_total_residues_original"] = total_residues
    return counts


def _matched_residue_pairs(native_structure, model_structure):
    model_map = _residue_map(model_structure)
    for native_chain in _structure_model(native_structure):
        model_chain_map = model_map.get(native_chain.id, {})
        for native_residue in native_chain:
            if native_residue.id[0] != " ":
                continue
            model_residue = model_chain_map.get(native_residue.id)
            if model_residue is not None:
                yield native_chain.id, native_residue, model_residue


def _matching_backbone_coords(native_residue, model_residue) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
    native_coords: List[np.ndarray] = []
    model_coords: List[np.ndarray] = []
    missing = 0
    for atom_name in BACKBONE_ATOMS:
        native_atom = native_residue.child_dict.get(atom_name)
        model_atom = model_residue.child_dict.get(atom_name)
        if native_atom is None or model_atom is None:
            missing += 1
            continue
        native_coords.append(native_atom.get_coord())
        model_coords.append(model_atom.get_coord())
    return native_coords, model_coords, missing


def _kabsch(ref: np.ndarray, mob: np.ndarray):
    ref_centered = ref - ref.mean(axis=0)
    mob_centered = mob - mob.mean(axis=0)
    cov = mob_centered.T @ ref_centered
    u, _, vt = np.linalg.svd(cov)
    det = np.sign(np.linalg.det(vt.T @ u.T))
    rot = u @ np.diag([1.0, 1.0, det]) @ vt
    trans = ref.mean(axis=0) - mob.mean(axis=0) @ rot
    return rot, trans


def _aligned_rmsd(align_nat, align_mod, target_nat, target_mod) -> float:
    if len(align_nat) < 3 or len(target_nat) == 0:
        return float("nan")
    align_nat = np.asarray(align_nat, dtype=np.float64)
    align_mod = np.asarray(align_mod, dtype=np.float64)
    target_nat = np.asarray(target_nat, dtype=np.float64)
    target_mod = np.asarray(target_mod, dtype=np.float64)
    if align_nat.shape != align_mod.shape or target_nat.shape != target_mod.shape:
        return float("nan")
    rot, trans = _kabsch(align_nat, align_mod)
    target_mod_sup = target_mod @ rot + trans
    diff = target_nat - target_mod_sup
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _backbone_lddt(nat_pts, mod_pts, atom_to_res, score_atom_mask) -> float:
    nat_pts = np.asarray(nat_pts, dtype=np.float64)
    mod_pts = np.asarray(mod_pts, dtype=np.float64)
    atom_to_res = np.asarray(atom_to_res, dtype=np.int32)
    score_atom_mask = np.asarray(score_atom_mask, dtype=bool)
    if len(nat_pts) < 2 or not score_atom_mask.any():
        return float("nan")

    d_nat = _pairwise_distances(nat_pts)
    d_mod = _pairwise_distances(mod_pts)
    residue_mask = atom_to_res[:, None] != atom_to_res[None, :]
    neighbor = residue_mask & (d_nat > 0) & (d_nat <= LDDT_CUTOFF_A)

    score_idx = np.where(score_atom_mask)[0]
    d_nat_rows = d_nat[score_idx]
    d_mod_rows = d_mod[score_idx]
    neighbor_rows = neighbor[score_idx]
    diff = np.abs(d_nat_rows - d_mod_rows)

    preserved = np.zeros_like(d_nat_rows, dtype=np.float64)
    for threshold in LDDT_THRESHOLDS_A:
        preserved += (diff < threshold).astype(np.float64)
    preserved /= len(LDDT_THRESHOLDS_A)

    # Micro-average (canonical global lDDT): every preserved distance counts
    # equally, i.e. total preserved distances / total considered distances,
    # rather than averaging per-atom then per-residue (macro-average).
    total_pairs = int(neighbor_rows.sum())
    if total_pairs == 0:
        return float("nan")
    return float(preserved[neighbor_rows].sum() / total_pairs)


def _loop_id(chain_id: str, resseq: int) -> int:
    for idx, (_name, c_id, start, end) in enumerate(DEFAULT_NAMED_LOOP_RANGES, start=1):
        if chain_id == c_id and start <= resseq <= end:
            return idx
    return 0


def _collect_loop_atom_pairs(native_structure, model_structure) -> LoopAtomPairs:
    native_coords: List[np.ndarray] = []
    model_coords: List[np.ndarray] = []
    atom_to_res: List[int] = []
    loop_ids: List[int] = []
    ref_native: List[np.ndarray] = []
    ref_model: List[np.ndarray] = []
    ref_atom_to_res: List[int] = []
    missing = 0
    residue_idx = 0

    for chain_id, native_residue, model_residue in _matched_residue_pairs(native_structure, model_structure):
        nat_atoms, mod_atoms, miss = _matching_backbone_coords(native_residue, model_residue)
        if chain_id in DEFAULT_LOOP_RANGES:
            # Antibody chains: framework + CDR atoms, scored via loop_ids.
            missing += miss
            if not nat_atoms:
                continue
            resseq = int(native_residue.id[1])
            lid = _loop_id(chain_id, resseq)
            for nat_coord, mod_coord in zip(nat_atoms, mod_atoms):
                native_coords.append(nat_coord)
                model_coords.append(mod_coord)
                atom_to_res.append(residue_idx)
                loop_ids.append(lid)
            residue_idx += 1
        else:
            # Antigen / other chains: neighbour-only lDDT reference atoms.
            if not nat_atoms:
                continue
            for nat_coord, mod_coord in zip(nat_atoms, mod_atoms):
                ref_native.append(nat_coord)
                ref_model.append(mod_coord)
                ref_atom_to_res.append(residue_idx)
            residue_idx += 1

    return LoopAtomPairs(
        native_coords=np.asarray(native_coords, dtype=np.float64).reshape(-1, 3),
        model_coords=np.asarray(model_coords, dtype=np.float64).reshape(-1, 3),
        atom_to_res=np.asarray(atom_to_res, dtype=np.int32),
        loop_ids=np.asarray(loop_ids, dtype=np.int8),
        ref_native_coords=np.asarray(ref_native, dtype=np.float64).reshape(-1, 3),
        ref_model_coords=np.asarray(ref_model, dtype=np.float64).reshape(-1, 3),
        ref_atom_to_res=np.asarray(ref_atom_to_res, dtype=np.int32),
        missing_backbone_atom_count=int(missing),
    )


def _masked_loop_rmsd(pairs: LoopAtomPairs, target_mask: np.ndarray) -> float:
    align_mask = ~target_mask
    return _aligned_rmsd(
        pairs.native_coords[align_mask],
        pairs.model_coords[align_mask],
        pairs.native_coords[target_mask],
        pairs.model_coords[target_mask],
    )


def _prepare_loop_lddt(pairs: LoopAtomPairs):
    """Precompute the (loop-atom x reference-atom) lDDT tensors once.

    Rows are the loop backbone atoms (the only atoms ever scored); columns are
    the full reference environment: every antibody (H/L) backbone atom plus the
    neighbour-only reference atoms (antigen chains). The distance matrices are
    built a single time and reused for the global and per-CDR masks, replacing
    the previous per-mask O(N^2) recomputation. The per-pair values are
    identical to _backbone_lddt; only the antigen reference columns are new.
    """
    nat_hl = pairs.native_coords
    if len(nat_hl) < 2:
        return None
    loop_rows = np.where(pairs.loop_ids > 0)[0]
    if loop_rows.size == 0:
        return None

    if len(pairs.ref_native_coords):
        col_nat = np.concatenate([nat_hl, pairs.ref_native_coords], axis=0)
        col_mod = np.concatenate([pairs.model_coords, pairs.ref_model_coords], axis=0)
        col_res = np.concatenate([pairs.atom_to_res, pairs.ref_atom_to_res])
    else:
        col_nat, col_mod, col_res = nat_hl, pairs.model_coords, pairs.atom_to_res

    row_nat = nat_hl[loop_rows]
    row_mod = pairs.model_coords[loop_rows]
    row_res = pairs.atom_to_res[loop_rows]

    d_nat = np.sqrt(np.sum((row_nat[:, None, :] - col_nat[None, :, :]) ** 2, axis=-1))
    d_mod = np.sqrt(np.sum((row_mod[:, None, :] - col_mod[None, :, :]) ** 2, axis=-1))
    neighbor = (row_res[:, None] != col_res[None, :]) & (d_nat > 0) & (d_nat <= LDDT_CUTOFF_A)

    diff = np.abs(d_nat - d_mod)
    preserved = np.zeros_like(d_nat)
    for threshold in LDDT_THRESHOLDS_A:
        preserved += diff < threshold
    preserved /= len(LDDT_THRESHOLDS_A)
    return loop_rows, neighbor, preserved


def _loop_lddt_from_prep(prep, target_mask: np.ndarray) -> float:
    if prep is None:
        return float("nan")
    loop_rows, neighbor, preserved = prep
    sub = target_mask[loop_rows]
    if not sub.any():
        return float("nan")
    nb = neighbor[sub]
    total = int(nb.sum())
    if total == 0:
        return float("nan")
    return float(preserved[sub][nb].sum() / total)


def compute_loop_row(native_structure, model_structure) -> Dict[str, float | int | str]:
    row: Dict[str, float | int | str] = {}
    pairs = _collect_loop_atom_pairs(native_structure, model_structure)
    prep = _prepare_loop_lddt(pairs)
    global_mask = pairs.loop_ids > 0
    row["cdr_rmsd"] = _masked_loop_rmsd(pairs, global_mask)
    row["cdr_lddt"] = _loop_lddt_from_prep(prep, global_mask)
    row["missing_backbone_atom_count"] = int(pairs.missing_backbone_atom_count)
    row["missing_backbone_report"] = (
        f"masked_backbone_atoms={int(pairs.missing_backbone_atom_count)}"
        if int(pairs.missing_backbone_atom_count) >= 5
        else ""
    )

    for loop_idx, (name, _chain_id, _start, _end) in enumerate(DEFAULT_NAMED_LOOP_RANGES, start=1):
        loop_mask = pairs.loop_ids == loop_idx
        row[f"{name}_loop_rmsd"] = _masked_loop_rmsd(pairs, loop_mask)
        row[f"{name}_loop_lddt"] = _loop_lddt_from_prep(prep, loop_mask)
    return row


def _is_galaxy_heavy_atom_name(name: str) -> bool:
    """Replicate Galaxy/PPDock's ``get_heavy()`` atom classification.

    PPDock (``Galaxy.anal``) decides whether an atom is "heavy" purely from the
    PDB atom-name columns. For 4-character atom names it follows the legacy PDB
    column convention that rotates the leading character to the end (e.g. the
    side-chain amide hydrogen ``HE21`` is stored as ``E21H``), then keeps the
    atom when the resulting name does not start with ``H``. As a consequence
    PPDock treats 4-character hydrogen names such as ``HE21``/``HD22``/``HH11``
    (when the second character is not ``H``) as heavy atoms. To reproduce
    PPDock's Fnat/interface residue selection bit-for-bit we mirror that rule
    here instead of using a chemically-correct hydrogen filter.
    """
    if not name:
        return False
    if len(name) == 4:
        # 4-char name: Galaxy rotates the first char to the end, so heaviness
        # is decided by the second character of the original name.
        return name[1] != "H"
    return name[0] != "H"


def _heavy_atom_coords(residue) -> List[np.ndarray]:
    coords = []
    for atom in residue:
        name = atom.get_name().strip()
        if not _is_galaxy_heavy_atom_name(name):
            continue
        coords.append(atom.get_coord())
    return coords


def _has_full_backbone(residue) -> bool:
    """Galaxy's ``check_bb``: residue must have N, CA, C and O."""
    return all(atom_name in residue.child_dict for atom_name in BACKBONE_ATOMS)


def _heavy_atom_contact_input(
    residue_map: Dict[str, Dict[tuple, object]],
    chains: Iterable[str],
    *,
    cdr_only: bool = False,
    require_full_backbone: bool = False,
) -> AtomContactInput:
    coords: List[np.ndarray] = []
    residue_keys: List[Tuple[str, tuple]] = []
    atom_to_residue_index: List[int] = []
    for chain_id in chains:
        for residue_id, residue in residue_map.get(chain_id, {}).items():
            if cdr_only and not _is_loop_residue(chain_id, int(residue_id[1])):
                continue
            if require_full_backbone and not _has_full_backbone(residue):
                continue
            residue_atom_coords = _heavy_atom_coords(residue)
            if not residue_atom_coords:
                continue
            residue_idx = len(residue_keys)
            residue_keys.append((chain_id, residue_id))
            for atom_coord in residue_atom_coords:
                coords.append(atom_coord)
                atom_to_residue_index.append(residue_idx)
    if coords:
        xyz = np.asarray(coords, dtype=np.float32)
        atom_to_residue_index_arr = np.asarray(atom_to_residue_index, dtype=np.int64)
    else:
        xyz = np.empty((0, 3), dtype=np.float32)
        atom_to_residue_index_arr = np.empty((0,), dtype=np.int64)
    return AtomContactInput(
        xyz=xyz,
        residue_keys=tuple(residue_keys),
        atom_to_residue_index=atom_to_residue_index_arr,
    )


def _squared_distance_matrix(a_xyz: torch.Tensor, b_xyz: torch.Tensor) -> torch.Tensor:
    return (
        torch.sum(a_xyz * a_xyz, dim=-1, keepdim=True)
        + torch.sum(b_xyz * b_xyz, dim=-1).unsqueeze(-2)
        - 2.0 * torch.matmul(a_xyz, b_xyz.transpose(-1, -2))
    )


def _contact_pairs_from_residue_hits(
    residue_hits: torch.Tensor,
    rec_keys_by_decoy: Sequence[Sequence[Tuple[str, tuple]]],
    lig_keys_by_decoy: Sequence[Sequence[Tuple[str, tuple]]],
    *,
    batched: bool,
) -> List[set[Tuple[str, tuple, str, tuple]]]:
    if batched:
        out = [set() for _ in rec_keys_by_decoy]
        for decoy_idx, rec_res_idx, lig_res_idx in residue_hits.cpu().tolist():
            rec_chain, rec_id = rec_keys_by_decoy[decoy_idx][rec_res_idx]
            lig_chain, lig_id = lig_keys_by_decoy[decoy_idx][lig_res_idx]
            out[decoy_idx].add((rec_chain, rec_id, lig_chain, lig_id))
        return out

    contacts: set[Tuple[str, tuple, str, tuple]] = set()
    rec_keys = rec_keys_by_decoy[0]
    lig_keys = lig_keys_by_decoy[0]
    for rec_res_idx, lig_res_idx in residue_hits.cpu().tolist():
        rec_chain, rec_id = rec_keys[rec_res_idx]
        lig_chain, lig_id = lig_keys[lig_res_idx]
        contacts.add((rec_chain, rec_id, lig_chain, lig_id))
    return [contacts]


def _unique_residue_hits_from_atom_hits(
    atom_hits: torch.Tensor,
    rec_atom_to_residue_index: torch.Tensor,
    lig_atom_to_residue_index: torch.Tensor,
    *,
    batched: bool,
) -> torch.Tensor:
    if atom_hits.numel() == 0:
        width = 3 if batched else 2
        return atom_hits.new_empty((0, width))

    if batched:
        decoy_idx = atom_hits[:, 0]
        rec_res_idx = rec_atom_to_residue_index[decoy_idx, atom_hits[:, 1]]
        lig_res_idx = lig_atom_to_residue_index[decoy_idx, atom_hits[:, 2]]
        residue_hits = torch.stack((decoy_idx, rec_res_idx, lig_res_idx), dim=1)
    else:
        rec_res_idx = rec_atom_to_residue_index[atom_hits[:, 0]]
        lig_res_idx = lig_atom_to_residue_index[atom_hits[:, 1]]
        residue_hits = torch.stack((rec_res_idx, lig_res_idx), dim=1)
    return torch.unique(residue_hits, dim=0)


def _contact_pairs_from_inputs(
    rec_input: AtomContactInput,
    lig_input: AtomContactInput,
    *,
    cutoff_a: float,
    device: torch.device,
) -> set[Tuple[str, tuple, str, tuple]]:
    if rec_input.xyz.shape[0] == 0 or lig_input.xyz.shape[0] == 0:
        return set()

    with torch.no_grad():
        rec_xyz = torch.as_tensor(rec_input.xyz, dtype=torch.float32, device=device)
        lig_xyz = torch.as_tensor(lig_input.xyz, dtype=torch.float32, device=device)
        rec_atom_to_residue_index = torch.as_tensor(
            rec_input.atom_to_residue_index,
            dtype=torch.long,
            device=device,
        )
        lig_atom_to_residue_index = torch.as_tensor(
            lig_input.atom_to_residue_index,
            dtype=torch.long,
            device=device,
        )
        dist2 = _squared_distance_matrix(rec_xyz, lig_xyz)
        dist2.clamp_min_(0.0)
        # Galaxy/PPDock uses a strict ``distance < cutoff`` comparison.
        atom_hits = torch.nonzero(dist2 < cutoff_a * cutoff_a, as_tuple=False)
        residue_hits = _unique_residue_hits_from_atom_hits(
            atom_hits,
            rec_atom_to_residue_index,
            lig_atom_to_residue_index,
            batched=False,
        )
    return _contact_pairs_from_residue_hits(
        residue_hits,
        [rec_input.residue_keys],
        [lig_input.residue_keys],
        batched=False,
    )[0]


def _contact_pairs(
    residue_map: Dict[str, Dict[tuple, object]],
    receptor_chains: Iterable[str],
    ligand_chains: Iterable[str],
    *,
    cutoff_a: float,
    cdr_only: bool = False,
    require_full_backbone: bool = False,
) -> set[Tuple[str, tuple, str, tuple]]:
    rec_input = _heavy_atom_contact_input(
        residue_map, receptor_chains, cdr_only=cdr_only, require_full_backbone=require_full_backbone,
    )
    lig_input = _heavy_atom_contact_input(
        residue_map, ligand_chains, cdr_only=False, require_full_backbone=require_full_backbone,
    )
    return _contact_pairs_from_inputs(
        rec_input,
        lig_input,
        cutoff_a=cutoff_a,
        device=torch.device("cpu"),
    )


def _batched_contact_pairs_from_inputs(
    rec_inputs: Sequence[AtomContactInput],
    lig_inputs: Sequence[AtomContactInput],
    *,
    cutoff_a: float,
    device: torch.device,
) -> List[set[Tuple[str, tuple, str, tuple]]]:
    if len(rec_inputs) != len(lig_inputs):
        raise ValueError("receptor and ligand contact batches must have the same length")
    if not rec_inputs:
        return []

    rec_counts = [item.xyz.shape[0] for item in rec_inputs]
    lig_counts = [item.xyz.shape[0] for item in lig_inputs]
    if max(rec_counts, default=0) == 0 or max(lig_counts, default=0) == 0:
        return [set() for _ in rec_inputs]

    rec_keys_by_decoy = [item.residue_keys for item in rec_inputs]
    lig_keys_by_decoy = [item.residue_keys for item in lig_inputs]
    clean_batch = len(set(rec_counts)) == 1 and len(set(lig_counts)) == 1

    if clean_batch:
        rec_xyz_np = np.stack([item.xyz for item in rec_inputs], axis=0)
        lig_xyz_np = np.stack([item.xyz for item in lig_inputs], axis=0)
        rec_atom_to_residue_index_np = np.stack([item.atom_to_residue_index for item in rec_inputs], axis=0)
        lig_atom_to_residue_index_np = np.stack([item.atom_to_residue_index for item in lig_inputs], axis=0)
        rec_mask = None
        lig_mask = None
    else:
        max_rec = max(rec_counts)
        max_lig = max(lig_counts)
        rec_xyz_np = np.zeros((len(rec_inputs), max_rec, 3), dtype=np.float32)
        lig_xyz_np = np.zeros((len(lig_inputs), max_lig, 3), dtype=np.float32)
        rec_atom_to_residue_index_np = np.full((len(rec_inputs), max_rec), -1, dtype=np.int64)
        lig_atom_to_residue_index_np = np.full((len(lig_inputs), max_lig), -1, dtype=np.int64)
        rec_mask_np = np.zeros((len(rec_inputs), max_rec), dtype=bool)
        lig_mask_np = np.zeros((len(lig_inputs), max_lig), dtype=bool)
        for idx, item in enumerate(rec_inputs):
            n_atoms = item.xyz.shape[0]
            if n_atoms:
                rec_xyz_np[idx, :n_atoms] = item.xyz
                rec_atom_to_residue_index_np[idx, :n_atoms] = item.atom_to_residue_index
                rec_mask_np[idx, :n_atoms] = True
        for idx, item in enumerate(lig_inputs):
            n_atoms = item.xyz.shape[0]
            if n_atoms:
                lig_xyz_np[idx, :n_atoms] = item.xyz
                lig_atom_to_residue_index_np[idx, :n_atoms] = item.atom_to_residue_index
                lig_mask_np[idx, :n_atoms] = True
        rec_mask = torch.as_tensor(rec_mask_np, dtype=torch.bool, device=device)
        lig_mask = torch.as_tensor(lig_mask_np, dtype=torch.bool, device=device)

    with torch.no_grad():
        rec_xyz = torch.as_tensor(rec_xyz_np, dtype=torch.float32, device=device)
        lig_xyz = torch.as_tensor(lig_xyz_np, dtype=torch.float32, device=device)
        rec_atom_to_residue_index = torch.as_tensor(
            rec_atom_to_residue_index_np,
            dtype=torch.long,
            device=device,
        )
        lig_atom_to_residue_index = torch.as_tensor(
            lig_atom_to_residue_index_np,
            dtype=torch.long,
            device=device,
        )
        dist2 = _squared_distance_matrix(rec_xyz, lig_xyz)
        dist2.clamp_min_(0.0)
        # Galaxy/PPDock uses a strict ``distance < cutoff`` comparison.
        contact_mask = dist2 < cutoff_a * cutoff_a
        if rec_mask is not None and lig_mask is not None:
            contact_mask = contact_mask & rec_mask[:, :, None] & lig_mask[:, None, :]

        atom_hits = torch.nonzero(contact_mask, as_tuple=False)
        residue_hits = _unique_residue_hits_from_atom_hits(
            atom_hits,
            rec_atom_to_residue_index,
            lig_atom_to_residue_index,
            batched=True,
        )
    return _contact_pairs_from_residue_hits(
        residue_hits,
        rec_keys_by_decoy,
        lig_keys_by_decoy,
        batched=True,
    )


def _decoy_contact_pairs_batched(
    model_maps: Sequence[Dict[str, Dict[tuple, object]]],
    receptor_chains: Iterable[str],
    ligand_chains: Iterable[str],
    *,
    cutoff_a: float,
    cdr_only: bool,
    device: torch.device,
    contact_batch_size: Optional[int] = None,
) -> List[set[Tuple[str, tuple, str, tuple]]]:
    rec_inputs = [
        _heavy_atom_contact_input(model_map, receptor_chains, cdr_only=cdr_only)
        for model_map in model_maps
    ]
    lig_inputs = [
        _heavy_atom_contact_input(model_map, ligand_chains, cdr_only=False)
        for model_map in model_maps
    ]
    if not rec_inputs:
        return []

    batch_size = int(contact_batch_size or 0)
    if batch_size <= 0:
        batch_size = len(rec_inputs)

    out: List[set[Tuple[str, tuple, str, tuple]]] = []
    for start in range(0, len(rec_inputs), batch_size):
        rec_chunk = rec_inputs[start:start + batch_size]
        lig_chunk = lig_inputs[start:start + batch_size]
        try:
            out.extend(_batched_contact_pairs_from_inputs(
                rec_chunk,
                lig_chunk,
                cutoff_a=cutoff_a,
                device=device,
            ))
            continue
        except RuntimeError:
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for rec_input, lig_input in zip(rec_chunk, lig_chunk):
            try:
                out.append(_contact_pairs_from_inputs(
                    rec_input,
                    lig_input,
                    cutoff_a=cutoff_a,
                    device=device,
                ))
            except RuntimeError:
                if device.type != "cuda":
                    raise
                torch.cuda.empty_cache()
                out.append(_contact_pairs_from_inputs(
                    rec_input,
                    lig_input,
                    cutoff_a=cutoff_a,
                    device=torch.device("cpu"),
                ))
    return out


def _collect_backbone_by_keys(native_map, model_map, keys: Sequence[Tuple[str, tuple]]):
    nat = []
    mod = []
    for chain_id, residue_id in sorted(keys):
        native_residue = native_map.get(chain_id, {}).get(residue_id)
        model_residue = model_map.get(chain_id, {}).get(residue_id)
        if native_residue is None or model_residue is None:
            continue
        # Galaxy uses get_backbone(): the residue contributes all four
        # backbone atoms or none at all.
        residue_nat = []
        residue_mod = []
        complete = True
        for atom_name in BACKBONE_ATOMS:
            native_atom = native_residue.child_dict.get(atom_name)
            model_atom = model_residue.child_dict.get(atom_name)
            if native_atom is None or model_atom is None:
                complete = False
                break
            residue_nat.append(native_atom.get_coord())
            residue_mod.append(model_atom.get_coord())
        if complete:
            nat.extend(residue_nat)
            mod.extend(residue_mod)
    return np.asarray(nat, dtype=np.float64), np.asarray(mod, dtype=np.float64)


def _native_backbone_cache(native_map) -> Dict[Tuple[str, tuple], Dict[str, np.ndarray]]:
    out: Dict[Tuple[str, tuple], Dict[str, np.ndarray]] = {}
    for chain_id, chain_residues in native_map.items():
        for residue_id, residue in chain_residues.items():
            atom_coords = {}
            for atom_name in BACKBONE_ATOMS:
                atom = residue.child_dict.get(atom_name)
                if atom is not None:
                    atom_coords[atom_name] = atom.get_coord()
            out[(chain_id, residue_id)] = atom_coords
    return out


def _collect_backbone_by_cached_keys(
    backbone_by_key: Dict[Tuple[str, tuple], Dict[str, np.ndarray]],
    model_map,
    keys: Sequence[Tuple[str, tuple]],
):
    nat = []
    mod = []
    for chain_id, residue_id in sorted(keys):
        native_atoms = backbone_by_key.get((chain_id, residue_id), {})
        model_residue = model_map.get(chain_id, {}).get(residue_id)
        if model_residue is None:
            continue
        # Galaxy uses get_backbone(): the residue contributes all four
        # backbone atoms or none at all.
        residue_nat = []
        residue_mod = []
        complete = True
        for atom_name in BACKBONE_ATOMS:
            native_coord = native_atoms.get(atom_name)
            model_atom = model_residue.child_dict.get(atom_name)
            if native_coord is None or model_atom is None:
                complete = False
                break
            residue_nat.append(native_coord)
            residue_mod.append(model_atom.get_coord())
        if complete:
            nat.extend(residue_nat)
            mod.extend(residue_mod)
    return np.asarray(nat, dtype=np.float64), np.asarray(mod, dtype=np.float64)


def _all_residue_keys(residue_map, chains, *, require_full_backbone: bool = False):
    keys = []
    for chain_id in chains:
        for residue_id, residue in residue_map.get(chain_id, {}).items():
            if require_full_backbone and not _has_full_backbone(residue):
                continue
            keys.append((chain_id, residue_id))
    return keys


def _build_native_interface_cache(
    native_structure,
    *,
    contact_cutoff_a: float,
    interface_cutoff_a: float,
    device: torch.device,
) -> NativeInterfaceCache:
    native_map = _residue_map(native_structure)
    receptor_chains = tuple(chain for chain in ANTIBODY_CHAINS if chain in native_map)
    ligand_chains = tuple(sorted(set(native_map) - set(receptor_chains)))
    backbone_by_key = _native_backbone_cache(native_map)
    sorted_native_keys = tuple(
        (chain_id, residue_id)
        for chain_id in sorted(native_map)
        for residue_id in sorted(native_map[chain_id])
    )

    if not receptor_chains or not ligand_chains:
        return NativeInterfaceCache(
            native_map=native_map,
            receptor_chains=receptor_chains,
            ligand_chains=ligand_chains,
            native_contacts=set(),
            native_cdr_contacts=set(),
            iface_keys=set(),
            receptor_keys=[],
            ligand_keys=[],
            backbone_by_key=backbone_by_key,
            sorted_native_keys=sorted_native_keys,
        )

    native_contacts = _contact_pairs_from_inputs(
        _heavy_atom_contact_input(native_map, receptor_chains, cdr_only=False, require_full_backbone=True),
        _heavy_atom_contact_input(native_map, ligand_chains, cdr_only=False, require_full_backbone=True),
        cutoff_a=contact_cutoff_a,
        device=device,
    )
    native_cdr_contacts = _contact_pairs_from_inputs(
        _heavy_atom_contact_input(native_map, receptor_chains, cdr_only=True, require_full_backbone=True),
        _heavy_atom_contact_input(native_map, ligand_chains, cdr_only=False, require_full_backbone=True),
        cutoff_a=contact_cutoff_a,
        device=device,
    )
    interface_contacts_for_rmsd = _contact_pairs_from_inputs(
        _heavy_atom_contact_input(native_map, receptor_chains, cdr_only=False, require_full_backbone=True),
        _heavy_atom_contact_input(native_map, ligand_chains, cdr_only=False, require_full_backbone=True),
        cutoff_a=interface_cutoff_a,
        device=device,
    )
    iface_keys = {
        (rec_chain, rec_id)
        for rec_chain, rec_id, _lig_chain, _lig_id in interface_contacts_for_rmsd
    } | {
        (lig_chain, lig_id)
        for _rec_chain, _rec_id, lig_chain, lig_id in interface_contacts_for_rmsd
    }

    return NativeInterfaceCache(
        native_map=native_map,
        receptor_chains=receptor_chains,
        ligand_chains=ligand_chains,
        native_contacts=native_contacts,
        native_cdr_contacts=native_cdr_contacts,
        iface_keys=iface_keys,
        receptor_keys=_all_residue_keys(native_map, receptor_chains, require_full_backbone=True),
        ligand_keys=_all_residue_keys(native_map, ligand_chains, require_full_backbone=True),
        backbone_by_key=backbone_by_key,
        sorted_native_keys=sorted_native_keys,
    )


def _interface_lddt(native_structure, model_structure, interface_keys: set[Tuple[str, tuple]]) -> float:
    native_map = _residue_map(native_structure)
    model_map = _residue_map(model_structure)
    nat_pts = []
    mod_pts = []
    atom_to_res = []
    score_mask = []
    residue_idx = 0
    for chain_id in sorted(native_map):
        for residue_id, native_residue in sorted(native_map[chain_id].items()):
            model_residue = model_map.get(chain_id, {}).get(residue_id)
            if model_residue is None:
                continue
            nat_atoms, mod_atoms, _missing = _matching_backbone_coords(native_residue, model_residue)
            if not nat_atoms:
                continue
            is_interface = (chain_id, residue_id) in interface_keys
            for nat_coord, mod_coord in zip(nat_atoms, mod_atoms):
                nat_pts.append(nat_coord)
                mod_pts.append(mod_coord)
                atom_to_res.append(residue_idx)
                score_mask.append(is_interface)
            residue_idx += 1
    return _backbone_lddt(nat_pts, mod_pts, atom_to_res, score_mask)


def _interface_lddt_cached(cache: NativeInterfaceCache, model_map) -> float:
    nat_pts = []
    mod_pts = []
    atom_to_res = []
    score_mask = []
    residue_idx = 0
    for chain_id, residue_id in cache.sorted_native_keys:
        model_residue = model_map.get(chain_id, {}).get(residue_id)
        if model_residue is None:
            continue
        native_atoms = cache.backbone_by_key.get((chain_id, residue_id), {})
        added = 0
        is_interface = (chain_id, residue_id) in cache.iface_keys
        for atom_name in BACKBONE_ATOMS:
            native_coord = native_atoms.get(atom_name)
            model_atom = model_residue.child_dict.get(atom_name)
            if native_coord is None or model_atom is None:
                continue
            nat_pts.append(native_coord)
            mod_pts.append(model_atom.get_coord())
            atom_to_res.append(residue_idx)
            score_mask.append(is_interface)
            added += 1
        if added:
            residue_idx += 1
    return _backbone_lddt(nat_pts, mod_pts, atom_to_res, score_mask)


def compute_interface_rows(
    native_structure,
    model_structure,
    *,
    contact_cutoff_a: float,
    interface_cutoff_a: float,
) -> Tuple[Dict[str, float | int], Dict[str, float]]:
    native_map = _residue_map(native_structure)
    model_map = _residue_map(model_structure)
    receptor_chains = tuple(chain for chain in ANTIBODY_CHAINS if chain in native_map)
    ligand_chains = tuple(sorted(set(native_map) - set(receptor_chains)))
    if not receptor_chains or not ligand_chains:
        nan_interface = {
            "interface_bb_lddt": float("nan"),
            "interface_rmsd": float("nan"),
            "irmsd": float("nan"),
            "lrmsd": float("nan"),
            "cdr_antigen_contact_count": 0,
            "cdr_antigen_contact_recovery": float("nan"),
            "fnat": float("nan"),
            "native_contact_count": 0,
            "decoy_contact_count": 0,
        }
        return nan_interface, {"dockq": float("nan"), "fnat": float("nan"), "irmsd": float("nan"), "lrmsd": float("nan")}

    native_contacts = _contact_pairs(
        native_map, receptor_chains, ligand_chains, cutoff_a=contact_cutoff_a, cdr_only=False,
        require_full_backbone=True,
    )
    decoy_contacts = _contact_pairs(
        model_map, receptor_chains, ligand_chains, cutoff_a=contact_cutoff_a, cdr_only=False,
    )
    native_cdr_contacts = _contact_pairs(
        native_map, receptor_chains, ligand_chains, cutoff_a=contact_cutoff_a, cdr_only=True,
        require_full_backbone=True,
    )
    decoy_cdr_contacts = _contact_pairs(
        model_map, receptor_chains, ligand_chains, cutoff_a=contact_cutoff_a, cdr_only=True,
    )

    fnat = (
        len(native_contacts & decoy_contacts) / len(native_contacts)
        if native_contacts else float("nan")
    )
    cdr_recovery = (
        len(native_cdr_contacts & decoy_cdr_contacts) / len(native_cdr_contacts)
        if native_cdr_contacts else float("nan")
    )

    interface_contacts_for_rmsd = _contact_pairs(
        native_map, receptor_chains, ligand_chains, cutoff_a=interface_cutoff_a, cdr_only=False,
        require_full_backbone=True,
    )
    iface_keys = {
        (rec_chain, rec_id)
        for rec_chain, rec_id, _lig_chain, _lig_id in interface_contacts_for_rmsd
    } | {
        (lig_chain, lig_id)
        for _rec_chain, _rec_id, lig_chain, lig_id in interface_contacts_for_rmsd
    }
    receptor_keys = _all_residue_keys(native_map, receptor_chains, require_full_backbone=True)
    ligand_keys = _all_residue_keys(native_map, ligand_chains, require_full_backbone=True)

    interface_nat, interface_mod = _collect_backbone_by_keys(native_map, model_map, list(iface_keys))
    receptor_nat, receptor_mod = _collect_backbone_by_keys(native_map, model_map, receptor_keys)
    ligand_nat, ligand_mod = _collect_backbone_by_keys(native_map, model_map, ligand_keys)

    irmsd = _aligned_rmsd(interface_nat, interface_mod, interface_nat, interface_mod)
    # DockQ convention (matches Galaxy reference step2_prep_input.py): the antigen
    # is the receptor used for superposition and lRMSD is measured on the antibody
    # (H/L). receptor_keys here are antibody chains, so align on the antigen
    # (ligand_*) and measure RMSD on the antibody (receptor_*).
    lrmsd = _aligned_rmsd(ligand_nat, ligand_mod, receptor_nat, receptor_mod)
    interface_bb_lddt = _interface_lddt(native_structure, model_structure, iface_keys)
    dockq = (
        (fnat + 1.0 / (1.0 + (irmsd / 1.5) ** 2) + 1.0 / (1.0 + (lrmsd / 8.5) ** 2)) / 3.0
        if math.isfinite(fnat) and math.isfinite(irmsd) and math.isfinite(lrmsd)
        else float("nan")
    )

    interface_row = {
        "interface_bb_lddt": interface_bb_lddt,
        "interface_rmsd": irmsd,
        "irmsd": irmsd,
        "lrmsd": lrmsd,
        "cdr_antigen_contact_count": int(len(native_cdr_contacts)),
        "cdr_antigen_contact_recovery": cdr_recovery,
        "fnat": fnat,
        "native_contact_count": int(len(native_contacts)),
        "decoy_contact_count": int(len(decoy_contacts)),
    }
    dockq_row = {"dockq": dockq, "fnat": fnat, "irmsd": irmsd, "lrmsd": lrmsd}
    return interface_row, dockq_row


def _nan_interface_rows_for_no_ligand() -> Tuple[Dict[str, float | int], Dict[str, float]]:
    nan_interface = {
        "interface_bb_lddt": float("nan"),
        "interface_rmsd": float("nan"),
        "irmsd": float("nan"),
        "lrmsd": float("nan"),
        "cdr_antigen_contact_count": 0,
        "cdr_antigen_contact_recovery": float("nan"),
        "fnat": float("nan"),
        "native_contact_count": 0,
        "decoy_contact_count": 0,
    }
    return nan_interface, {"dockq": float("nan"), "fnat": float("nan"), "irmsd": float("nan"), "lrmsd": float("nan")}


def _compute_interface_row_from_cache(
    cache: NativeInterfaceCache,
    model_map,
    decoy_contacts: set[Tuple[str, tuple, str, tuple]],
    decoy_cdr_contacts: set[Tuple[str, tuple, str, tuple]],
) -> Tuple[Dict[str, float | int], Dict[str, float]]:
    if not cache.receptor_chains or not cache.ligand_chains:
        return _nan_interface_rows_for_no_ligand()

    fnat = (
        len(cache.native_contacts & decoy_contacts) / len(cache.native_contacts)
        if cache.native_contacts else float("nan")
    )
    cdr_recovery = (
        len(cache.native_cdr_contacts & decoy_cdr_contacts) / len(cache.native_cdr_contacts)
        if cache.native_cdr_contacts else float("nan")
    )

    interface_nat, interface_mod = _collect_backbone_by_cached_keys(
        cache.backbone_by_key,
        model_map,
        list(cache.iface_keys),
    )
    receptor_nat, receptor_mod = _collect_backbone_by_cached_keys(
        cache.backbone_by_key,
        model_map,
        cache.receptor_keys,
    )
    ligand_nat, ligand_mod = _collect_backbone_by_cached_keys(
        cache.backbone_by_key,
        model_map,
        cache.ligand_keys,
    )

    irmsd = _aligned_rmsd(interface_nat, interface_mod, interface_nat, interface_mod)
    # DockQ convention (matches Galaxy reference step2_prep_input.py): the antigen
    # is the receptor used for superposition and lRMSD is measured on the antibody
    # (H/L). cache.receptor_keys are antibody chains, so align on the antigen
    # (ligand_*) and measure RMSD on the antibody (receptor_*).
    lrmsd = _aligned_rmsd(ligand_nat, ligand_mod, receptor_nat, receptor_mod)
    interface_bb_lddt = _interface_lddt_cached(cache, model_map)
    dockq = (
        (fnat + 1.0 / (1.0 + (irmsd / 1.5) ** 2) + 1.0 / (1.0 + (lrmsd / 8.5) ** 2)) / 3.0
        if math.isfinite(fnat) and math.isfinite(irmsd) and math.isfinite(lrmsd)
        else float("nan")
    )

    interface_row = {
        "interface_bb_lddt": interface_bb_lddt,
        "interface_rmsd": irmsd,
        "irmsd": irmsd,
        "lrmsd": lrmsd,
        "cdr_antigen_contact_count": int(len(cache.native_cdr_contacts)),
        "cdr_antigen_contact_recovery": cdr_recovery,
        "fnat": fnat,
        "native_contact_count": int(len(cache.native_contacts)),
        "decoy_contact_count": int(len(decoy_contacts)),
    }
    dockq_row = {"dockq": dockq, "fnat": fnat, "irmsd": irmsd, "lrmsd": lrmsd}
    return interface_row, dockq_row


def compute_interface_rows_for_models_fast(
    native_structure,
    model_structures: Sequence[object],
    *,
    contact_cutoff_a: float,
    interface_cutoff_a: float,
    device: torch.device,
    contact_batch_size: Optional[int] = None,
) -> List[Tuple[Dict[str, float | int], Dict[str, float]]]:
    cache = _build_native_interface_cache(
        native_structure,
        contact_cutoff_a=contact_cutoff_a,
        interface_cutoff_a=interface_cutoff_a,
        device=device,
    )
    if not model_structures:
        return []
    if not cache.receptor_chains or not cache.ligand_chains:
        return [_nan_interface_rows_for_no_ligand() for _ in model_structures]

    model_maps = [_residue_map(model_structure) for model_structure in model_structures]
    decoy_contacts_by_model = _decoy_contact_pairs_batched(
        model_maps,
        cache.receptor_chains,
        cache.ligand_chains,
        cutoff_a=contact_cutoff_a,
        cdr_only=False,
        device=device,
        contact_batch_size=contact_batch_size,
    )
    decoy_cdr_contacts_by_model = _decoy_contact_pairs_batched(
        model_maps,
        cache.receptor_chains,
        cache.ligand_chains,
        cutoff_a=contact_cutoff_a,
        cdr_only=True,
        device=device,
        contact_batch_size=contact_batch_size,
    )

    rows = []
    for model_map, decoy_contacts, decoy_cdr_contacts in zip(
        model_maps,
        decoy_contacts_by_model,
        decoy_cdr_contacts_by_model,
    ):
        try:
            rows.append(_compute_interface_row_from_cache(
                cache,
                model_map,
                decoy_contacts,
                decoy_cdr_contacts,
            ))
        except Exception as exc:
            interface_row = {
                "interface_bb_lddt": float("nan"),
                "interface_rmsd": float("nan"),
                "irmsd": float("nan"),
                "lrmsd": float("nan"),
                "cdr_antigen_contact_count": float("nan"),
                "cdr_antigen_contact_recovery": float("nan"),
                "fnat": float("nan"),
                "native_contact_count": float("nan"),
                "decoy_contact_count": float("nan"),
                "interface_metric_report": f"interface_metric_failed: {exc}",
            }
            dockq_row = {
                "dockq": float("nan"),
                "fnat": float("nan"),
                "irmsd": float("nan"),
                "lrmsd": float("nan"),
            }
            rows.append((interface_row, dockq_row))
    return rows


def _is_boltz2_source(source: str) -> bool:
    return _source_key(source).startswith("boltz2")


def _boltz2_model_number(model, identity: DecoyIdentity) -> Optional[int]:
    pdb_path = getattr(model, "pdb_path", None)
    if pdb_path is not None:
        match = re.search(r"_model_(\d+)\.pdb$", Path(pdb_path).name)
        if match:
            return int(match.group(1))
    return int(identity.sample) if identity.sample is not None else None


def _candidate_boltz2_confidence_paths(model, target_id: str, model_num: int) -> List[Path]:
    paths: List[Path] = []
    pdb_path = getattr(model, "pdb_path", None)
    if pdb_path is None:
        return paths

    decoy_path = Path(pdb_path)
    folder_name = decoy_path.parent.name or target_id
    filename = f"confidence_{folder_name}_model_{model_num}.json"
    paths.append(decoy_path.with_name(filename))

    for parent in decoy_path.parents:
        if not parent.name.endswith("_boltz2_chothia_w_ag_renum"):
            continue
        dataset_root = parent.parent
        prefix = parent.name.split("_", 1)[0]
        if prefix.isdigit():
            paths.append(
                dataset_root
                / f"{int(prefix) - 1}_boltz2"
                / f"boltz_results_{folder_name}"
                / "predictions"
                / folder_name
                / filename
            )
        for candidate_root in sorted(dataset_root.glob("*_boltz2")):
            paths.append(
                candidate_root
                / f"boltz_results_{folder_name}"
                / "predictions"
                / folder_name
                / filename
            )
        break

    deduped: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def _boltz2_confidence_score(model, target_id: str, identity: DecoyIdentity) -> float:
    model_num = _boltz2_model_number(model, identity)
    if model_num is not None:
        for path in _candidate_boltz2_confidence_paths(model, target_id, model_num):
            if not path.exists():
                continue
            try:
                with open(path, encoding="utf-8") as handle:
                    payload = json.load(handle)
                return float(payload.get("confidence_score", float("nan")))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                break

    try:
        return float(getattr(model, "ranking_score", float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def _ranking_overrides_for_models(source: str, target_id: str, models: Sequence[object]):
    if not _is_boltz2_source(source):
        return {}, {}

    scored = []
    score_by_key = {}
    for model_position, model in enumerate(models):
        identity = _decoy_identity(model, model_position)
        key = (identity.seed, identity.sample)
        score = _boltz2_confidence_score(model, target_id, identity)
        score_by_key[key] = score
        scored.append((key, score))

    finite = [(key, score) for key, score in scored if math.isfinite(score)]
    finite.sort(key=lambda item: (-item[1], item[0][1]))
    ranking_by_key = {key: rank for rank, (key, _score) in enumerate(finite, start=1)}
    return ranking_by_key, score_by_key


def _select_best_ranking_model(
    source: str,
    target_id: str,
    models: Sequence[object],
    ranking_by_key: Dict[Tuple[Optional[int], int], int],
    score_by_key: Dict[Tuple[Optional[int], int], float],
) -> List[Tuple[int, object]]:
    if not models:
        return []

    identities = [_decoy_identity(model, model_position) for model_position, model in enumerate(models)]
    if ranking_by_key:
        best_key = min(ranking_by_key, key=lambda key: ranking_by_key[key])
        for model_position, model in enumerate(models):
            identity = identities[model_position]
            if (identity.seed, identity.sample) == best_key:
                return [(model_position, model)]

    scored = []
    for model_position, model in enumerate(models):
        identity = identities[model_position]
        key = (identity.seed, identity.sample)
        score = score_by_key.get(key)
        if score is None and _is_boltz2_source(source):
            score = _boltz2_confidence_score(model, target_id, identity)
        if score is not None and math.isfinite(score):
            scored.append((float(score), model_position, model))
    if scored:
        _score, model_position, model = max(scored, key=lambda item: (item[0], -item[1]))
        return [(model_position, model)]

    ranked = []
    for model_position, model in enumerate(models):
        try:
            ranking = int(getattr(model, "ranking", 0))
        except (TypeError, ValueError):
            continue
        if ranking > 0:
            ranked.append((ranking, model_position, model))
    if ranked:
        _ranking, model_position, model = min(ranked, key=lambda item: (item[0], item[1]))
        return [(model_position, model)]

    return [(0, models[0])]


def _common_decoy_columns(
    target_id: str,
    source: str,
    target_path: str,
    target: Target,
    model,
    identity: DecoyIdentity,
    ranking_override: Optional[int] = None,
    ranking_score_override: Optional[float] = None,
):
    native_path = str(getattr(target, "pdb_path", ""))
    decoy_path = str(getattr(model, "pdb_path", ""))
    return {
        "target_id": target_id,
        "source": source,
        "seed": identity.seed,
        "sample": identity.sample,
        "decoy_id": identity.decoy_id,
        "native_path": native_path,
        "decoy_path": decoy_path,
        "target_pickle_path": str(target_path),
        "ranking": ranking_override if ranking_override is not None else getattr(model, "ranking", -1),
        "ranking_score": (
            ranking_score_override
            if ranking_score_override is not None
            else getattr(model, "ranking_score", float("nan"))
        ),
    }


def _target_row(target_id: str, source: str, target_path: str, target: Target):
    native_map = _residue_map(target.gt_structure)
    receptor_chains = tuple(chain for chain in ANTIBODY_CHAINS if chain in native_map)
    ligand_chains = tuple(sorted(set(native_map) - set(receptor_chains)))
    row = {
        "target_id": target_id,
        "source": source,
        "native_path": str(getattr(target, "pdb_path", "")),
        "target_pickle_path": str(target_path),
        "has_holo_antigen": bool(ligand_chains),
        "antibody_chains": ",".join(receptor_chains),
        "antigen_chains": ",".join(ligand_chains),
    }
    row.update(_target_loop_counts(target.gt_structure))
    return row


def _write_parquet(rows: List[dict], path: Path, overwrite: bool, columns: Optional[List[str]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    if rows:
        frame = pd.DataFrame(rows)
        if columns is not None:
            extra_columns = [col for col in frame.columns if col not in columns]
            frame = frame.reindex(columns=columns + extra_columns)
    else:
        frame = pd.DataFrame(columns=columns)
    frame.to_parquet(path, index=False)


def _append_csv_rows(rows: List[dict], path: Path, columns: Optional[List[str]] = None):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if columns is not None:
        extra_columns = [col for col in frame.columns if col not in columns]
        frame = frame.reindex(columns=columns + extra_columns)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _reset_incremental_csv_dir(csv_dir: Path):
    for name in (
        "targets.csv",
        "loop_metrics.csv",
        "interface_metrics.csv",
        "dockq_metrics.csv",
        "progress.csv",
    ):
        path = csv_dir / name
        if path.exists():
            path.unlink()


def _dedupe_rows(rows: List[dict], key_columns: Sequence[str]) -> List[dict]:
    seen = set()
    deduped = []
    for row in rows:
        key = tuple(row.get(col) for col in key_columns)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _ensure_parquet_engine():
    try:
        import pyarrow  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    try:
        import fastparquet  # noqa: F401
        return
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Writing parquet requires pyarrow or fastparquet in this Python environment. "
            "Install one of them, or run with --dry-run to validate metric computation without writing files."
        ) from exc


def _resolve_target_pickle_candidates(spec: DatasetSpec, target_id: str, source: str):
    registry = SourceRegistry(spec)
    candidates = [
        cand for cand in registry.get_candidates(target_id, epoch=0)
        if cand.source_name == source and cand.file_type == "target_model_pickle"
    ]
    return candidates


def _source_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _resolve_sources(spec: DatasetSpec, requested: Optional[Sequence[str]]) -> List[str]:
    if not requested:
        return sorted(spec.enabled_sources())

    by_lower = {name.lower(): [name] for name in spec.sources}
    by_key: Dict[str, List[str]] = {}
    for name in spec.sources:
        by_key.setdefault(_source_key(name), []).append(name)
        by_key.setdefault(re.sub(r"\d+$", "", _source_key(name)), []).append(name)

    resolved: List[str] = []
    unknown: List[str] = []
    for raw_name in requested:
        query = raw_name.lower()
        matches = by_lower.get(query) or by_key.get(_source_key(raw_name))
        if not matches:
            prefix_matches = [
                name for name in spec.sources
                if _source_key(name).startswith(_source_key(raw_name))
            ]
            matches = prefix_matches or None
        if not matches:
            unknown.append(raw_name)
            continue
        for match in matches:
            if match not in resolved:
                resolved.append(match)

    if unknown:
        raise ValueError(f"Unknown sources requested: {unknown}")
    return resolved


def _metric_groups(raw_groups: Sequence[str]) -> set[str]:
    groups = {group.lower() for group in raw_groups}
    if "all" in groups:
        return {"loop", "interface", "dockq"}
    valid = {"loop", "interface", "dockq"}
    invalid = sorted(groups - valid)
    if invalid:
        raise ValueError(f"Unknown metric groups: {invalid}")
    return groups


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _resolve_torch_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"--device {device_arg!r} requested CUDA, but torch.cuda.is_available() is false")
    return device


def run(args):
    spec = load_dataset_spec(args.dataset_config)
    if not args.dry_run:
        _ensure_parquet_engine()
    metric_groups = _metric_groups(args.metric_groups)
    compute_loop = "loop" in metric_groups
    compute_interface = "interface" in metric_groups
    compute_dockq = "dockq" in metric_groups
    needs_interface_calculation = compute_interface or compute_dockq
    device = _resolve_torch_device(args.device)
    use_tqdm = bool(sys.stderr.isatty() and not args.no_tqdm)
    target_ids = _load_target_ids(spec, args.split, args.target_list)
    if args.max_targets is not None:
        target_ids = target_ids[: args.max_targets]

    enabled_sources = set(spec.enabled_sources())
    sources = _resolve_sources(spec, args.sources)
    for source in sources:
        if source not in enabled_sources:
            print(f"[WARN] source={source} is disabled in dataset config; no candidates may be resolved", flush=True)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = DATA_ROOT / output_dir
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for source in sources:
        incremental_csv_dir = None
        if args.incremental_csv and not args.dry_run:
            if args.incremental_csv_dir:
                incremental_csv_dir = Path(args.incremental_csv_dir)
                if not incremental_csv_dir.is_absolute():
                    incremental_csv_dir = output_dir / incremental_csv_dir
            else:
                incremental_csv_dir = output_dir / source / "incremental_csv"
            incremental_csv_dir.mkdir(parents=True, exist_ok=True)
            if args.overwrite_incremental_csv:
                _reset_incremental_csv_dir(incremental_csv_dir)

        source_t0 = time.perf_counter()
        target_rows: List[dict] = []
        loop_rows: List[dict] = []
        interface_rows: List[dict] = []
        dockq_rows: List[dict] = []
        n_missing_reports = 0
        n_targets_done = 0
        n_decoys_done = 0

        target_iter = enumerate(target_ids, start=1)
        if use_tqdm:
            target_iter = tqdm(
                target_iter,
                total=len(target_ids),
                desc=f"{source}:targets",
                dynamic_ncols=True,
            )

        for target_i, target_id in target_iter:
            target_t0 = time.perf_counter()
            decoys_before_target = n_decoys_done
            target_loaded = False
            current_target_rows: List[dict] = []
            current_loop_rows: List[dict] = []
            current_interface_rows: List[dict] = []
            current_dockq_rows: List[dict] = []
            candidates = _resolve_target_pickle_candidates(spec, target_id, source)
            if candidates:
                for cand in candidates:
                    target_path = cand.target_model_path
                    if not target_path:
                        continue
                    try:
                        target = _load_target_pickle(target_path)
                    except Exception as exc:
                        print(f"[WARN] failed to load target pickle source={source} target={target_id} path={target_path}: {exc}", flush=True)
                        continue

                    target_row = _target_row(cand.pdb_id or target_id, source, target_path, target)
                    target_rows.append(target_row)
                    current_target_rows.append(target_row)
                    target_loaded = True
                    n_targets_done += 1

                    models = target.models
                    if args.max_decoys_per_target is not None:
                        models = models[: args.max_decoys_per_target]
                    ranking_by_key, ranking_score_by_key = _ranking_overrides_for_models(
                        source,
                        cand.pdb_id or target_id,
                        models,
                    )
                    decoy_records = []

                    if args.best_ranking_only:
                        model_items = _select_best_ranking_model(
                            source,
                            cand.pdb_id or target_id,
                            models,
                            ranking_by_key,
                            ranking_score_by_key,
                        )
                    else:
                        model_items = list(enumerate(models))

                    decoy_iter = iter(model_items)
                    if use_tqdm:
                        decoy_iter = tqdm(
                            decoy_iter,
                            total=len(model_items),
                            desc=f"{source}:{target_id}:decoys",
                            leave=False,
                            dynamic_ncols=True,
                        )
                    for model_position, model in decoy_iter:
                        identity = _decoy_identity(model, model_position)
                        identity_key = (identity.seed, identity.sample)
                        common = _common_decoy_columns(
                            cand.pdb_id or target_id,
                            source,
                            target_path,
                            target,
                            model,
                            identity,
                            ranking_override=ranking_by_key.get(identity_key),
                            ranking_score_override=ranking_score_by_key.get(identity_key),
                        )
                        decoy_records.append((model, common))

                        if compute_loop:
                            try:
                                loop_row = dict(common)
                                loop_row.update(compute_loop_row(target.gt_structure, model.md_structure))
                                if int(loop_row.get("missing_backbone_atom_count", 0)) >= args.missing_atom_report_threshold:
                                    n_missing_reports += 1
                                    print(
                                        "[WARN] masked backbone atoms "
                                        f"target={common['target_id']} source={source} decoy_id={identity.decoy_id} "
                                        f"count={loop_row['missing_backbone_atom_count']}",
                                        flush=True,
                                    )
                                loop_rows.append(loop_row)
                                current_loop_rows.append(loop_row)
                            except Exception as exc:
                                failed = dict(common)
                                failed.update({
                                    "cdr_rmsd": float("nan"),
                                    "cdr_lddt": float("nan"),
                                    "missing_backbone_atom_count": float("nan"),
                                    "missing_backbone_report": f"loop_metric_failed: {exc}",
                                })
                                for name in LOOP_COLUMN_PREFIXES:
                                    failed[f"{name}_loop_rmsd"] = float("nan")
                                    failed[f"{name}_loop_lddt"] = float("nan")
                                loop_rows.append(failed)
                                current_loop_rows.append(failed)

                        n_decoys_done += 1

                    if needs_interface_calculation and decoy_records:
                        try:
                            fast_rows = compute_interface_rows_for_models_fast(
                                target.gt_structure,
                                [model.md_structure for model, _common in decoy_records],
                                contact_cutoff_a=args.contact_cutoff,
                                interface_cutoff_a=args.interface_cutoff,
                                device=device,
                                contact_batch_size=args.contact_batch_size,
                            )
                        except Exception as exc:
                            fast_rows = []
                            for _model, _common in decoy_records:
                                interface_row = {
                                    "interface_bb_lddt": float("nan"),
                                    "interface_rmsd": float("nan"),
                                    "irmsd": float("nan"),
                                    "lrmsd": float("nan"),
                                    "cdr_antigen_contact_count": float("nan"),
                                    "cdr_antigen_contact_recovery": float("nan"),
                                    "fnat": float("nan"),
                                    "native_contact_count": float("nan"),
                                    "decoy_contact_count": float("nan"),
                                    "interface_metric_report": f"interface_metric_failed: {exc}",
                                }
                                dockq_row = {
                                    "dockq": float("nan"),
                                    "fnat": float("nan"),
                                    "irmsd": float("nan"),
                                    "lrmsd": float("nan"),
                                }
                                fast_rows.append((interface_row, dockq_row))

                        for (_model, common), (interface_row, dockq_row) in zip(decoy_records, fast_rows):
                            if compute_interface:
                                full_interface_row = dict(common)
                                full_interface_row.update(interface_row)
                                interface_rows.append(full_interface_row)
                                current_interface_rows.append(full_interface_row)

                            if compute_dockq:
                                full_dockq_row = dict(common)
                                full_dockq_row.update(dockq_row)
                                dockq_rows.append(full_dockq_row)
                                current_dockq_rows.append(full_dockq_row)

            target_elapsed = time.perf_counter() - target_t0
            target_decoys = n_decoys_done - decoys_before_target
            if incremental_csv_dir is not None:
                _append_csv_rows(current_target_rows, incremental_csv_dir / "targets.csv", TARGET_COLUMNS)
                if compute_loop:
                    _append_csv_rows(current_loop_rows, incremental_csv_dir / "loop_metrics.csv", LOOP_COLUMNS)
                if compute_interface:
                    _append_csv_rows(current_interface_rows, incremental_csv_dir / "interface_metrics.csv", INTERFACE_COLUMNS)
                if compute_dockq:
                    _append_csv_rows(current_dockq_rows, incremental_csv_dir / "dockq_metrics.csv", DOCKQ_COLUMNS)
                elapsed = time.perf_counter() - source_t0
                avg_target_sec = elapsed / max(1, target_i)
                eta = avg_target_sec * max(0, len(target_ids) - target_i)
                _append_csv_rows(
                    [{
                        "source": source,
                        "target_id": target_id,
                        "target_index": target_i,
                        "total_targets": len(target_ids),
                        "target_loaded": bool(target_loaded),
                        "target_decoys": target_decoys,
                        "cumulative_targets_done": n_targets_done,
                        "cumulative_decoys_done": n_decoys_done,
                        "target_elapsed_sec": target_elapsed,
                        "elapsed": _format_duration(elapsed),
                        "avg_target_sec": avg_target_sec,
                        "eta": _format_duration(eta),
                        "best_ranking_only": bool(args.best_ranking_only),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    }],
                    incremental_csv_dir / "progress.csv",
                )
            if (
                target_loaded
                and args.target_log_interval > 0
                and n_targets_done % args.target_log_interval == 0
            ):
                print(
                    f"[TARGET_DONE] source={source} target={target_id} "
                    f"decoys={target_decoys} elapsed_sec={target_elapsed:.1f}",
                    flush=True,
                )

            if args.progress_interval > 0 and target_i % args.progress_interval == 0:
                elapsed = time.perf_counter() - source_t0
                avg_target_sec = elapsed / max(1, target_i)
                eta = avg_target_sec * max(0, len(target_ids) - target_i)
                print(
                    f"[PROGRESS] source={source} targets={target_i}/{len(target_ids)} "
                    f"decoys={n_decoys_done} elapsed={_format_duration(elapsed)} "
                    f"avg_target_sec={avg_target_sec:.1f} eta={_format_duration(eta)}",
                    flush=True,
                )

        source_dir = output_dir / source
        if not args.dry_run:
            target_rows = _dedupe_rows(target_rows, ["target_id", "source"])
            decoy_key = ["target_id", "source", "seed", "sample"]
            loop_rows = _dedupe_rows(loop_rows, decoy_key)
            interface_rows = _dedupe_rows(interface_rows, decoy_key)
            dockq_rows = _dedupe_rows(dockq_rows, decoy_key)
            _write_parquet(target_rows, source_dir / "targets" / "target_metrics.parquet", args.overwrite, TARGET_COLUMNS)
            if compute_loop:
                _write_parquet(loop_rows, source_dir / "metrics" / "loop_metrics.parquet", args.overwrite, LOOP_COLUMNS)
            if compute_interface:
                _write_parquet(interface_rows, source_dir / "metrics" / "interface_metrics.parquet", args.overwrite, INTERFACE_COLUMNS)
            if compute_dockq:
                _write_parquet(dockq_rows, source_dir / "metrics" / "dockq_metrics.parquet", args.overwrite, DOCKQ_COLUMNS)

        elapsed = time.perf_counter() - source_t0
        print(
            f"[DONE] source={source} targets={n_targets_done} decoys={n_decoys_done} "
            f"missing_reports={n_missing_reports} elapsed_sec={elapsed:.1f} "
            f"out={source_dir} dry_run={bool(args.dry_run)}",
            flush=True,
        )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True, help="Dataset YAML used to resolve source target pickles.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory where source-level parquet files are written. "
            "Relative paths are resolved under /home/sujin/projects/cdr-scoring/cdr-data."
        ),
    )
    parser.add_argument("--sources", nargs="+", default=None, help="Source names to process. Defaults to all enabled sources.")
    parser.add_argument(
        "--metric-groups",
        nargs="+",
        default=["loop"],
        choices=("loop", "interface", "dockq", "all"),
        help="Metric groups to compute. Default: loop.",
    )
    parser.add_argument("--split", choices=("train", "valid"), default="train", help="Target list split from dataset config.")
    parser.add_argument("--target-list", default="", help="Optional explicit text file with one target id per line.")
    parser.add_argument("--max-targets", type=int, default=None, help="Small-run limit for number of targets.")
    parser.add_argument("--max-decoys-per-target", type=int, default=None, help="Small-run limit per target pickle.")
    parser.add_argument(
        "--best-ranking-only",
        action="store_true",
        help="For each target, compute metrics only for the decoy with the best Boltz ranking/confidence score.",
    )
    parser.add_argument("--device", default="auto", help="Torch device for tensorized contact metrics. Default: auto.")
    parser.add_argument(
        "--contact-batch-size",
        type=int,
        default=0,
        help="Number of decoys per tensorized contact chunk. Use 0 to batch all decoys for a target.",
    )
    parser.add_argument("--contact-cutoff", type=float, default=5.0, help="Heavy-atom contact cutoff for fnat/contact recovery.")
    parser.add_argument("--interface-cutoff", type=float, default=10.0, help="Heavy-atom cutoff defining interface residues for iRMSD/lRMSD.")
    parser.add_argument("--missing-atom-report-threshold", type=int, default=5, help="Warn when this many backbone atoms are masked.")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars even in an interactive terminal.")
    parser.add_argument("--progress-interval", type=int, default=50, help="Print one [PROGRESS] line every N target ids. Use 0 to disable.")
    parser.add_argument("--target-log-interval", type=int, default=1, help="Print one [TARGET_DONE] line every N loaded targets. Use 0 to disable.")
    parser.add_argument("--incremental-csv", action="store_true", help="Append per-target CSV outputs as each target finishes.")
    parser.add_argument(
        "--incremental-csv-dir",
        default="",
        help="Directory for incremental CSV files. Relative paths are resolved under output-dir. Default: {output-dir}/{source}/incremental_csv.",
    )
    parser.add_argument(
        "--overwrite-incremental-csv",
        action="store_true",
        help="Delete existing known incremental CSV files before processing each source.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing parquet files.")
    parser.add_argument("--dry-run", action="store_true", help="Compute metrics and print summary without writing parquet files.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
