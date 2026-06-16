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

REPO_ROOT = Path(__file__).resolve().parents[1]
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
    missing_backbone_atom_count: int


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
    "global_loop_rmsd", "global_loop_lddt",
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
    if method in ("commat", "pertmd"):
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

    residue_scores = []
    score_res_ids = atom_to_res[score_idx]
    for res_id in np.unique(score_res_ids):
        row_mask = score_res_ids == res_id
        atom_scores = []
        for local_idx, is_match in enumerate(row_mask):
            if not is_match:
                continue
            nbr = neighbor_rows[local_idx]
            if nbr.any():
                atom_scores.append(float(preserved[local_idx, nbr].mean()))
        if atom_scores:
            residue_scores.append(float(np.mean(atom_scores)))
    return float(np.mean(residue_scores)) if residue_scores else float("nan")


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
    missing = 0
    residue_idx = 0

    for chain_id, native_residue, model_residue in _matched_residue_pairs(native_structure, model_structure):
        if chain_id not in DEFAULT_LOOP_RANGES:
            continue
        resseq = int(native_residue.id[1])
        lid = _loop_id(chain_id, resseq)
        nat_atoms, mod_atoms, miss = _matching_backbone_coords(native_residue, model_residue)
        missing += miss
        if not nat_atoms:
            continue
        for nat_coord, mod_coord in zip(nat_atoms, mod_atoms):
            native_coords.append(nat_coord)
            model_coords.append(mod_coord)
            atom_to_res.append(residue_idx)
            loop_ids.append(lid)
        residue_idx += 1

    return LoopAtomPairs(
        native_coords=np.asarray(native_coords, dtype=np.float64),
        model_coords=np.asarray(model_coords, dtype=np.float64),
        atom_to_res=np.asarray(atom_to_res, dtype=np.int32),
        loop_ids=np.asarray(loop_ids, dtype=np.int8),
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


def _masked_loop_lddt(pairs: LoopAtomPairs, target_mask: np.ndarray) -> float:
    return _backbone_lddt(
        pairs.native_coords,
        pairs.model_coords,
        pairs.atom_to_res,
        target_mask,
    )


def compute_loop_row(native_structure, model_structure) -> Dict[str, float | int | str]:
    row: Dict[str, float | int | str] = {}
    pairs = _collect_loop_atom_pairs(native_structure, model_structure)
    global_mask = pairs.loop_ids > 0
    row["global_loop_rmsd"] = _masked_loop_rmsd(pairs, global_mask)
    row["global_loop_lddt"] = _masked_loop_lddt(pairs, global_mask)
    row["missing_backbone_atom_count"] = int(pairs.missing_backbone_atom_count)
    row["missing_backbone_report"] = (
        f"masked_backbone_atoms={int(pairs.missing_backbone_atom_count)}"
        if int(pairs.missing_backbone_atom_count) >= 5
        else ""
    )

    for loop_idx, (name, _chain_id, _start, _end) in enumerate(DEFAULT_NAMED_LOOP_RANGES, start=1):
        loop_mask = pairs.loop_ids == loop_idx
        row[f"{name}_loop_rmsd"] = _masked_loop_rmsd(pairs, loop_mask)
        row[f"{name}_loop_lddt"] = _masked_loop_lddt(pairs, loop_mask)
    return row


def _heavy_atom_coords(residue) -> List[np.ndarray]:
    coords = []
    for atom in residue:
        name = atom.get_name().strip()
        if not name:
            continue
        if atom.element == "H" or name.startswith("H") or (len(name) >= 2 and name[0].isdigit() and name[1] == "H"):
            continue
        coords.append(atom.get_coord())
    return coords


def _has_contact(coords_a, coords_b, cutoff_sq: float) -> bool:
    for a in coords_a:
        for b in coords_b:
            diff = a - b
            if float(np.dot(diff, diff)) <= cutoff_sq:
                return True
    return False


def _contact_pairs(
    residue_map: Dict[str, Dict[tuple, object]],
    receptor_chains: Iterable[str],
    ligand_chains: Iterable[str],
    *,
    cutoff_a: float,
    cdr_only: bool = False,
) -> set[Tuple[str, tuple, str, tuple]]:
    cutoff_sq = cutoff_a * cutoff_a
    contacts = set()
    for rec_chain in receptor_chains:
        for lig_chain in ligand_chains:
            for rec_id, rec_residue in residue_map.get(rec_chain, {}).items():
                rec_resseq = int(rec_id[1])
                if cdr_only and not _is_loop_residue(rec_chain, rec_resseq):
                    continue
                rec_atoms = _heavy_atom_coords(rec_residue)
                if not rec_atoms:
                    continue
                for lig_id, lig_residue in residue_map.get(lig_chain, {}).items():
                    lig_atoms = _heavy_atom_coords(lig_residue)
                    if lig_atoms and _has_contact(rec_atoms, lig_atoms, cutoff_sq):
                        contacts.add((rec_chain, rec_id, lig_chain, lig_id))
    return contacts


def _collect_backbone_by_keys(native_map, model_map, keys: Sequence[Tuple[str, tuple]]):
    nat = []
    mod = []
    for chain_id, residue_id in sorted(keys):
        native_residue = native_map.get(chain_id, {}).get(residue_id)
        model_residue = model_map.get(chain_id, {}).get(residue_id)
        if native_residue is None or model_residue is None:
            continue
        for atom_name in BACKBONE_ATOMS:
            native_atom = native_residue.child_dict.get(atom_name)
            model_atom = model_residue.child_dict.get(atom_name)
            if native_atom is None or model_atom is None:
                continue
            nat.append(native_atom.get_coord())
            mod.append(model_atom.get_coord())
    return np.asarray(nat, dtype=np.float64), np.asarray(mod, dtype=np.float64)


def _all_residue_keys(residue_map, chains):
    return [(chain_id, residue_id) for chain_id in chains for residue_id in residue_map.get(chain_id, {})]


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
    )
    decoy_contacts = _contact_pairs(
        model_map, receptor_chains, ligand_chains, cutoff_a=contact_cutoff_a, cdr_only=False,
    )
    native_cdr_contacts = _contact_pairs(
        native_map, receptor_chains, ligand_chains, cutoff_a=contact_cutoff_a, cdr_only=True,
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
    )
    iface_keys = {
        (rec_chain, rec_id)
        for rec_chain, rec_id, _lig_chain, _lig_id in interface_contacts_for_rmsd
    } | {
        (lig_chain, lig_id)
        for _rec_chain, _rec_id, lig_chain, lig_id in interface_contacts_for_rmsd
    }
    receptor_keys = _all_residue_keys(native_map, receptor_chains)
    ligand_keys = _all_residue_keys(native_map, ligand_chains)

    interface_nat, interface_mod = _collect_backbone_by_keys(native_map, model_map, list(iface_keys))
    receptor_nat, receptor_mod = _collect_backbone_by_keys(native_map, model_map, receptor_keys)
    ligand_nat, ligand_mod = _collect_backbone_by_keys(native_map, model_map, ligand_keys)

    irmsd = _aligned_rmsd(interface_nat, interface_mod, interface_nat, interface_mod)
    lrmsd = _aligned_rmsd(receptor_nat, receptor_mod, ligand_nat, ligand_mod)
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


def _is_boltz2_source(source: str) -> bool:
    return _source_key(source) == "boltz2"


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


def run(args):
    spec = load_dataset_spec(args.dataset_config)
    if not args.dry_run:
        _ensure_parquet_engine()
    metric_groups = _metric_groups(args.metric_groups)
    compute_loop = "loop" in metric_groups
    compute_interface = "interface" in metric_groups
    compute_dockq = "dockq" in metric_groups
    needs_interface_calculation = compute_interface or compute_dockq
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

                    target_rows.append(_target_row(cand.pdb_id or target_id, source, target_path, target))
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

                    decoy_iter = enumerate(models)
                    if use_tqdm:
                        decoy_iter = tqdm(
                            decoy_iter,
                            total=len(models),
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
                            except Exception as exc:
                                failed = dict(common)
                                failed.update({
                                    "global_loop_rmsd": float("nan"),
                                    "global_loop_lddt": float("nan"),
                                    "missing_backbone_atom_count": float("nan"),
                                    "missing_backbone_report": f"loop_metric_failed: {exc}",
                                })
                                for name in LOOP_COLUMN_PREFIXES:
                                    failed[f"{name}_loop_rmsd"] = float("nan")
                                    failed[f"{name}_loop_lddt"] = float("nan")
                                loop_rows.append(failed)

                        if needs_interface_calculation:
                            try:
                                interface_row, dockq_row = compute_interface_rows(
                                    target.gt_structure,
                                    model.md_structure,
                                    contact_cutoff_a=args.contact_cutoff,
                                    interface_cutoff_a=args.interface_cutoff,
                                )
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
                                dockq_row = {"dockq": float("nan"), "fnat": float("nan"), "irmsd": float("nan"), "lrmsd": float("nan")}

                            if compute_interface:
                                full_interface_row = dict(common)
                                full_interface_row.update(interface_row)
                                interface_rows.append(full_interface_row)

                            if compute_dockq:
                                full_dockq_row = dict(common)
                                full_dockq_row.update(dockq_row)
                                dockq_rows.append(full_dockq_row)
                        n_decoys_done += 1

            target_elapsed = time.perf_counter() - target_t0
            target_decoys = n_decoys_done - decoys_before_target
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
    parser.add_argument("--contact-cutoff", type=float, default=5.0, help="Heavy-atom contact cutoff for fnat/contact recovery.")
    parser.add_argument("--interface-cutoff", type=float, default=10.0, help="Heavy-atom cutoff defining interface residues for iRMSD/lRMSD.")
    parser.add_argument("--missing-atom-report-threshold", type=int, default=5, help="Warn when this many backbone atoms are masked.")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars even in an interactive terminal.")
    parser.add_argument("--progress-interval", type=int, default=50, help="Print one [PROGRESS] line every N target ids. Use 0 to disable.")
    parser.add_argument("--target-log-interval", type=int, default=1, help="Print one [TARGET_DONE] line every N loaded targets. Use 0 to disable.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing parquet files.")
    parser.add_argument("--dry-run", action="store_true", help="Compute metrics and print summary without writing parquet files.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
