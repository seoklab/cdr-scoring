"""On-the-fly H3 local lDDT for finetune aux loss (v1: ComMat / Boltz2 / Xtal only).

Uses the same backbone lDDT definition as ``benchmark/h3_evaluation.py`` +
``benchmark/pdb_fast`` (CoordDict + NativeCache + ModelEvalSlice).
"""
from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import FrozenSet, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmark.h3_evaluation import (  # noqa: E402
    BACKBONE,
    CACHE_VERSION,
    LDDT_CUTOFF_A,
    NativeCache,
    build_model_eval_slice,
    h3_local_lddt_from_slice,
)
from benchmark.pdb_fast import CoordDict, ResKey  # noqa: E402

logger = logging.getLogger(__name__)

# YAML source names (case-insensitive match)
H3_LDDT_V1_SOURCES: FrozenSet[str] = frozenset({"commat", "boltz2", "xtal"})

XTAL_SOURCE_NAMES: FrozenSet[str] = frozenset({"xtal"})


def source_supports_h3_lddt(source_name: str) -> bool:
    return source_name.lower() in H3_LDDT_V1_SOURCES


def is_crystal_self_decoy(source_name: str, model) -> bool:
    """Explicit crystal/self — not rmsd≈0 alone."""
    s = source_name.lower()
    if s in XTAL_SOURCE_NAMES or s.startswith("xtal"):
        return True
    method = (getattr(model, "method", None) or "").lower()
    if method in ("xtal", "crystal"):
        return True
    return False


def _chains_in_coords(coords: CoordDict) -> list[str]:
    return sorted({k[0] for k in coords})


def _unwrap_bio_model(bio_obj):
    """BioPython Structure wraps Model(s); coords live under Model → Chain."""
    if not hasattr(bio_obj, "child_list") or not bio_obj.child_list:
        return bio_obj
    first = bio_obj.child_list[0]
    # Structure → Model (Model's children are Chains with residue child_list)
    if hasattr(first, "child_list") and first.child_list:
        sub = first.child_list[0]
        if hasattr(sub, "child_list"):
            return first
    return bio_obj


def _chains_from_bio(bio_obj):
    """Return chain list from BioPython Structure or Model."""
    bio_obj = _unwrap_bio_model(bio_obj)
    if not hasattr(bio_obj, "child_list") or not bio_obj.child_list:
        return []
    return list(bio_obj.child_list)


def bio_model_to_coords(bio_obj, chain_ids: Optional[set] = None) -> CoordDict:
    """BioPython Structure/Model → backbone CoordDict (N/CA/C/O)."""
    out: CoordDict = {}
    for chain in _chains_from_bio(bio_obj):
        cid = chain.id
        if chain_ids is not None and cid not in chain_ids:
            continue
        for residue in chain:
            if residue.id[0] != " ":
                continue
            resseq = residue.id[1]
            icode = (residue.id[2] or "").strip()
            key: ResKey = (cid, int(resseq), icode)
            atoms = {}
            for name in BACKBONE:
                if name in residue:
                    atoms[name] = np.asarray(residue[name].get_coord(), dtype=np.float64)
            if atoms:
                out[key] = atoms
    return out


def _h3_keys_from_range(
    coords: CoordDict,
    heavy_chain: str,
    h3_range: Tuple[int, int],
) -> frozenset[ResKey]:
    h3_start, h3_end = h3_range
    keys = [
        key for key in coords
        if key[0] == heavy_chain and h3_start <= key[1] <= h3_end
    ]
    return frozenset(keys)


def build_native_cache_from_coords(
    nat_res: CoordDict,
    target_id: str,
    h3_range: Tuple[int, int],
    heavy_chain: str = "H",
    light_chain: str = "L",
) -> NativeCache:
    """Build NativeCache from native CoordDict (same logic as h3_evaluation.NativeCache.build)."""
    from scipy.spatial.distance import cdist

    from benchmark.h3_evaluation import (
        _cross_group_interface_residues,
        _residue_contacts,
        heavy_coords_from_res,
        FNAT_CONTACT_CUTOFF_A,
    )

    nat_heavy = heavy_chain
    nat_light = light_chain
    mod_heavy = heavy_chain
    mod_light = light_chain
    all_chains = _chains_in_coords(nat_res)
    nat_ag = [c for c in all_chains if c not in (nat_heavy, nat_light)]
    mod_ag = list(nat_ag)

    h3_keys = _h3_keys_from_range(nat_res, nat_heavy, h3_range)
    mod_h3_keys = frozenset((mod_heavy, k[1], k[2]) for k in h3_keys)

    chain_map = {nat_heavy: mod_heavy, nat_light: mod_light}
    for nc, mc in zip(nat_ag, mod_ag):
        chain_map[nc] = mc

    ab_chains = frozenset({nat_heavy, nat_light})
    ag_chains = frozenset(nat_ag)
    nat_chains = [nat_heavy, nat_light] + nat_ag

    matched_keys: list[ResKey] = []
    for nat_key in sorted(nat_res):
        if nat_key[0] in nat_chains:
            matched_keys.append(nat_key)

    bb_specs: list[tuple[ResKey, ResKey, str]] = []
    atom_to_res_list: list[int] = []
    is_h3_score_list: list[bool] = []
    nat_key_atom_indices: dict[ResKey, list[int]] = {}
    h3_local_align_list: list[int] = []
    h3_list: list[int] = []
    rec_list: list[int] = []
    lig_list: list[int] = []
    res_idx = 0
    atom_i = 0

    for nat_key in matched_keys:
        n_chain, _, _ = nat_key
        mod_key = (chain_map[n_chain], nat_key[1], nat_key[2])
        res_atoms = nat_res.get(nat_key, {})
        if not any(name in res_atoms for name in BACKBONE):
            continue

        res_atom_start = atom_i
        for name in BACKBONE:
            if name not in res_atoms:
                continue
            bb_specs.append((nat_key, mod_key, name))
            atom_to_res_list.append(res_idx)
            is_h3 = nat_key in h3_keys
            is_h3_score_list.append(is_h3)
            if is_h3:
                h3_list.append(atom_i)
            elif n_chain == nat_heavy or n_chain == nat_light:
                h3_local_align_list.append(atom_i)
                rec_list.append(atom_i)
            if n_chain in ag_chains:
                lig_list.append(atom_i)
            atom_i += 1

        if atom_i > res_atom_start:
            nat_key_atom_indices[nat_key] = list(range(res_atom_start, atom_i))
        res_idx += 1

    if not bb_specs:
        raise ValueError(f"{target_id}: no matched backbone atoms in native")

    nat_pts = np.stack([nat_res[nk][a] for nk, _mk, a in bb_specs], axis=0)
    atom_to_res = np.asarray(atom_to_res_list, dtype=np.int32)
    is_h3_score = np.asarray(is_h3_score_list, dtype=bool)
    d_nat = cdist(nat_pts, nat_pts)
    res_ids = atom_to_res[:, None]
    neighbor = (res_ids != res_ids.T) & (d_nat > 0) & (d_nat <= LDDT_CUTOFF_A)

    nat_heavy_by_res = {k: heavy_coords_from_res(nat_res[k]) for k in matched_keys if k in nat_res}
    ab_keys = [k for k in matched_keys if k[0] in ab_chains]
    ag_keys = [k for k in matched_keys if k[0] in ag_chains]
    h3_key_list = [k for k in matched_keys if k in h3_keys]

    ab_ag_contacts = (
        _residue_contacts(ab_keys, nat_heavy_by_res, ag_keys, nat_heavy_by_res, FNAT_CONTACT_CUTOFF_A)
        if ag_keys else frozenset()
    )
    ab_ag_interface = (
        _cross_group_interface_residues(ab_keys, nat_heavy_by_res, ag_keys, nat_heavy_by_res)
        if ag_keys else frozenset()
    )
    h3_ag_contacts = (
        _residue_contacts(h3_key_list, nat_heavy_by_res, ag_keys, nat_heavy_by_res, FNAT_CONTACT_CUTOFF_A)
        if ag_keys else frozenset()
    )

    return NativeCache(
        target_id=target_id,
        cache_version=CACHE_VERSION,
        is_holo=len(ag_chains) > 0,
        nat_heavy=nat_heavy,
        mod_heavy=mod_heavy,
        chain_map=chain_map,
        ab_chains=ab_chains,
        ag_chains=ag_chains,
        h3_keys=h3_keys,
        mod_h3_keys=mod_h3_keys,
        n_h3_residues=sum(1 for k in matched_keys if k in h3_keys),
        matched_keys=matched_keys,
        nat_pts=nat_pts,
        atom_to_res=atom_to_res,
        is_h3_score=is_h3_score,
        d_nat=d_nat,
        neighbor=neighbor,
        bb_specs=bb_specs,
        nat_key_atom_indices=nat_key_atom_indices,
        h3_local_align_indices=np.asarray(h3_local_align_list, dtype=np.int32),
        h3_indices=np.asarray(h3_list, dtype=np.int32),
        rec_indices=np.asarray(rec_list, dtype=np.int32),
        lig_indices=np.asarray(lig_list, dtype=np.int32),
        nat_heavy_by_res=nat_heavy_by_res,
        ab_ag_contacts=ab_ag_contacts,
        ab_ag_interface_residues=ab_ag_interface,
        h3_ag_contacts=h3_ag_contacts,
        n_native_ab_ag_contacts=len(ab_ag_contacts),
        n_native_h3_ag_contacts=len(h3_ag_contacts),
    )


def build_native_cache_from_gt_structure(
    gt_structure,
    target_id: str,
    h3_range: Tuple[int, int],
) -> NativeCache:
    nat_res = bio_model_to_coords(gt_structure)
    return build_native_cache_from_coords(nat_res, target_id, h3_range)


@lru_cache(maxsize=256)
def _cached_native_cache(pickle_path: str, target_id: str, h3_start: int, h3_end: int) -> NativeCache:
    """Process-local cache keyed by native pickle path."""
    import pickle as _pickle

    from data_loading.pdb2dict import Target  # noqa: WPS433

    with open(pickle_path, "rb") as fh:
        target = _pickle.load(fh)
    if not isinstance(target, Target) or target.gt_structure is None:
        raise ValueError(f"{pickle_path}: not a Target with gt_structure")
    return build_native_cache_from_gt_structure(
        target.gt_structure, target_id, (h3_start, h3_end),
    )


def resolve_native_pickle_path(per_source: dict) -> Optional[str]:
    """Prefer Xtal pickle, else any v1 source target pickle."""
    for preferred in ("Xtal", "ComMat", "Boltz2"):
        if preferred in per_source:
            _gp, _rmsds, _vi, tpk, _ag = per_source[preferred]
            if tpk:
                return tpk
    for sname, entry in per_source.items():
        if source_supports_h3_lddt(sname):
            tpk = entry[3]
            if tpk:
                return tpk
    return None


def h3_local_lddt_from_model(
    native_cache: NativeCache,
    md_structure,
) -> float:
    """Compute q for one decoy BioPython structure."""
    if md_structure is None:
        return float("nan")
    mod_res = bio_model_to_coords(md_structure)
    sl = build_model_eval_slice(native_cache, mod_res)
    if sl is None:
        return float("nan")
    return h3_local_lddt_from_slice(sl)


def compute_q(
    source_name: str,
    model,
    md_structure,
    native_cache: Optional[NativeCache],
) -> float:
    if not source_supports_h3_lddt(source_name):
        return float("nan")
    if is_crystal_self_decoy(source_name, model):
        return 1.0
    if native_cache is None:
        return float("nan")
    return h3_local_lddt_from_model(native_cache, md_structure)
