from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from evaluation.loop_metrics import BACKBONE_ATOMS, load_structure


INTERFACE_CUTOFF_A = 10.0
DEFAULT_RECEPTOR_CHAINS = ("H", "L")


@dataclass(frozen=True)
class DockQStyleMetricResult:
    native_path: str
    model_path: str
    irmsd: float
    lrmsd: float
    receptor_chains: Tuple[str, ...]
    ligand_chains: Tuple[str, ...]
    interface_receptor_residues: Tuple[Tuple[str, int], ...]
    interface_ligand_residues: Tuple[Tuple[str, int], ...]
    n_interface_backbone_atoms: int
    n_ligand_backbone_atoms: int
    n_receptor_align_atoms: int


def compute_dockq_style_metrics(
    native_path: Union[str, Path],
    model_path: Union[str, Path],
    *,
    receptor_chains: Sequence[str] = DEFAULT_RECEPTOR_CHAINS,
    interface_cutoff_a: float = INTERFACE_CUTOFF_A,
) -> DockQStyleMetricResult:
    native_structure = load_structure(native_path)
    model_structure = load_structure(model_path)
    return compute_dockq_style_metrics_from_structures(
        native_structure,
        model_structure,
        native_path=str(native_path),
        model_path=str(model_path),
        receptor_chains=receptor_chains,
        interface_cutoff_a=interface_cutoff_a,
    )


def compute_dockq_style_metrics_from_structures(
    native_structure,
    model_structure,
    *,
    native_path: str = "",
    model_path: str = "",
    receptor_chains: Sequence[str] = DEFAULT_RECEPTOR_CHAINS,
    interface_cutoff_a: float = INTERFACE_CUTOFF_A,
) -> DockQStyleMetricResult:
    native_model = next(native_structure.get_models())
    model_model = next(model_structure.get_models())

    native_residue_map = _build_residue_map(native_model)
    model_residue_map = _build_residue_map(model_model)

    receptor_chain_set = {chain for chain in receptor_chains if chain in native_residue_map}
    ligand_chain_set = set(native_residue_map) - receptor_chain_set

    iface_rec_keys, iface_lig_keys = _find_interface_residue_keys(
        native_residue_map,
        receptor_chain_set,
        ligand_chain_set,
        cutoff_a=interface_cutoff_a,
    )

    interface_nat, interface_mod = _collect_backbone_atoms(
        native_residue_map,
        model_residue_map,
        list(iface_rec_keys | iface_lig_keys),
    )
    receptor_nat, receptor_mod = _collect_backbone_atoms(
        native_residue_map,
        model_residue_map,
        _all_residue_keys(native_residue_map, receptor_chain_set),
    )
    ligand_nat, ligand_mod = _collect_backbone_atoms(
        native_residue_map,
        model_residue_map,
        _all_residue_keys(native_residue_map, ligand_chain_set),
    )

    irmsd = _superposed_rmsd(interface_nat, interface_mod, interface_nat, interface_mod)
    # DockQ convention (matches Galaxy reference step2_prep_input.py): the antigen
    # is the receptor used for superposition and lRMSD is measured on the antibody
    # (H/L). receptor_* here are the antibody backbone atoms, so align on the
    # antigen (ligand_*) and measure RMSD on the antibody (receptor_*).
    lrmsd = _superposed_rmsd(ligand_nat, ligand_mod, receptor_nat, receptor_mod)

    return DockQStyleMetricResult(
        native_path=native_path,
        model_path=model_path,
        irmsd=irmsd,
        lrmsd=lrmsd,
        receptor_chains=tuple(sorted(receptor_chain_set)),
        ligand_chains=tuple(sorted(ligand_chain_set)),
        interface_receptor_residues=tuple(sorted((chain, key[1]) for chain, key in iface_rec_keys)),
        interface_ligand_residues=tuple(sorted((chain, key[1]) for chain, key in iface_lig_keys)),
        n_interface_backbone_atoms=len(interface_nat),
        n_ligand_backbone_atoms=len(ligand_nat),
        n_receptor_align_atoms=len(receptor_nat),
    )


def _build_residue_map(model) -> dict:
    residue_map = {}
    for chain in model:
        chain_map = {}
        for residue in chain:
            if residue.id[0] != " ":
                continue
            chain_map[residue.id] = residue
        residue_map[chain.id] = chain_map
    return residue_map


def _find_interface_residue_keys(
    residue_map: dict,
    receptor_chains: Iterable[str],
    ligand_chains: Iterable[str],
    *,
    cutoff_a: float,
) -> Tuple[set, set]:
    interface_receptor = set()
    interface_ligand = set()
    cutoff_sq = cutoff_a * cutoff_a

    for receptor_chain in receptor_chains:
        for ligand_chain in ligand_chains:
            for rec_key, rec_residue in residue_map.get(receptor_chain, {}).items():
                rec_atoms = _residue_heavy_atom_coords(rec_residue)
                if not rec_atoms:
                    continue
                for lig_key, lig_residue in residue_map.get(ligand_chain, {}).items():
                    lig_atoms = _residue_heavy_atom_coords(lig_residue)
                    if not lig_atoms:
                        continue
                    if _has_contact(rec_atoms, lig_atoms, cutoff_sq):
                        interface_receptor.add((receptor_chain, rec_key))
                        interface_ligand.add((ligand_chain, lig_key))
    return interface_receptor, interface_ligand


def _residue_backbone_coords(residue) -> List[np.ndarray]:
    coords = []
    for atom_name in BACKBONE_ATOMS:
        atom = residue.child_dict.get(atom_name)
        if atom is not None:
            coords.append(atom.get_coord())
    return coords


def _residue_heavy_atom_coords(residue) -> List[np.ndarray]:
    coords = []
    for atom in residue:
        atom_name = atom.get_name().strip()
        if not atom_name:
            continue
        if atom.element == "H" or atom_name.startswith("H") or (
            len(atom_name) >= 2 and atom_name[0].isdigit() and atom_name[1] == "H"
        ):
            continue
        coords.append(atom.get_coord())
    return coords


def _has_contact(rec_atoms, lig_atoms, cutoff_sq: float) -> bool:
    for rec_coord in rec_atoms:
        for lig_coord in lig_atoms:
            diff = rec_coord - lig_coord
            if float(np.dot(diff, diff)) <= cutoff_sq:
                return True
    return False


def _collect_backbone_atoms(
    native_residue_map: dict,
    model_residue_map: dict,
    residue_keys: Sequence[Tuple[str, tuple]],
) -> Tuple[np.ndarray, np.ndarray]:
    native_coords: List[np.ndarray] = []
    model_coords: List[np.ndarray] = []
    for chain_id, residue_id in sorted(residue_keys):
        native_residue = native_residue_map.get(chain_id, {}).get(residue_id)
        model_residue = model_residue_map.get(chain_id, {}).get(residue_id)
        if native_residue is None or model_residue is None:
            continue
        for atom_name in BACKBONE_ATOMS:
            native_atom = native_residue.child_dict.get(atom_name)
            model_atom = model_residue.child_dict.get(atom_name)
            if native_atom is None or model_atom is None:
                continue
            native_coords.append(native_atom.get_coord())
            model_coords.append(model_atom.get_coord())
    return np.asarray(native_coords, dtype=np.float64), np.asarray(model_coords, dtype=np.float64)


def _all_residue_keys(residue_map: dict, chains: Iterable[str]) -> List[Tuple[str, tuple]]:
    keys = []
    for chain_id in sorted(chains):
        for residue_id in residue_map.get(chain_id, {}):
            keys.append((chain_id, residue_id))
    return keys


def _superposed_rmsd(
    align_nat: np.ndarray,
    align_mod: np.ndarray,
    target_nat: np.ndarray,
    target_mod: np.ndarray,
) -> float:
    if len(align_nat) < 3 or len(target_nat) == 0:
        return float("nan")
    if align_nat.shape != align_mod.shape or target_nat.shape != target_mod.shape:
        return float("nan")

    rot, trans = _kabsch(align_nat, align_mod)
    target_mod_sup = target_mod @ rot + trans
    diff = target_nat - target_mod_sup
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _kabsch(ref: np.ndarray, mob: np.ndarray):
    ref_centered = ref - ref.mean(axis=0)
    mob_centered = mob - mob.mean(axis=0)
    cov = mob_centered.T @ ref_centered
    u, _, vt = np.linalg.svd(cov)
    det = np.sign(np.linalg.det(vt.T @ u.T))
    rot = u @ np.diag([1.0, 1.0, det]) @ vt
    trans = ref.mean(axis=0) - mob.mean(axis=0) @ rot
    return rot, trans
