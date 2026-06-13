from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser, Structure


BACKBONE_ATOMS = ("N", "CA", "C", "O")
LDDT_CUTOFF_A = 15.0
LDDT_THRESHOLDS_A = (0.5, 1.0, 2.0, 4.0)

# Chothia CDR definitions used in this project.
DEFAULT_LOOP_RANGES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "H": ((26, 32), (52, 56), (95, 102)),
    "L": ((24, 34), (50, 56), (89, 97)),
}

DEFAULT_NAMED_LOOP_RANGES: Tuple[Tuple[str, str, int, int], ...] = (
    ("H1", "H", 26, 32),
    ("H2", "H", 52, 56),
    ("H3", "H", 95, 102),
    ("L1", "L", 24, 34),
    ("L2", "L", 50, 56),
    ("L3", "L", 89, 97),
)


@dataclass(frozen=True)
class LoopMetricResult:
    native_path: str
    model_path: str
    loop_rmsd: float
    loop_lddt: float
    n_align_atoms: int
    n_loop_atoms: int
    n_backbone_atoms: int
    n_loop_residues: int
    align_residues: Tuple[Tuple[str, int], ...]
    loop_residues: Tuple[Tuple[str, int], ...]
    per_cdr_rmsd: Dict[str, float]
    per_cdr_lddt: Dict[str, float]


def load_structure(path: Union[str, Path]) -> Structure.Structure:
    path = Path(path)
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() == ".cif" else PDBParser(QUIET=True)
    return parser.get_structure(path.stem, str(path))


def compute_loop_metrics(
    native_path: Union[str, Path],
    model_path: Union[str, Path],
    *,
    loop_ranges: Dict[str, Tuple[Tuple[int, int], ...]] = DEFAULT_LOOP_RANGES,
    rmsd_atom_type: str = "all",
) -> LoopMetricResult:
    native_structure = load_structure(native_path)
    model_structure = load_structure(model_path)
    return compute_loop_metrics_from_structures(
        native_structure,
        model_structure,
        native_path=str(native_path),
        model_path=str(model_path),
        loop_ranges=loop_ranges,
        rmsd_atom_type=rmsd_atom_type,
    )


def compute_loop_metrics_from_structures(
    native_structure,
    model_structure,
    *,
    native_path: str = "",
    model_path: str = "",
    loop_ranges: Dict[str, Tuple[Tuple[int, int], ...]] = DEFAULT_LOOP_RANGES,
    rmsd_atom_type: str = "all",
) -> LoopMetricResult:
    native_model = next(native_structure.get_models())
    model_model = next(model_structure.get_models())

    matched_pairs = list(_iter_matched_residue_pairs(native_model, model_model))
    (
        align_nat,
        align_mod,
        loop_nat,
        loop_mod,
        align_residues,
        loop_residues,
    ) = _collect_loop_rmsd_atoms(
        matched_pairs,
        loop_ranges=loop_ranges,
        atom_type=rmsd_atom_type,
    )
    loop_rmsd = _aligned_rmsd(align_nat, align_mod, loop_nat, loop_mod)

    nat_pts, mod_pts, atom_to_res, loop_atom_mask, loop_residue_count = _collect_backbone_pairs(
        matched_pairs,
        loop_ranges=loop_ranges,
    )
    loop_lddt = _backbone_lddt(nat_pts, mod_pts, atom_to_res, loop_atom_mask)
    per_cdr_rmsd, per_cdr_lddt = _compute_per_cdr_metrics(
        matched_pairs,
        loop_ranges=loop_ranges,
        rmsd_atom_type=rmsd_atom_type,
    )

    return LoopMetricResult(
        native_path=native_path,
        model_path=model_path,
        loop_rmsd=loop_rmsd,
        loop_lddt=loop_lddt,
        n_align_atoms=len(align_nat),
        n_loop_atoms=len(loop_nat),
        n_backbone_atoms=len(nat_pts),
        n_loop_residues=loop_residue_count,
        align_residues=tuple(sorted(align_residues)),
        loop_residues=tuple(sorted(loop_residues)),
        per_cdr_rmsd=per_cdr_rmsd,
        per_cdr_lddt=per_cdr_lddt,
    )


def _compute_per_cdr_metrics(
    matched_pairs,
    *,
    loop_ranges: Dict[str, Tuple[Tuple[int, int], ...]],
    rmsd_atom_type: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    per_cdr_rmsd: Dict[str, float] = {}
    per_cdr_lddt: Dict[str, float] = {}
    available = {
        (chain_id, start, end)
        for chain_id, ranges in loop_ranges.items()
        for start, end in ranges
    }
    for name, chain_id, start, end in DEFAULT_NAMED_LOOP_RANGES:
        if (chain_id, start, end) not in available:
            continue
        single_range = {chain_id: ((start, end),)}
        (
            align_nat,
            align_mod,
            loop_nat,
            loop_mod,
            _align_residues,
            _loop_residues,
        ) = _collect_loop_rmsd_atoms(
            matched_pairs,
            loop_ranges=single_range,
            atom_type=rmsd_atom_type,
        )
        per_cdr_rmsd[name] = _aligned_rmsd(align_nat, align_mod, loop_nat, loop_mod)

        nat_pts, mod_pts, atom_to_res, loop_atom_mask, _loop_residue_count = _collect_backbone_pairs(
            matched_pairs,
            loop_ranges=single_range,
        )
        per_cdr_lddt[name] = _backbone_lddt(nat_pts, mod_pts, atom_to_res, loop_atom_mask)
    return per_cdr_rmsd, per_cdr_lddt


def _iter_matched_residue_pairs(native_model, model_model):
    model_residue_map = {}
    for chain in model_model:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            model_residue_map[(chain.id, residue.id)] = residue

    for native_chain in native_model:
        for native_residue in native_chain:
            if native_residue.id[0] != " ":
                continue
            model_residue = model_residue_map.get((native_chain.id, native_residue.id))
            if model_residue is None:
                continue
            yield native_chain.id, native_residue, model_residue


def _is_loop_residue(
    chain_id: str,
    resseq: int,
    loop_ranges: Dict[str, Tuple[Tuple[int, int], ...]],
) -> bool:
    for start, end in loop_ranges.get(chain_id, ()):
        if start <= resseq <= end:
            return True
    return False


def _collect_loop_rmsd_atoms(
    matched_pairs,
    *,
    loop_ranges: Dict[str, Tuple[Tuple[int, int], ...]],
    atom_type: str,
):
    align_nat: List[np.ndarray] = []
    align_mod: List[np.ndarray] = []
    loop_nat: List[np.ndarray] = []
    loop_mod: List[np.ndarray] = []
    align_residues = set()
    loop_residues = set()

    valid_loop_chains = set(loop_ranges.keys())
    for chain_id, native_residue, model_residue in matched_pairs:
        if chain_id not in valid_loop_chains:
            continue
        resseq = int(native_residue.id[1])
        is_loop = _is_loop_residue(chain_id, resseq, loop_ranges)
        nat_atoms, mod_atoms = _matching_atom_coords(native_residue, model_residue, atom_type)
        if is_loop:
            loop_nat.extend(nat_atoms)
            loop_mod.extend(mod_atoms)
            if nat_atoms:
                loop_residues.add((chain_id, resseq))
        else:
            align_nat.extend(nat_atoms)
            align_mod.extend(mod_atoms)
            if nat_atoms:
                align_residues.add((chain_id, resseq))

    return (
        np.asarray(align_nat, dtype=np.float64),
        np.asarray(align_mod, dtype=np.float64),
        np.asarray(loop_nat, dtype=np.float64),
        np.asarray(loop_mod, dtype=np.float64),
        align_residues,
        loop_residues,
    )


def _matching_atom_coords(native_residue, model_residue, atom_type: str):
    native_coords = []
    model_coords = []
    for native_atom in native_residue:
        atom_name = native_atom.get_name()
        if atom_type == "backbone" and atom_name not in BACKBONE_ATOMS:
            continue
        if atom_type == "heavy" and _is_hydrogen(atom_name):
            continue
        model_atom = model_residue.child_dict.get(atom_name)
        if model_atom is None:
            continue
        if atom_type == "heavy" and _is_hydrogen(model_atom.get_name()):
            continue
        native_coords.append(native_atom.get_coord())
        model_coords.append(model_atom.get_coord())
    return native_coords, model_coords


def _is_hydrogen(atom_name: str) -> bool:
    name = atom_name.strip()
    if not name:
        return False
    if name[0] == "H":
        return True
    if len(name) >= 2 and name[0].isdigit() and name[1] == "H":
        return True
    return False


def _aligned_rmsd(
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


def _collect_backbone_pairs(
    matched_pairs,
    *,
    loop_ranges: Dict[str, Tuple[Tuple[int, int], ...]],
):
    nat_pts: List[np.ndarray] = []
    mod_pts: List[np.ndarray] = []
    atom_to_res: List[int] = []
    loop_atom_mask: List[bool] = []
    residue_idx = 0
    loop_residue_keys = set()

    valid_loop_chains = set(loop_ranges.keys())
    for chain_id, native_residue, model_residue in matched_pairs:
        if chain_id not in valid_loop_chains:
            continue
        resseq = int(native_residue.id[1])
        is_loop = _is_loop_residue(chain_id, resseq, loop_ranges)
        start_count = len(nat_pts)
        for atom_name in BACKBONE_ATOMS:
            native_atom = native_residue.child_dict.get(atom_name)
            model_atom = model_residue.child_dict.get(atom_name)
            if native_atom is None or model_atom is None:
                continue
            nat_pts.append(native_atom.get_coord())
            mod_pts.append(model_atom.get_coord())
            atom_to_res.append(residue_idx)
            loop_atom_mask.append(is_loop)
        if len(nat_pts) > start_count:
            if is_loop:
                loop_residue_keys.add((chain_id, native_residue.id))
            residue_idx += 1

    return (
        np.asarray(nat_pts, dtype=np.float64),
        np.asarray(mod_pts, dtype=np.float64),
        np.asarray(atom_to_res, dtype=np.int32),
        np.asarray(loop_atom_mask, dtype=bool),
        len(loop_residue_keys),
    )


def _backbone_lddt(
    nat_pts: np.ndarray,
    mod_pts: np.ndarray,
    atom_to_res: np.ndarray,
    loop_atom_mask: np.ndarray,
) -> float:
    if len(nat_pts) < 2 or not loop_atom_mask.any():
        return float("nan")

    d_nat = _pairwise_distances(nat_pts)
    d_mod = _pairwise_distances(mod_pts)
    residue_mask = atom_to_res[:, None] != atom_to_res[None, :]
    neighbor = residue_mask & (d_nat > 0) & (d_nat <= LDDT_CUTOFF_A)

    score_idx = np.where(loop_atom_mask)[0]
    d_nat_rows = d_nat[score_idx]
    d_mod_rows = d_mod[score_idx]
    neighbor_rows = neighbor[score_idx]
    diff = np.abs(d_nat_rows - d_mod_rows)

    preserved = np.zeros_like(d_nat_rows, dtype=np.float64)
    for thr in LDDT_THRESHOLDS_A:
        preserved += (diff < thr).astype(np.float64)
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
    if not residue_scores:
        return float("nan")
    return float(np.mean(residue_scores))


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))
