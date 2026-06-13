import dgl
import torch
import numpy as np
from Bio.PDB.Polypeptide import is_aa
from data_loading import coords_rosetta6d
from .constants import MAX_REL_INDEX, AA3_IDX, restype_name_to_atom14_names, MAX_NUM_ATOM, REF_CHAIN
import random
import re
import logging


logger = logging.getLogger(__name__)


DEFAULT_CDR_RANGES = {
    "H": ((26, 32), (52, 56), (95, 102)),
    "L": ((24, 34), (50, 56), (89, 97)),
}

CDR_LOOP_IDS = {
    ("H", 26, 32): 1,
    ("H", 52, 56): 2,
    ("H", 95, 102): 3,
    ("L", 24, 34): 4,
    ("L", 50, 56): 5,
    ("L", 89, 97): 6,
}


def _normalize_cdr_ranges(cdr_ranges):
    if cdr_ranges is None:
        return None
    if isinstance(cdr_ranges, dict):
        return {
            str(chain_id): tuple((int(start), int(end)) for start, end in ranges)
            for chain_id, ranges in cdr_ranges.items()
        }

    out = {}
    for item in cdr_ranges:
        if len(item) == 3:
            chain_id, start, end = item
            out.setdefault(str(chain_id), []).append((int(start), int(end)))
        elif len(item) == 2:
            chain_id, ranges = item
            out.setdefault(str(chain_id), []).extend(
                (int(start), int(end)) for start, end in ranges
            )
        else:
            raise ValueError(f"Unsupported CDR range entry: {item!r}")
    return {chain_id: tuple(ranges) for chain_id, ranges in out.items()}


def _is_cdr_residue(chain_id, resseq, cdr_ranges):
    ranges = cdr_ranges.get(chain_id, ())
    return any(start <= int(resseq) <= end for start, end in ranges)


def _cdr_loop_id(chain_id, resseq, cdr_ranges):
    for start, end in cdr_ranges.get(chain_id, ()):
        if start <= int(resseq) <= end:
            return CDR_LOOP_IDS.get((chain_id, start, end), 1)
    return 0


def _get_model_metric(model, metric_name, *, task_scope="full_cdr"):
    aliases = {
        "loop_rmsd": ("loop_rmsd", "full_cdr_loop_rmsd", "cdr_loop_rmsd"),
        "loop_lddt": ("loop_lddt", "full_cdr_loop_lddt", "cdr_loop_lddt"),
        "h3_rmsd": ("h3_rmsd", "loop_rmsd"),
        "h3_lddt": ("h3_lddt", "loop_lddt"),
    }
    if task_scope == "h3":
        aliases = {
            **aliases,
            "loop_rmsd": ("h3_rmsd", "loop_rmsd"),
            "loop_lddt": ("h3_lddt", "loop_lddt"),
        }
    for attr in aliases.get(metric_name, (metric_name,)):
        value = getattr(model, attr, None)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float("nan")


def _parse_seed_sample_from_filename(filename: str):
    """Parse seed and sample from decoy PDB filename.

    Supported patterns (order of precedence):
        AF3:    sd-{seed}_sp-{sample}  or  seed-{seed}_sample-{sample}
        Boltz2: seed_{seed}_model_{sample}

    Returns:
        (seed: int|None, sample: int|None)
    """
    if not filename:
        return None, None
    # AF3 pattern: sd-{seed}_sp-{sample}
    m = re.search(r"sd-(\d+)_sp-(\d+)", filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    # AF3 pattern (alt): seed-{seed}_sample-{sample}
    m = re.search(r"seed-(\d+)_sample-(\d+)", filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Boltz2 pattern: seed_{seed}_model_{sample}
    m = re.search(r"seed_(\d+)_model_(\d+)", filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def parse_model_numbers_from_pdb(pdb_path) -> list:
    """1-based MODEL numbers from a multi-MODEL PDB file (order preserved).

    Used for ComMat / PertMD-style decoys where every structure lives in one PDB.
    Does not load coordinates — safe for quick validation without target pickles.
    """
    nums = []
    path = str(pdb_path)
    with open(path) as fp:
        for line in fp:
            if not line.startswith("MODEL"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                nums.append(int(parts[1]))
            else:
                nums.append(len(nums) + 1)
    return nums


def decoy_identity_from_model(model, index: int):
    """Build {file, seed, sample} for one decoy without rebuilding target pickles.

    Priority:
      1. Parse seed/sample from filename (AF3, Boltz2, …).
      2. Else use model.model_idx if set (e.g. Boltz2).
      3. Else index + 1 (list order == MODEL # in multi-MODEL PDB; ComMat/PertMD).

    Does not use model.ranking — ranking is method/score rank, not sample id.

    ComMat / PertMD: seed is always None (no seed in filenames or pickles).
    """
    filename = ""
    if hasattr(model, "pdb_path") and model.pdb_path is not None:
        filename = (
            str(model.pdb_path.name)
            if hasattr(model.pdb_path, "name")
            else str(model.pdb_path).split("/")[-1]
        )

    seed, sample = _parse_seed_sample_from_filename(filename)

    model_seed = getattr(model, "seed", None)
    if model_seed == -1:
        model_seed = None
    if seed is None and model_seed is not None:
        seed = int(model_seed)

    if sample is None:
        model_idx = getattr(model, "model_idx", None)
        if model_idx is not None:
            sample = int(model_idx)
        else:
            sample = index + 1

    method = (getattr(model, "method", "") or "").lower()
    if method in ("commat", "pertmd"):
        seed = None

    return {"file": filename, "seed": seed, "sample": sample}


# Heavy atom order for backbone (5 atoms)
HEAVY_ATOMS = ["CA", "N", "C", "O", "CB"]

def get_residue_atom_coords(res, expected_atoms, max_atoms):
    """
    Returns a numpy array of shape (max_atoms, 3) with atom coordinates.
    Missing atoms are filled with zeros.
    """
    coords = []
    for atom_name in expected_atoms:
        coords.append(res[atom_name].get_coord() if atom_name and atom_name in res else np.zeros(3))
    while len(coords) < max_atoms:
        coords.append(np.zeros(3))
    return np.array(coords)

def create_dictionary_from_model(
    model,
    use_all_atom=False,
    h3_range=(95, 102),
    cdr_ranges=DEFAULT_CDR_RANGES,
    task_scope="full_cdr",
):
    """
    Processes a Bio.PDB structure (Model object) and returns a dictionary with 
    pre-computed features.
    
    Node-related keys: 'aa_type', 'res_no', 'chain_id', 'str_mask', 'ulr_mask', 'get_res_idx', 'coord_s', 'is_gly'
    Torsion keys from get_coords6d: 'dist6d', 'phi6d', 'omega6d', 'theta6d', 'phi_res', 'psi_res'
    
    - Effective residue numbers are reassigned consecutively per chain.
    - ULR mask is full-CDR by default. Pass ``task_scope="h3"`` or
      ``cdr_ranges=None`` to keep the legacy H3-only mask.
    """
    ca_coords_list = []      # List of CA coordinates (L,3)
    all_coords_list = []     # List of full coordinate tensors (L, max_atoms,3)
    orig_resseq_list = []    # Original residue numbers (integer part)
    chain_ids = []           # Chain IDs
    aa_types = []            # Residue names

    if use_all_atom:
        def get_expected_atoms(resname):
            return restype_name_to_atom14_names.get(resname, restype_name_to_atom14_names['UNK'])
        max_atoms = MAX_NUM_ATOM
    else:
        expected_atoms = HEAVY_ATOMS
        max_atoms = len(HEAVY_ATOMS)
    
    # Iterate over residues in all chains
    for chain in model.md_structure.get_chains():
        for res in chain:
            if not is_aa(res, standard=True) or 'CA' not in res:
                continue
            ca_coords_list.append(res['CA'].get_coord())
            if use_all_atom:
                resname = res.get_resname()
                exp_atoms = get_expected_atoms(resname)
            else:
                exp_atoms = expected_atoms
            all_coords_list.append(get_residue_atom_coords(res, exp_atoms, max_atoms))
            orig_resseq_list.append(res.id[1])
            chain_ids.append(chain.id)
            aa_types.append(res.get_resname())
    
    if len(ca_coords_list) == 0:
        raise ValueError("No valid CA atoms found in the structure.")
    
    L = len(ca_coords_list)
    all_coords = np.stack(all_coords_list, axis=0)    # (L, max_atoms,3)
    
    # Reassign effective residue numbers consecutively per chain.
    effective_numbers = [0] * L
    chain_to_indices = {}
    for idx, cid in enumerate(chain_ids):
        chain_to_indices.setdefault(cid, []).append(idx)
    for cid, indices in chain_to_indices.items():
        start = orig_resseq_list[indices[0]]
        effective_numbers[indices[0]] = start
        for i in range(1, len(indices)):
            effective_numbers[indices[i]] = effective_numbers[indices[i-1]] + 1

    # Use effective residue numbers for 'res_no'
    res_no_tensor = torch.tensor(effective_numbers, dtype=torch.int32)  # (L,)
    chain_id_tensor = torch.tensor([REF_CHAIN.index(cid) for cid in chain_ids], dtype=torch.int32)

    aa_type_tensor = torch.tensor([AA3_IDX.get(aa, AA3_IDX['UNK']) for aa in aa_types], dtype=torch.int64)  # (L,)
    
    cdr_ranges = None if task_scope == "h3" else _normalize_cdr_ranges(cdr_ranges)
    if cdr_ranges:
        ulr_mask = torch.tensor(
            [
                1 if _is_cdr_residue(cid, res, cdr_ranges) else 0
                for res, cid in zip(orig_resseq_list, chain_ids)
            ],
            dtype=torch.int64,
        )
        loop_id = torch.tensor(
            [
                _cdr_loop_id(cid, res, cdr_ranges)
                for res, cid in zip(orig_resseq_list, chain_ids)
            ],
            dtype=torch.int64,
        )
    else:
        ulr_mask = torch.tensor(
            [
                1 if (cid == 'H' and h3_range[0] <= res <= h3_range[1]) else 0
                for res, cid in zip(orig_resseq_list, chain_ids)
            ],
            dtype=torch.int64,
        )
        loop_id = torch.tensor(
            [
                3 if (cid == 'H' and h3_range[0] <= res <= h3_range[1]) else 0
                for res, cid in zip(orig_resseq_list, chain_ids)
            ],
            dtype=torch.int64,
        )
    if int(ulr_mask.sum().item()) == 0:
        logger.warning(
            "create_dictionary_from_model: zero CDR residues detected for %s",
            getattr(model, "pdb_path", getattr(model, "method", "<unknown>")),
        )
    str_mask = torch.ones(L, dtype=torch.int32) if use_all_atom else None

    # Create a dummy "get_res_idx" mapping
    # For each chain, mapping original res number -> index
    get_res_idx = {}
    for idx, cid in enumerate(chain_ids):
        cid_int = REF_CHAIN.index(cid)
        if cid_int not in get_res_idx:
            get_res_idx[cid_int] = {}
        get_res_idx[cid_int][orig_resseq_list[idx]] = idx

    # Create coordinate tensor for decoy: assume B decoys; here we set B=1 for a single model
    # In practice, if multiple decoys exist, this tensor is expanded accordingly.
    coord_s = torch.tensor(all_coords, dtype=torch.float32)  # (L, max_atoms, 3)
    coord_s = coord_s.unsqueeze(0)  # (B=1, L, max_atoms, 3)

    is_gly = (aa_type_tensor == AA3_IDX["GLY"])  # (L,)

    coords = coords_rosetta6d.Coords_rosetta6d()
    dist6d, phi6d, omega6d, theta6d, phi_res, psi_res = coords.get_coords6d(coord_s, is_gly, 20)

    dic = {}
    dic['aa_type'] = aa_type_tensor   # (L,)
    dic['res_no'] = res_no_tensor      # (L,)
    dic['chain_id'] = chain_id_tensor  # (L,)
    dic['str_mask'] = str_mask         # (L,) if use_all_atom else None
    dic['ulr_mask'] = ulr_mask         # (L,)
    dic['loop_id'] = loop_id           # (L,), 0 non-CDR; 1-6 H1/H2/H3/L1/L2/L3
    dic['get_res_idx'] = get_res_idx   # dict mapping chain -> {orig_res_no: index}
    dic['coord_s'] = coord_s           # (1, L, max_atoms, 3)
    dic['is_gly'] = is_gly             # (L,)
    dic['dist6d'] = dist6d
    dic['phi6d'] = phi6d
    dic['omega6d'] = omega6d
    dic['theta6d'] = theta6d
    dic['phi_res'] = phi_res
    dic['psi_res'] = psi_res

    return dic

def build_edge_mask(dic, dist_cut_off=10.0):
    """
    Computes edge masks and relative index using the dictionary data.
    Adds keys:
      - 'noncov_mask': (B, L, L) mask for non-covalent connections.
      - 'cov_mask': (B, L, L) mask for covalent bonds (effective number difference == 1).
      - 'rel_index': (L, L) matrix of effective residue differences (clamped).
    """
    B, L, _, _ = dic['coord_s'].shape
    # Compute pairwise chain mask
    chain_id = dic['chain_id'].unsqueeze(-1) - dic['chain_id'].unsqueeze(-2)
    pairwise_chain_mask = (chain_id == 0)
    # Compute pairwise differences using effective residue numbers
    res_no = dic['res_no']
    rel_residue_index = res_no.unsqueeze(-1) - res_no.unsqueeze(-2)
    rel_residue_index = torch.clamp(rel_residue_index, min=-MAX_REL_INDEX, max=MAX_REL_INDEX)
    rel_residue_index[~pairwise_chain_mask] = MAX_REL_INDEX
    # Covalent bond mask: effective number difference == 1 within same chain
    covalent_bond_mask = (torch.abs(rel_residue_index) == 1) & pairwise_chain_mask
    dic['cov_mask'] = covalent_bond_mask.unsqueeze(0).expand(B, L, L)
    # Non-covalent mask is defined from CA distances computed later in graph building; here we store rel_residue_index
    dic['rel_index'] = rel_residue_index
    return dic

def _build_edge_mask_from_distance(diff, dist_cut_off=10.0, max_neighbors=0):
    """Build an edge mask from pairwise distances.

    If ``max_neighbors`` > 0, each node keeps at most that many nearest
    neighbors within ``dist_cut_off``.
    """
    radius_mask = (diff > 0) & (diff < dist_cut_off)
    if max_neighbors is None or int(max_neighbors) <= 0:
        return radius_mask

    n_res = diff.shape[0]
    k = min(int(max_neighbors) + 1, n_res)
    if k <= 1:
        return torch.zeros_like(radius_mask)

    dvals, nidx = torch.topk(diff, k=k, largest=False, dim=1)
    row_idx = torch.arange(n_res, device=diff.device).unsqueeze(1).expand_as(nidx)
    knn_mask = torch.zeros_like(radius_mask)
    valid = (dvals > 0) & (dvals < dist_cut_off)
    knn_mask[row_idx[valid], nidx[valid]] = True
    return radius_mask & knn_mask


def build_graph(dic, use_all_atom=False, dist_cut_off=10.0, max_neighbors=0):
    """
    Generates a DGLGraph from the precomputed dictionary.
    
    Node features:
      - 'h': Amino acid type index (shape: (L,))
      - 'ulr': ULR mask (shape: (L,))
      - 'str_mask': Structure mask (all-atom mode only, shape: (L,))
      - 'bb_torsion': Backbone torsion angles (phi, psi) for each residue (shape: (L, 2))
      - 'pos': CA coordinates (shape: (L, 3))
      - Additionally, in all-atom mode: 'all_atom_rel_pos' and 'info_aa_atom'
             in heavy mode: 'l1' (optional)
    
    Edge features:
      - 'e_ij': Basic edge features (CA distance and bond type one-hot), (num_edges, 3), 27 if heavy mode
      - 'rel_index': Concatenated relative residue index (clamped) and chain feature, (num_edges, 2)
      - 'pair_torsion': Pair torsion angles (omega, phi, theta) for each edge, (num_edges, 3)
      - In all-atom mode: 'rel_ca_pos': Relative CA position difference (num_edges, 3)
    
    Parameters:
        dic: Precomputed dictionary with keys from create_dictionary_from_model.
        use_all_atom (bool): If True, use full atom coordinates; otherwise use heavy atoms.
    
    Returns:
        g: DGLGraph with the above node and edge features.
    """
    B, L, max_atoms, _ = dic['coord_s'].shape
    if use_all_atom:
        ca_index = 1
        ca_tensor = dic['coord_s'][0, :, ca_index, :3]
        diff = torch.norm(ca_tensor.unsqueeze(1) - ca_tensor.unsqueeze(0), dim=-1)  # (L, L)
        edge_mask = _build_edge_mask_from_distance(
            diff,
            dist_cut_off=dist_cut_off,
            max_neighbors=max_neighbors,
        )
        src, dst = edge_mask.nonzero(as_tuple=True)
        # CA-CA distance and bond type (3-dim) as before.
        eff = dic['res_no']
        bond_types = [1 if abs(eff[i].item() - eff[j].item()) == 1 else 0 for i, j in zip(src, dst)]
        bond_types = torch.tensor(bond_types, dtype=torch.long)
        bond_type_onehot = torch.nn.functional.one_hot(bond_types, num_classes=2).float()
        edge_dists = diff[src, dst].unsqueeze(1)
        basic_edge_feat = torch.cat([edge_dists, bond_type_onehot], dim=1)  # (num_edges, 3)
    else:
        # Heavy mode: use first 5 heavy atoms (e.g., ["CA", "C", "N", "O", "CB"]) for distance features.
        # Extract heavy atom coordinates: shape (L, 5, 3)
        heavy_coords = dic['coord_s'][0, :, :5, :3]
        # For edge mask, use CA coordinates (assuming CA is at index 0)
        ca_tensor = dic['coord_s'][0, :, 0, :3]
        diff = torch.norm(ca_tensor.unsqueeze(1) - ca_tensor.unsqueeze(0), dim=-1)  # (L, L)
        edge_mask = _build_edge_mask_from_distance(
            diff,
            dist_cut_off=dist_cut_off,
            max_neighbors=max_neighbors,
        )
        src, dst = edge_mask.nonzero(as_tuple=True)
        # For each edge (i, j), compute 5x5 distance matrix between heavy atoms of residue i and residue j.
        edge_feat_list = []
        for i, j in zip(src, dst):
            # heavy_coords[i]: (5, 3), heavy_coords[j]: (5, 3)
            # torch.cdist computes pairwise distances: output shape (1, 5, 5)
            dists_ij = torch.cdist(heavy_coords[i].unsqueeze(0), heavy_coords[j].unsqueeze(0))
            dists_ij = dists_ij.view(-1)  # flatten to (25,)
            edge_feat_list.append(dists_ij)
        edge_feats = torch.stack(edge_feat_list, dim=0)  # (num_edges, 25)
        # Compute bond type as before.
        eff = dic['res_no']
        bond_types = [1 if abs(eff[i].item() - eff[j].item()) == 1 else 0 for i, j in zip(src, dst)]
        bond_types = torch.tensor(bond_types, dtype=torch.long)
        bond_type_onehot = torch.nn.functional.one_hot(bond_types, num_classes=2).float()
        # Concatenate 25-dim heavy atom distances with 2-dim bond type => (num_edges, 27)
        basic_edge_feat = torch.cat([edge_feats, bond_type_onehot], dim=1)
    
    g = dgl.graph((src, dst), num_nodes=L)
    

    g.ndata['h'] = dic['aa_type']
    g.ndata['ulr'] = dic['ulr_mask']
    if 'loop_id' in dic:
        g.ndata['loop_id'] = dic['loop_id']
    if use_all_atom:
        g.ndata['str_mask'] = dic['str_mask']
    bb_torsion = torch.cat((dic['phi_res'][0].unsqueeze(-1).long(),
                            dic['psi_res'][0].unsqueeze(-1).long()), dim=1)
    g.ndata['bb_torsion'] = bb_torsion
    g.ndata['pos'] = ca_tensor
    
    pair_torsion = torch.stack((dic['omega6d'][0][src, dst],
                                 dic['phi6d'][0][src, dst],
                                 dic['theta6d'][0][src, dst]), dim=1)
    g.edata['pair_torsion'] = pair_torsion
    
    rel_index = dic['rel_index']
    rel_index_edge = rel_index[src, dst].unsqueeze(-1).long()
    chain_feat = torch.tensor([1 if dic['chain_id'][i] == dic['chain_id'][j] else 0 
                                 for i, j in zip(src, dst)],
                                dtype=torch.int64).unsqueeze(-1)
    g.edata['rel_index'] = torch.cat((rel_index_edge, chain_feat), dim=1)
    
    g.edata['e_ij'] = basic_edge_feat
    
    if use_all_atom:
        all_coord = dic['coord_s'][0, :, :, :3]
        all_atom_rel_pos = all_coord - ca_tensor.unsqueeze(1)
        g.ndata['all_atom_rel_pos'] = all_atom_rel_pos
        info_aa_atom = dic['coord_s'][0, :, :, 3:]
        g.ndata['info_aa_atom'] = info_aa_atom
        rel_ca_pos = ca_tensor[src] - ca_tensor[dst]
        g.edata['rel_ca_pos'] = rel_ca_pos
    else:
        heavy_coords = dic['coord_s'][0, :, 1:5, :3]  # (L, 4, 3)
        l1 = heavy_coords - ca_tensor.unsqueeze(1)           # (L, 4, 3)
        g.ndata['l1'] = l1


    return g



def generate_graphs_from_target(
    target,
    dist_cutoff_center=10.0,
    random_range=0.0,
    max_neighbors=0,
    use_all_atom=False,
    h3_range=(95, 102),
    cdr_ranges=DEFAULT_CDR_RANGES,
    task_scope="full_cdr",
    label_metric="loop_rmsd",
):
    """
    For each model in the Target object, creates a dictionary via create_dictionary_from_model,
    applies build_edge_mask, and then generates a DGLGraph via build_graph.

    Returns:
        tuple: (graphs, rmsds, ranks, decoy_meta)
            decoy_meta: list of dicts with keys {file, seed, sample} per decoy
    """
    graphs = []
    rmsds = []
    ranks = []
    decoy_meta = []
    for i, model in enumerate(target.models):
        dic = create_dictionary_from_model(
            model,
            use_all_atom=use_all_atom,
            h3_range=h3_range,
            cdr_ranges=cdr_ranges,
            task_scope=task_scope,
        )
        dist_cutoff = dist_cutoff_center + random.uniform(-random_range, random_range) if random_range > 0 else dist_cutoff_center
        dic = build_edge_mask(dic, dist_cut_off=dist_cutoff)
        g = build_graph(
            dic,
            use_all_atom=use_all_atom,
            dist_cut_off=dist_cutoff,
            max_neighbors=max_neighbors,
        )
        graphs.append(g)
        rmsds.append(_get_model_metric(model, label_metric, task_scope=task_scope))

        ranking = getattr(model, 'ranking', None)
        if ranking == -1:
            ranking = None
        ranks.append(ranking)

        decoy_meta.append(decoy_identity_from_model(model, i))

    return graphs, rmsds, ranks, decoy_meta
