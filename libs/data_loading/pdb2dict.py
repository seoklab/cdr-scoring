import argparse
import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np
try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None
from Bio.PDB import PDBParser, Structure
from Bio.PDB.Atom import Atom
from Bio.SVDSuperimposer import SVDSuperimposer
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

from evaluation.loop_metrics import compute_loop_metrics_from_structures


class ParsingRule(TypedDict, total=False):
    rank_pattern: str
    final_pattern: str
    rank_offset: int
    final_ranking: int
    csv_path: str


PARSING_RULES: Dict[str, ParsingRule] = {
    "ABB2": {
        "rank_pattern": r"rank(\d+)_unrefined\.pdb",
        "final_pattern": r"final_model\.pdb",
        "rank_offset": 1,
        "final_ranking": 0,
    },
    "igfold4_local_opt": {
        "rank_pattern": r"local_opt_([0-3])\.pdb",
        "final_pattern": "",
        "rank_offset": 1,
        "final_ranking": -1,
    },
    "igfold4_org": {
        "rank_pattern": r"ranked_(\d+)\.pdb",
        "final_pattern": "",
        "rank_offset": 1,
        "final_ranking": -1,
    },
    "igfold4_ag_local_opt": {
        "rank_pattern": r"local_opt_([0-3])\.pdb",
        "final_pattern": "",
        "rank_offset": 1,
        "final_ranking": -1,
    },
    "AF3": {
        "rank_pattern": r".*seed-(\d+)_sample-(\d+)_model\.pdb$",
        "final_pattern": "",
        "rank_offset": 0,
        "final_ranking": -1,
        "csv_path": "af3/ranking_scores.csv",
    },
    "boltz2": {
        "rank_pattern": r".*_model_(\d+)\.pdb$",
        "final_pattern": "",
        "rank_offset": 1,
        "final_ranking": -1,
    },
}


# Dataset-specific path configurations for boltz2
DATASET_CONFIG: Dict[str, Dict[str, str]] = {
    "0_igfold_197": {
        "base_dir": "/home/sujin/DB/h3-loop-modeling/ab_ag/0_igfold_197",
        "crystal_dir": "00_crystal",
        "boltz2_pred_dir": "31_boltz2",
        "boltz2_chothia_dir": "32_boltz2_chothia_w_ag_renum",
        "output_dir": "33_boltz2_pdb2dict",
        "pdb_list_pickle": "0_info/info.pkl",
    },
    "1_tr_abag": {
        "base_dir": "/home/sujin/DB/h3-loop-modeling/ab_ag/1_tr_abag",
        "crystal_dir": "00_crystal",
        "boltz2_pred_dir": "11_boltz2",
        "boltz2_chothia_dir": "12_boltz2_chothia_w_ag_renum",
        "output_dir": "13_boltz2_pdb2dict",
        "pdb_list_pickle": "0_info/info.pkl",
    },
}


HEAVY_ATOMS = {"CA", "CB", "O", "N", "C"}


@dataclass
class Model:
    method: str
    md_structure: Structure
    pdb_path: Path
    full_rmsd: float = -1.0
    h3_rmsd: float = -1.0
    loop_rmsd: float = float("nan")
    loop_lddt: float = float("nan")
    ag_rmsd: float = -1.0
    ag_local_rmsd: float = -1.0
    h3_lddt: float = float("nan")
    DockQ: float = float("nan")
    ranking: int = -1
    ranking_score: float = float("nan")
    # Boltz2 specific metrics
    confidence_score: float = float("nan")
    iptm: float = float("nan")
    h3_plddt: float = float("nan")

    @classmethod
    def from_pdb(
        cls,
        method: str,
        pdb_path: Path,
        score_map: Optional[Dict[Tuple[int, int], float]] = None,
        rank_map: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> "Model":
        filename = pdb_path.name
        ranking = -1
        ranking_score = float("nan")
        method_lower = method.lower()

        if method_lower.startswith("af3"):
            parts = filename.replace('.pdb', '').split('_')
            if len(parts) >= 2:
                try:
                    seed = int(parts[-2])
                    sample = int(parts[-1])
                    if score_map:
                        ranking_score = score_map.get((seed, sample), float("nan"))
                    if rank_map:
                        ranking = rank_map.get((seed, sample), -1)
                except (ValueError, IndexError):
                    # Fallback to old pattern if new pattern fails
                    match = re.search(r"seed-(\d+)_sample-(\d+)", filename)
                    if match:
                        seed = int(match.group(1))
                        sample = int(match.group(2))
                        if score_map:
                            ranking_score = score_map.get((seed, sample), float("nan"))
                        if rank_map:
                            ranking = rank_map.get((seed, sample), -1)
        else:
            rules = (
                PARSING_RULES.get(method)
                or PARSING_RULES.get(method_lower)
                or PARSING_RULES.get(method.upper())
            )
            if rules:
                rank_pat = rules.get("rank_pattern")
                if rank_pat:
                    match_rank = re.search(rank_pat, filename)
                    if match_rank:
                        rank_offset = rules.get("rank_offset", 0)
                        try:
                            ranking = int(match_rank.group(1)) + rank_offset
                        except ValueError:
                            ranking = -1
                if ranking == -1:
                    final_pat = rules.get("final_pattern")
                    if final_pat:
                        match_final = re.search(final_pat, filename)
                        if match_final:
                            ranking = rules.get("final_ranking", 0)

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(method, str(pdb_path))
        return cls(
            method=method,
            md_structure=structure,
            pdb_path=pdb_path,
            ranking=ranking,
            ranking_score=ranking_score,
        )

    @classmethod
    def from_boltz2_pdb(
        cls,
        pdb_path: Path,
        confidence_json_path: Optional[Path] = None,
        plddt_npz_path: Optional[Path] = None,
    ) -> "Model":
        """Create Model from Boltz2 PDB file with optional confidence and plddt data."""
        filename = pdb_path.name
        ranking = -1
        ranking_score = float("nan")
        confidence_score = float("nan")
        iptm = float("nan")
        h3_plddt = float("nan")

        # Parse model number from filename (e.g., 6xlz_H_L_AB_model_0.pdb -> model 0)
        match = re.search(r"_model_(\d+)\.pdb$", filename)
        if match:
            ranking = int(match.group(1)) + 1  # 1-indexed ranking

        # Load confidence score and iptm from JSON
        if confidence_json_path and confidence_json_path.exists():
            try:
                with open(confidence_json_path, 'r') as f:
                    conf_data = json.load(f)
                confidence_score = float(conf_data.get("confidence_score", float("nan")))
                iptm = float(conf_data.get("protein_iptm", float("nan")))
                ranking_score = confidence_score  # Use confidence_score as ranking_score
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        # Load plddt and calculate H3 plddt
        if plddt_npz_path and plddt_npz_path.exists():
            try:
                plddt_data = np.load(plddt_npz_path)
                plddt_values = plddt_data['plddt']
                
                # Parse the model structure to find H3 residue indices
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure("model", str(pdb_path))
                model = list(structure.get_models())[0]
                
                # Count residues to find H3 loop indices in plddt array
                # plddt is ordered by chain (H, L, A, B, ...)
                residue_idx = 0
                h3_indices = []
                
                for chain in model:
                    for residue in chain:
                        if chain.id == "H":
                            res_id = residue.get_id()
                            res_seq = res_id[1]
                            # H3 loop: residues 95-102 (Chothia numbering)
                            if 95 <= res_seq <= 102:
                                h3_indices.append(residue_idx)
                        residue_idx += 1
                
                if h3_indices and len(plddt_values) > max(h3_indices):
                    h3_plddt_values = plddt_values[h3_indices]
                    h3_plddt = float(np.mean(h3_plddt_values))
            except Exception:
                pass

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("boltz2", str(pdb_path))
        return cls(
            method="boltz2",
            md_structure=structure,
            pdb_path=pdb_path,
            ranking=ranking,
            ranking_score=ranking_score,
            confidence_score=confidence_score,
            iptm=iptm,
            h3_plddt=h3_plddt,
        )


def categorize_ag_local_rmsd(rmsd: float) -> str:
    """Categorize ag_local_rmsd into bins."""
    if np.isnan(rmsd):
        return "nan"
    elif rmsd < 1.5:
        return "0-1.5"
    elif rmsd < 3.0:
        return "1.5-3"
    elif rmsd < 4.5:
        return "3-4.5"
    else:
        return ">4.5"


@dataclass
class Target:
    pdb_id: str
    pdb_path: Path
    db_dir: Path
    dataset_config: Optional[Dict[str, str]] = None
    gt_structure: Structure = field(init=False)
    models: List[Model] = field(default_factory=list)

    def __post_init__(self):
        parser = PDBParser(QUIET=True)
        self.gt_structure = parser.get_structure("target", str(self.pdb_path))

    def _match_residue_atoms(self, gt_res, md_res, atom_type: str) -> Tuple[List[Atom], List[Atom]]:
        gt_atoms: List[Atom] = []
        md_atoms: List[Atom] = []
        for gt_atom in gt_res:
            atom_name = gt_atom.get_name()
            if atom_type == "heavy" and atom_name not in HEAVY_ATOMS:
                continue
            if atom_name in md_res.child_dict:
                md_atom = md_res.child_dict[atom_name]
                gt_atoms.append(gt_atom)
                md_atoms.append(md_atom)
        return gt_atoms, md_atoms

    def match_common_atoms(
        self,
        model: Model,
        atom_type: str = "all",
        region: str = "all",
        use_h3_cutoff: bool = False,
    ) -> Tuple[List[Atom], List[Atom], List[Atom], List[Atom]]:
        align_gt_atoms: List[Atom] = []
        align_md_atoms: List[Atom] = []
        target_gt_atoms: List[Atom] = []
        target_md_atoms: List[Atom] = []

        gt_model = list(self.gt_structure.get_models())[0]
        md_model = list(model.md_structure.get_models())[0]

        # Collect H3 CA atoms for antigen filtering (when use_h3_cutoff is True)
        h3_ca_coords: List[np.ndarray] = []
        if use_h3_cutoff and region == "Ag":
            for gt_chain in gt_model:
                if gt_chain.id == "H":
                    for gt_res in gt_chain:
                        res_seq = gt_res.get_id()[1]
                        if 95 <= res_seq <= 102:
                            if 'CA' in gt_res:
                                h3_ca_coords.append(gt_res['CA'].get_coord())

        for gt_chain in gt_model:
            md_chain = next((chain for chain in md_model if chain.id == gt_chain.id), None)
            if md_chain is None:
                continue            

            for gt_res in gt_chain:
                md_res = next((res for res in md_chain if res.get_id() == gt_res.get_id()), None)
                if md_res is None:
                    continue

                align_here = False
                target_here = False

                if region == "Ab-H3":
                    if gt_chain.id not in {"H", "L"}:
                        continue
                    is_h_chain = gt_chain.id == "H"
                    res_seq = gt_res.get_id()[1]
                    is_h3 = is_h_chain and 95 <= res_seq <= 102
                    if not is_h3:
                        align_here = True
                    if is_h3:
                        target_here = True
                elif region == "Ag":
                    if gt_chain.id in {"H", "L"}:
                        align_here = True
                    else:
                        target_here = True
                else:
                    align_here = True
                    target_here = True

                if align_here or target_here:
                    gt_atoms, md_atoms = self._match_residue_atoms(gt_res, md_res, atom_type)
                    if align_here:
                        align_gt_atoms.extend(gt_atoms)
                        align_md_atoms.extend(md_atoms)
                    if target_here:
                        target_gt_atoms.extend(gt_atoms)
                        target_md_atoms.extend(md_atoms)

        # Filter antigen atoms by 10A cutoff from H3 CA atoms
        if use_h3_cutoff and region == "Ag" and h3_ca_coords:
            filtered_gt_atoms = []
            filtered_md_atoms = []
            cutoff = 10.0
            h3_ca_array = np.array(h3_ca_coords)  # Shape: (N_h3_ca, 3)
            
            for gt_atom, md_atom in zip(target_gt_atoms, target_md_atoms):
                # Check if antigen atom is within 10A of any H3 CA atom
                gt_coord = gt_atom.get_coord()
                # Calculate distances to all H3 CA atoms at once
                distances = np.linalg.norm(h3_ca_array - gt_coord, axis=1)
                min_distance = np.min(distances)
                
                if min_distance <= cutoff:
                    filtered_gt_atoms.append(gt_atom)
                    filtered_md_atoms.append(md_atom)
            
            target_gt_atoms = filtered_gt_atoms
            target_md_atoms = filtered_md_atoms

        if region == "all":
            target_gt_atoms = align_gt_atoms[:]
            target_md_atoms = align_md_atoms[:]

        return align_gt_atoms, align_md_atoms, target_gt_atoms, target_md_atoms

    def rmsd(
        self,
        align_gt: List[Atom],
        align_md: List[Atom],
        target_gt: List[Atom],
        target_md: List[Atom],
    ) -> float:
        if not align_gt or not align_md or not target_gt or not target_md:
            return float("nan")
        align_gt_coords = np.array([atom.get_coord() for atom in align_gt])
        align_md_coords = np.array([atom.get_coord() for atom in align_md])
        sup = SVDSuperimposer()
        sup.set(align_gt_coords, align_md_coords)
        sup.run()
        rot, transl = sup.get_rotran()

        target_gt_coords = np.array([atom.get_coord() for atom in target_gt])
        target_md_coords = np.array([atom.get_coord() for atom in target_md])
        target_md_coords_sup = np.dot(target_md_coords, rot) + transl
        diff = target_gt_coords - target_md_coords_sup
        return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))

    @classmethod
    def from_pdb(
        cls,
        pdb_id: str,
        db_dir: Path,
        method: str,
        atom_type: str = "all",
        dataset_config: Optional[Dict[str, str]] = None,
    ) -> "Target":
        method_lower = method.lower()

        # Crystal structure path - depends on method
        if method_lower == "boltz2":
            folder_name = f"{pdb_id}"
            if dataset_config:
                base_dir = Path(dataset_config["base_dir"])
                pdb_path = base_dir / dataset_config["crystal_dir"] / f"{folder_name}.pdb"
            else:
                pdb_path = Path('/home/sujin/DB/h3-loop-modeling/ab_ag/0_igfold_197/00_crystal') / f"{folder_name}.pdb"
        else:
            pdb_path = Path('/home/sujin/DB/h3-loop-modeling/ab_ag/0_igfold_197/04_xtal_chothia_ag_renum') / f"{pdb_id}.pdb"

        target = cls(
            pdb_id=pdb_id,
            pdb_path=pdb_path,
            db_dir=db_dir,
            dataset_config=dataset_config,
        )

        score_map: Optional[Dict[Tuple[int, int], float]] = None
        rank_map: Optional[Dict[Tuple[int, int], int]] = None

        # Model files logic
        if method_lower == "abb2" or method_lower == "igfold4":
            model_folder = db_dir / pdb_id / f"{method_lower}-chothia"
            rank_files = list(model_folder.glob("rank*_unrefined.pdb"))
            final_model = list(model_folder.glob("final_model.pdb"))
            model_files = sorted(rank_files) + sorted(final_model)
        elif method_lower == "igfold4_local_opt":
            model_folder = db_dir / pdb_id / "chothia"
            model_files = sorted(model_folder.glob("local_opt_*.pdb"))
        elif method_lower == "igfold4_org":
            model_folder = db_dir / pdb_id / "chothia"
            model_files = sorted(model_folder.glob("ranked_*.pdb"))
        elif method_lower == "igfold4_ag_local_opt":
            model_folder = db_dir / pdb_id / "addAg"
            model_files = sorted(model_folder.glob("local_opt_*.pdb"))
        elif method_lower.startswith("af3"):
            # AF3 model files from reorganized directory
            model_folder = Path('/home/sujin/DB/h3-loop-modeling/ab_ag/0_igfold_197/14_af3_chothia_ag_renum') / pdb_id
            model_files = sorted(model_folder.glob(f"{pdb_id}_*_*.pdb"))
            
            # AF3 ranking scores from original directory
            ranking_candidates = [
                Path('/home/sujin/DB/h3-loop-modeling/ab_ag/0_igfold_197/10_af3') / pdb_id / "ranking.csv",
                # db_dir / pdb_id_lower / "ranking_scores.csv",
                # db_dir / pdb_id_lower / f"{pdb_id_lower}_ranking_scores.csv",
            ]
            for csv_path in ranking_candidates:
                if csv_path.exists():
                    if pd is None:
                        raise ModuleNotFoundError("pandas is required to read AF3 ranking CSV files")
                    df = pd.read_csv(csv_path)
                    required_cols = {"seed", "sample", "ranking_score"}
                    if required_cols.issubset(df.columns):
                        df = df.sort_values(by="ranking_score", ascending=False).reset_index(drop=True)
                        score_map = {}
                        ordered_keys: List[Tuple[int, int]] = []
                        for row in df.itertuples(index=False):
                            key = (int(getattr(row, "seed")), int(getattr(row, "sample")))
                            score_map[key] = float(getattr(row, "ranking_score"))
                            ordered_keys.append(key)
                        rank_map = {key: idx + 1 for idx, key in enumerate(ordered_keys)}
                        break
            if score_map is None:
                score_map = {}
            if rank_map is None:
                rank_map = {}
        elif method_lower == "boltz2":
            # Boltz2 model files
            # pdb_id is like "6xlz", folder name is like "6xlz_H_L_AB"
            folder_name = f"{pdb_id}"
            if dataset_config:
                base_dir = Path(dataset_config["base_dir"])
                model_folder = base_dir / dataset_config["boltz2_chothia_dir"] / folder_name
                boltz2_pred_dir = base_dir / dataset_config["boltz2_pred_dir"] / f"boltz_results_{folder_name}" / "predictions" / folder_name
            else:
                model_folder = Path('/home/sujin/DB/h3-loop-modeling/ab_ag/0_igfold_197/32_boltz2_chothia_w_ag_renum') / folder_name
                boltz2_pred_dir = Path('/home/sujin/DB/h3-loop-modeling/ab_ag/0_igfold_197/31_boltz2') / f"boltz_results_{folder_name}" / "predictions" / folder_name
            model_files = sorted(model_folder.glob(f"{folder_name}_model_*.pdb"))
        else:
            model_folder = db_dir / pdb_id
            model_files = sorted(model_folder.glob("*.pdb"))

        for model_file in model_files:
            if method_lower.startswith("af3"):
                model = Model.from_pdb(
                    method=method,
                    pdb_path=model_file,
                    score_map=score_map,
                    rank_map=rank_map,
                )
            elif method_lower == "boltz2":
                # Extract model number from filename
                match = re.search(r"_model_(\d+)\.pdb$", model_file.name)
                model_num = int(match.group(1)) if match else 0
                
                confidence_json = boltz2_pred_dir / f"confidence_{folder_name}_model_{model_num}.json"
                plddt_npz = boltz2_pred_dir / f"plddt_{folder_name}_model_{model_num}.npz"
                
                model = Model.from_boltz2_pdb(
                    pdb_path=model_file,
                    confidence_json_path=confidence_json,
                    plddt_npz_path=plddt_npz,
                )
            else:
                model = Model.from_pdb(method=method, pdb_path=model_file)

            align_gt, align_md, target_gt, target_md = target.match_common_atoms(
                model, atom_type=atom_type, region="all"
            )
            model.full_rmsd = target.rmsd(align_gt, align_md, target_gt, target_md)

            ab_align_gt, ab_align_md, h3_gt, h3_md = target.match_common_atoms(
                model, atom_type=atom_type, region="Ab-H3"
            )
            model.h3_rmsd = target.rmsd(ab_align_gt, ab_align_md, h3_gt, h3_md)
            try:
                loop_metrics = compute_loop_metrics_from_structures(
                    target.gt_structure,
                    model.md_structure,
                    rmsd_atom_type="backbone",
                )
                model.loop_rmsd = loop_metrics.loop_rmsd
                model.loop_lddt = loop_metrics.loop_lddt
            except Exception:
                model.loop_rmsd = float("nan")
                model.loop_lddt = float("nan")

            ab_align_gt, ab_align_md, ag_gt, ag_md = target.match_common_atoms(
                model, atom_type=atom_type, region="Ag"
            )
            model.ag_rmsd = target.rmsd(ab_align_gt, ab_align_md, ag_gt, ag_md)

            # Calculate ag_local_rmsd (antigen atoms within 10A of H3 CA atoms)
            ab_align_gt_local, ab_align_md_local, ag_local_gt, ag_local_md = target.match_common_atoms(
                model, atom_type=atom_type, region="Ag", use_h3_cutoff=True
            )
            model.ag_local_rmsd = target.rmsd(ab_align_gt_local, ab_align_md_local, ag_local_gt, ag_local_md)

            target.models.append(model)

        return target

    def split_targets(self, split_size: int = 10) -> List["Target"]:
        targets = []
        for i in range(0, len(self.models), split_size):
            target_subset = Target(
                pdb_id=self.pdb_id,
                pdb_path=self.pdb_path,
                db_dir=self.db_dir,
            )
            target_subset.gt_structure = self.gt_structure
            target_subset.models = self.models[i : i + split_size]
            targets.append(target_subset)
        return targets


def main():
    parser = argparse.ArgumentParser(description="Calculate RMSD for pdb targets.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed RMSD per model",
    )
    parser.add_argument(
        "--atom_type",
        choices=["all", "heavy"],
        default="all",
        help="Calculate RMSD using all atoms or heavy atoms only (CA, CB, O, N, C)",
    )
    parser.add_argument(
        "--method",
        default="ABB2",
        help="Method to use (default: ABB2)",
    )
    parser.add_argument(
        "--pdb_list_pickle",
        type=Path,
        default=None,
        help="Pickle file containing list of pdb ids (auto-detected based on method if not provided)",
    )
    parser.add_argument(
        "--db_dir",
        type=Path,
        default=None,
        help="Root directory containing method-specific decoys",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Directory where target pickles will be written",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Run in test mode, only process the first target",
    )
    parser.add_argument(
        "--save_dataframe",
        action="store_true",
        help="Save results to a CSV dataframe",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="0_igfold_197",
        choices=list(DATASET_CONFIG.keys()),
        help="Dataset to use (default: 0_igfold_197)",
    )
    args = parser.parse_args()

    method_lower = args.method.lower()

    # Get dataset configuration
    dataset_config = DATASET_CONFIG.get(args.dataset) if method_lower == "boltz2" else None

    # Set default pdb_list_pickle based on method and dataset
    if args.pdb_list_pickle is None:
        if method_lower == "boltz2" and dataset_config:
            base_dir = Path(dataset_config["base_dir"])
            args.pdb_list_pickle = base_dir / dataset_config["pdb_list_pickle"]
        else:
            args.pdb_list_pickle = Path("/home/sujin/DB/h3-loop-modeling/ab_ag/igfold_ordered/igfold-list-197.pkl")

    # Set default output_root based on method and dataset
    if args.output_root is None:
        if method_lower == "boltz2" and dataset_config:
            base_dir = Path(dataset_config["base_dir"])
            args.output_root = base_dir / dataset_config["output_dir"]
        else:
            args.output_root = Path("/home/sujin/DB/h3-loop-modeling/ab_ag/igfold_ordered/decoy")

    with open(args.pdb_list_pickle, "rb") as f:
        pdb_list = pickle.load(f)
        if isinstance(pdb_list, dict):
            if 'list' in pdb_list:
                pdb_list = pdb_list['list']
            elif 'pdb_ids' in pdb_list:
                pdb_list = pdb_list['pdb_ids']
            else:
                # Try to get keys as pdb_ids
                pdb_list = list(pdb_list.keys())

    # Collect results for dataframe
    results_data = []
    
    # Collect ag_local_rmsd categories for Boltz2
    ag_local_rmsd_categories: Dict[str, Dict[str, List[int]]] = {}

    if args.db_dir is not None:
        db_dir = args.db_dir
    else:
        if method_lower.startswith("igfold"):
            db_dir = Path("/home/sujin/DB/h3-loop-modeling/ab_ag/igfold_ordered/igfold_4_local_opt")
        elif method_lower.startswith("af3"):
            db_dir = Path("/home/sujin/DB/h3-loop-modeling/ab_ag/igfold_ordered/decoy/af3")
        elif method_lower == "boltz2" and dataset_config:
            base_dir = Path(dataset_config["base_dir"])
            db_dir = base_dir / dataset_config["boltz2_chothia_dir"]
        else:
            db_dir = Path("/home/sujin/DB/h3-loop-modeling/ab_ag/igfold_ordered/decoy")

    for pdb_id in tqdm(pdb_list, desc="Write dictionary for pdb targets"):
        target = Target.from_pdb(
            pdb_id=pdb_id,
            db_dir=db_dir,
            method=args.method,
            atom_type=args.atom_type,
            dataset_config=dataset_config,
        )

        target_pickle_path = args.output_root / f"{pdb_id}.pkl"
        target_pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_pickle_path, "wb") as f:
            pickle.dump(target, f)

        # Collect ag_local_rmsd categories for Boltz2
        if method_lower == "boltz2":
            ag_local_rmsd_categories[pdb_id] = {
                "0-1.5": [],
                "1.5-3": [],
                "3-4.5": [],
                ">4.5": [],
                "nan": [],
            }
            for model in target.models:
                category = categorize_ag_local_rmsd(model.ag_local_rmsd)
                model_num = model.ranking - 1  # Convert back to 0-indexed model number
                ag_local_rmsd_categories[pdb_id][category].append(model_num)

        # Collect model results for dataframe
        for model in target.models:
            row_data = {
                'pdb_id': pdb_id,
                'method': args.method,
                'model_file': model.pdb_path.name,
                'ranking': model.ranking,
                'ranking_score': model.ranking_score,
                'full_rmsd': model.full_rmsd,
                'h3_rmsd': model.h3_rmsd,
                'loop_rmsd': model.loop_rmsd,
                'loop_lddt': model.loop_lddt,
                'ag_rmsd': model.ag_rmsd,
                'ag_local_rmsd': model.ag_local_rmsd,
            }
            # Add Boltz2-specific metrics
            if method_lower == "boltz2":
                row_data['confidence_score'] = model.confidence_score
                row_data['iptm'] = model.iptm
                row_data['h3_plddt'] = model.h3_plddt
            results_data.append(row_data)

        if target.models and np.isnan(target.models[0].full_rmsd):
            print(pdb_id, "nan in rmsd")

        if args.verbose:
            print(f"Target: {pdb_id}, Method: {args.method}")
            print(f"Total models: {len(target.models)}")
            for model in sorted(target.models, key=lambda m: m.ranking):
                base_info = (
                    f"Model (ranking: {model.ranking}) "
                    f"Full RMSD: {model.full_rmsd:.4f}, "
                    f"H3 RMSD: {model.h3_rmsd:.4f}, "
                    f"Loop RMSD: {model.loop_rmsd:.4f}, "
                    f"Loop lDDT: {model.loop_lddt:.4f}, "
                    f"Ag RMSD: {model.ag_rmsd:.4f}, "
                    f"Ag Local RMSD: {model.ag_local_rmsd:.4f}"
                )
                if method_lower == "boltz2":
                    base_info += (
                        f", Conf: {model.confidence_score:.4f}, "
                        f"iPTM: {model.iptm:.4f}, "
                        f"H3 pLDDT: {model.h3_plddt:.4f}"
                    )
                print(base_info)
        if args.test_mode:
            print("=== TEST MODE ===")
            print("[test_mode] Stopping after first valid target.")
            break

    # Save ag_local_rmsd categories for Boltz2
    if method_lower == "boltz2" and ag_local_rmsd_categories:
        categories_json_path = args.output_root / "ag_local_rmsd_categories.json"
        with open(categories_json_path, 'w') as f:
            json.dump(ag_local_rmsd_categories, f, indent=2)
        print(f"\nag_local_rmsd categories saved to: {categories_json_path}")
        
        # Print summary statistics
        total_counts = {"0-1.5": 0, "1.5-3": 0, "3-4.5": 0, ">4.5": 0, "nan": 0}
        for pdb_id, categories in ag_local_rmsd_categories.items():
            for cat, models in categories.items():
                total_counts[cat] += len(models)
        print("Category distribution:")
        for cat, count in total_counts.items():
            print(f"  {cat}: {count}")

    # Generate and save dataframe
    if args.save_dataframe and results_data:
        if pd is None:
            raise ModuleNotFoundError("pandas is required to save result dataframes")
        df = pd.DataFrame(results_data)
        df_csv_path = args.output_root / f"{args.method}_results.csv"
        
        df.to_csv(df_csv_path, index=False)
        
        print(f"\nResults saved to: {df_csv_path}")
        print(f"Total models processed: {len(df)}")
        print(f"Total targets processed: {df['pdb_id'].nunique()}")


if __name__ == "__main__":
    main()
