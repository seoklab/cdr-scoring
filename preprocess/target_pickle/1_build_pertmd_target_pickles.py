#!/usr/bin/env python
"""Build Target pickles from pertMD or Boltz2 s10n10 decoy directories.

The output pickle uses data_loading.pdb2dict.Target/Model so it can be consumed
by the existing target_model_pickle dataset source and precompute scripts.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Structure import Structure as PDBStructure

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_ROOT = REPO_ROOT / "libs"
if str(LIBS_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBS_ROOT))

from data_loading.pdb2dict import Model, Target  # noqa: E402
from evaluation.loop_metrics import compute_loop_metrics_from_structures  # noqa: E402


PDB_PARSER = PDBParser(QUIET=True)
MMCIF_PARSER = MMCIFParser(QUIET=True)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    low = value.lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    if low in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_flat_yaml(path: Optional[Path]) -> Dict[str, Any]:
    """Load a flat key: value YAML file without requiring PyYAML."""
    if path is None:
        return {}
    out: Dict[str, Any] = {}
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if line.startswith((" ", "\t")):
                raise ValueError(f"Only flat YAML is supported: {path}: {raw_line.rstrip()}")
            if ":" not in line:
                raise ValueError(f"Invalid config line: {path}: {raw_line.rstrip()}")
            key, value = line.split(":", 1)
            out[key.strip().replace("-", "_")] = _parse_scalar(value)
    return out


def parse_range_spec(raw: str) -> List[int]:
    values: List[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(token))
    return values


def read_target_list(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    if path.suffix == ".pkl":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        values = payload.get("list") if isinstance(payload, dict) else payload
        if isinstance(values, dict):
            values = list(values.keys())
        return [str(item) for item in values]
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def read_exclude_list(path: Optional[Path]) -> set[str]:
    if path is None:
        return set()
    with open(path, encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def resolve_native_path(native_dir: Path, target_id: str, template: str) -> Path:
    path = native_dir / template.format(target=target_id)
    if path.is_file():
        return path
    fallback = native_dir / f"{target_id}.pdb"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Native PDB not found for {target_id}: {path}")


def parse_structure(path: Path, struct_id: str):
    if path.suffix.lower() == ".cif":
        return MMCIF_PARSER.get_structure(struct_id, str(path))
    return PDB_PARSER.get_structure(struct_id, str(path))


def structure_from_model(bio_model, struct_id: str) -> PDBStructure:
    model_copy = bio_model
    parent = model_copy.get_parent()
    if parent is not None:
        parent.detach_child(model_copy.id)
    model_copy.id = 0
    structure = PDBStructure(struct_id)
    structure.add(model_copy)
    return structure


def overlay_model_onto_crystal(crystal_struct: PDBStructure, decoy_model) -> PDBStructure:
    """Overlay any matching chain/residue/atom coordinates onto a crystal copy."""
    merged = copy.deepcopy(crystal_struct)
    merged_model = next(merged.get_models())
    target_chains = {chain.id: chain for chain in merged_model}

    for src_chain in decoy_model:
        target_chain = target_chains.get(src_chain.id)
        if target_chain is None:
            continue
        target_residues = {residue.id: residue for residue in target_chain}
        for src_residue in src_chain:
            if src_residue.id[0] != " ":
                continue
            target_residue = target_residues.get(src_residue.id)
            if target_residue is None:
                continue
            for src_atom in src_residue:
                atom_name = src_atom.get_name()
                if atom_name in target_residue.child_dict:
                    target_residue.child_dict[atom_name].set_coord(src_atom.get_coord())
    return merged


def parse_original_residue_id(raw_resid: str) -> tuple:
    """Parse original PDB residue ids such as 100 or 100A into Bio.PDB ids."""
    raw = str(raw_resid).strip()
    match = re.fullmatch(r"(-?\d+)([A-Za-z]?)", raw)
    if match is None:
        raise ValueError(f"Unsupported residue id in pertMD map: {raw_resid!r}")
    resseq = int(match.group(1))
    icode = match.group(2) or " "
    return (" ", resseq, icode)


def load_pertmd_residue_remap(map_path: Path) -> Dict[Tuple[str, int], tuple]:
    with open(map_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    remap: Dict[Tuple[str, int], tuple] = {}
    for entry in payload.get("map", []):
        if len(entry) != 3:
            raise ValueError(f"Invalid pertMD map entry in {map_path}: {entry!r}")
        chain_id, original_resid, galaxy_resid = entry
        remap[(str(chain_id), int(galaxy_resid))] = parse_original_residue_id(str(original_resid))
    return remap


def remap_model_residue_numbers(bio_model, remap: Dict[Tuple[str, int], tuple]) -> int:
    """Rewrite GalaxyRefine continuous residue ids back to native chain-local ids."""
    remapped = 0
    for chain in bio_model:
        residue_updates = []
        final_ids = []
        for residue in list(chain):
            if residue.id[0] != " ":
                final_ids.append(residue.id)
                continue
            new_id = remap.get((chain.id, int(residue.id[1])))
            if new_id is None:
                final_ids.append(residue.id)
                continue
            residue_updates.append((residue, new_id))
            final_ids.append(new_id)

        duplicate_final_ids = {res_id for res_id in final_ids if final_ids.count(res_id) > 1}
        if duplicate_final_ids:
            examples = ", ".join(str(res_id) for res_id in sorted(duplicate_final_ids)[:5])
            raise ValueError(f"Residue remap would create duplicate ids in chain {chain.id}: {examples}")

        # Avoid transient Bio.PDB child_dict collisions for shifts such as
        # 120->1, 121->2, ... by moving all remapped residues out of the way
        # before assigning their final native residue ids.
        for idx, (residue, _new_id) in enumerate(residue_updates, start=1):
            residue.id = (" ", -10_000_000 - idx, " ")
        for residue, new_id in residue_updates:
            old_id = residue.id
            if old_id != new_id:
                residue.id = new_id
                remapped += 1
    return remapped


def compute_rmsd_metrics(target: Target, model: Model, atom_type: str) -> None:
    try:
        align_gt, align_md, target_gt, target_md = target.match_common_atoms(
            model, atom_type=atom_type, region="all"
        )
        model.full_rmsd = target.rmsd(align_gt, align_md, target_gt, target_md)
    except Exception:
        model.full_rmsd = float("nan")

    try:
        align_gt, align_md, target_gt, target_md = target.match_common_atoms(
            model, atom_type=atom_type, region="Ab-H3"
        )
        model.h3_rmsd = target.rmsd(align_gt, align_md, target_gt, target_md)
    except Exception:
        model.h3_rmsd = float("nan")

    try:
        align_gt, align_md, target_gt, target_md = target.match_common_atoms(
            model, atom_type=atom_type, region="Ag"
        )
        model.ag_rmsd = target.rmsd(align_gt, align_md, target_gt, target_md)
    except Exception:
        model.ag_rmsd = float("nan")

    try:
        align_gt, align_md, target_gt, target_md = target.match_common_atoms(
            model, atom_type=atom_type, region="Ag", use_h3_cutoff=True
        )
        model.ag_local_rmsd = target.rmsd(align_gt, align_md, target_gt, target_md)
    except Exception:
        model.ag_local_rmsd = float("nan")

    try:
        loop_metrics = compute_loop_metrics_from_structures(
            target.gt_structure,
            model.md_structure,
            rmsd_atom_type="backbone" if atom_type == "heavy" else atom_type,
        )
        model.loop_rmsd = loop_metrics.loop_rmsd
        model.loop_lddt = loop_metrics.loop_lddt
    except Exception:
        model.loop_rmsd = float("nan")
        model.loop_lddt = float("nan")


def load_confidence(path: Path) -> Tuple[float, float]:
    if not path.is_file():
        return float("nan"), float("nan")
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        confidence = float(payload.get("confidence_score", float("nan")))
        iptm = float(payload.get("protein_iptm", payload.get("iptm", float("nan"))))
        return confidence, iptm
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return float("nan"), float("nan")


def compute_h3_plddt(structure, plddt_path: Path) -> float:
    if not plddt_path.is_file():
        return float("nan")
    try:
        values = np.load(plddt_path)["plddt"]
        model = next(structure.get_models())
        residue_idx = 0
        h3_indices: List[int] = []
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if chain.id == "H" and 95 <= int(residue.id[1]) <= 102:
                    h3_indices.append(residue_idx)
                residue_idx += 1
        if h3_indices and len(values) > max(h3_indices):
            return float(np.mean(values[h3_indices]))
    except Exception:
        pass
    return float("nan")


def build_pertmd_target(args: argparse.Namespace, target_id: str) -> Target:
    native_path = resolve_native_path(args.native_dir, target_id, args.native_template)
    model_pdb = args.input_dir / target_id / args.model_relative_path
    if not model_pdb.is_file():
        raise FileNotFoundError(f"Missing pertMD model PDB: {model_pdb}")

    target = Target(pdb_id=target_id, pdb_path=native_path, db_dir=args.input_dir / target_id)
    multi = PDB_PARSER.get_structure("pertmd", str(model_pdb))
    bio_models = list(multi.get_models())
    if args.max_models is not None:
        bio_models = bio_models[: args.max_models]

    residue_remap: Optional[Dict[Tuple[str, int], tuple]] = None
    residue_map_path: Optional[Path] = None
    if args.remap_pertmd_residues:
        residue_map_dir = args.residue_map_dir or (args.native_dir.parent / "0_info")
        residue_map_path = residue_map_dir / f"{target_id}.pertmd_cdr_map.json"
        if not residue_map_path.is_file():
            raise FileNotFoundError(f"Missing pertMD residue map: {residue_map_path}")
        residue_remap = load_pertmd_residue_remap(residue_map_path)

    decoy_mode = args.decoy_mode
    if decoy_mode == "auto":
        first_model = bio_models[0] if bio_models else None
        first_resseq = None
        if first_model is not None:
            for chain in first_model:
                for residue in chain:
                    if residue.id[0] == " ":
                        first_resseq = int(residue.id[1])
                        break
                if first_resseq is not None:
                    break
        decoy_mode = "full" if first_resseq == 1 else "overlay"

    for idx, bio_model in enumerate(bio_models):
        remapped_residues = 0
        if residue_remap is not None:
            remapped_residues = remap_model_residue_numbers(bio_model, residue_remap)

        if decoy_mode == "overlay":
            structure = overlay_model_onto_crystal(target.gt_structure, bio_model)
        elif decoy_mode == "full":
            structure = structure_from_model(bio_model, f"{target_id}_model_{idx + 1}")
        else:
            raise ValueError(f"Unknown decoy_mode: {args.decoy_mode}")

        model = Model(
            method=args.method,
            md_structure=structure,
            pdb_path=model_pdb,
            ranking=idx + 1,
        )
        setattr(model, "model_idx", idx + 1)
        setattr(model, "seed", None)
        if residue_map_path is not None:
            setattr(model, "residue_map_path", str(residue_map_path))
            setattr(model, "remapped_residue_count", remapped_residues)
        if args.compute_metrics:
            compute_rmsd_metrics(target, model, args.atom_type)
        target.models.append(model)

    return target


def boltz_prediction_dir(input_dir: Path, target_id: str, seed: int) -> Path:
    return (
        input_dir
        / target_id
        / f"seed_{seed}"
        / f"boltz_results_{target_id}"
        / "predictions"
        / target_id
    )


def boltz_chothia_pdb(chothia_dir: Path, target_id: str, seed: int, sample: int) -> Path:
    return chothia_dir / target_id / f"{target_id}_seed_{seed}_sample_{sample}.pdb"


def build_boltz2_s10n10_target(args: argparse.Namespace, target_id: str) -> Target:
    native_path = resolve_native_path(args.native_dir, target_id, args.native_template)
    target = Target(pdb_id=target_id, pdb_path=native_path, db_dir=args.input_dir / target_id)
    seeds = parse_range_spec(args.seeds)
    samples = parse_range_spec(args.samples)
    chothia_dir = getattr(args, "chothia_dir", None)

    model_entries = []
    for seed in seeds:
        pred_dir = boltz_prediction_dir(args.input_dir, target_id, seed)
        for sample in samples:
            # Confidence / plDDT always come from the raw Boltz prediction dir.
            confidence_path = pred_dir / f"confidence_{target_id}_model_{sample}.json"
            plddt_path = pred_dir / f"plddt_{target_id}_model_{sample}.npz"
            # Coordinates come from the Chothia-renumbered PDB when --chothia-dir is
            # given, otherwise from the raw (sequentially numbered) CIF.
            if chothia_dir is not None:
                struct_path = boltz_chothia_pdb(chothia_dir, target_id, seed, sample)
            else:
                struct_path = pred_dir / f"{target_id}_model_{sample}.cif"
            if not struct_path.is_file():
                if args.strict:
                    raise FileNotFoundError(f"Missing Boltz2 decoy structure: {struct_path}")
                continue
            confidence, iptm = load_confidence(confidence_path)
            model_entries.append((seed, sample, struct_path, plddt_path, confidence, iptm))

    for position, (seed, sample, struct_path, plddt_path, confidence, iptm) in enumerate(model_entries, start=1):
        structure = parse_structure(struct_path, f"{target_id}_seed_{seed}_sample_{sample}")
        model = Model(
            method=args.method,
            md_structure=structure,
            pdb_path=struct_path,
            ranking=position,
            ranking_score=confidence,
            confidence_score=confidence,
            iptm=iptm,
            h3_plddt=compute_h3_plddt(structure, plddt_path),
        )
        setattr(model, "seed", seed)
        setattr(model, "model_idx", sample)
        if args.compute_metrics:
            compute_rmsd_metrics(target, model, args.atom_type)
        target.models.append(model)

    finite_scores = [
        (idx, model.ranking_score)
        for idx, model in enumerate(target.models)
        if math.isfinite(float(model.ranking_score))
    ]
    finite_scores.sort(key=lambda item: (-float(item[1]), target.models[item[0]].seed, target.models[item[0]].model_idx))
    for rank, (idx, _score) in enumerate(finite_scores, start=1):
        target.models[idx].ranking = rank

    return target


def discover_targets(args: argparse.Namespace) -> List[str]:
    target_ids = read_target_list(args.target_list)
    excludes = read_exclude_list(args.exclude_list)
    if target_ids is None:
        target_ids = []
        for path in sorted(args.input_dir.iterdir()):
            if not path.is_dir():
                continue
            if args.source_type == "pertmd":
                if (path / args.model_relative_path).is_file():
                    target_ids.append(path.name)
            else:
                if any(path.glob("seed_*/boltz_results_*/predictions/*")):
                    target_ids.append(path.name)
    target_ids = [target_id for target_id in target_ids if target_id not in excludes]
    if args.limit is not None:
        target_ids = target_ids[: args.limit]
    return target_ids


def process_one(job: Dict[str, Any]) -> Dict[str, Any]:
    args = argparse.Namespace(**job["args"])
    target_id = job["target_id"]
    out_path = args.output_dir / f"{target_id}.pkl"
    summary = {
        "target_id": target_id,
        "source_type": args.source_type,
        "method": args.method,
        "status": "ok",
        "n_models": 0,
        "expected_models": args.expected_models or "",
        "input_path": "",
        "native_path": "",
        "output_pickle": str(out_path),
        "error": "",
    }
    try:
        if out_path.exists() and not args.overwrite:
            summary["status"] = "skipped"
            return summary

        native_path = resolve_native_path(args.native_dir, target_id, args.native_template)
        summary["native_path"] = str(native_path)
        if args.source_type == "pertmd":
            summary["input_path"] = str(args.input_dir / target_id / args.model_relative_path)
            target = build_pertmd_target(args, target_id)
        elif args.source_type == "boltz2_s10n10":
            summary["input_path"] = str(args.input_dir / target_id)
            target = build_boltz2_s10n10_target(args, target_id)
        else:
            raise ValueError(f"Unknown source_type: {args.source_type}")

        summary["n_models"] = len(target.models)
        if args.expected_models is not None and len(target.models) != args.expected_models:
            message = f"expected {args.expected_models} models, got {len(target.models)}"
            if args.strict:
                raise ValueError(message)
            summary["status"] = "warn"
            summary["error"] = message

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as handle:
            pickle.dump(target, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "error"
        summary["error"] = f"{type(exc).__name__}: {exc}"[:500]
    return summary


def write_summary(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "target_id",
        "source_type",
        "method",
        "status",
        "n_models",
        "expected_models",
        "input_path",
        "native_path",
        "output_pickle",
        "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["target_id"]):
            writer.writerow({col: row.get(col, "") for col in columns})


def build_parser(defaults: Optional[Dict[str, Any]] = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-yaml", type=Path, default=None, help="Optional flat YAML config.")
    parser.add_argument("--source-type", choices=("pertmd", "boltz2_s10n10"), default=defaults.get("source_type"))
    parser.add_argument("--input-dir", type=Path, default=defaults.get("input_dir"))
    parser.add_argument(
        "--chothia-dir",
        type=Path,
        default=defaults.get("chothia_dir"),
        help="For boltz2_s10n10: read decoy coordinates from Chothia-renumbered PDBs "
        "({target}/{target}_seed_{seed}_sample_{sample}.pdb) instead of the raw CIFs. "
        "Confidence/plDDT are still read from --input-dir.",
    )
    parser.add_argument("--output-dir", type=Path, default=defaults.get("output_dir"))
    parser.add_argument("--native-dir", type=Path, default=defaults.get("native_dir"))
    parser.add_argument("--native-template", default=defaults.get("native_template", "{target}.pdb"))
    parser.add_argument("--target-list", type=Path, default=defaults.get("target_list"))
    parser.add_argument("--exclude-list", type=Path, default=defaults.get("exclude_list"))
    parser.add_argument("--model-relative-path", type=Path, default=defaults.get("model_relative_path", "input/model/model.pdb"))
    parser.add_argument("--decoy-mode", choices=("overlay", "full", "auto"), default=defaults.get("decoy_mode", "auto"))
    parser.add_argument("--residue-map-dir", type=Path, default=defaults.get("residue_map_dir"))
    parser.add_argument("--remap-pertmd-residues", action="store_true", default=bool(defaults.get("remap_pertmd_residues", False)))
    parser.add_argument("--no-remap-pertmd-residues", dest="remap_pertmd_residues", action="store_false")
    parser.add_argument("--method", default=defaults.get("method"))
    parser.add_argument("--expected-models", type=int, default=defaults.get("expected_models"))
    parser.add_argument("--max-models", type=int, default=defaults.get("max_models"))
    parser.add_argument("--workers", type=int, default=int(defaults.get("workers", 4)))
    parser.add_argument("--atom-type", choices=("all", "heavy"), default=defaults.get("atom_type", "all"))
    parser.add_argument("--compute-metrics", action="store_true", default=bool(defaults.get("compute_metrics", False)))
    parser.add_argument("--no-compute-metrics", dest="compute_metrics", action="store_false")
    parser.add_argument("--seeds", default=str(defaults.get("seeds", "42-51")))
    parser.add_argument("--samples", default=str(defaults.get("samples", "0-9")))
    parser.add_argument("--limit", type=int, default=defaults.get("limit"))
    parser.add_argument("--summary-name", default=defaults.get("summary_name", "build_summary.csv"))
    parser.add_argument("--strict", action="store_true", default=bool(defaults.get("strict", False)))
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("--overwrite", action="store_true", default=bool(defaults.get("overwrite", False)))
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.add_argument("--write-summary", action="store_true", default=bool(defaults.get("write_summary", True)))
    parser.add_argument("--no-summary", dest="write_summary", action="store_false")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config-yaml", type=Path, default=None)
    config_args, _unknown = config_parser.parse_known_args(argv)
    defaults = load_flat_yaml(config_args.config_yaml)
    parser = build_parser(defaults)
    args = parser.parse_args(argv)

    for path_field in (
        "input_dir",
        "chothia_dir",
        "output_dir",
        "native_dir",
        "target_list",
        "exclude_list",
        "model_relative_path",
    ):
        value = getattr(args, path_field)
        if value is not None and not isinstance(value, Path):
            setattr(args, path_field, Path(value))

    required = ("source_type", "input_dir", "output_dir", "native_dir")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))
    if args.method is None:
        args.method = "pertmd" if args.source_type == "pertmd" else "boltz2"
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    target_ids = discover_targets(args)
    if not target_ids:
        print("[error] no targets to process", file=sys.stderr)
        return 1

    print(
        f"[info] source_type={args.source_type} targets={len(target_ids)} "
        f"input={args.input_dir} output={args.output_dir} workers={args.workers}",
        flush=True,
    )

    serializable_args = vars(args).copy()
    for key, value in list(serializable_args.items()):
        if isinstance(value, Path):
            serializable_args[key] = value
    jobs = [{"target_id": target_id, "args": serializable_args} for target_id in target_ids]

    rows: List[Dict[str, Any]] = []
    if args.workers <= 1:
        for job in jobs:
            row = process_one(job)
            rows.append(row)
            print(f"[{row['status']}] {row['target_id']} n_models={row['n_models']} {row['error']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_one, job) for job in jobs]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(f"[{row['status']}] {row['target_id']} n_models={row['n_models']} {row['error']}", flush=True)

    if args.write_summary:
        write_summary(rows, args.output_dir / args.summary_name)

    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"[done] elapsed_sec={time.time() - t0:.1f} counts={counts}", flush=True)
    return 2 if counts.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
