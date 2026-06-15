import csv
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import dgl
import torch
from Bio.PDB import MMCIFParser, PDBParser
from torch.utils.data import DataLoader, Dataset

from data_loading.constants import REF_CHAIN
from data_loading.graph_generation_from_target import (
    build_edge_mask,
    build_graph,
    create_dictionary_from_model,
)


CDRRange = Tuple[str, int, int]


DEFAULT_CDR_RANGES = (
    "H:26-32,52-56,95-102;"
    "L:24-34,50-56,89-97;"
)


@dataclass(frozen=True)
class StructureSample:
    path: Path
    sample_id: str
    target_id: str = ""
    native_path: Optional[Path] = None
    method: str = "structure"
    seed: Optional[int] = None
    sample: Optional[int] = None
    ranking: Optional[int] = None
    ranking_score: Optional[float] = None
    label: Optional[float] = None


def parse_cdr_ranges(spec: str) -> List[CDRRange]:
    """Parse ranges like ``H:26-32,52-56,95-102;L:24-34,50-56,89-97``."""
    ranges: List[CDRRange] = []
    if not spec:
        return ranges
    for chain_block in spec.split(";"):
        chain_block = chain_block.strip()
        if not chain_block:
            continue
        chain_id, raw_ranges = chain_block.split(":", 1)
        chain_id = chain_id.strip()
        for raw_range in raw_ranges.split(","):
            raw_range = raw_range.strip()
            if not raw_range:
                continue
            start, end = raw_range.split("-", 1)
            ranges.append((chain_id, int(start), int(end)))
    return ranges


def discover_structure_samples(input_dir: Union[str, Path]) -> List[StructureSample]:
    """Find PDB/mmCIF structures without loading coordinates into memory."""
    root = Path(input_dir)
    if root.is_file():
        return [StructureSample(path=root, sample_id=root.stem, target_id=root.stem)]
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    files = sorted(root.glob("seed-*sample-*/*_model.cif"))
    files.extend(sorted(root.glob("seed-*sample-*/*_model.pdb")))
    if not files:
        files = sorted(root.rglob("*_model.cif")) + sorted(root.rglob("*_model.pdb"))

    samples: List[StructureSample] = []
    for path in files:
        match = re.search(r"seed-(\d+)_sample-(\d+)", str(path))
        seed = int(match.group(1)) if match else None
        sample = int(match.group(2)) if match else None
        sample_id = path.parent.name if path.parent.name.startswith("seed-") else path.stem
        samples.append(
            StructureSample(
                path=path,
                sample_id=sample_id,
                target_id=root.name,
                method="af3" if match else "structure",
                seed=seed,
                sample=sample,
            )
        )
    return samples


def _match_path_case_insensitive(root: Path, stem: str, suffix: str = "") -> Optional[Path]:
    stem_lower = stem.lower()
    for path in root.iterdir():
        name = path.name.lower()
        if suffix:
            if name == f"{stem_lower}{suffix.lower()}":
                return path
        elif path.stem.lower() == stem_lower:
            return path
    return None


def resolve_target_native_path(
    target_id: str,
    native_root: Union[str, Path],
) -> Path:
    native_root = Path(native_root)
    match = _match_path_case_insensitive(native_root, target_id, suffix=".pdb")
    if match is None:
        raise FileNotFoundError(f"Native structure not found for target {target_id}: {native_root}")
    return match


def resolve_target_af3_dir(
    target_id: str,
    af3_root: Union[str, Path],
) -> Path:
    af3_root = Path(af3_root)
    exact_match = af3_root / str(target_id)
    if exact_match.is_dir():
        return exact_match
    match = _match_path_case_insensitive(af3_root, target_id)
    if match is None or not match.is_dir():
        raise FileNotFoundError(f"AF3 directory not found for target {target_id}: {af3_root}")
    return match


def discover_structure_samples_from_info(
    info_pkl: Union[str, Path],
    *,
    af3_root: Union[str, Path],
    native_root: Union[str, Path],
    target_key: str = "list",
) -> List[StructureSample]:
    with Path(info_pkl).open("rb") as fp:
        info = pickle.load(fp)

    if isinstance(info, dict):
        target_ids = info.get(target_key, [])
    else:
        target_ids = info
    if not isinstance(target_ids, (list, tuple)):
        raise ValueError(f"Unsupported target list in {info_pkl}: {type(target_ids)}")

    samples: List[StructureSample] = []
    for target_id in target_ids:
        af3_dir = resolve_target_af3_dir(str(target_id), af3_root)
        native_path = resolve_target_native_path(str(target_id), native_root)
        target_samples = attach_af3_ranking(
            discover_structure_samples(af3_dir),
            load_af3_ranking_scores(af3_dir),
        )
        for sample in target_samples:
            samples.append(
                StructureSample(
                    path=sample.path,
                    sample_id=sample.sample_id,
                    target_id=str(target_id),
                    native_path=native_path,
                    method=sample.method,
                    seed=sample.seed,
                    sample=sample.sample,
                    ranking=sample.ranking,
                    ranking_score=sample.ranking_score,
                    label=sample.label,
                )
            )
    return samples


def load_af3_ranking_scores(input_dir: Union[str, Path]) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Load AF3 ranking CSV and attach per-(seed, sample) score and rank.

    Preferred CSV format:
      seed,sample,ranking_score

    If an explicit rank column is absent, rank is derived by sorting all rows by
    descending ranking_score (higher score = better AF3 rank).
    """
    root = Path(input_dir)
    candidates = sorted(root.glob("*ranking_scores.csv"))
    if not candidates:
        return {}

    with candidates[0].open(newline="") as fp:
        rows = list(csv.DictReader(fp))

    parsed_rows = []
    for row in rows:
        seed = None
        sample = None
        if row.get("seed") not in ("", None) and row.get("sample") not in ("", None):
            try:
                seed = int(row["seed"])
                sample = int(row["sample"])
            except ValueError:
                seed = None
                sample = None
        if seed is None or sample is None:
            blob = " ".join(str(v) for v in row.values())
            match = re.search(r"seed-(\d+)_sample-(\d+)", blob)
            if not match:
                match = re.search(r"seed[_ -]?(\d+).*sample[_ -]?(\d+)", blob)
            if match:
                seed = int(match.group(1))
                sample = int(match.group(2))
        if seed is None or sample is None:
            continue

        score = float("nan")
        for key in ("ranking_score", "score", "confidence", "aggregate_score"):
            if key in row and row[key] not in ("", None):
                try:
                    score = float(row[key])
                    break
                except ValueError:
                    pass

        explicit_rank = None
        for key in ("rank", "ranking", "af3_rank"):
            if key in row and row[key] not in ("", None):
                try:
                    explicit_rank = int(row[key])
                    break
                except ValueError:
                    pass

        parsed_rows.append(
            {
                "seed": seed,
                "sample": sample,
                "ranking_score": score,
                "ranking": explicit_rank,
            }
        )

    if not parsed_rows:
        return {}

    if not any(row["ranking"] is not None for row in parsed_rows):
        sortable = sorted(
            parsed_rows,
            key=lambda row: row["ranking_score"],
            reverse=True,
        )
        for rank, row in enumerate(sortable, start=1):
            row["ranking"] = rank

    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    for row in parsed_rows:
        out[(row["seed"], row["sample"])] = {
            "ranking": float(row["ranking"]) if row["ranking"] is not None else float("nan"),
            "ranking_score": row["ranking_score"],
        }
    return out


def attach_af3_ranking(
    samples: Sequence[StructureSample],
    ranking_info: Dict[Tuple[int, int], Dict[str, float]],
) -> List[StructureSample]:
    if not ranking_info:
        return list(samples)

    enriched = []
    for sample in samples:
        info = ranking_info.get((sample.seed, sample.sample), {})
        ranking = sample.ranking
        if "ranking" in info:
            try:
                rank_value = float(info["ranking"])
                if math.isfinite(rank_value):
                    ranking = int(rank_value)
            except (TypeError, ValueError):
                pass
        enriched.append(
            StructureSample(
                path=sample.path,
                sample_id=sample.sample_id,
                method=sample.method,
                seed=sample.seed,
                sample=sample.sample,
                ranking=ranking,
                ranking_score=info.get("ranking_score", sample.ranking_score),
                label=sample.label,
            )
        )
    return enriched


def _parse_structure(path: Path):
    if path.suffix.lower() == ".cif":
        return MMCIFParser(QUIET=True).get_structure(path.stem, str(path))
    return PDBParser(QUIET=True).get_structure(path.stem, str(path))


def _apply_cdr_mask(dic: dict, cdr_ranges: Sequence[CDRRange]) -> None:
    if not cdr_ranges:
        return
    chain_ids = dic["chain_id"]
    res_no = dic["res_no"]
    mask = torch.zeros_like(dic["ulr_mask"], dtype=torch.int64)
    loop_id = torch.zeros_like(dic["ulr_mask"], dtype=torch.int64)
    for i, (chain_idx, resnum) in enumerate(zip(chain_ids.tolist(), res_no.tolist())):
        chain_id = REF_CHAIN[int(chain_idx)]
        for loop_idx, (range_chain, start, end) in enumerate(cdr_ranges, start=1):
            if chain_id == range_chain and start <= int(resnum) <= end:
                mask[i] = 1
                loop_id[i] = loop_idx
                break
    dic["ulr_mask"] = mask
    dic["loop_id"] = loop_id


class OnTheFlyStructureGraphDataset(Dataset):
    """Build one DGL graph from one PDB/mmCIF structure at access time.

    This module is reusable for training and inference. For training, pass
    ``StructureSample.label`` values and consume ``meta["label"]`` in the loop.
    For inference, leave labels unset and use the same graph construction path.
    """

    def __init__(
        self,
        samples: Sequence[StructureSample],
        *,
        cdr_ranges: Sequence[CDRRange],
        dist_cutoff: float = 10.0,
        max_neighbors: int = 60,
        use_all_atom: bool = False,
    ):
        self.samples = list(samples)
        self.cdr_ranges = list(cdr_ranges)
        self.dist_cutoff = float(dist_cutoff)
        self.max_neighbors = int(max_neighbors)
        self.use_all_atom = bool(use_all_atom)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        structure = _parse_structure(sample.path)
        model_obj = SimpleNamespace(
            method=sample.method,
            md_structure=structure,
            pdb_path=sample.path,
            h3_rmsd=float("nan") if sample.label is None else float(sample.label),
            ranking=-1 if sample.ranking is None else sample.ranking,
            seed=sample.seed,
            model_idx=sample.sample,
        )
        dic = create_dictionary_from_model(
            model_obj,
            use_all_atom=self.use_all_atom,
            cdr_ranges=self.cdr_ranges,
            target_id=sample.target_id or sample.sample_id,
        )
        _apply_cdr_mask(dic, self.cdr_ranges)
        dic = build_edge_mask(dic, dist_cut_off=self.dist_cutoff)
        graph = build_graph(
            dic,
            use_all_atom=self.use_all_atom,
            dist_cut_off=self.dist_cutoff,
            max_neighbors=self.max_neighbors,
        )
        meta = {
            "sample_id": sample.sample_id,
            "target_id": sample.target_id,
            "native_path": None if sample.native_path is None else str(sample.native_path),
            "method": sample.method,
            "path": str(sample.path),
            "seed": sample.seed,
            "sample": sample.sample,
            "ranking": sample.ranking,
            "ranking_score": sample.ranking_score,
            "label": sample.label,
            "n_nodes": int(graph.num_nodes()),
            "n_edges": int(graph.num_edges()),
        }
        return graph, meta


def collate_structure_graphs(batch):
    graphs, metas = zip(*batch)
    return dgl.batch(list(graphs)), list(metas)


def make_structure_graph_loader(
    samples: Sequence[StructureSample],
    *,
    batch_size: int = 4,
    cdr_ranges: str = DEFAULT_CDR_RANGES,
    dist_cutoff: float = 10.0,
    max_neighbors: int = 60,
    use_all_atom: bool = False,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    shuffle: bool = False,
) -> DataLoader:
    dataset = OnTheFlyStructureGraphDataset(
        samples,
        cdr_ranges=parse_cdr_ranges(cdr_ranges),
        dist_cutoff=dist_cutoff,
        max_neighbors=max_neighbors,
        use_all_atom=use_all_atom,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_structure_graphs,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0 and persistent_workers),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


def make_structure_inference_loader(
    input_dir: Union[str, Path],
    *,
    batch_size: int = 4,
    cdr_ranges: str = DEFAULT_CDR_RANGES,
    dist_cutoff: float = 10.0,
    max_neighbors: int = 60,
    use_all_atom: bool = False,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
) -> DataLoader:
    samples = discover_structure_samples(input_dir)
    samples = attach_af3_ranking(samples, load_af3_ranking_scores(input_dir))
    return make_structure_graph_loader(
        samples,
        batch_size=batch_size,
        cdr_ranges=cdr_ranges,
        dist_cutoff=dist_cutoff,
        max_neighbors=max_neighbors,
        use_all_atom=use_all_atom,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        shuffle=False,
    )
