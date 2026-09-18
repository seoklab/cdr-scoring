"""
dataset/config.py
=================
Load YAML dataset specification and resolve epoch-dependent parameters.

Key classes
-----------
- SourceSpec   : per-source settings (path template, file_type, weight, cap, …)
- ScheduleSpec : piecewise / linear schedule for a single scalar param
- DatasetSpec  : top-level container loaded from YAML
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def metric_higher_is_better(metric_name: str) -> bool:
    """True when larger values are better (lDDT-family), False for RMSD-family."""
    return "lddt" in str(metric_name).lower()


def default_near_native_cutoff(metric_name: str) -> float:
    """Default near-native vs non-native cutoff for a given label metric."""
    return 0.8 if metric_higher_is_better(metric_name) else 2.0


# ──────────────────────────────────────────────────────────────
# Schedule
# ──────────────────────────────────────────────────────────────
@dataclass
class ScheduleSpec:
    """Piecewise-constant or linear ramp schedule.

    YAML example (piecewise):
        schedule:
          - {epoch: 1,   value: 0.0}
          - {epoch: 50,  value: 0.5}
          - {epoch: 100, value: 1.0}
        interpolation: linear          # "step" (default) or "linear"

    ``get(epoch)`` returns the effective value at that epoch.
    """
    breakpoints: List[Dict[str, float]]  # list of {epoch: int, value: float}
    interpolation: str = "step"          # "step" | "linear"

    def get(self, epoch: int) -> float:
        if not self.breakpoints:
            raise ValueError("ScheduleSpec has no breakpoints")
        bps = sorted(self.breakpoints, key=lambda b: b["epoch"])
        if epoch <= bps[0]["epoch"]:
            return bps[0]["value"]
        if epoch >= bps[-1]["epoch"]:
            return bps[-1]["value"]
        # find surrounding breakpoints
        for i in range(len(bps) - 1):
            if bps[i]["epoch"] <= epoch < bps[i + 1]["epoch"]:
                if self.interpolation == "linear":
                    t = (epoch - bps[i]["epoch"]) / (bps[i + 1]["epoch"] - bps[i]["epoch"])
                    return bps[i]["value"] + t * (bps[i + 1]["value"] - bps[i]["value"])
                else:  # step
                    return bps[i]["value"]
        return bps[-1]["value"]


def _maybe_schedule(raw: Any) -> Any:
    """If *raw* is a dict with a 'schedule' key, wrap it in ScheduleSpec."""
    if isinstance(raw, dict) and "schedule" in raw:
        return ScheduleSpec(
            breakpoints=raw["schedule"],
            interpolation=raw.get("interpolation", "step"),
        )
    return raw


# ──────────────────────────────────────────────────────────────
# Source
# ──────────────────────────────────────────────────────────────
@dataclass
class SourceSpec:
    name: str
    enabled: bool = True
    file_type: str = "graph_pickle"          # "graph_pickle" | "target_model_pickle"
    weight: Any = 1.0                        # float or ScheduleSpec
    per_target_cap: int = 9999               # max decoys drawn from this source per target
    max_fraction: float = 1.0                # safety cap: fraction of total decoys
    graph_template: str = ""                 # e.g. "{db_root}/pertMD/graph/se3/{pdb}.dat"
    rmsd_template: str = ""                  # e.g. "{db_root}/pertMD/decoy/{pdb}/{pdb}_fp.rmsd"
    target_pickle_template: str = ""         # for target_model_pickle file_type
    metrics_template: str = ""               # for Boltz2 ag_local_rmsd csv/json
    metrics_format: str = "csv"              # "csv" | "json"
    pdb_id_type: str = "old"                 # "old" | "new" — which PDB ID namespace this source uses
    domain: str = "antibody"                 # "antibody" | "general_protein" — decoy provenance for
                                             # v2 per-head source-typing (GP=general_protein; else antibody,
                                             # apo/holo distinguished by finite DockQ at decoy level).

    def effective_weight(self, epoch: int) -> float:
        if isinstance(self.weight, ScheduleSpec):
            return self.weight.get(epoch)
        return float(self.weight)


# ──────────────────────────────────────────────────────────────
# Top-level spec
# ──────────────────────────────────────────────────────────────
@dataclass
class GraphBuildArgs:
    """Arguments forwarded to ``generate_graphs_from_target``."""
    dist_cutoff_center: float = 10.0
    random_range: float = 0.0
    max_neighbors: int = 60           # 0 means disabled (radius-only)
    use_all_atom: bool = False
    h3_range: List[int] = field(default_factory=lambda: [95, 102])
    cdr_ranges: Dict[str, List[List[int]]] = field(default_factory=lambda: {
        "H": [[26, 32], [52, 56], [95, 102]],
        "L": [[24, 34], [50, 56], [89, 97]],
    })
    cdr_context_cutoff: float = 15.0
    max_context_residues: int = 120
    graph_crop_debug: bool = False
    on_the_fly: bool = True             # skip caching for target_model_pickle (preserves random_range)


@dataclass
class RmsdFilterSpec:
    """RMSD-based pre-filtering before graph loading.

    Only decoys whose RMSD falls within [min_rmsd, max_rmsd] are retained.
    Applied *before* mixing / decoy selection so that graph pickles for
    completely-excluded sources are never loaded into memory.
    """
    min_rmsd: float = 0.0
    max_rmsd: float = float('inf')


@dataclass
class PrecomputedMetricsSpec:
    """Use precomputed loop-metric parquet files instead of on-the-fly metric
    computation during training.

    YAML example::

        precomputed_metrics:
          enabled: true
          root: /home/sujin/projects/cdr-scoring/cdr-data/metrics_precomputed
          require: true          # drop decoys with no precomputed metric (no fallback)
          metrics_filename: metrics/loop_metrics.parquet
          sources:               # optional source_name -> subdir override
            Boltz2: Boltz2
            ComMat: ComMat
    """
    enabled: bool = False
    root: str = ""
    require: bool = True
    metrics_filename: str = "metrics/loop_metrics.parquet"
    sources: Dict[str, str] = field(default_factory=dict)


@dataclass
class PdbIdMapping:
    """Bidirectional old <-> new PDB ID mapping.

    Loaded from info.pkl which has  ``{'list_old': {new_id: old_id, ...}}``.
    """
    old_to_new: Dict[str, str]    # old_pdb_id -> new_pdb_id
    new_to_old: Dict[str, str]    # new_pdb_id -> old_pdb_id

    @classmethod
    def from_info_pkl(cls, path: str) -> 'PdbIdMapping':
        import pickle as _pkl
        with open(path, 'rb') as f:
            info = _pkl.load(f)
        # info['list_old'] = {new_pdb_id: old_pdb_id}
        new_to_old: Dict[str, str] = info.get('list_old', {})
        old_to_new: Dict[str, str] = {v: k for k, v in new_to_old.items()}
        return cls(old_to_new=old_to_new, new_to_old=new_to_old)

    def get_new(self, pdb_id: str) -> str:
        """Return new-format PDB ID.  If already new or unknown, return as-is."""
        return self.old_to_new.get(pdb_id, pdb_id)

    def get_old(self, pdb_id: str) -> str:
        """Return old-format PDB ID.  If already old or unknown, return as-is."""
        return self.new_to_old.get(pdb_id, pdb_id)


@dataclass
class DatasetSpec:
    db_root: str = ""
    cache_root: str = ""
    seed: int = 42
    n_decoy: int = 64                   # total decoys per target returned to model
    task_scope: str = "full_cdr"        # "full_cdr" (M0) or legacy "h3"
    label_metric: str = "loop_rmsd"     # main scalar target; loop_rmsd=lower-better, loop_lddt=higher-better
    ranking_metric: str = "loop_rmsd"   # metric used for filtering/tier sampling
    near_native_cutoff: Optional[float] = None  # None -> metric-dependent default (2.0 rmsd / 0.8 lddt)
    min_diversity_threshold: int = 3     # min available sources to trigger diversity fill
    profile_training: bool = False
    profile_data_loading: bool = False
    profile_log_interval: int = 20
    graph_build: GraphBuildArgs = field(default_factory=GraphBuildArgs)
    sources: Dict[str, SourceSpec] = field(default_factory=dict)
    # schedulable scalars
    xtal_gate_prob: Any = 0.0            # float or ScheduleSpec
    boltz2_ag_local_rmsd_cutoff: Any = 2.0  # float or ScheduleSpec
    # YAML-driven list paths (when set, set_data() uses these instead of hardcoded paths)
    train_list: str = ""           # abag train list (file path)
    train_list_from_pkl: str = ""  # when set, load train list from this info.pkl instead
    train_list_pkl_key: str = "list"  # key in the pkl for the list (e.g. 1_tr_abag uses 'list')
    valid_list: str = ""
    valid_list_from_pkl: str = ""  # optional: load valid list from this pkl
    valid_list_pkl_key: str = "valid_list"  # key in the pkl for valid list
    gp_list: str = ""              # general protein train list (optional)
    num_gp: Optional[int] = None   # number of GP targets to sample per epoch (None = use all)
    num_abag: Optional[int] = None # number of AbAg targets to sample per epoch (None = use all)
    num_valid_abag: Optional[int] = None  # validation AbAg targets per epoch (None = use all)
    # RMSD range pre-filter (applied before mixing / graph loading)
    rmsd_filter: Optional[RmsdFilterSpec] = None
    # PDB ID mapping for old <-> new format conversion
    pdb_id_mapping_path: str = ""           # path to info.pkl
    pdb_id_mapping: Optional[PdbIdMapping] = field(default=None, repr=False)
    # Precomputed per-decoy metrics (skip on-the-fly loop metric computation)
    precomputed_metrics: Optional[PrecomputedMetricsSpec] = None
    # v1 DockQ gate: during TRAINING, keep only holo decoys with DockQ >= this
    # value (apo decoys, i.e. NaN DockQ, are always kept). None disables the gate.
    # Validation is never gated (top-1 CAPRI must see the full decoy pool).
    dockq_gate: Optional[float] = None
    # exp8 fnat gate: keep only holo decoys with fnat > this value (apo/GP, i.e.
    # NaN fnat, are always kept). None disables it. Same rule the exp8 SML applies
    # in the loss, moved into candidate selection so unusable poses are never
    # loaded, and so the training pool matches the fnat>0.5-gated validation
    # manifests. Strict >, matching the project-wide "fnat>0.5" filter.
    fnat_gate: Optional[float] = None

    # ── convenience helpers ──
    def effective_xtal_prob(self, epoch: int) -> float:
        if isinstance(self.xtal_gate_prob, ScheduleSpec):
            return self.xtal_gate_prob.get(epoch)
        return float(self.xtal_gate_prob)

    def effective_boltz2_cutoff(self, epoch: int) -> float:
        if isinstance(self.boltz2_ag_local_rmsd_cutoff, ScheduleSpec):
            return self.boltz2_ag_local_rmsd_cutoff.get(epoch)
        return float(self.boltz2_ag_local_rmsd_cutoff)

    def enabled_sources(self) -> Dict[str, SourceSpec]:
        return {k: v for k, v in self.sources.items() if v.enabled}

    def label_higher_is_better(self) -> bool:
        """True when the label metric is higher-is-better (lDDT-family)."""
        return metric_higher_is_better(self.label_metric)

    def effective_near_native_cutoff(self) -> float:
        """Cutoff separating near-native from non-native for SML / tier sampling."""
        if self.near_native_cutoff is not None:
            return float(self.near_native_cutoff)
        return default_near_native_cutoff(self.label_metric)

    def resolve_pdb_id(self, pdb_id: str, target_type: str) -> str:
        """Convert *pdb_id* to old / new format as needed by *target_type*."""
        if self.pdb_id_mapping is None:
            return pdb_id
        if target_type == "new":
            return self.pdb_id_mapping.get_new(pdb_id)
        else:  # "old" or default
            return self.pdb_id_mapping.get_old(pdb_id)


# ──────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────
def load_dataset_spec(yaml_path: str | Path) -> DatasetSpec:
    """Parse YAML file into a ``DatasetSpec``."""
    with open(yaml_path, encoding='utf-8') as f:
        raw: dict = yaml.safe_load(f)

    # graph_build
    gb_raw = raw.get("graph_build", {})
    graph_build = GraphBuildArgs(
        dist_cutoff_center=gb_raw.get("dist_cutoff_center", 10.0),
        random_range=gb_raw.get("random_range", 0.0),
        max_neighbors=gb_raw.get("max_neighbors", 60),
        use_all_atom=gb_raw.get("use_all_atom", False),
        h3_range=gb_raw.get("h3_range", [95, 102]),
        cdr_ranges=gb_raw.get("cdr_ranges", {
            "H": [[26, 32], [52, 56], [95, 102]],
            "L": [[24, 34], [50, 56], [89, 97]],
        }),
        cdr_context_cutoff=float(gb_raw.get("cdr_context_cutoff", 15.0)),
        max_context_residues=int(gb_raw.get("max_context_residues", 120)),
        graph_crop_debug=bool(gb_raw.get("graph_crop_debug", False)),
        on_the_fly=gb_raw.get("on_the_fly", True),
    )

    # sources
    sources: Dict[str, SourceSpec] = {}
    for name, src_raw in raw.get("sources", {}).items():
        weight_raw = src_raw.get("weight", 1.0)
        sources[name] = SourceSpec(
            name=name,
            enabled=src_raw.get("enabled", True),
            file_type=src_raw.get("file_type", "graph_pickle"),
            weight=_maybe_schedule(weight_raw),
            per_target_cap=src_raw.get("per_target_cap", 9999),
            max_fraction=src_raw.get("max_fraction", 1.0),
            graph_template=src_raw.get("graph_template", ""),
            rmsd_template=src_raw.get("rmsd_template", ""),
            target_pickle_template=src_raw.get("target_pickle_template", ""),
            metrics_template=src_raw.get("metrics_template", ""),
            metrics_format=src_raw.get("metrics_format", "csv"),
            pdb_id_type=src_raw.get("pdb_id_type", "old"),
            domain=src_raw.get("domain", "antibody"),
        )

    # rmsd_filter
    rf_raw = raw.get("rmsd_filter", None)
    rmsd_filter = None
    if rf_raw:
        rmsd_filter = RmsdFilterSpec(
            min_rmsd=float(rf_raw.get("min_rmsd", 0.0)),
            max_rmsd=float(rf_raw.get("max_rmsd", float('inf'))),
        )

    # precomputed_metrics
    pm_raw = raw.get("precomputed_metrics", None)
    precomputed_metrics = None
    if pm_raw and pm_raw.get("enabled", False):
        precomputed_metrics = PrecomputedMetricsSpec(
            enabled=True,
            root=pm_raw.get("root", ""),
            require=bool(pm_raw.get("require", True)),
            metrics_filename=pm_raw.get("metrics_filename", "metrics/loop_metrics.parquet"),
            sources={str(k): str(v) for k, v in (pm_raw.get("sources", {}) or {}).items()},
        )

    # "all" (string) or null → None → use entire list without subsampling
    _raw_gp = raw.get("num_gp", None)
    _raw_abag = raw.get("num_abag", None)
    _raw_valid_abag = raw.get("num_valid_abag", None)
    _num_gp = None if (_raw_gp is None or str(_raw_gp).lower() == "all") else int(_raw_gp)
    _num_abag = None if (_raw_abag is None or str(_raw_abag).lower() == "all") else int(_raw_abag)
    _num_valid_abag = (
        None if (_raw_valid_abag is None or str(_raw_valid_abag).lower() == "all")
        else int(_raw_valid_abag)
    )

    spec = DatasetSpec(
        db_root=raw.get("db_root", ""),
        cache_root=raw.get("cache_root", ""),
        seed=raw.get("seed", 42),
        n_decoy=raw.get("n_decoy", 64),
        task_scope=raw.get("task_scope", "full_cdr"),
        label_metric=raw.get("label_metric", "loop_rmsd"),
        ranking_metric=raw.get("ranking_metric", "loop_rmsd"),
        near_native_cutoff=raw.get("near_native_cutoff", None),
        min_diversity_threshold=raw.get("min_diversity_threshold", 3),
        profile_training=bool(raw.get("profile_training", False)),
        profile_data_loading=bool(raw.get("profile_data_loading", False)),
        profile_log_interval=int(raw.get("profile_log_interval", 20)),
        graph_build=graph_build,
        sources=sources,
        xtal_gate_prob=_maybe_schedule(raw.get("xtal_gate_prob", 0.0)),
        boltz2_ag_local_rmsd_cutoff=_maybe_schedule(raw.get("boltz2_ag_local_rmsd_cutoff", 2.0)),
        train_list=raw.get("train_list", ""),
        train_list_from_pkl=raw.get("train_list_from_pkl", ""),
        train_list_pkl_key=raw.get("train_list_pkl_key", "list"),
        valid_list=raw.get("valid_list", ""),
        valid_list_from_pkl=raw.get("valid_list_from_pkl", ""),
        valid_list_pkl_key=raw.get("valid_list_pkl_key", "valid_list"),
        gp_list=raw.get("gp_list", ""),
        num_gp=_num_gp,
        num_abag=_num_abag,
        num_valid_abag=_num_valid_abag,
        rmsd_filter=rmsd_filter,
        pdb_id_mapping_path=raw.get("pdb_id_mapping", ""),
        precomputed_metrics=precomputed_metrics,
        dockq_gate=(None if raw.get("dockq_gate", None) is None else float(raw.get("dockq_gate"))),
        fnat_gate=(None if raw.get("fnat_gate", None) is None else float(raw.get("fnat_gate"))),
    )
    # ── CLI overrides via environment (set by train.py from argparser) ──
    # This keeps a single source of truth: both train.py's spec and the
    # dataset's internally-loaded spec honor --label_metric / --near_native_cutoff.
    _env_label = os.environ.get("CDR_LABEL_METRIC")
    if _env_label:
        spec.label_metric = _env_label
        # Full-lDDT mode: ranking/tier logic follows the same metric as the label.
        spec.ranking_metric = _env_label
    _env_cutoff = os.environ.get("CDR_NEAR_NATIVE_CUTOFF")
    if _env_cutoff not in (None, ""):
        try:
            spec.near_native_cutoff = float(_env_cutoff)
        except ValueError:
            pass

    # Load PDB ID mapping if path provided
    if spec.pdb_id_mapping_path:
        spec.pdb_id_mapping = PdbIdMapping.from_info_pkl(spec.pdb_id_mapping_path)
    return spec
