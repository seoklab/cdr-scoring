import logging
import pathlib
from typing import List
import sys,pickle
_CURRENT_LIBS_DIR = str(pathlib.Path(__file__).resolve().parents[1])
if _CURRENT_LIBS_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_LIBS_DIR)
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []
from torch.optim import AdamW
from data_loading.data_module import HUDataModule
from data_loading.pdb2dict import Target, Model
from model.transformer import Sujin_with_SE3, Sujin_with_SE3_allatom
from model.fiber import Fiber
from runtime.arguments import PARSER
from runtime.callbacks import QM9LRSchedulerCallback, BaseCallback
from runtime.loggers import LoggerCollection, DLLogger, WandbLogger, Logger
from runtime.utils import to_cuda,to_cpu,get_local_rank, init_distributed, seed_everything, \
    using_tensor_cores, increase_l2_fetch_granularity
from config import config as se3_config
from utils import (
    initialize_epoch_loss, update_epoch_loss, finalize_epoch_loss,
    reduce_epoch_loss, report_epoch_loss, aggregate_top1_h3_val_metrics,
)
from runtime.sujin_loss import FinalLoss as Sujin_loss
from runtime.sujin_loss import (
    TotalLoss, TierDPOLoss, OrdinalH3LddtAuxLoss, compute_top1_h3_validation_metrics,
    TOP1_H3_VAL_LIST_KEYS, InterfaceSoftRankLoss, TierBalancedSoftRankLoss,
    Af3MatchedTierPairLoss, CdrMatchedFnatContrastLoss,
    InterfaceFnatRegLoss, grad_scale_wrt,
)
from runtime.phase_config import get_phase_config
from runtime.pair_sampling import (
    build_top_region_pairs,
    PairSamplingConfig, build_training_pairs, build_decoy_tiers, detect_xtal_mask,
    build_af3_matched_pairs, AF3_LDDT_TIER_PAIR_P,
)
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_descriptor')
import os,random
from runtime.constants import *
import traceback
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:256')
import copy
import warnings
import time


import dgl

DEFAULT_CKPT_ROOT = pathlib.Path('/home/sujin/projects/cdr-scoring/cdr-data/ckpt')
DEFAULT_INFERENCE_DIR = pathlib.Path('/home/sujin/projects/cdr-scoring/cdr-data/inference')


def _run_result_dir(args) -> pathlib.Path:
    """Directory for checkpoints and per-epoch info files for this run."""
    root = pathlib.Path(getattr(args, 'ckpt_root', DEFAULT_CKPT_ROOT) or DEFAULT_CKPT_ROOT)
    return root / str(args.param_name)


def _inference_output_dir(args) -> pathlib.Path:
    root = pathlib.Path(getattr(args, 'inference_dir', DEFAULT_INFERENCE_DIR) or DEFAULT_INFERENCE_DIR)
    return root


def _extract_eval_metrics(batch):
    """Find optional loop_rmsd/loop_lddt eval metrics appended by YAML datasets."""
    for item in batch[3:]:
        if isinstance(item, dict):
            return item
        if isinstance(item, (list, tuple)) and len(item) == 1 and isinstance(item[0], dict):
            return item[0]
    return None


def _metric_tensor(eval_metrics, name, device):
    if not eval_metrics or name not in eval_metrics:
        return None
    value = eval_metrics[name]
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32).reshape(-1)
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], torch.Tensor):
        return value[0].to(device=device, dtype=torch.float32).reshape(-1)
    try:
        return torch.tensor(value, dtype=torch.float32, device=device).reshape(-1)
    except Exception:
        return None


def _add_eval_top1_metrics(out, eval_metrics, scores, label_metric):
    """Attach active-score top1 metrics for both loop RMSD and loop lDDT."""
    if not eval_metrics:
        return
    score_vec = scores.detach().reshape(-1)
    if score_vec.numel() == 0:
        return
    higher_score_is_better = 'lddt' in str(label_metric).lower()
    top1_idx = torch.argsort(score_vec, descending=higher_score_is_better)[0]
    device = score_vec.device

    loop_rmsd = _metric_tensor(eval_metrics, 'loop_rmsd', device)
    if loop_rmsd is not None and loop_rmsd.numel() == score_vec.numel():
        finite = torch.isfinite(loop_rmsd)
        if finite.any():
            top1 = loop_rmsd[top1_idx]
            oracle = loop_rmsd[finite].min()
            out['eval_top1_loop_rmsd'] = top1.detach()
            out['eval_oracle_loop_rmsd'] = oracle.detach()
            out['eval_loop_rmsd_regret'] = (top1 - oracle).detach()

    loop_lddt = _metric_tensor(eval_metrics, 'loop_lddt', device)
    if loop_lddt is not None and loop_lddt.numel() == score_vec.numel():
        finite = torch.isfinite(loop_lddt)
        if finite.any():
            top1 = loop_lddt[top1_idx]
            oracle = loop_lddt[finite].max()
            out['eval_top1_loop_lddt'] = top1.detach()
            out['eval_oracle_loop_lddt'] = oracle.detach()
            out['eval_loop_lddt_regret'] = (oracle - top1).detach()

    # H3 lDDT of the score-top1 decoy (validation tracking of the 2nd objective).
    h3_lddt = _metric_tensor(eval_metrics, 'h3_lddt', device)
    if h3_lddt is not None and h3_lddt.numel() == score_vec.numel():
        if torch.isfinite(h3_lddt[top1_idx]):
            out['eval_top1_h3_lddt'] = h3_lddt[top1_idx].detach()
        fin = torch.isfinite(h3_lddt)
        if fin.any():
            out['eval_oracle_h3_lddt'] = h3_lddt[fin].max().detach()

    # Top-1 CAPRI classification (validation metric only, NOT a training target):
    # pick the best-scored NON-crystal decoy and bucket its DockQ into CAPRI
    # classes. Emitted only for holo targets (finite dockq) so each epoch mean is
    # a fraction over holo targets.
    dockq = _metric_tensor(eval_metrics, 'dockq', device)
    loop_rmsd_v = _metric_tensor(eval_metrics, 'loop_rmsd', device)
    if (dockq is not None and dockq.numel() == score_vec.numel()
            and loop_rmsd_v is not None and loop_rmsd_v.numel() == score_vec.numel()):
        nonxtal = loop_rmsd_v >= 0.01              # crystal has loop_rmsd == 0
        if nonxtal.any():
            idxs = torch.nonzero(nonxtal, as_tuple=False).reshape(-1)
            order = torch.argsort(score_vec[idxs], descending=higher_score_is_better)
            top1_nx = idxs[order[0]]
            dq = dockq[top1_nx]
            if torch.isfinite(dq):
                dqf = float(dq.item())
                out['eval_top1_dockq'] = dq.detach()
                out['eval_top1_capri_acceptable'] = torch.tensor(1.0 if dqf >= 0.23 else 0.0, device=device)
                out['eval_top1_capri_medium'] = torch.tensor(1.0 if dqf >= 0.49 else 0.0, device=device)
                out['eval_top1_capri_high'] = torch.tensor(1.0 if dqf >= 0.80 else 0.0, device=device)
                dqall = dockq[idxs]
                dqfin = dqall[torch.isfinite(dqall)]
                if dqfin.numel() > 0:
                    out['eval_oracle_dockq'] = dqfin.max().detach()


def _read_list_file(path):
    """Read a PDB list file (one ID per line)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _read_list_from_pkl(pkl_path: str, key: str = "list"):
    """Read a PDB list from a pickle file (e.g. info.pkl with key 'list')."""
    with open(pkl_path, "rb") as f:
        info = pickle.load(f)
    lst = info.get(key)
    if lst is None:
        return []
    if isinstance(lst, dict):
        lst = list(lst.keys())
    elif not isinstance(lst, (list, tuple)):
        lst = [lst]
    return [str(x) for x in lst]


def _unique_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _load_excluded_pdbs(args):
    """Load excluded PDB IDs from CLI args (comma-list + optional file)."""
    excluded = set()
    raw = getattr(args, 'exclude_pdb_ids', '') or ''
    if raw:
        excluded.update(x.strip() for x in raw.split(',') if x.strip())

    fp = getattr(args, 'exclude_pdb_file', None)
    if fp:
        try:
            with open(fp) as f:
                excluded.update(line.strip() for line in f if line.strip())
        except Exception as e:
            logging.warning('Failed to read exclude_pdb_file=%s: %s', fp, e)
    return excluded


def _apply_pdb_exclusion(items, excluded):
    """Filter out excluded IDs while preserving order."""
    if not excluded:
        return list(items), 0
    out = [x for x in items if x not in excluded]
    return out, len(items) - len(out)


def _prepare_yaml_abag_pools(spec, excluded_pdbs=None):
    """Build AbAg train/valid pools for YAML mode with no overlap."""
    if getattr(spec, "train_list_from_pkl", ""):
        abag_all = _read_list_from_pkl(spec.train_list_from_pkl, getattr(spec, "train_list_pkl_key", "list"))
    else:
        abag_all = _read_list_file(spec.train_list) if spec.train_list else []
    abag_all = _unique_preserve_order(abag_all)

    if getattr(spec, "valid_list_from_pkl", ""):
        valid_cfg = _read_list_from_pkl(spec.valid_list_from_pkl, getattr(spec, "valid_list_pkl_key", "valid_list"))
    else:
        valid_cfg = _read_list_file(spec.valid_list) if spec.valid_list else []
    valid_cfg = _unique_preserve_order(valid_cfg)

    excluded_pdbs = excluded_pdbs or set()
    abag_all, n_excluded_train_pool = _apply_pdb_exclusion(abag_all, excluded_pdbs)
    valid_cfg, n_excluded_valid_pool = _apply_pdb_exclusion(valid_cfg, excluded_pdbs)

    split_enabled = bool(
        getattr(spec, "split_valid_from_train_pkl", False)
        and getattr(spec, "train_list_from_pkl", "")
    )

    if split_enabled:
        ratio = float(getattr(spec, "valid_split_ratio", 0.1))
        ratio = min(max(ratio, 0.0), 0.5)
        seed = getattr(spec, "valid_split_seed", None)
        if seed is None:
            seed = int(getattr(spec, "seed", 42))

        shuffled = list(abag_all)
        _rng = random.Random(seed)
        _rng.shuffle(shuffled)
        if len(shuffled) >= 2:
            n_valid = int(round(len(shuffled) * ratio))
            n_valid = max(1, min(n_valid, len(shuffled) - 1))
        else:
            n_valid = len(shuffled)
        valid_pool = shuffled[:n_valid]
        train_pool = shuffled[n_valid:]
        valid_source = "split_from_train_pkl"
    else:
        valid_pool = list(valid_cfg)
        valid_set = set(valid_pool)
        train_pool = [p for p in abag_all if p not in valid_set]
        valid_source = "configured_valid_list"

    # Final safety: always enforce no overlap
    valid_set = set(valid_pool)
    train_pool = [p for p in train_pool if p not in valid_set]
    overlap_after = len(set(train_pool) & set(valid_pool))

    meta = {
        "valid_source": valid_source,
        "abag_all": len(abag_all),
        "train_pool": len(train_pool),
        "valid_pool": len(valid_pool),
        "overlap_after": overlap_after,
        "split_enabled": split_enabled,
        "split_ratio": float(getattr(spec, "valid_split_ratio", 0.1)),
        "n_excluded_train_pool": n_excluded_train_pool,
        "n_excluded_valid_pool": n_excluded_valid_pool,
    }
    return train_pool, valid_pool, meta


def _batch_node_info(batched_graph):
    """Return a compact node-count summary string for a batched DGL graph."""
    try:
        if hasattr(batched_graph, "batch_num_nodes"):
            bn = batched_graph.batch_num_nodes()
            bn_list = bn.tolist() if hasattr(bn, "tolist") else list(bn)
            if bn_list:
                total_nodes = int(sum(int(x) for x in bn_list))
                return (
                    f"total_nodes={total_nodes}, n_graphs={len(bn_list)}, "
                    f"min_nodes={int(min(bn_list))}, max_nodes={int(max(bn_list))}"
                )
        return f"total_nodes={int(batched_graph.num_nodes())}"
    except Exception:
        return "total_nodes=unknown"


def _get_total_nodes(batched_graph):
    """Best-effort total node count for a batched DGL graph."""
    try:
        return int(batched_graph.num_nodes())
    except Exception:
        try:
            if hasattr(batched_graph, 'batch_num_nodes'):
                bn = batched_graph.batch_num_nodes()
                bn_list = bn.tolist() if hasattr(bn, 'tolist') else list(bn)
                return int(sum(int(x) for x in bn_list))
        except Exception:
            pass
    return -1


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _graph_batch_stats(batched_graph):
    stats = {
        "n_graphs": 1,
        "total_nodes": -1,
        "total_edges": -1,
        "max_nodes_per_graph": -1,
        "max_edges_per_graph": -1,
    }
    try:
        stats["total_nodes"] = int(batched_graph.num_nodes())
    except Exception:
        pass
    try:
        stats["total_edges"] = int(batched_graph.num_edges())
    except Exception:
        pass
    try:
        if hasattr(batched_graph, "batch_num_nodes"):
            node_counts = batched_graph.batch_num_nodes()
            node_counts = node_counts.tolist() if hasattr(node_counts, "tolist") else list(node_counts)
            if node_counts:
                stats["n_graphs"] = len(node_counts)
                stats["max_nodes_per_graph"] = int(max(node_counts))
                if stats["total_nodes"] < 0:
                    stats["total_nodes"] = int(sum(int(x) for x in node_counts))
    except Exception:
        pass
    try:
        if hasattr(batched_graph, "batch_num_edges"):
            edge_counts = batched_graph.batch_num_edges()
            edge_counts = edge_counts.tolist() if hasattr(edge_counts, "tolist") else list(edge_counts)
            if edge_counts:
                stats["max_edges_per_graph"] = int(max(edge_counts))
                if stats["total_edges"] < 0:
                    stats["total_edges"] = int(sum(int(x) for x in edge_counts))
    except Exception:
        pass
    return stats


class _TrainingProfiler:
    def __init__(self, enabled, interval, local_rank):
        self.enabled = bool(enabled) and local_rank == 0
        self.interval = max(1, int(interval or 20))
        self.local_rank = local_rank
        self.count = 0
        self.totals = {}
        self.latest_stats = {}

    def add(self, timings, stats):
        if not self.enabled:
            return
        self.count += 1
        for key, value in timings.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value)
        self.latest_stats = dict(stats)
        if self.count % self.interval == 0:
            self.log()
            self.totals = {}

    def log(self):
        if not self.enabled or self.count <= 0 or not self.totals:
            return
        denom = float(self.interval if self.count % self.interval == 0 else self.count % self.interval)
        mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        timing_txt = " ".join(
            f"{key}={self.totals.get(key, 0.0) / denom:.4f}s"
            for key in (
                "dataloader",
                "transfer",
                "forward",
                "loss",
                "backward",
                "optimizer",
                "total",
            )
        )
        stats_txt = " ".join(
            f"{key}={self.latest_stats.get(key, -1)}"
            for key in (
                "n_graphs",
                "total_nodes",
                "total_edges",
                "max_nodes_per_graph",
                "max_edges_per_graph",
            )
        )
        print(
            f"[TRAIN_PROFILE] steps={self.count} avg_over={int(denom)} "
            f"{timing_txt} {stats_txt} "
            f"gpu_allocated_gb={mem_alloc:.3f} gpu_reserved_gb={mem_reserved:.3f}",
            flush=True,
        )


def _should_skip_oversized_batch(batched_graph, pdb, epoch_idx, batch_idx, train_tag, args, local_rank):
    """Synchronously skip oversized batches across ranks to avoid DDP desync."""
    max_nodes = int(getattr(args, 'max_batch_nodes', 0) or 0)
    if max_nodes <= 0:
        return False

    total_nodes = _get_total_nodes(batched_graph)
    local_skip = (total_nodes > max_nodes) if total_nodes >= 0 else False

    if not dist.is_initialized():
        if local_skip:
            print(
                f'[WARN] skip oversized batch: mode={train_tag}, epoch={epoch_idx}, '
                f'batch_idx={batch_idx}, pdb={pdb}, total_nodes={total_nodes}, '
                f'max_batch_nodes={max_nodes}',
                flush=True,
            )
        return local_skip

    sync_device = torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    flag = torch.tensor([1 if local_skip else 0], dtype=torch.int32, device=sync_device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    global_skip = bool(flag.item())
    if not global_skip:
        return False

    offenders = [None for _ in range(dist.get_world_size())]
    local_offender = (str(pdb), int(total_nodes)) if local_skip else None
    try:
        dist.all_gather_object(offenders, local_offender)
    except Exception:
        offenders = [local_offender]

    if local_rank == 0:
        offenders = [x for x in offenders if x is not None]
        if offenders:
            offender_txt = ', '.join(f'{pid}(nodes={n})' for pid, n in offenders)
            print(
                f'[WARN] skip oversized batch: mode={train_tag}, epoch={epoch_idx}, '
                f'batch_idx={batch_idx}, offenders={offender_txt}, '
                f'max_batch_nodes={max_nodes}',
                flush=True,
            )
        else:
            print(
                f'[WARN] skip oversized batch: mode={train_tag}, epoch={epoch_idx}, '
                f'batch_idx={batch_idx}, max_batch_nodes={max_nodes}',
                flush=True,
            )
    return True


def set_data(mode, num_gp, num_abag,random_seed,args):
    # random seed have to be defined for synchronizing between process 
    if mode =='train':
        random.seed(random_seed)#
        # a= [1,2,3,4,5,6,7,8] // gpu = 4
        # gpu 1~4 : [1,2],[3,4],[5,6],[7,8]
        # random.shuffle(a)
        # gpu1: [87654321] gpu2:[12/34/56/78]
        # gpu1 87 gpu2:12
        # Get training set
        train_gp = []; train_abag=[]
        if args.all_atom:
            gp_list = open('/home/sujin/DB/h3-loop-modeling/general_protein/graph-all-atom/list/gp-allatom.list') # all-atom graph list
        else:
            gp_list = open('/home/sujin/projects/h3-loop-modeling/data/list/0_FINAL/gp/train_val.list') # train_val.list
        for pdb in gp_list:
            train_gp.append(pdb.rstrip('\n'))
        abag_list = open('/home/sujin/projects/h3-loop-modeling/data/list/0_FINAL/abag/train_95.list')
        # abag_list = open('/home/sujin/projects/h3-loop-modeling/data/list/0_FINAL/abag/tmp.list')
        for pdb in abag_list:
            train_abag.append(pdb.rstrip('\n'))
        if random_seed == 1: # random_seed was epoch_idx in training loop
            print('[gp]    ',len(train_gp),num_gp)
            print('[abag]  ',len(train_abag),num_abag)
        if num_gp == None:
            num_gp=len(train_gp)
        if num_abag==None:
            num_abag=len(train_abag)

        train_list=random.sample(train_gp,k=num_gp)+random.sample(train_abag,k=num_abag)
        random.shuffle(train_list)
        # Get validation set
        valid_gp = []; valid_abag=[]
        # VALIDATION using only abag dataset
        # gp_list = open('/home/sujin/projects/h3-loop-modeling/data/step3_decoy_distribution/list/1027_nnd_4_val.list')
        # for pdb in gp_list:
        #     valid_gp.append(pdb.rstrip('\n'))
        gp_list=[] # no gp validation set ! abag only
        abag_list = open('/home/sujin/projects/h3-loop-modeling/data/list/0_FINAL/abag/valid_05.list')
        for pdb in abag_list:
            valid_abag.append(pdb.rstrip('\n'))
        # valid_list=valid_gp+valid_abag
        valid_list=valid_abag # validation only for abag dataset (152) [23.11.27]
        return train_list, valid_list
def save_state(model: nn.Module, optimizer: Optimizer, epoch: int, path: str, callbacks: List[BaseCallback]):
    if get_local_rank() == 0: 
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        checkpoint = {
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        for callback in callbacks:
            callback.on_checkpoint_save(checkpoint)

        torch.save(checkpoint, str(path))
        logging.info(f'Saved checkpoint to {str(path)}')
def load_state(model: nn.Module, optimizer: Optimizer, path: pathlib.Path, callbacks: List[BaseCallback]):
    """ Loads model, optimizer and epoch states from path """
    checkpoint = torch.load(str(path), map_location={'cuda:0': f'cuda:{get_local_rank()}'})
    if 'scheduler_state_dict' in checkpoint and isinstance(checkpoint['scheduler_state_dict'], dict):
        checkpoint['scheduler_state_dict']['gamma'] = 0.75
    if isinstance(model, DistributedDataParallel):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for callback in callbacks:
        callback.on_checkpoint_load(checkpoint)

    logging.info(f'Loaded checkpoint from {str(path)}')
    return checkpoint['epoch']

_NO_DECAY_HEADS = ('linear_out_1', 'final_head', 'interface_head', 'ord_head')


def _param_groups(model, args, local_rank, tag):
    """AdamW parameter groups: keep weight decay off the output heads and 1-D params.

    AdamW decays EVERY parameter it owns on every step, gradient or not:
    ``w <- w - lr*wd*w``. With lr=5e-4, wd=0.1 and 1000 steps per epoch that is
    ~5 % shrinkage per epoch applied to things that should not shrink. Measured on
    exp6, where the intrinsic head gets no gradient at all in the DPO path: its norm
    went 0.4795 -> 0.3295 (-31 %) in 13 epochs purely from decay, on track for -80 %
    by epoch 60. The same mechanism collapsed an earlier gate-OFF run (1.97 -> 0.06).

    Excluded from decay:
      * the four output heads -- a 1x64 read-out has no capacity to overfit, and
        shrinking it just rescales the scores (or erases them)
      * every 1-D parameter (biases, group-norm scales/offsets), the standard rule

    The backbone weight matrices keep the configured decay, so regularisation of the
    part that can actually overfit is unchanged.
    """
    if not getattr(args, 'no_decay_heads', False):
        return model.parameters()
    decay, no_decay, nd_names = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or any(h in name for h in _NO_DECAY_HEADS):
            no_decay.append(p); nd_names.append(name)
        else:
            decay.append(p)
    if local_rank == 0:
        print(f'[{tag}] weight decay split: {len(decay)} tensors decayed at '
              f'{args.weight_decay}, {len(no_decay)} excluded (heads + 1-D params)')
        print(f'[{tag}] excluded heads: '
              f'{[n for n in nd_names if any(h in n for h in _NO_DECAY_HEADS)]}')
    return [{'params': decay, 'weight_decay': float(args.weight_decay)},
            {'params': no_decay, 'weight_decay': 0.0}]


def _resolve_gen_source_ids(args, spec, local_rank, tag):
    """exp6: map --dpo_gen_sources NAMES to the integer ids the data layer emits.

    Source ids are indices into ``sorted(spec.sources)`` (mirrors
    ``data_module._src_index``), so they shift whenever the YAML changes; resolving
    by name is the only safe way to reach them from the loss loop.

    Raises when ``--dpo_gen_only_pairs`` is set but nothing resolves. A silent
    fallback here would run the whole experiment with the intervention switched
    off while every log line still claimed it was on.
    """
    names = [s.strip() for s in
             str(getattr(args, 'dpo_gen_sources', '') or '').split(',') if s.strip()]
    order = sorted(getattr(spec, 'sources', {}) or {})
    index = {s: i for i, s in enumerate(order)}
    ids = [index[s] for s in names if s in index]
    unknown = [s for s in names if s not in index]
    if getattr(args, 'dpo_gen_only_pairs', False):
        if local_rank == 0:
            print(f'[{tag} exp6 gen-gen] source order = {order}')
            print(f'[{tag} exp6 gen-gen] generative {names} -> ids {ids}')
            if unknown:
                print(f'[{tag} exp6 gen-gen] WARNING not in this config: {unknown}')
        if not ids:
            raise SystemExit(
                f'[{tag} exp6] --dpo_gen_only_pairs is set but --dpo_gen_sources='
                f'{names!r} resolved to nothing against {order}. Refusing to run with '
                f'the intervention silently disabled.')
    return ids


def _resolve_named_source_ids(names_csv, spec, local_rank, tag, required):
    """exp8: map source NAMES to the integer ids the data layer emits.

    Same contract as ``_resolve_gen_source_ids`` — ids are indices into
    ``sorted(spec.sources)`` and shift with the YAML, so resolving by name is the
    only safe route. ``required`` makes an unresolvable name fatal rather than a
    silent no-op (the exp6 lesson: --dpo_gen_only_pairs ran a whole campaign with
    the intervention off while every log line said it was on).
    """
    names = [s.strip() for s in str(names_csv or '').split(',') if s.strip()]
    order = sorted(getattr(spec, 'sources', {}) or {})
    index = {s: i for i, s in enumerate(order)}
    ids = [index[s] for s in names if s in index]
    unknown = [s for s in names if s not in index]
    if names and local_rank == 0:
        print(f'[{tag} exp8 demote] source order = {order}')
        print(f'[{tag} exp8 demote] {names} -> ids {ids}')
    if required and names and (not ids or unknown):
        raise SystemExit(
            f'[{tag} exp8] --sml_demote_sources={names!r} did not fully resolve '
            f'against {order} (unknown={unknown}). Refusing to run with the cap '
            f'silently disabled.')
    return ids


_COLLAPSE_DIAG_KEYS = ('pred_std', 'label_std', 'informative_pair_frac',
                       'intrinsic_grad_norm', 'pair_weight_sum', 'pair_weight_mean',
                       'head_weight_norm')
_INFORMATIVE_DELTA = 0.05


def _add_collapse_diagnostics(loss_dic, scores, labels, loss, near_cut):
    """Per-batch score/label spread + intrinsic ranking gradient, for collapse triage.

    Emitted on EVERY step (zeros when undefined) so all DDP ranks share the keyset
    — reduce_epoch_loss only all-reduces keys common to every rank. Purely
    observational: nothing here feeds the optimizer.
    """
    for _k in _COLLAPSE_DIAG_KEYS:
        loss_dic.setdefault(_k, 0.0)
    try:
        s = scores.detach().float().reshape(-1)
        l = torch.as_tensor(labels).detach().float().reshape(-1)
        if s.numel() != l.numel() or s.numel() < 2:
            return
        fin = torch.isfinite(s) & torch.isfinite(l)
        if int(fin.sum()) < 2:
            return
        sv, lv = s[fin], l[fin]
        loss_dic['pred_std'] = float(sv.std())
        loss_dic['label_std'] = float(lv.std())
        n = sv.numel()
        ii, jj = torch.triu_indices(n, n, offset=1, device=sv.device)
        loss_dic['informative_pair_frac'] = float(
            ((lv[ii] - lv[jj]).abs() > _INFORMATIVE_DELTA).float().mean())
        # SML contributes only near x non pairs, so that product IS the pair budget
        near = (lv >= float(near_cut))
        w = float(near.sum()) * float((~near).sum())
        loss_dic['pair_weight_sum'] = w
        loss_dic['pair_weight_mean'] = w / max(n * (n - 1) / 2.0, 1.0)
        # gradient of the intrinsic ranking loss wrt the score vector
        if isinstance(loss, torch.Tensor) and loss.requires_grad and scores.requires_grad:
            g = torch.autograd.grad(loss, scores, retain_graph=True,
                                    create_graph=False, allow_unused=True)[0]
            if g is not None:
                loss_dic['intrinsic_grad_norm'] = float(g.detach().float().norm())
    except Exception:
        pass


def _head_weight_norm(model):
    """||W|| of the linear layer that emits pred['out'] (linear_out_1)."""
    net = model.module if hasattr(model, 'module') else model
    w = getattr(net, 'linear_out_1', None)
    try:
        return float(w.weight.detach().float().norm()) if w is not None else 0.0
    except Exception:
        return 0.0


def run_epoch(model,dataloader, epoch_idx, grad_scaler, optimizer, local_rank,
        callbacks,is_train,args):
    epoch_loss = initialize_epoch_loss()
    ##
    if is_train: train_tag='train';training=True;inference=False
    else:train_tag='valid';training=False;inference=False
    ##
    loss_fn=Sujin_loss()
    # Optional H3-lDDT ordinal auxiliary head (v0 second objective). When
    # --use_ord_aux_loss is set, the pretrain path also supervises P(H3_lddt >=
    # 0.6/0.8/0.9) from the per-decoy h3_lddt eval channel, so the objective is
    # "rank by cdr_lddt (SML) + predict h3_lddt (ordinal aux)".
    ord_loss_fn = None
    if getattr(args, 'use_ord_aux_loss', False):
        ord_loss_fn = OrdinalH3LddtAuxLoss(
            cutoff_mode=args.ord_cutoff_mode,
            cutoff_a=args.ord_cutoff_a,
            cutoff_b=args.ord_cutoff_b,
            cutoff_c=args.ord_cutoff_c,
        )
    # Optional interface-compatibility head (v2 1st-priority): WITHIN-TARGET soft
    # RankNet on pred['interface'] against fnat (native antibody-antigen contact
    # fraction). Multi-task with the intrinsic SML on the shared backbone. apo/GP
    # decoys have NaN fnat and are masked inside InterfaceSoftRankLoss. One batch
    # == one target here, so the per-target pair mean IS the per-batch loss and
    # the epoch mean over batches is the target-wise mean.
    iface_loss_fn = None
    _lambda_iface = float(getattr(args, 'lambda_interface', 0.0) or 0.0)
    _iface_tau = float(getattr(args, 'interface_tau_fnat', 0.1) or 0.1)
    _tier_balanced = getattr(args, 'tier_balanced_rank', False)
    _af3_matched = getattr(args, 'af3_matched_pairs', False)
    _cdr_matched = getattr(args, 'cdr_matched_pairs', False)
    if getattr(args, 'use_interface_head', False) and _lambda_iface > 0.0:
        if _cdr_matched:
            # exp5: budget goes to pairs that look alike on cdr_lddt but differ
            # sharply on fnat — the only pairs where the interface head can carry
            # information the intrinsic head does not.
            iface_loss_fn = CdrMatchedFnatContrastLoss(
                tau_fnat=_iface_tau,
                dead_zone=float(getattr(args, 'fnat_dead_zone', 0.05)),
                lddt_match_tol=float(getattr(args, 'lddt_match_tol', 0.05)),
                fnat_contrast_min=float(getattr(args, 'fnat_contrast_min', 0.2)),
                cond_frac=float(getattr(args, 'cond_pair_frac', 0.60)),
                min_cond_pairs=int(getattr(args, 'min_cond_pairs', 8)),
            )
        elif _af3_matched:
            # exp3: drop dead-zone pairs, then weight the survivors by the AF3
            # cell frequency raised to -power (0.5 = 1/sqrt(p)).
            iface_loss_fn = Af3MatchedTierPairLoss(
                tau_fnat=_iface_tau,
                dead_zone=float(getattr(args, 'fnat_dead_zone', 0.05)),
                power=float(getattr(args, 'cell_weight_power', 0.5)),
            )
        elif _tier_balanced:
            # exp1 variant: same continuous soft-rank target, but pair MASS is
            # allocated by absolute fnat tier and by generation source instead of
            # by however many pairs each combination happens to contribute.
            iface_loss_fn = TierBalancedSoftRankLoss(
                tau_fnat=_iface_tau,
                inter_tier_frac=float(getattr(args, 'tier_inter_frac', 0.75)),
                same_source_frac=float(getattr(args, 'tier_same_source_frac', 0.75)),
                mode=str(getattr(args, 'tier_pair_mode', 'weight')),
                pairs_per_target=int(getattr(args, 'tier_pairs_per_target', 512)),
                seed=int(getattr(args, 'seed', 0) or 0),
            )
        else:
            # exp4 uses this branch with a dead zone but no cell weighting
            iface_loss_fn = InterfaceSoftRankLoss(
                tau_fnat=_iface_tau,
                dead_zone=float(getattr(args, 'softrank_dead_zone', 0.0) or 0.0))
    # Optional absolute-fnat regression auxiliary on the SAME interface logit
    # (experiment 2). The rank loss only sees s_i - s_j and so leaves the head's
    # absolute scale free; this grounds sigmoid(s) on fnat itself. Disabled at
    # lambda 0 so experiment 1's configuration is unchanged.
    fnat_reg_loss_fn = None
    _lambda_fnat_reg = float(getattr(args, 'lambda_fnat_reg', 0.0) or 0.0)
    _fnat_reg_beta = float(getattr(args, 'fnat_reg_beta', 0.1) or 0.1)
    if iface_loss_fn is not None and _lambda_fnat_reg > 0.0:
        # SAME tau as the rank loss: tau*s is the scale the rank loss already
        # learns (delta_s -> delta_fnat/tau), so the two terms cannot disagree.
        fnat_reg_loss_fn = InterfaceFnatRegLoss(beta=_fnat_reg_beta, tau_fnat=_iface_tau)
    # Effective label metric / near-native cutoff (set from the YAML spec after
    # CLI-env override). Drives loss-direction + cutoff for loop_rmsd vs loop_lddt.
    _label_metric = getattr(args, 'label_metric', None) or 'loop_rmsd'
    _near_cut = getattr(args, 'near_native_cutoff', None)
    if _near_cut is None:
        _near_cut = 0.8 if 'lddt' in _label_metric.lower() else 2.0
    info_dict={}
    _head_w = _head_weight_norm(model)
    profiler = _TrainingProfiler(
        getattr(args, "profile_training", False),
        getattr(args, "profile_log_interval", 20),
        local_rank,
    )
    last_step_end = time.perf_counter()
    
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), unit='batch',
                         #desc=f'{train_tag} Epoch {epoch_idx}', disable=(True)):
                         desc=f'{train_tag} Epoch {epoch_idx}', disable=(local_rank != 0)):
        batch_ready_t = time.perf_counter()
        dataloader_time = batch_ready_t - last_step_end
        _cuda_sync()
        step_start_t = time.perf_counter()
        transfer_start_t = step_start_t
        # NOTE: `rmsd_s` is the per-decoy TRAINING LABEL for the configured
        # label_metric, NOT necessarily an RMSD. With label_metric=loop_lddt it
        # holds cdr_lddt (higher-is-better, range [0,1]). The legacy top{k}_rmsd
        # loss keys are likewise the label metric, not Angstrom RMSD.
        batched_graph, rmsd_s = to_cuda(batch[0:2])
        _cuda_sync()
        transfer_time = time.perf_counter() - transfer_start_t
        pdb = batch[2]
        eval_metrics = _extract_eval_metrics(batch)
        node_info = _batch_node_info(batched_graph)
        graph_stats = _graph_batch_stats(batched_graph)
        if _should_skip_oversized_batch(batched_graph, pdb, epoch_idx, i, train_tag, args, local_rank):
            del batched_graph, rmsd_s
            torch.cuda.empty_cache()
            last_step_end = time.perf_counter()
            continue
        try:
            for callback in callbacks:
                callback.on_batch_start()
            with torch.cuda.amp.autocast(enabled=args.amp):
                _cuda_sync()
                forward_start_t = time.perf_counter()
                pred = model(batched_graph)
                _cuda_sync()
                forward_time = time.perf_counter() - forward_start_t
                device=pred['out'].device

                # TODO: Add loss function for nodewise score
                if args.nodewise_score:
                    nodewise_score = pred['nodewise_score']
                    print('nodewise_score ',len(nodewise_score), nodewise_score[0].shape)

                _cuda_sync()
                loss_start_t = time.perf_counter()
                # exp6b/exp6all: 2-D tier SML. near = T1 (good on BOTH axes),
                # non = T3 (bad on both); T2 (loop right / pose wrong) and T4 are
                # dropped from the loss entirely. Off unless --sml_tier_2d is set.
                # exp8 (--sml_fnat_gated) supersedes the 2-D tier path: the pose
                # filter (fnat) and the loop cutoffs are separate, and PertMD_all's
                # share of the non class is capped. See SoftMarginLoss_rmsd.
                _fg = bool(getattr(args, 'sml_fnat_gated', False))
                _t2d = bool(getattr(args, 'sml_tier_2d', False)) and not _fg
                _tier_l = float(getattr(args, 'sml_tier_lddt', 0.8)) if _t2d else None
                _tier_f = float(getattr(args, 'sml_tier_fnat', 0.5)) if _t2d else None
                _tier_lo = float(getattr(args, 'sml_tier_lddt_low', 0.70)) if _t2d else None
                _sml_fnat = (_metric_tensor(eval_metrics, 'fnat', device)
                             if (_t2d or _fg) else None)
                _near_c = float(getattr(args, 'sml_near_cut', 0.90)) if _fg else None
                _non_c = float(getattr(args, 'sml_non_cut', 0.85)) if _fg else None
                _fnat_g = float(getattr(args, 'sml_fnat_gate', 0.50)) if _fg else None
                _dem_ids = getattr(args, '_sml_demote_ids', None) if _fg else None
                _sml_sid = (_metric_tensor(eval_metrics, 'source_id', device)
                            if (_fg and _dem_ids) else None)
                loss,loss_dic=loss_fn(pdb,rmsd_s,pred['out'],training,device,inference,rmsd_cutoff=_near_cut,loss_type=args.loss_type,label_metric=_label_metric,
                                      fnat=_sml_fnat, tier_lddt=_tier_l, tier_fnat=_tier_f,
                                      tier_lddt_low=_tier_lo,
                                      fnat_gate=_fnat_g, near_cut=_near_c, non_cut=_non_c,
                                      source_id=_sml_sid, demote_ids=_dem_ids,
                                      demote_max_frac=float(getattr(args, 'sml_demote_max_frac', 0.5)))
                # ── collapse diagnostics (logging only; does not touch the loss) ──
                # Names match analyze/diag_collapse.py so a live run and an offline
                # checkpoint sweep can be read on the same axes.
                _add_collapse_diagnostics(loss_dic, pred['out'], rmsd_s, loss, _near_cut)
                loss_dic['head_weight_norm'] = _head_w
                if ord_loss_fn is not None:
                    _h3_q = _metric_tensor(eval_metrics, 'h3_lddt', device)
                    loss = _apply_ord_aux_loss(loss, loss_dic, pred, _h3_q, None, ord_loss_fn, args, device)
                # Interface-compatibility multi-task term: within-target soft RankNet
                # on pred['interface'] against fnat. Always touch the interface head so
                # DDP sees it every step; add the real loss only when the target keeps
                # at least one valid pair. The five log keys are emitted on EVERY step
                # (even as zeros) so all DDP ranks hold the same keyset - reduce_epoch_loss
                # only all-reduces keys common to every rank.
                if 'interface' in pred:
                    if iface_loss_fn is not None:
                        _fnat = _metric_tensor(eval_metrics, 'fnat', device)
                        _if_loss = None
                        _reg_loss = None
                        _if_stats = {'num_rank_targets': 0.0, 'num_rank_pairs': 0.0,
                                     'mean_abs_fnat_pair_delta': 0.0, 'mean_pair_confidence': 0.0,
                                     'frac_inter_tier': 0.0, 'frac_cross_source': 0.0,
                                     'n_tier_combos': 0.0, 'frac_quota_redistributed': 0.0,
                                     'frac_pairs_kept': 0.0, 'n_cells_used': 0.0,
                                     'n_targets_all_dead': 0.0, 'n_cond_pairs': 0.0,
                                     'frac_cond_mass': 0.0, 'frac_cond_same_source': 0.0,
                                     'mean_abs_lddt_delta_cond': 0.0, 'n_targets_no_cond': 0.0}
                        _reg_stats = {'num_reg_targets': 0.0, 'num_reg_decoys': 0.0,
                                      'mean_abs_fnat_err': 0.0, 'mean_fnat_pred': 0.0,
                                      'frac_pred_out_of_range': 0.0}
                        if _fnat is not None and _fnat.numel() == pred['interface'].numel():
                            if _cdr_matched:
                                # rmsd_s IS cdr_lddt here (label_metric=loop_lddt)
                                _sid = _metric_tensor(eval_metrics, 'source_id', device)
                                _if_loss, _if_stats = iface_loss_fn(
                                    _fnat, pred['interface'], rmsd_s, source_id=_sid)
                            elif _af3_matched:
                                _if_loss, _if_stats = iface_loss_fn(_fnat, pred['interface'])
                            elif _tier_balanced:
                                _sid = _metric_tensor(eval_metrics, 'source_id', device)
                                _if_loss, _if_stats = iface_loss_fn(
                                    _fnat, pred['interface'], source_id=_sid)
                            else:
                                _if_loss, _if_stats = iface_loss_fn(_fnat, pred['interface'])
                            if fnat_reg_loss_fn is not None:
                                _reg_loss, _reg_stats = fnat_reg_loss_fn(_fnat, pred['interface'])
                        # L_interface = L_softrank + lambda_fnat_reg * L_reg, then the
                        # existing lambda_interface scales the whole interface objective.
                        _iface_total = None
                        for _term, _w in ((_if_loss, 1.0), (_reg_loss, _lambda_fnat_reg)):
                            if isinstance(_term, torch.Tensor) and _term.requires_grad:
                                _contrib = _w * _term
                                _iface_total = _contrib if _iface_total is None else _iface_total + _contrib
                        # Per-component diagnostics. Gradient scale is taken w.r.t. the
                        # shared interface logit, the only fair comparison point: the raw
                        # BCE and SmoothL1 values are not on a common scale.
                        loss_dic['interface_soft_rank_loss'] = (
                            _if_loss.detach() if isinstance(_if_loss, torch.Tensor) else 0.0)
                        loss_dic['interface_fnat_reg_loss'] = (
                            _reg_loss.detach() if isinstance(_reg_loss, torch.Tensor) else 0.0)
                        loss_dic['grad_scale_soft_rank'] = grad_scale_wrt(_if_loss, pred['interface'])
                        loss_dic['grad_scale_fnat_reg'] = grad_scale_wrt(_reg_loss, pred['interface'])
                        for _sk in ('num_rank_targets', 'num_rank_pairs',
                                    'mean_abs_fnat_pair_delta', 'mean_pair_confidence',
                                    'frac_inter_tier', 'frac_cross_source',
                                    'n_tier_combos', 'frac_quota_redistributed',
                                    'frac_pairs_kept', 'n_cells_used',
                                    'n_targets_all_dead', 'n_cond_pairs',
                                    'frac_cond_mass', 'frac_cond_same_source',
                                    'mean_abs_lddt_delta_cond', 'n_targets_no_cond'):
                            loss_dic[_sk] = float(_if_stats.get(_sk, 0.0))
                        for _sk in ('num_reg_targets', 'num_reg_decoys',
                                    'mean_abs_fnat_err', 'mean_fnat_pred',
                                    'frac_pred_out_of_range'):
                            loss_dic[_sk] = float(_reg_stats[_sk])
                        if _iface_total is not None:
                            loss = loss + _lambda_iface * _iface_total
                        else:
                            loss = loss + 0.0 * pred['interface'].sum()
                    else:
                        loss = loss + 0.0 * pred['interface'].sum()
                if not training:
                    _add_eval_top1_metrics(loss_dic, eval_metrics, pred['out'], _label_metric)
                # Detach GPU tensors in loss_dic to prevent holding computation graph
                info_dict[pdb] = {k: (v.detach().cpu().item() if isinstance(v, torch.Tensor) else v) for k, v in loss_dic.items()}
                loss = loss/args.accumulate_grad_batches
                _cuda_sync()
                loss_time = time.perf_counter() - loss_start_t
            ###
            epoch_loss=update_epoch_loss(epoch_loss,loss_dic)
            backward_time = 0.0
            optimizer_time = 0.0
            if is_train:
                def _zero_dummy_pretrain():
                    d = pred['out'].sum() * 0.0
                    ol = pred.get('ord_logits')
                    if ol is not None:
                        d = d + ol.sum() * 0.0
                    ifc = pred.get('interface')
                    if ifc is not None:
                        d = d + ifc.sum() * 0.0
                    return d

                if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
                    loss = _zero_dummy_pretrain()
                elif torch.isnan(loss):
                    print('save nan pred and rmsd for that target as pickle')
                    nan_path = _run_result_dir(args) / 'nan'
                    nan_path.mkdir(parents=True, exist_ok=True)
                    with open(nan_path / f'{pdb}.dat','wb')as fp:
                        pickle.dump([pred['out'].detach().cpu(),rmsd_s.detach().cpu()],fp)
                    loss = _zero_dummy_pretrain()
                _cuda_sync()
                backward_start_t = time.perf_counter()
                grad_scaler.scale(loss).backward()
                _cuda_sync()
                backward_time = time.perf_counter() - backward_start_t
                # Free GPU memory from this batch before the next optimizer step
                del batched_graph, pred, loss_dic
                if (i + 1) % args.accumulate_grad_batches == 0 or (i + 1) == len(dataloader):
                    _cuda_sync()
                    optimizer_start_t = time.perf_counter()
                    if args.gradient_clip:
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    _cuda_sync()
                    optimizer_time = time.perf_counter() - optimizer_start_t
                del loss
                torch.cuda.empty_cache()
            else:
                del batched_graph, pred, loss, loss_dic
                torch.cuda.empty_cache()
            _cuda_sync()
            total_time = time.perf_counter() - step_start_t
            profiler.add(
                {
                    "dataloader": dataloader_time,
                    "transfer": transfer_time,
                    "forward": forward_time,
                    "loss": loss_time,
                    "backward": backward_time,
                    "optimizer": optimizer_time,
                    "total": total_time,
                },
                graph_stats,
            )
            last_step_end = time.perf_counter()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                msg = (
                    f'[OOM] train loop failed at pdb={pdb}, epoch={epoch_idx}, '
                    f'batch_idx={i}, mode={train_tag}, {node_info}'
                )
                print(msg, flush=True)
                torch.cuda.empty_cache()
                raise RuntimeError(msg) from e
            raise
    profiler.log()
    with torch.no_grad():
        epoch_loss=finalize_epoch_loss(epoch_loss)
    result_dir = _run_result_dir(args)
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / f'{train_tag}.{epoch_idx}.{local_rank}.info', 'wb') as fp:
        pickle.dump(info_dict,fp)
    return epoch_loss

def train(model: nn.Module,
          callbacks: List[BaseCallback],
          logger: Logger,
          args):
    device = torch.cuda.current_device()
    model.to(device=device)
    local_rank = get_local_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    data_module = HUDataModule()
    if dist.is_initialized():
        # NOTE: no static_graph here. The H3 ordinal aux head participates via a
        # per-batch-varying path (real BCE when H3 labels exist, else a
        # zero-magnitude touch), so the autograd graph changes across iterations,
        # which is incompatible with static_graph=True under DDP. Use
        # find_unused_parameters=True instead.
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
        )
    model.train()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    optimizer = AdamW(_param_groups(model, args, get_local_rank(), 'pretrain'),
                      lr=args.learning_rate, betas=(args.momentum, 0.999),
                      weight_decay=args.weight_decay)

    for callback in callbacks:
        callback.on_fit_start(optimizer, args)
    epoch_start = 0
    if args.load_ckpt_path and os.path.exists(args.load_ckpt_path):
        epoch_start = load_state(model, optimizer, args.load_ckpt_path, callbacks)
        if get_local_rank() == 0:
            print(f'load checkpoint model from path: {args.load_ckpt_path} (epoch={epoch_start})')
    elif os.path.exists('last.pt'):
        epoch_start = load_state(model, optimizer, 'last.pt', callbacks)
        if get_local_rank() == 0:
            print(f'load checkpoint model from path: last.pt (epoch={epoch_start})')
    result_dir = _run_result_dir(args)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Fine-tune init: WEIGHTS ONLY, fresh optimizer, epoch 0. `--load_ckpt_path`
    # cannot be used for this because load_state() also restores the optimizer
    # state and the epoch counter, i.e. it resumes A's LR schedule and momentum
    # instead of starting a new run from A's parameters.
    if getattr(args, 'init_weights_from', None) and os.path.exists(args.init_weights_from):
        _ck = torch.load(args.init_weights_from,
                         map_location={'cuda:0': f'cuda:{get_local_rank()}'})
        _sd = _ck['state_dict'] if 'state_dict' in _ck else _ck
        _tgt = model.module if isinstance(model, DistributedDataParallel) else model
        _missing, _unexpected = _tgt.load_state_dict(_sd, strict=False)
        epoch_start = 0
        if get_local_rank() == 0:
            print(f'[init_weights_from] loaded WEIGHTS ONLY from {args.init_weights_from} '
                  f'(source epoch={_ck.get("epoch", "?")}), starting at epoch 0 with a '
                  f'fresh optimizer')
            if _missing:
                print(f'[init_weights_from] missing keys ({len(_missing)}): {_missing[:8]}')
            if _unexpected:
                print(f'[init_weights_from] unexpected keys ({len(_unexpected)}): {_unexpected[:8]}')

    # get checkpoint model  (skipped when --init_weights_from already seeded the model,
    # otherwise this would silently overwrite the fine-tune init)
    if (not getattr(args, 'init_weights_from', None)) and \
            epoch_start == 0 and args.save_model_path and os.path.exists(args.save_model_path):
        checkpoint = torch.load(args.save_model_path, map_location={'cuda:0': f'cuda:{get_local_rank()}'})
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        print(f'load checkpoint model from path: {args.save_model_path}')
        
    NUM_GP=3057; NUM_ABAG=1462
    dataset_config = getattr(args, 'dataset_config', None)
    excluded_pdbs = _load_excluded_pdbs(args)
    if excluded_pdbs and get_local_rank() == 0:
        print(f'[list exclusion] loaded {len(excluded_pdbs)} excluded pdb ids')
    # Load YAML spec once (used for list paths and forwarded to MyDataset)
    _yaml_spec = None
    if dataset_config:
        _libs_dir = _CURRENT_LIBS_DIR
        if _libs_dir not in sys.path:
            sys.path.insert(0, _libs_dir)
        from dataset.config import load_dataset_spec
        _yaml_spec = load_dataset_spec(dataset_config)
        # Effective label metric / cutoff (post CLI-env override) → args, so the
        # loss loop (run_epoch) flips direction consistently with the dataset.
        args.label_metric = _yaml_spec.label_metric
        args.near_native_cutoff = _yaml_spec.effective_near_native_cutoff()
        if get_local_rank() == 0:
            print(f'[label_metric] {args.label_metric} '
                  f'(higher_is_better={_yaml_spec.label_higher_is_better()}, '
                  f'near_native_cutoff={args.near_native_cutoff})')
        args._gen_source_ids = _resolve_gen_source_ids(
            args, _yaml_spec, get_local_rank(), 'pretrain')
        args._sml_demote_ids = _resolve_named_source_ids(
            getattr(args, 'sml_demote_sources', ''), _yaml_spec, get_local_rank(),
            'pretrain', required=bool(getattr(args, 'sml_fnat_gated', False)))
        if get_local_rank() == 0 and getattr(args, 'sml_fnat_gated', False):
            print(f'[pretrain exp8] SML fnat-gated: drop finite fnat<={args.sml_fnat_gate}, '
                  f'near>={args.sml_near_cut}, non<={args.sml_non_cut}, '
                  f'dead band ({args.sml_non_cut}, {args.sml_near_cut}); '
                  f'demote ids={args._sml_demote_ids} max_frac={args.sml_demote_max_frac}')
        args.profile_training = bool(getattr(_yaml_spec, "profile_training", getattr(args, "profile_training", False)))
        args.profile_log_interval = int(getattr(_yaml_spec, "profile_log_interval", getattr(args, "profile_log_interval", 20)))
        if get_local_rank() == 0 and args.profile_training:
            print(f'[TRAIN_PROFILE] enabled interval={args.profile_log_interval}', flush=True)

    yaml_train_abag_pool = None
    yaml_valid_abag_pool = None
    if _yaml_spec and (_yaml_spec.train_list or getattr(_yaml_spec, "train_list_from_pkl", "")):
        yaml_train_abag_pool, yaml_valid_abag_pool, yaml_meta = _prepare_yaml_abag_pools(_yaml_spec, excluded_pdbs=excluded_pdbs)
        if get_local_rank() == 0:
            print(
                f"[YAML split] source={yaml_meta['valid_source']}, "
                f"abag_all={yaml_meta['abag_all']}, train_pool={yaml_meta['train_pool']}, "
                f"valid_pool={yaml_meta['valid_pool']}, overlap_after={yaml_meta['overlap_after']}, "
                f"split={yaml_meta['split_enabled']} ratio={yaml_meta['split_ratio']}, "
                f"excluded(train/valid)=({yaml_meta['n_excluded_train_pool']}/{yaml_meta['n_excluded_valid_pool']})"
            )

    end_epoch = epoch_start + int(args.epochs)
    if get_local_rank() == 0:
        print(f'[pretrain] epochs: running {args.epochs} epoch(s), '
              f'{epoch_start + 1} -> {end_epoch}', flush=True)
    for epoch_idx in range(epoch_start + 1, end_epoch + 1):
        # ── Build training / validation lists ──
        if _yaml_spec and (_yaml_spec.train_list or getattr(_yaml_spec, "train_list_from_pkl", "")):
            # YAML-driven: read abag + gp lists, random sample like set_data()
            _epoch_rng = random.Random(epoch_idx)
            random.seed(epoch_idx)
            abag_all = yaml_train_abag_pool if yaml_train_abag_pool is not None else []
            num_abag = _yaml_spec.num_abag if _yaml_spec.num_abag is not None else len(abag_all)
            num_abag = min(num_abag, len(abag_all))
            gp_all = _read_list_file(_yaml_spec.gp_list) if _yaml_spec.gp_list else []
            num_gp = (_yaml_spec.num_gp if _yaml_spec.num_gp is not None else len(gp_all)) if gp_all else 0
            num_gp = min(num_gp, len(gp_all))
            _pretrain_phase = getattr(args, 'pretrain_phase', None)
            if _pretrain_phase in ('A', 'B'):
                # v2 target-level GP/holo/apo ratio (explicit A/B; no auto switch).
                _total = num_abag + num_gp
                training_list, _pinfo = _sample_pretrain_targets(
                    _pretrain_phase, abag_all, gp_all, _yaml_spec, _epoch_rng, _total)
                if epoch_idx == epoch_start + 1 and get_local_rank() == 0:
                    print(f'[pretrain_phase {_pretrain_phase}] target mix '
                          f'GP/holo/apo = {_pinfo["n_gp"]}/{_pinfo["n_holo"]}/{_pinfo["n_apo"]} '
                          f'(pools gp={_pinfo["gp_pool"]} holo={_pinfo["holo_pool"]} apo={_pinfo["apo_pool"]}, '
                          f'total={len(training_list)})')
            else:
                training_list = random.sample(abag_all, k=num_abag) if num_abag > 0 else []
                if gp_all and num_gp > 0:
                    training_list += random.sample(gp_all, k=num_gp)
                random.shuffle(training_list)
                if epoch_idx == epoch_start + 1 and get_local_rank() == 0:
                    print(f'[YAML] abag: {num_abag}/{len(abag_all)}, gp: {num_gp}/{len(gp_all)}, total: {len(training_list)}')
            valid_all = list(yaml_valid_abag_pool) if yaml_valid_abag_pool is not None else []
            num_valid = (
                _yaml_spec.num_valid_abag
                if _yaml_spec.num_valid_abag is not None
                else len(valid_all)
            )
            num_valid = min(num_valid, len(valid_all))
            validation_list = (
                random.sample(valid_all, k=num_valid) if num_valid > 0 else []
            )
        else:
            # Fallback: original hardcoded lists
            training_list, validation_list = set_data('train',1000,1000,epoch_idx,args) # set to be (1000,1000)
            if excluded_pdbs:
                training_list, n_excl_train = _apply_pdb_exclusion(training_list, excluded_pdbs)
                validation_list, n_excl_valid = _apply_pdb_exclusion(validation_list, excluded_pdbs)
                if get_local_rank() == 0 and (n_excl_train or n_excl_valid):
                    print(f'[list exclusion fallback] removed train={n_excl_train}, valid={n_excl_valid}')
        train_dataloader=data_module.train_dataloader(training_list,args.decoytype, dataset_config=dataset_config)
        # propagate epoch to YAML-driven dataset so schedulable params update
        if data_module.ds_train is not None and hasattr(data_module.ds_train, 'set_epoch'):
            data_module.ds_train.set_epoch(epoch_idx)
        #### Training
        model.train()
        is_train=True
        epoch_loss=run_epoch(model,train_dataloader, epoch_idx, grad_scaler, optimizer,
                local_rank, callbacks,is_train,args)
        if dist.is_initialized():
            epoch_loss=reduce_epoch_loss(epoch_loss,world_size)
        if get_local_rank()==0:
            report_epoch_loss(epoch_loss,epoch_idx,mode='train',out_tag=args.param_name)
        args.save_ckpt_path = result_dir / f'{args.param_name}_{epoch_idx}.pt'
        # NOTE: this used to force ckpt_interval=2, silently ignoring --ckpt_interval.
        # The checkpoint cadence must be <= the validation cadence, otherwise a
        # "best" epoch identified by validation has no checkpoint to load.
        _ckpt_iv = max(1, int(getattr(args, 'ckpt_interval', 1) or 1))
        if epoch_idx % _ckpt_iv == 0:
            save_state(model, optimizer, epoch_idx, args.save_ckpt_path, callbacks)
        if args.wandb:
            current_lr = optimizer.param_groups[0]['lr']
            _train_log = {"train/loss": epoch_loss['final_loss'],
                "learning_rate": current_lr,
                "epoch": epoch_idx}
            # H3 ordinal auxiliary (v1/v2 pretrain 2nd objective), guarded by presence.
            for _k, _name in (("loss_aux_ord", "train/loss_aux_ord"),
                              ("mono_penalty", "train/mono_penalty"),
                              ("n_aux_valid",  "train/n_aux_valid"),
                              # interface soft-rank diagnostics (present only when the
                              # interface head is active); num_rank_* are per-target means
                              ("interface_soft_rank_loss", "train/interface_soft_rank_loss"),
                              ("mean_abs_fnat_pair_delta", "train/mean_abs_fnat_pair_delta"),
                              ("mean_pair_confidence",     "train/mean_pair_confidence"),
                              ("num_rank_targets",         "train/num_rank_targets"),
                              ("num_rank_pairs",           "train/num_rank_pairs"),
                              ("frac_inter_tier",         "train/frac_inter_tier"),
                              ("frac_cross_source",       "train/frac_cross_source"),
                              ("n_tier_combos",           "train/n_tier_combos"),
                              ("frac_quota_redistributed","train/frac_quota_redistributed"),
                              ("frac_pairs_kept",      "train/frac_pairs_kept"),
                              ("n_cells_used",         "train/n_cells_used"),
                              ("n_targets_all_dead",   "train/n_targets_all_dead"),
                              ("n_cond_pairs",         "train/n_cond_pairs"),
                              ("frac_cond_mass",       "train/frac_cond_mass"),
                              ("frac_cond_same_source","train/frac_cond_same_source"),
                              ("mean_abs_lddt_delta_cond","train/mean_abs_lddt_delta_cond"),
                              ("n_targets_no_cond",    "train/n_targets_no_cond"),
                              # collapse diagnostics
                              ("pred_std",             "train/pred_std"),
                              ("label_std",            "train/label_std"),
                              ("informative_pair_frac","train/informative_pair_frac"),
                              ("intrinsic_grad_norm",  "train/intrinsic_grad_norm"),
                              ("pair_weight_sum",      "train/pair_weight_sum"),
                              ("pair_weight_mean",     "train/pair_weight_mean"),
                              ("head_weight_norm",     "train/head_weight_norm"),
                              # absolute-fnat regression auxiliary (experiment 2)
                              ("interface_fnat_reg_loss", "train/interface_fnat_reg_loss"),
                              ("grad_scale_soft_rank",    "train/grad_scale_soft_rank"),
                              ("grad_scale_fnat_reg",     "train/grad_scale_fnat_reg"),
                              ("mean_abs_fnat_err",       "train/mean_abs_fnat_err"),
                              ("mean_fnat_pred",          "train/mean_fnat_pred"),
                              ("num_reg_targets",         "train/num_reg_targets"),
                              ("num_reg_decoys",          "train/num_reg_decoys"),
                              ("frac_pred_out_of_range",  "train/frac_pred_out_of_range"),
                              # exp7: 2-D tier SML class sizes. sml_n_T3 is the one to
                              # watch -- when it hits 0 the loss silently returns 0 for
                              # that target, so a falling T3 count means training is
                              # quietly doing less than the loss curve suggests.
                              ("sml_n_T1",            "train/sml_n_T1"),
                              ("sml_n_T3",            "train/sml_n_T3"),
                              ("sml_n_skipped",       "train/sml_n_skipped"),
                              ("sml_tier_fallback",   "train/sml_tier_fallback"),
                              # exp8: how much the two gates actually remove.
                              ("sml_n_fnat_gated",      "train/sml_n_fnat_gated"),
                              ("sml_n_demote_dropped",  "train/sml_n_demote_dropped")):
                if _k in epoch_loss:
                    _v = epoch_loss[_k]
                    _train_log[_name] = _v.item() if hasattr(_v, "item") else _v
            logger.log_metrics(_train_log)
        
        #### Validation
        # was hardcoded `% 2`, which made --eval_interval a dead flag.
        _eval_iv = max(1, int(getattr(args, 'eval_interval', 1) or 1))
        if epoch_idx % _eval_iv == 0:
            model.eval()
            is_train=False
            valid_dataloader=data_module.val_dataloader(validation_list,args.decoytype, dataset_config=dataset_config)
            if data_module.ds_val is not None and hasattr(data_module.ds_val, 'set_epoch'):
                data_module.ds_val.set_epoch(epoch_idx)
            with torch.no_grad(), warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='None of the inputs have requires_grad=True')
                epoch_loss=run_epoch(model,valid_dataloader, epoch_idx, grad_scaler, optimizer,
                        local_rank, callbacks,is_train,args)
            if dist.is_initialized():
                epoch_loss=reduce_epoch_loss(epoch_loss,world_size)
            if get_local_rank()==0:
                report_epoch_loss(epoch_loss,epoch_idx,mode='valid',out_tag=args.param_name)
            if args.wandb:
                current_lr = optimizer.param_groups[0]['lr']
                log_dict = {
                    "valid/loss": epoch_loss['final_loss'],
                    "valid/best_rank": epoch_loss['best_rank'],
                    "learning_rate": current_lr,
                    "epoch": epoch_idx,
                }
                # The legacy top{k}_rmsd keys actually hold the label metric
                # (cdr_lddt when label_metric=loop_lddt), so log them under clear
                # cdr_lddt names, and add the H3 / DockQ / CAPRI validation
                # metrics computed by _add_eval_top1_metrics. Guarded by presence
                # so missing keys (e.g. no holo targets) are simply skipped.
                _valid_log_aliases = {
                    "top1_rmsd":  "valid/top1_cdr_lddt",
                    "top3_rmsd":  "valid/top3_cdr_lddt",
                    "top5_rmsd":  "valid/top5_cdr_lddt",
                    "top10_rmsd": "valid/top10_cdr_lddt",
                    "eval_oracle_loop_lddt": "valid/oracle_cdr_lddt",
                    "eval_loop_lddt_regret": "valid/cdr_lddt_regret",
                    "eval_top1_h3_lddt":     "valid/top1_h3_lddt",
                    "eval_oracle_h3_lddt":   "valid/oracle_h3_lddt",
                    "eval_top1_dockq":       "valid/top1_dockq",
                    "eval_oracle_dockq":     "valid/oracle_dockq",
                    "eval_top1_capri_acceptable": "valid/top1_capri_acceptable",
                    "eval_top1_capri_medium":     "valid/top1_capri_medium",
                    "eval_top1_capri_high":       "valid/top1_capri_high",
                    # interface soft-rank diagnostics (interface head only)
                    "interface_soft_rank_loss": "valid/interface_soft_rank_loss",
                    "mean_abs_fnat_pair_delta": "valid/mean_abs_fnat_pair_delta",
                    "mean_pair_confidence":     "valid/mean_pair_confidence",
                    "num_rank_targets":         "valid/num_rank_targets",
                    "num_rank_pairs":           "valid/num_rank_pairs",
                    "frac_inter_tier":          "valid/frac_inter_tier",
                    "frac_cross_source":        "valid/frac_cross_source",
                    "n_tier_combos":            "valid/n_tier_combos",
                    "frac_quota_redistributed": "valid/frac_quota_redistributed",
                    "frac_pairs_kept":       "valid/frac_pairs_kept",
                    "n_cells_used":          "valid/n_cells_used",
                    "n_targets_all_dead":    "valid/n_targets_all_dead",
                    "n_cond_pairs":          "valid/n_cond_pairs",
                    "frac_cond_mass":        "valid/frac_cond_mass",
                    "frac_cond_same_source": "valid/frac_cond_same_source",
                    "mean_abs_lddt_delta_cond":"valid/mean_abs_lddt_delta_cond",
                    "n_targets_no_cond":     "valid/n_targets_no_cond",
                    "pred_std":              "valid/pred_std",
                    "label_std":             "valid/label_std",
                    "informative_pair_frac": "valid/informative_pair_frac",
                    "intrinsic_grad_norm":   "valid/intrinsic_grad_norm",
                    "pair_weight_sum":       "valid/pair_weight_sum",
                    "pair_weight_mean":      "valid/pair_weight_mean",
                    "head_weight_norm":      "valid/head_weight_norm",
                    # absolute-fnat regression auxiliary (experiment 2)
                    "interface_fnat_reg_loss": "valid/interface_fnat_reg_loss",
                    "mean_abs_fnat_err":       "valid/mean_abs_fnat_err",
                    "mean_fnat_pred":          "valid/mean_fnat_pred",
                    "num_reg_targets":         "valid/num_reg_targets",
                    "num_reg_decoys":          "valid/num_reg_decoys",
                    "frac_pred_out_of_range":  "valid/frac_pred_out_of_range",
                }
                for _k, _name in _valid_log_aliases.items():
                    if _k in epoch_loss:
                        _v = epoch_loss[_k]
                        log_dict[_name] = _v.item() if hasattr(_v, "item") else _v
                logger.log_metrics(log_dict)
            # v2 fixed-manifest validation (both heads on identical candidate pool).
            # ALL ranks must enter: targets are sharded and rows all_gathered, so a
            # rank-0-only call would starve the other ranks (NCCL watchdog abort).
            _run_manifest_pools(model, args, dataset_config, device, epoch_idx, logger)
        for callback in callbacks:
            callback.on_epoch_end()
    for callback in callbacks:
        callback.on_fit_end()



def _auto_phase(epoch_idx, total_epochs):
    """Derive pair sampling phase from training progress."""
    if total_epochs <= 0:
        return 2
    frac = epoch_idx / total_epochs
    if frac <= 0.33:
        return 1
    elif frac <= 0.66:
        return 2
    return 3


def _get_enabled_phase_config(args):
    if not getattr(args, 'use_phase_config', False):
        return None
    return get_phase_config(int(getattr(args, 'current_phase', 1)))


# ── v2 pretrain target-level source phase (GP/holo/apo ratio) ──
_PRETRAIN_PHASE_RATIOS = {          # (GP, holo, apo)
    'A': (0.40, 0.40, 0.20),
    'B': (0.20, 0.65, 0.15),
}
_HOLO_SET_CACHE = {}


def _load_holo_set(spec):
    """Set of holo (antigen-bearing) target_ids in the parquet (new) namespace.

    holo := target has at least one finite DockQ in the Boltz2_s10n10 store
    (which covers ~all targets). Cached per db_root.
    """
    key = getattr(spec, 'db_root', '') or ''
    if key in _HOLO_SET_CACHE:
        return _HOLO_SET_CACHE[key]
    holo = set()
    try:
        import pandas as pd
        pm = getattr(spec, 'precomputed_metrics', None)
        sub = pm.sources.get('Boltz2_s10n10') if (pm and pm.sources) else None
        if sub:
            p = os.path.join(pm.root, sub, 'metrics', 'dockq_metrics.parquet')
            df = pd.read_parquet(p, columns=['target_id', 'dockq'])
            g = df.groupby('target_id')['dockq'].apply(lambda s: s.notna().any())
            holo = set(g[g].index.astype(str))
    except Exception as e:
        logging.warning('[pretrain_phase] holo-set load failed: %s', e)
    _HOLO_SET_CACHE[key] = holo
    return holo


def _sample_pretrain_targets(phase, abag_all, gp_all, spec, rng, total):
    """Sample a target list hitting the GP/holo/apo ratio for the given phase.

    holo/apo are per-target (antigen present or not); GP vs antibody is list-level.
    Returns (target_list, info_dict). Sampling is without replacement when the
    pool is large enough, else with replacement (small pools).
    """
    r_gp, r_holo, r_apo = _PRETRAIN_PHASE_RATIOS[phase]
    holo_set = _load_holo_set(spec)
    holo_ab, apo_ab = [], []
    for t in abag_all:
        nid = spec.resolve_pdb_id(t, 'new') if getattr(spec, 'pdb_id_mapping', None) else t
        (holo_ab if str(nid) in holo_set else apo_ab).append(t)
    n_gp = round(total * r_gp)
    n_holo = round(total * r_holo)
    n_apo = total - n_gp - n_holo

    def _samp(pool, k):
        if not pool or k <= 0:
            return []
        if k <= len(pool):
            return [pool[i] for i in rng.sample(range(len(pool)), k)]
        return [pool[rng.randrange(len(pool))] for _ in range(k)]

    lst = _samp(gp_all, n_gp) + _samp(holo_ab, n_holo) + _samp(apo_ab, n_apo)
    rng.shuffle(lst)
    info = dict(phase=phase, n_gp=n_gp, n_holo=n_holo, n_apo=n_apo,
                holo_pool=len(holo_ab), apo_pool=len(apo_ab), gp_pool=len(gp_all))
    return lst, info


def _manifest_validate(net, manifest_path, pool, dataset_config, args, device, epoch_idx,
                       shard_rank=0, shard_world=1):
    """Evaluate this rank's SHARD of a FIXED manifest pool.

    intrinsic and final heads are scored on the SAME forward batch per target.
    Returns (per_target_rows, manifest_meta); the caller all_gathers the rows and
    takes a target-wise equal-weight mean of the score-argmax top1 — no label
    dedup, no tier/source-weight/phase influence (manifest is fixed)."""
    from data_loading.data_module import MyDataset
    ds = MyDataset([], is_train=False, dataset_config=dataset_config)
    meta = ds.load_val_manifest(manifest_path)
    ds.set_epoch(epoch_idx)
    net.eval()
    # i_*/f_*/x_* = intrinsic / final / interface head top1 selections.
    # For each head we log the picked decoy's cdr_lddt, h3_lddt, fnat, dockq so we
    # can see whether the interface head (x_) selects a better docking pose than
    # the loop-quality heads. oracle_* are per-target ceilings for each metric.
    acc = {k: [] for k in (
        'i_cdr', 'f_cdr', 'x_cdr', 'delta', 'oracle', 'i_gap', 'f_gap', 'ncand',
        'i_h3', 'f_h3', 'x_h3', 'oracle_h3',
        'i_fnat', 'f_fnat', 'x_fnat', 'oracle_fnat',
        'i_dockq', 'f_dockq', 'x_dockq', 'oracle_dockq',
        'n_holo',
        # exp6 gen-gen restricted metrics (checkpoint selection)
        'g_i_cdr', 'g_f_cdr', 'g_oracle', 'g_mean', 'g_ncand',
        'g_pair_acc', 'g_pair_acc_final', 'g_n_pairs',
        # exp9 checkpoint-selection signal: pair accuracy restricted to the TOP
        # region (both decoys >= 0.85). The full-range number cannot tell exp8's
        # frozen top apart from a model that actually orders it, because the
        # <=0.85 vs >=0.90 step alone already scores ~0.55-0.60 there.
        'g_top_pair_acc', 'g_top_n_pairs')}
    # DDP: shard targets across ranks (rank r takes idx r, r+world, ...). Every rank
    # runs this, so no rank sits idle in a collective while another evaluates.
    _idxs = range(shard_rank, len(ds.inp_dat), shard_world) if shard_world > 1 else range(len(ds.inp_dat))
    with torch.no_grad():
        for idx in _idxs:
            try:
                item = ds[idx]
            except Exception:
                continue
            graph, cdr_label = item[0], item[1]
            em = next((it for it in item[3:] if isinstance(it, dict)), None)
            if em is None:
                continue
            graph = graph.to(device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = net(graph)
            s_i = pred['out'].detach().float().reshape(-1)
            s_f = pred.get('final', pred['out']).detach().float().reshape(-1)
            s_x = pred.get('interface', pred['out']).detach().float().reshape(-1)
            cdr = em['loop_lddt'].float().reshape(-1).to(s_i.device)
            h3 = em['h3_lddt'].float().reshape(-1).to(s_i.device) if 'h3_lddt' in em else None
            fnat = em['fnat'].float().reshape(-1).to(s_i.device) if 'fnat' in em else None
            dockq = em['dockq'].float().reshape(-1).to(s_i.device) if 'dockq' in em else None
            n = cdr.numel()
            if n == 0 or s_i.numel() != n:
                continue
            fin = torch.isfinite(cdr)
            if not fin.any():
                continue
            neg = torch.full_like(s_i, -1e9)
            i_top = int(torch.argmax(torch.where(torch.isfinite(s_i), s_i, neg)))
            f_top = int(torch.argmax(torch.where(torch.isfinite(s_f), s_f, neg)))
            x_top = int(torch.argmax(torch.where(torch.isfinite(s_x), s_x, neg)))
            oracle = float(cdr[fin].max())
            i_cdr = float(cdr[i_top]); f_cdr = float(cdr[f_top]); x_cdr = float(cdr[x_top])
            acc['i_cdr'].append(i_cdr); acc['f_cdr'].append(f_cdr); acc['x_cdr'].append(x_cdr)
            acc['delta'].append(f_cdr - i_cdr)   # per-target final - intrinsic
            acc['oracle'].append(oracle)
            acc['i_gap'].append(oracle - i_cdr)
            acc['f_gap'].append(oracle - f_cdr)
            acc['ncand'].append(int(n))

            # ── exp6: the SAME metrics restricted to generative candidates ──
            # This is the checkpoint-selection signal. The multisource numbers above
            # are dominated by perturbation decoys, which are trivially rankable
            # (90.6 % pair accuracy vs 64.3 % gen-gen) and therefore saturate without
            # tracking test behaviour at all. Everything here is suffixed g_/gen_.
            _sid = em.get('source_id')
            _gen_ids = getattr(args, '_gen_source_ids', None)
            if _sid is not None and _gen_ids:
                _sid = _sid.float().reshape(-1).to(s_i.device)
                if _sid.numel() == n:
                    gmask = torch.zeros_like(_sid, dtype=torch.bool)
                    for _g in _gen_ids:
                        gmask |= (_sid.round().long() == int(_g))
                    gmask &= fin
                    if int(gmask.sum()) >= 2:
                        _gneg = torch.full_like(s_i, -1e9)
                        gi = int(torch.argmax(torch.where(gmask & torch.isfinite(s_i), s_i, _gneg)))
                        gf = int(torch.argmax(torch.where(gmask & torch.isfinite(s_f), s_f, _gneg)))
                        g_or = float(cdr[gmask].max())
                        acc['g_i_cdr'].append(float(cdr[gi]))
                        acc['g_f_cdr'].append(float(cdr[gf]))
                        acc['g_oracle'].append(g_or)
                        acc['g_mean'].append(float(cdr[gmask].mean()))
                        acc['g_ncand'].append(int(gmask.sum()))
                        # Pairwise ranking accuracy on informative gen-gen pairs --
                        # the exact quantity the AF3/Boltz2 diagnosis reports. Computed
                        # for BOTH heads: DPO trains `final`, so that is the selection
                        # signal, while `intrinsic` shows whether the SML anchor drifts.
                        for _tag, _sv in (('g_pair_acc', s_i), ('g_pair_acc_final', s_f)):
                            _y = cdr[gmask]
                            _s = _sv[gmask]
                            _ok = torch.isfinite(_s)
                            _y2, _s2 = _y[_ok], _s[_ok]
                            if _y2.numel() < 2:
                                continue
                            dy = _y2.unsqueeze(1) - _y2.unsqueeze(0)
                            dsc = _s2.unsqueeze(1) - _s2.unsqueeze(0)
                            iu = torch.triu_indices(_y2.numel(), _y2.numel(), offset=1)
                            dy = dy[iu[0], iu[1]]; dsc = dsc[iu[0], iu[1]]
                            infm = dy.abs() >= _INFORMATIVE_DELTA
                            if int(infm.sum()) > 0:
                                corr = (torch.sign(dsc[infm]) == torch.sign(dy[infm])).float()
                                acc[_tag].append(float(corr.mean()))
                                if _tag == 'g_pair_acc':
                                    acc['g_n_pairs'].append(int(infm.sum()))
                                    # exp9: same pairs, restricted to both decoys
                                    # inside the top region
                                    _tc = float(getattr(args, 'dpo_top_region_cut', 0.85))
                                    _ytop = (_y2 >= _tc)
                                    _both = (_ytop.unsqueeze(1) & _ytop.unsqueeze(0))[iu[0], iu[1]]
                                    _tm = infm & _both
                                    if int(_tm.sum()) > 0:
                                        _tcorr = (torch.sign(dsc[_tm]) == torch.sign(dy[_tm])).float()
                                        acc['g_top_pair_acc'].append(float(_tcorr.mean()))
                                        acc['g_top_n_pairs'].append(int(_tm.sum()))

            # ── exp8 diagnostic: per-decoy dump for the first N manifest targets ──
            # Answers "what happens to the 0.85-0.90 dead band?" — those decoys get
            # no gradient from SML, so whether their score lands between the two
            # trained classes or drifts is only visible by looking. Rank-sharded
            # (each rank writes its own file) and gated on the manifest index so the
            # SAME targets are dumped every epoch regardless of world size.
            _ssn = int(getattr(args, 'score_scatter_targets', 0) or 0)
            if _ssn > 0 and idx < _ssn:
                try:
                    _sd = (getattr(args, 'score_scatter_dir', None)
                           or os.path.join(str(getattr(args, 'log_dir', '.')), 'score_scatter'))
                    os.makedirs(_sd, exist_ok=True)
                    _fp = os.path.join(_sd, f'{pool}_ep{int(epoch_idx):03d}_rank{shard_rank}.csv')
                    _tgt = str(ds.inp_dat[idx])      # manifest inp_dat is a target-id list
                    _sid_v = em.get('source_id')
                    _sid_v = (_sid_v.float().reshape(-1) if _sid_v is not None else None)
                    _new = not os.path.exists(_fp)
                    with open(_fp, 'a') as _fh:
                        if _new:
                            _fh.write('epoch,pool,target,idx,decoy,source_id,cdr_lddt,'
                                      'fnat,score_intrinsic,score_final,score_interface\n')
                        for _j in range(n):
                            _f_v = float(fnat[_j]) if fnat is not None and _j < fnat.numel() else float('nan')
                            _s_v = float(_sid_v[_j]) if _sid_v is not None and _j < _sid_v.numel() else float('nan')
                            _fh.write(f'{int(epoch_idx)},{pool},{_tgt},{idx},{_j},{_s_v:.0f},'
                                      f'{float(cdr[_j]):.6f},{_f_v:.6f},'
                                      f'{float(s_i[_j]):.6f},{float(s_f[_j]):.6f},'
                                      f'{float(s_x[_j]):.6f}\n')
                except Exception as _e:
                    logging.warning('[score_scatter] %s: %s', pool, _e)

            def _pick(vec, top):
                if vec is None or top >= vec.numel():
                    return None
                v = vec[top]
                return float(v) if torch.isfinite(v) else None

            for _tag, _top in (('i', i_top), ('f', f_top), ('x', x_top)):
                hv = _pick(h3, _top)
                if hv is not None:
                    acc[f'{_tag}_h3'].append(hv)
                fv = _pick(fnat, _top)
                if fv is not None:
                    acc[f'{_tag}_fnat'].append(fv)
                dv = _pick(dockq, _top)
                if dv is not None:
                    acc[f'{_tag}_dockq'].append(dv)
            # per-target oracles for the interface metrics (holo targets only)
            if h3 is not None:
                _h3f = h3[torch.isfinite(h3)]
                if _h3f.numel():
                    acc['oracle_h3'].append(float(_h3f.max()))
            if fnat is not None:
                _ff = fnat[torch.isfinite(fnat)]
                if _ff.numel():
                    acc['oracle_fnat'].append(float(_ff.max()))
                    acc['n_holo'].append(1)   # target has an interface (holo)
            if dockq is not None:
                _df = dockq[torch.isfinite(dockq)]
                if _df.numel():
                    acc['oracle_dockq'].append(float(_df.max()))

    # Return this rank's per-target rows; aggregation happens after all_gather.
    return acc, meta


def _run_manifest_pools(model, args, dataset_config, device, epoch_idx, logger):
    """Evaluate the fixed manifest pools and log to W&B.

    MUST be called by EVERY rank: targets are sharded across ranks and the
    per-target rows are all_gathered, so the collectives stay symmetric. (A
    rank-0-only evaluation starves the other ranks and trips the NCCL watchdog.)
    Aggregation is a target-wise equal-weight mean over the gathered rows.
    """
    import numpy as _np
    interval = int(getattr(args, 'manifest_val_interval', 1) or 1)
    if interval > 0 and (epoch_idx % interval) != 0:
        return {}
    net = model.module if hasattr(model, 'module') else model
    rank = get_local_rank()
    world = dist.get_world_size() if dist.is_initialized() else 1
    # Pool list must be identical on all ranks (same args) so the number of
    # all_gather calls matches across ranks.
    pools = [(p, path) for p, path in
             (('multisource', getattr(args, 'val_manifest_multisource', None)),
              ('boltz2', getattr(args, 'val_manifest_boltz2', None)))
             if path and os.path.exists(path)]
    all_metrics = {}
    for pool, path in pools:
        try:
            acc, meta = _manifest_validate(net, path, pool, dataset_config, args,
                                           device, epoch_idx, rank, world)
        except Exception as e:
            logging.warning('[manifest_val:%s] rank%d failed: %s', pool, rank, e)
            acc, meta = {k: [] for k in ('i_cdr', 'f_cdr', 'delta', 'i_h3', 'f_h3',
                                         'oracle', 'i_gap', 'f_gap', 'ncand')}, {}
        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, acc)
            merged = {k: [] for k in acc}
            for part in gathered:
                if not part:
                    continue
                for k in merged:
                    merged[k].extend(part.get(k, []))
        else:
            merged = acc

        def _m(x):
            return float(_np.mean(x)) if x else float('nan')
        nc = merged['ncand']
        metrics = {
            f'val/{pool}/intrinsic_top1_cdr_lddt': _m(merged['i_cdr']),
            f'val/{pool}/final_top1_cdr_lddt': _m(merged['f_cdr']),
            f'val/{pool}/final_minus_intrinsic_top1_cdr_lddt': _m(merged['delta']),
            f'val/{pool}/intrinsic_top1_h3_lddt': _m(merged['i_h3']),
            f'val/{pool}/final_top1_h3_lddt': _m(merged['f_h3']),
            f'val/{pool}/oracle_top1_cdr_lddt': _m(merged['oracle']),
            f'val/{pool}/intrinsic_cdr_lddt_gap': _m(merged['i_gap']),
            f'val/{pool}/final_cdr_lddt_gap': _m(merged['f_gap']),
            f'val/{pool}/n_targets': float(len(merged['i_cdr'])),
            f'val/{pool}/candidates_mean': _m(nc),
            f'val/{pool}/candidates_min': float(min(nc)) if nc else float('nan'),
            f'val/{pool}/candidates_max': float(max(nc)) if nc else float('nan'),
            # interface head (x_) top1 + fnat/dockq of every head's top1 pick
            # (holo targets only; apo has no fnat/dockq). This is where we see
            # whether the interface head selects a better docking pose.
            f'val/{pool}/interface_top1_cdr_lddt': _m(merged['x_cdr']),
            f'val/{pool}/interface_top1_h3_lddt': _m(merged['x_h3']),
            f'val/{pool}/n_holo': float(len(merged['n_holo'])),
            f'val/{pool}/intrinsic_top1_fnat': _m(merged['i_fnat']),
            f'val/{pool}/final_top1_fnat': _m(merged['f_fnat']),
            f'val/{pool}/interface_top1_fnat': _m(merged['x_fnat']),
            f'val/{pool}/oracle_top1_fnat': _m(merged['oracle_fnat']),
            f'val/{pool}/intrinsic_top1_dockq': _m(merged['i_dockq']),
            f'val/{pool}/final_top1_dockq': _m(merged['f_dockq']),
            f'val/{pool}/interface_top1_dockq': _m(merged['x_dockq']),
            f'val/{pool}/oracle_top1_dockq': _m(merged['oracle_dockq']),
            f'val/{pool}/oracle_top1_h3_lddt': _m(merged['oracle_h3']),
            # ── exp6 gen-gen restricted metrics ──
            # `gen_pair_acc` is the checkpoint-selection signal: it is the same
            # quantity the AF3/Boltz2 test diagnosis reports, so validation and test
            # finally measure one thing. `gen_top1_minus_mean` is the advantage over
            # picking at random from the same generative pool -- on post-2021 test the
            # model scores BELOW that baseline, which is what has to move.
            f'val/{pool}/gen_intrinsic_top1_cdr_lddt': _m(merged['g_i_cdr']),
            f'val/{pool}/gen_final_top1_cdr_lddt': _m(merged['g_f_cdr']),
            f'val/{pool}/gen_oracle_top1_cdr_lddt': _m(merged['g_oracle']),
            f'val/{pool}/gen_pool_mean_cdr_lddt': _m(merged['g_mean']),
            f'val/{pool}/gen_top1_minus_mean': _m([a - b for a, b in
                                                   zip(merged['g_i_cdr'], merged['g_mean'])]),
            f'val/{pool}/gen_pair_acc': _m(merged['g_pair_acc']),
            f'val/{pool}/gen_pair_acc_final': _m(merged['g_pair_acc_final']),
            f'val/{pool}/gen_final_top1_minus_mean': _m([a - b for a, b in
                                                        zip(merged['g_f_cdr'], merged['g_mean'])]),
            f'val/{pool}/gen_n_targets': float(len(merged['g_i_cdr'])),
            f'val/{pool}/gen_n_pairs': float(sum(merged['g_n_pairs']) if merged['g_n_pairs'] else 0),
            f'val/{pool}/gen_candidates_mean': _m(merged['g_ncand']),
            # exp9 selection signal (see the acc-dict comment)
            f'val/{pool}/gen_top_pair_acc': _m(merged['g_top_pair_acc']),
            f'val/{pool}/gen_top_n_pairs': float(
                sum(merged['g_top_n_pairs']) if merged['g_top_n_pairs'] else 0),
            f'val/{pool}/gen_top_n_targets': float(len(merged['g_top_pair_acc'])),
        }
        all_metrics.update(metrics)
        if rank == 0:
            print(f'  [manifest_val:{pool}] hash={meta.get("hash")} '
                  f'n_targets={metrics[f"val/{pool}/n_targets"]:.0f} '
                  f'cand(mean/min/max)={metrics[f"val/{pool}/candidates_mean"]:.1f}/'
                  f'{metrics[f"val/{pool}/candidates_min"]:.0f}/{metrics[f"val/{pool}/candidates_max"]:.0f} '
                  f'intrinsic_top1={metrics[f"val/{pool}/intrinsic_top1_cdr_lddt"]:.4f} '
                  f'final_top1={metrics[f"val/{pool}/final_top1_cdr_lddt"]:.4f} '
                  f'delta={metrics[f"val/{pool}/final_minus_intrinsic_top1_cdr_lddt"]:+.4f} '
                  f'oracle={metrics[f"val/{pool}/oracle_top1_cdr_lddt"]:.4f}', flush=True)
            print(f'  [manifest_val:{pool}] TOP-REGION '
                  f'n={metrics[f"val/{pool}/gen_top_n_targets"]:.0f} '
                  f'top_pair_acc={metrics[f"val/{pool}/gen_top_pair_acc"]:.4f} '
                  f'n_pairs={metrics[f"val/{pool}/gen_top_n_pairs"]:.0f}', flush=True)
            print(f'  [manifest_val:{pool}] GEN-GEN n={metrics[f"val/{pool}/gen_n_targets"]:.0f} '
                  f'pair_acc(intr/final)={metrics[f"val/{pool}/gen_pair_acc"]:.4f}/'
                  f'{metrics[f"val/{pool}/gen_pair_acc_final"]:.4f} '
                  f'top1(intr/final)={metrics[f"val/{pool}/gen_intrinsic_top1_cdr_lddt"]:.4f}/'
                  f'{metrics[f"val/{pool}/gen_final_top1_cdr_lddt"]:.4f} '
                  f'pool_mean={metrics[f"val/{pool}/gen_pool_mean_cdr_lddt"]:.4f} '
                  f'(final-mean={metrics[f"val/{pool}/gen_final_top1_minus_mean"]:+.4f}) '
                  f'oracle={metrics[f"val/{pool}/gen_oracle_top1_cdr_lddt"]:.4f} '
                  f'n_pairs={metrics[f"val/{pool}/gen_n_pairs"]:.0f}', flush=True)
            print(f'  [manifest_val:{pool}] n_holo={metrics[f"val/{pool}/n_holo"]:.0f} '
                  f'fnat(intr/final/iface/oracle)='
                  f'{metrics[f"val/{pool}/intrinsic_top1_fnat"]:.3f}/'
                  f'{metrics[f"val/{pool}/final_top1_fnat"]:.3f}/'
                  f'{metrics[f"val/{pool}/interface_top1_fnat"]:.3f}/'
                  f'{metrics[f"val/{pool}/oracle_top1_fnat"]:.3f}  '
                  f'dockq(intr/iface/oracle)='
                  f'{metrics[f"val/{pool}/intrinsic_top1_dockq"]:.3f}/'
                  f'{metrics[f"val/{pool}/interface_top1_dockq"]:.3f}/'
                  f'{metrics[f"val/{pool}/oracle_top1_dockq"]:.3f}', flush=True)
    if all_metrics and rank == 0 and getattr(args, 'wandb', False):
        all_metrics['epoch'] = epoch_idx
        logger.log_metrics(all_metrics)
    return all_metrics


def _apply_ord_aux_loss(loss, out, pi, h3_lddt_s, h3_loop_len, ord_loss_fn, args, device):
    """Add ordinal H3 lDDT aux loss when enabled and valid decoys exist.

    IMPORTANT: with DDP static_graph, ord_head must always participate in the
    backward graph so that gradient all-reduce covers ord_head parameters in
    every iteration.  When the real aux loss cannot be computed we add a
    zero-magnitude term ``0 * ord_logits.sum()`` to keep the gradient path.
    """
    ord_logits = pi.get('ord_logits')

    if ord_loss_fn is None or h3_lddt_s is None:
        if ord_logits is not None:
            loss = loss + 0.0 * ord_logits.sum()
        out['loss_aux_ord'] = 0.0
        out['mono_penalty'] = 0.0
        out['n_aux_valid'] = 0.0
        return loss
    if ord_logits is None:
        out['loss_aux_ord'] = 0.0
        out['mono_penalty'] = 0.0
        out['n_aux_valid'] = 0.0
        return loss
    aux_ord, mono, n_valid = ord_loss_fn(ord_logits, h3_lddt_s)
    out['loss_aux_ord'] = aux_ord.detach()
    out['mono_penalty'] = mono.detach()
    out['n_aux_valid'] = float(n_valid)
    out['ord_cutoff_60'] = float(args.ord_cutoff_a)
    out['ord_cutoff_80'] = float(args.ord_cutoff_b)
    out['ord_cutoff_90'] = float(args.ord_cutoff_c)
    if n_valid > 0:
        loss = loss + args.lambda_aux_ord * aux_ord + args.lambda_mono * mono
    else:
        loss = loss + 0.0 * ord_logits.sum()
    return loss


def run_epoch_dpo(model, pre_trained, loss_fn, dataloader, epoch_idx,
                  grad_scaler, optimizer, local_rank, callbacks, is_train, args,
                  tier_dpo=None, pair_cfg=None, ord_loss_fn=None):
    """Batch loop for DPO finetuning: policy model vs frozen reference model.

    When *tier_dpo* is not None, tier-based eject/top DPO losses are computed
    from pair_sampling output and added to the total loss.
    """
    epoch_loss = initialize_epoch_loss()
    for _k in ('dpo_eject_loss', 'dpo_top_loss', 'n_eject_pairs', 'n_top_pairs'):
        epoch_loss[_k] = []
    for _k in ('compactness_loss', 'compactness_n_pairs', 'compactness_gap_mean',
               'compactness_gap_max', 'compactness_ordering_viol', 'compactness_band_viol'):
        epoch_loss[_k] = []
    for _k in ('dpo_eject_weighted', 'dpo_top_weighted', 'compactness_weighted',
               'compactness_skipped_no_x_or_a'):
        epoch_loss[_k] = []
    for _k in ('tier_n_X', 'tier_n_A', 'tier_n_B', 'tier_n_C', 'tier_n_D'):
        epoch_loss[_k] = []
    # exp6 gen-gen / AF3-matched pair diagnostics
    for _k in ('dpo_within_loss', 'dpo_within_weighted', 'n_within_pairs',
               'gen_n_decoys', 'gen_n_pool', 'gen_n_candidate_pairs',
               'gen_n_cells_present'):
        epoch_loss[_k] = []
    for _cell in AF3_LDDT_TIER_PAIR_P:
        epoch_loss[f'paircell_{_cell.replace("-", "")}'] = []
    train_tag = 'train' if is_train else 'valid'
    is_training = is_train
    info_dict = {}
    phase_cfg = _get_enabled_phase_config(args)
    # v2 Phase C: rank on the final head (`final`) instead of intrinsic (`out`).
    # The frozen reference always contributes its trained intrinsic score (`out`).
    _pol_key = 'final' if getattr(args, 'use_multihead', False) else 'out'
    # exp9: DPO must train the head test inference reads. exp6 trained `final`,
    # which is not the deployed scorer; --dpo_policy_head intrinsic forces `out`.
    if str(getattr(args, 'dpo_policy_head', 'final')) == 'intrinsic':
        _pol_key = 'out'
    _top_region = bool(getattr(args, 'dpo_top_region', False))
    _hib = 'lddt' in str(getattr(args, 'label_metric', '') or '').lower()
    _near_cut = getattr(args, 'near_native_cutoff', None)

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), unit='batch',
                         desc=f'{train_tag} Epoch {epoch_idx}', disable=(local_rank != 0)):
        batched_graph, rmsd_s = to_cuda(batch[0:2])
        pdb = batch[2]
        device = rmsd_s.device
        eval_metrics = _extract_eval_metrics(batch)
        total_non_xtal_pool = batch[3] if len(batch) > 3 else None
        h3_lddt_s = None
        h3_loop_len = None
        if (
            len(batch) > 4
            and isinstance(batch[4], torch.Tensor)
            and batch[4].numel() > 1
        ):
            h3_lddt_s = batch[4].to(device=device)
            if len(batch) > 5 and isinstance(batch[5], torch.Tensor):
                h3_loop_len = batch[5].to(device=device)
        node_info = _batch_node_info(batched_graph)
        if _should_skip_oversized_batch(batched_graph, pdb, epoch_idx, i, train_tag, args, local_rank):
            del batched_graph, rmsd_s
            torch.cuda.empty_cache()
            continue

        try:
            for callback in callbacks:
                callback.on_batch_start()

            with torch.cuda.amp.autocast(enabled=args.amp):
                pi = model(batched_graph)
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=True), warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='None of the inputs have requires_grad=True')
                    ref = pre_trained(batched_graph)

            # Extract ref outputs (cast FP16→FP32) and free graph/ref memory early
            ref_ns = [t.float() for t in ref['nodewise_score']]
            ref_out_raw = ref['out'].float()
            del ref, batched_graph
            torch.cuda.empty_cache()

            pi_ns = pi['nodewise_score']
            sizes = set(t.size(0) for t in pi_ns)
            if len(sizes) > 1 and local_rank == 0:
                print(f'[nodewise pad] {pdb}: ULR node sizes vary {sizes}')
            if phase_cfg is not None:
                base_loss, out, loss_components = loss_fn.forward(
                    pi_ns, ref_ns,
                    pi[_pol_key], rmsd_s, pdb, device, is_training,
                    return_components=True,
                    higher_is_better=_hib, near_native_cutoff=_near_cut,
                )
            else:
                # exp9: stabilise with the SAME SML rule exp8 was pretrained on.
                _smlkw = None
                if bool(getattr(args, 'sml_fnat_gated', False)):
                    _smlkw = dict(
                        fnat=_metric_tensor(eval_metrics, 'fnat', device),
                        near_cut=float(getattr(args, 'sml_near_cut', 0.90)),
                        non_cut=float(getattr(args, 'sml_non_cut', 0.85)),
                        fnat_gate=float(getattr(args, 'sml_fnat_gate', 0.50)),
                        source_id=_metric_tensor(eval_metrics, 'source_id', device),
                        demote_ids=getattr(args, '_sml_demote_ids', None),
                        demote_max_frac=float(getattr(args, 'sml_demote_max_frac', 0.5)),
                    )
                base_loss, out = loss_fn.forward(
                    pi_ns, ref_ns,
                    pi[_pol_key], rmsd_s, pdb, device, is_training,
                    higher_is_better=_hib, near_native_cutoff=_near_cut,
                    sml_kwargs=_smlkw, skip_legacy_dpo=_top_region,
                )
                loss_components = None
            del ref_ns

            # ── Tier-based DPO losses (step 2) ──
            if tier_dpo is not None and pair_cfg is not None:
                pi_out = pi[_pol_key].reshape(-1)      # squeeze() -> 0-dim when n==1
                ref_out = ref_out_raw.reshape(-1)
                del ref_out_raw
                is_xtal = detect_xtal_mask(rmsd_s, higher_is_better=_hib)
                if phase_cfg is not None:
                    phase = phase_cfg.phase_id
                else:
                    dpo_phase = getattr(args, 'dpo_phase', 0)
                    phase = dpo_phase if dpo_phase > 0 else _auto_phase(epoch_idx, args.epochs)
                pool_size = int(total_non_xtal_pool) if total_non_xtal_pool is not None else None
                _h3_lddt_for_pairs = (
                    h3_lddt_s.squeeze()
                    if h3_lddt_s is not None and h3_lddt_s.dim() > 1
                    else h3_lddt_s
                )
                # exp6: restrict the DPO pairs to same-target GENERATIVE-vs-generative
                # comparisons and match their tier-cell mix to the AF3 test set.
                # `_gen_source_ids` is resolved from the YAML by NAME in main(), because
                # source ids are indices into sorted(spec.sources) and therefore change
                # with the config.
                _gen_ids = getattr(args, '_gen_source_ids', None)
                _want_af3_pairs = bool(getattr(args, 'dpo_gen_only_pairs', False))
                if _want_af3_pairs and not _gen_ids:
                    raise RuntimeError(
                        '[exp6] --dpo_gen_only_pairs is set but args._gen_source_ids is '
                        'empty -- the source-name resolution did not run on this code '
                        'path. Refusing to fall back to the ungated eject/top sampler, '
                        'which would run the experiment with the intervention off.')
                _use_af3_pairs = _want_af3_pairs
                _gen_mask = None
                if _use_af3_pairs:
                    _sid = _metric_tensor(eval_metrics, 'source_id', device)
                    if _sid is None or _sid.numel() != rmsd_s.numel():
                        raise RuntimeError(
                            f'[exp6] source_id missing or wrong length for pdb={pdb} '
                            f'(got {None if _sid is None else _sid.numel()}, '
                            f'expected {rmsd_s.numel()}). The gen-gen pair mask cannot '
                            'be built; refusing to train on unrestricted pairs.')
                    _gm = torch.zeros_like(_sid, dtype=torch.bool)
                    for _g in _gen_ids:
                        _gm |= (_sid.round().long() == int(_g))
                    _gen_mask = _gm
                with torch.no_grad():
                    if _top_region:
                        # exp9: every pair inside the top region, where exp8's
                        # dead-band diagnostic showed no ordering ever forms.
                        pairs, pair_summary = build_top_region_pairs(
                            rmsds=rmsd_s,
                            allowed_mask=_gen_mask,
                            cfg=pair_cfg,
                            rng=np.random.default_rng(epoch_idx * 10000 + i),
                            higher_is_better=_hib,
                            region_cut=float(getattr(args, 'dpo_top_region_cut', 0.85)),
                            band_lo=float(getattr(args, 'dpo_top_band_lo', 0.90)),
                            min_delta=float(getattr(args, 'dpo_top_min_delta', 0.05)),
                            min_delta_relaxed=float(getattr(args, 'dpo_top_min_delta_relaxed', 0.03)),
                            band_quota=float(getattr(args, 'dpo_top_band_quota', 0.0)),
                            is_xtal=is_xtal,
                        )
                    elif _use_af3_pairs:
                        pairs, pair_summary = build_af3_matched_pairs(
                            rmsds=rmsd_s,
                            allowed_mask=_gen_mask,
                            cfg=pair_cfg,
                            rng=np.random.default_rng(epoch_idx * 10000 + i),
                            higher_is_better=_hib,
                            min_delta=float(getattr(args, 'dpo_pair_min_delta', 0.02)),
                            is_xtal=is_xtal,
                        )
                    else:
                        pairs, pair_summary = build_training_pairs(
                            scores=pi_out.detach(),
                            rmsds=rmsd_s,
                            is_xtal=is_xtal,
                            structure_id=pdb,
                            phase=phase,
                            cfg=pair_cfg,
                            rng=random.Random(epoch_idx * 10000 + i),
                            total_non_xtal_pool_size=pool_size,
                            h3_lddt=_h3_lddt_for_pairs,
                            higher_is_better=_hib,
                        )
                eject_loss, n_eject = tier_dpo.dpo_eject_loss(pi_out, ref_out, pairs, device, higher_is_better=_hib)
                top_loss, n_top = tier_dpo.dpo_top_loss(pi_out, ref_out, pairs, device, higher_is_better=_hib)
                within_loss, n_within = tier_dpo.dpo_within_loss(
                    pi_out, ref_out, pairs, device, higher_is_better=_hib)
                # exp6 pair-composition diagnostics. Emitted on EVERY step (zeros when
                # the path is off) so all DDP ranks share the keyset - reduce_epoch_loss
                # only all-reduces keys present on every rank.
                out['n_within_pairs'] = float(n_within)
                for _k in ('top_used_delta', 'top_n_boundary_cand',
                           'top_n_inside_cand', 'top_n_boundary_used'):
                    out[_k] = float(pair_summary.get(_k, 0.0))
                out['gen_n_decoys'] = float(int(_gen_mask.sum()) if _gen_mask is not None else 0)
                out['gen_n_pool'] = float(pair_summary.get('n_pool', 0))
                out['gen_n_candidate_pairs'] = float(pair_summary.get('n_candidate_pairs', 0))
                out['gen_n_cells_present'] = float(pair_summary.get('n_cells_present', 0))
                _cc = pair_summary.get('af3_cell_counts', {}) or {}
                for _cell in AF3_LDDT_TIER_PAIR_P:
                    out[f'paircell_{_cell.replace("-", "")}'] = float(_cc.get(_cell, 0))
                del ref_out, pairs
                tier_counts = pair_summary.get('tier_counts', {})
                for _tier in ('X', 'A', 'B', 'C', 'D'):
                    out[f'tier_n_{_tier}'] = float(tier_counts.get(_tier, 0))
                # ── step 3: compactness band loss ──
                # Explicitly SKIP when the target has no crystal (tier X) or no
                # tier-A decoy: the xtal-A band is undefined without both.
                _lambda_compact = phase_cfg.lambda_compactness if phase_cfg is not None else getattr(args, 'lambda_compactness', 0.0)
                _tiers_c = pair_summary.get('tiers', {})
                _compact_skipped = not (_lambda_compact > 0.0 and _tiers_c.get('A') and _tiers_c.get('X'))
                if not _compact_skipped:
                    _delta_a = getattr(args, 'delta_A', 1.0)
                    compact_loss, compact_metrics = tier_dpo.compactness_loss(
                        pi_out, _tiers_c['X'], _tiers_c['A'], _delta_a, device,
                        higher_is_better=_hib)
                else:
                    compact_loss = torch.tensor(0.0, device=device)
                    compact_metrics = {'n_pairs_used': 0, 'x_to_a_gap_mean': 0.0,
                                       'x_to_a_gap_max': 0.0, 'ordering_violation_rate': 0.0,
                                       'band_violation_rate': 0.0}
                out['compactness_skipped_no_x_or_a'] = float(_compact_skipped)
                # exp6 adds the within-tier DPO term. Its lambda is a plain arg (not
                # part of PhaseConfig) so the phase presets stay byte-identical to the
                # C1/C2/C3 campaign; it is 0.0 unless the run asks for it.
                _lambda_within = float(getattr(args, 'lambda_dpo_within', 0.0))
                if phase_cfg is not None and loss_components is not None:
                    loss = (
                        phase_cfg.lambda_sml * loss_components['loss_sml']
                        + phase_cfg.lambda_dpo_eject * eject_loss
                        + phase_cfg.lambda_dpo_top * top_loss
                        + _lambda_within * within_loss
                        + phase_cfg.lambda_compactness * compact_loss
                    )
                else:
                    loss = (
                        base_loss
                        + args.lambda_dpo_eject * eject_loss
                        + args.lambda_dpo_top * top_loss
                        + _lambda_within * within_loss
                        + _lambda_compact * compact_loss
                    )
                out['dpo_within_loss'] = within_loss.detach()
                out['dpo_within_weighted'] = (_lambda_within * within_loss).detach()
                out['lambda_dpo_within'] = float(_lambda_within)
                out['final_loss'] = loss.detach()
                out['phase'] = float(phase)
                if phase_cfg is not None:
                    out['lambda_sml'] = float(phase_cfg.lambda_sml)
                    out['lambda_dpo_eject'] = float(phase_cfg.lambda_dpo_eject)
                    out['lambda_dpo_top'] = float(phase_cfg.lambda_dpo_top)
                    out['lambda_compactness'] = float(phase_cfg.lambda_compactness)
                out['sampler_target_eject'] = float(pair_summary['target_budget']['eject'])
                out['sampler_target_top'] = float(pair_summary['target_budget']['top'])
                out['sampler_actual_eject'] = float(pair_summary['actual_pairs']['eject'])
                out['sampler_actual_top'] = float(pair_summary['actual_pairs']['top'])
                out['sampler_hard_d_pairs'] = float(pair_summary['eject_info']['subtype_counts'].get('hard_d', 0))
                out['sampler_random_d_pairs'] = float(pair_summary['eject_info']['subtype_counts'].get('random_d', 0))
                out['sampler_xtal_or_a_vs_c_pairs'] = float(pair_summary['eject_info']['subtype_counts'].get('xtal_or_a_vs_c', 0))
                out['sampler_xtal_vs_b_pairs'] = float(pair_summary['top_info']['subtype_counts'].get('xtal_vs_b', 0))
                out['sampler_xtal_vs_c_pairs'] = float(pair_summary['top_info']['subtype_counts'].get('xtal_vs_c', 0))
                out['sampler_xtal_vs_a_pairs'] = float(pair_summary['top_info']['subtype_counts'].get('xtal_vs_a', 0))
                out['sampler_a_local_best_vs_b_pairs'] = float(pair_summary['top_info']['subtype_counts'].get('a_local_best_vs_b', 0))
                out['dpo_eject_loss'] = eject_loss.detach()
                out['dpo_top_loss'] = top_loss.detach()
                out['n_eject_pairs'] = float(n_eject)
                out['n_top_pairs'] = float(n_top)
                out['compactness_loss'] = compact_loss.detach()
                # Raw vs lambda-weighted contribution (logged separately in W&B).
                _le = phase_cfg.lambda_dpo_eject if phase_cfg is not None else getattr(args, 'lambda_dpo_eject', 1.0)
                _lt = phase_cfg.lambda_dpo_top if phase_cfg is not None else getattr(args, 'lambda_dpo_top', 1.0)
                out['dpo_eject_weighted'] = (_le * eject_loss).detach()
                out['dpo_top_weighted'] = (_lt * top_loss).detach()
                out['compactness_weighted'] = (_lambda_compact * compact_loss).detach()
                out['compactness_n_pairs'] = float(compact_metrics['n_pairs_used'])
                out['compactness_gap_mean'] = compact_metrics['x_to_a_gap_mean']
                out['compactness_gap_max'] = compact_metrics['x_to_a_gap_max']
                out['compactness_ordering_viol'] = compact_metrics['ordering_violation_rate']
                out['compactness_band_viol'] = compact_metrics['band_violation_rate']
            else:
                del ref_out_raw
                loss = base_loss
                out['dpo_eject_loss'] = 0.0
                out['dpo_top_loss'] = 0.0
                out['n_eject_pairs'] = 0.0
                out['n_top_pairs'] = 0.0
                out['dpo_eject_weighted'] = 0.0
                out['dpo_top_weighted'] = 0.0
                out['dpo_within_loss'] = 0.0
                out['dpo_within_weighted'] = 0.0
                out['lambda_dpo_within'] = 0.0
                out['n_within_pairs'] = 0.0
                out['gen_n_decoys'] = 0.0
                out['gen_n_pool'] = 0.0
                out['gen_n_candidate_pairs'] = 0.0
                out['gen_n_cells_present'] = 0.0
                for _cell in AF3_LDDT_TIER_PAIR_P:
                    out[f'paircell_{_cell.replace("-", "")}'] = 0.0
                out['compactness_weighted'] = 0.0
                out['compactness_skipped_no_x_or_a'] = 0.0
                out['compactness_loss'] = 0.0
                out['compactness_n_pairs'] = 0.0
                out['compactness_gap_mean'] = 0.0
                out['compactness_gap_max'] = 0.0
                out['compactness_ordering_viol'] = 0.0
                out['compactness_band_viol'] = 0.0
                if phase_cfg is not None:
                    out['phase'] = float(phase_cfg.phase_id)
                    out['lambda_sml'] = float(phase_cfg.lambda_sml)
                    out['lambda_dpo_eject'] = float(phase_cfg.lambda_dpo_eject)
                    out['lambda_dpo_top'] = float(phase_cfg.lambda_dpo_top)
                    out['lambda_compactness'] = float(phase_cfg.lambda_compactness)
                    out['sampler_target_eject'] = 0.0
                    out['sampler_target_top'] = 0.0
                    out['sampler_actual_eject'] = 0.0
                    out['sampler_actual_top'] = 0.0
                    out['sampler_hard_d_pairs'] = 0.0
                    out['sampler_random_d_pairs'] = 0.0
                    out['sampler_xtal_or_a_vs_c_pairs'] = 0.0
                    out['sampler_xtal_vs_b_pairs'] = 0.0
                    out['sampler_xtal_vs_c_pairs'] = 0.0
                    out['sampler_xtal_vs_a_pairs'] = 0.0
                    out['sampler_a_local_best_vs_b_pairs'] = 0.0
                    out['final_loss'] = (phase_cfg.lambda_sml * loss_components['loss_sml']).detach()
                    loss = phase_cfg.lambda_sml * loss_components['loss_sml']
                else:
                    out['final_loss'] = base_loss.detach()

            if not is_training:
                with torch.no_grad():
                    # reshape(-1), NOT squeeze(): a single-candidate target squeezes
                    # to a 0-dim tensor and Ranking() then indexes it with a list,
                    # raising IndexError. exp8's fnat gate makes such targets real --
                    # it drops unusable poses, so a validation target can be left with
                    # one decoy (e.g. 2vyr_H_X_A). Ranking is undefined there anyway,
                    # so skip rather than fabricate a top1.
                    _scores = pi[_pol_key].reshape(-1)
                    _rmsd = rmsd_s.reshape(-1)
                    _lddt = (h3_lddt_s.reshape(-1)
                             if isinstance(h3_lddt_s, torch.Tensor) else h3_lddt_s)
                    if _rmsd.numel() >= 2 and _scores.numel() == _rmsd.numel():
                        out.update(compute_top1_h3_validation_metrics(_rmsd, _scores, _lddt))

            loss = _apply_ord_aux_loss(
                loss, out, pi, h3_lddt_s, h3_loop_len, ord_loss_fn, args, device,
            )
            out['final_loss'] = loss.detach()
            if not is_training:
                _add_eval_top1_metrics(
                    out,
                    eval_metrics,
                    pi[_pol_key],
                    getattr(args, 'label_metric', None) or 'loop_rmsd',
                )
            info_dict[pdb] = {
                k: (v.detach().cpu().item() if isinstance(v, torch.Tensor) else v)
                for k, v in out.items()
            }
            loss = loss / args.accumulate_grad_batches

            epoch_loss = update_epoch_loss(epoch_loss, out)

            if is_train:
                def _zero_dummy():
                    """Dummy loss touching all heads for DDP static_graph."""
                    d = pi['out'].sum() * 0.0
                    for _hk in ('interface', 'final', 'ord_logits'):
                        _hv = pi.get(_hk)
                        if _hv is not None:
                            d = d + _hv.sum() * 0.0
                    return d

                # v2 multi-head: intrinsic (`out`) and interface heads are not in the
                # Phase-C ranking loss, so keep them in the autograd graph every step
                # (DDP static_graph requires all params to receive a gradient).
                if (getattr(args, 'use_multihead', False)
                        and isinstance(loss, torch.Tensor) and loss.requires_grad):
                    loss = loss + 0.0 * pi['out'].sum() + 0.0 * pi['interface'].sum()

                if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
                    loss = _zero_dummy()
                elif torch.isnan(loss):
                    nan_path = _run_result_dir(args) / 'nan'
                    nan_path.mkdir(parents=True, exist_ok=True)
                    with open(nan_path / f'{pdb}.dat', 'wb') as fp:
                        pickle.dump([pi['out'].detach().cpu(), rmsd_s.detach().cpu()], fp)
                    loss = _zero_dummy()
                grad_scaler.scale(loss).backward()
                del pi, out
                if (i + 1) % args.accumulate_grad_batches == 0 or (i + 1) == len(dataloader):
                    if args.gradient_clip:
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                del loss
                torch.cuda.empty_cache()
            else:
                del pi, loss, out
                torch.cuda.empty_cache()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                msg = (
                    f'[OOM] dpo loop failed at pdb={pdb}, epoch={epoch_idx}, '
                    f'batch_idx={i}, mode={train_tag}, {node_info}'
                )
                print(msg, flush=True)
                torch.cuda.empty_cache()
                raise RuntimeError(msg) from e
            raise

    with torch.no_grad():
        top1_agg = {}
        if not is_training:
            top1_agg = aggregate_top1_h3_val_metrics(epoch_loss)
            for key in TOP1_H3_VAL_LIST_KEYS:
                epoch_loss.pop(key, None)
        epoch_loss = finalize_epoch_loss(epoch_loss)
        if top1_agg:
            device = torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else torch.device('cpu')
            for key, value in top1_agg.items():
                epoch_loss[key] = torch.tensor(value, device=device)
    result_dir = _run_result_dir(args)
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / f'{train_tag}.{epoch_idx}.{local_rank}.info', 'wb') as fp:
        pickle.dump(info_dict, fp)
    return epoch_loss


def finetune(
        model: nn.Module,
        callbacks: List[BaseCallback],
        logger: Logger,
        args
        ):
    device = torch.cuda.current_device()
    local_rank = get_local_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # ── v2 Phase-C checkpoint flow: reference (frozen) vs policy-init (trainable). ──
    #   reference_ckpt : Phase B best, FIXED for all of C1/C2/C3 (the DPO reference).
    #   policy_init    : C1<-Phase B best, C2<-C1 best, C3<-C2 best (trainable start).
    # Back-compat fallbacks: reference <- save_model_path <- load_ckpt_path.
    reference_ckpt_path = (
        getattr(args, 'reference_ckpt', None)
        or args.save_model_path
        or (str(args.load_ckpt_path) if args.load_ckpt_path else None)
    )
    if not (reference_ckpt_path and os.path.exists(reference_ckpt_path)):
        raise FileNotFoundError(
            f'No valid reference checkpoint. reference_ckpt={getattr(args, "reference_ckpt", None)}, '
            f'save_model_path={args.save_model_path}'
        )
    policy_init_path = getattr(args, 'policy_init_ckpt', None) or reference_ckpt_path
    fresh_c1 = os.path.abspath(policy_init_path) == os.path.abspath(reference_ckpt_path)

    # 1) frozen reference = reference_ckpt
    ref_ck = torch.load(reference_ckpt_path, map_location={'cuda:0': f'cuda:{local_rank}'})
    model.load_state_dict(ref_ck['state_dict'], strict=False)
    pre_trained = copy.deepcopy(model)
    pre_trained.half()
    for param in pre_trained.parameters():
        param.requires_grad = False
    if local_rank == 0:
        print(f'[finetune] reference (frozen, FP16) = {reference_ckpt_path}')

    # 2) trainable policy init
    if not fresh_c1:
        pol_ck = torch.load(policy_init_path, map_location={'cuda:0': f'cuda:{local_rank}'})
        model.load_state_dict(pol_ck['state_dict'], strict=False)
        if local_rank == 0:
            print(f'[finetune] policy init = {policy_init_path} (continue prior Phase-C)')
    elif local_rank == 0:
        print(f'[finetune] policy init = reference (fresh C1 start)')

    # 3) v2: seed final head from the trained intrinsic head ONLY on a fresh C1
    #    start (a C2/C3 policy_init already carries a trained final head).
    if getattr(args, 'use_multihead', False) and fresh_c1 and hasattr(model, 'final_head'):
        with torch.no_grad():
            model.final_head.weight.copy_(model.linear_out_1.weight)
        if local_rank == 0:
            print('[finetune] v2: final_head seeded from intrinsic (fresh C1)')

    model.to(device=device)
    pre_trained.to(device=device)
    torch.cuda.empty_cache()

    optimizer = AdamW(_param_groups(model, args, local_rank, 'finetune'),
                      lr=args.learning_rate, betas=(args.momentum, 0.999),
                      weight_decay=args.weight_decay)
    epoch_start = 0

    # Optional resume for interrupted finetuning.
    # IMPORTANT: reference model remains fixed to args.save_model_path.
    if args.load_ckpt_path and os.path.exists(args.load_ckpt_path):
        resume_ckpt = torch.load(args.load_ckpt_path, map_location={'cuda:0': f'cuda:{local_rank}'})
        model.load_state_dict(resume_ckpt['state_dict'], strict=False)
        if 'optimizer_state_dict' in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        epoch_start = int(resume_ckpt.get('epoch', 0))
        if local_rank == 0:
            print(f'[finetune resume] loaded checkpoint: {args.load_ckpt_path} (epoch={epoch_start})')

    if dist.is_initialized():
        # find_unused_parameters=True (NOT static_graph): Phase-C head usage varies
        # per batch — the DPO loss touches `final` (+backbone), while degenerate/NaN
        # batches fall back to a dummy that touches other heads. static_graph=True
        # records the first iteration's param set and crashes when a later batch
        # uses a different set ("training graph has changed ... not compatible with
        # static_graph"). find_unused_parameters handles the variation, matching the
        # pretrain wrap. Small per-iter overhead, negligible for this 952k-param model.
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        if local_rank == 0:
            print('[finetune] DDP find_unused_parameters=True (no static_graph)', flush=True)

    num_epochs = args.epochs
    data_module = HUDataModule()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # DPO pair count scales with decoy count
    n_pairs = 16
    dataset_config = getattr(args, 'dataset_config', None)
    excluded_pdbs = _load_excluded_pdbs(args)
    if excluded_pdbs and local_rank == 0:
        print(f'[list exclusion] loaded {len(excluded_pdbs)} excluded pdb ids')
    _yaml_spec = None
    if dataset_config:
        _libs_dir = _CURRENT_LIBS_DIR
        if _libs_dir not in sys.path:
            sys.path.insert(0, _libs_dir)
        from dataset.config import load_dataset_spec
        _yaml_spec = load_dataset_spec(dataset_config)
        n_pairs = max(16, _yaml_spec.n_decoy // 4)
        # exp6 spreads the budget over 10 AF3 tier-pair cells, so the default 16
        # leaves the rarer cells with 0-1 pairs; --total_pair_budget overrides it.
        if int(getattr(args, 'total_pair_budget', 0) or 0) > 0:
            n_pairs = int(args.total_pair_budget)
        args.label_metric = _yaml_spec.label_metric
        args.near_native_cutoff = _yaml_spec.effective_near_native_cutoff()
        # exp6: resolve generative source NAMES -> the ids the data layer emits.
        # This MUST happen in the finetune path too -- the pretrain path loads its own
        # copy of the spec, and without this args._gen_source_ids stays unset and the
        # whole gen-gen intervention silently no-ops.
        args._gen_source_ids = _resolve_gen_source_ids(args, _yaml_spec, local_rank, 'finetune')
        args._sml_demote_ids = _resolve_named_source_ids(
            getattr(args, 'sml_demote_sources', ''), _yaml_spec, local_rank,
            'finetune', required=bool(getattr(args, 'sml_fnat_gated', False)))
        if local_rank == 0 and getattr(args, 'sml_fnat_gated', False):
            print(f'[finetune exp8] SML fnat-gated: drop finite fnat<={args.sml_fnat_gate}, '
                  f'near>={args.sml_near_cut}, non<={args.sml_non_cut}; '
                  f'demote ids={args._sml_demote_ids} max_frac={args.sml_demote_max_frac}')
        if local_rank == 0:
            print(f'[finetune] YAML dataset config loaded: {dataset_config}')
            print(f'[finetune] n_decoy={_yaml_spec.n_decoy}, n_pairs={n_pairs}')
            print(f'[finetune] label_metric={args.label_metric} near_native_cutoff={args.near_native_cutoff}')

    yaml_train_abag_pool = None
    yaml_valid_abag_pool = None
    if _yaml_spec and (_yaml_spec.train_list or getattr(_yaml_spec, "train_list_from_pkl", "")):
        yaml_train_abag_pool, yaml_valid_abag_pool, yaml_meta = _prepare_yaml_abag_pools(_yaml_spec, excluded_pdbs=excluded_pdbs)
        if local_rank == 0:
            print(
                f"[YAML split finetune] source={yaml_meta['valid_source']}, "
                f"abag_all={yaml_meta['abag_all']}, train_pool={yaml_meta['train_pool']}, "
                f"valid_pool={yaml_meta['valid_pool']}, overlap_after={yaml_meta['overlap_after']}, "
                f"split={yaml_meta['split_enabled']} ratio={yaml_meta['split_ratio']}, "
                f"excluded(train/valid)=({yaml_meta['n_excluded_train_pool']}/{yaml_meta['n_excluded_valid_pool']})"
            )

    loss_fn = TotalLoss(n_pairs=n_pairs)
    ord_loss_fn = None
    if getattr(args, 'use_ord_aux_loss', False):
        ord_loss_fn = OrdinalH3LddtAuxLoss(
            cutoff_mode=args.ord_cutoff_mode,
            cutoff_a=args.ord_cutoff_a,
            cutoff_b=args.ord_cutoff_b,
            cutoff_c=args.ord_cutoff_c,
        )
        if local_rank == 0:
            print(
                f'[finetune] ordinal H3 lDDT aux enabled: '
                f'fixed cutoffs p60/p80/p90=({args.ord_cutoff_a},{args.ord_cutoff_b},{args.ord_cutoff_c}), '
                f'lambda_aux={args.lambda_aux_ord}, lambda_mono={args.lambda_mono}'
            )
    result_dir = _run_result_dir(args)
    result_dir.mkdir(parents=True, exist_ok=True)

    # ── Tier-based DPO (optional, enabled via --use_tier_dpo) ──
    tier_dpo = None
    pair_cfg = None
    if getattr(args, 'use_tier_dpo', False):
        tier_dpo = TierDPOLoss(beta=args.dpo_beta)
        pair_cfg = PairSamplingConfig(
            total_pair_budget=n_pairs,
            use_lddt_tiers=getattr(args, 'use_lddt_tiers', False),
        )
        if getattr(args, 'use_lddt_tiers', False) and local_rank == 0:
            print('[finetune] lDDT tier mode: X=crystal, A=lddt>=0.9&rmsd<=2, '
                  'B=lddt>=0.8, C=lddt>=0.6, D=lddt<0.6; '
                  'eject=(X,A,B)vsD, top=XvsA+XvsB')
        if getattr(args, 'use_phase_config', False):
            phase_cfg = get_phase_config(args.current_phase)
            if local_rank == 0:
                print(
                    f'[finetune] phase config enabled: phase={phase_cfg.phase_id}, '
                    f'pair(eject/top)=({phase_cfg.eject_ratio:.2f}/{phase_cfg.top_ratio:.2f}), '
                    f'lambdas(sml/eject/top/compact)=('
                    f'{phase_cfg.lambda_sml:.2f}/{phase_cfg.lambda_dpo_eject:.2f}/'
                    f'{phase_cfg.lambda_dpo_top:.2f}/{phase_cfg.lambda_compactness:.2f})'
                )
        elif local_rank == 0:
            print(f'[finetune] tier DPO enabled: beta={args.dpo_beta}, '
                  f'lambda_eject={args.lambda_dpo_eject}, lambda_top={args.lambda_dpo_top}, '
                  f'pair_budget={n_pairs}, phase={args.dpo_phase or "auto"}')

    for callback in callbacks:
        callback.on_fit_start(optimizer, args)

    # `--epochs` is a COUNT of epochs to run, matching the pretrain path. It used to
    # be read here as an ABSOLUTE end epoch, so resuming from epoch 13 with
    # --epochs 47 ran only to 47 instead of 60 -- silently 13 epochs short.
    _end_epoch = epoch_start + int(num_epochs)
    if local_rank == 0:
        print(f'[finetune] epochs: running {num_epochs} epoch(s), '
              f'{epoch_start + 1} -> {_end_epoch}', flush=True)
    for epoch_idx in range(epoch_start + 1, _end_epoch + 1):
        # ── Build training / validation lists (abag only, GP excluded for finetune) ──
        if _yaml_spec and (_yaml_spec.train_list or getattr(_yaml_spec, "train_list_from_pkl", "")):
            random.seed(epoch_idx)
            abag_all = yaml_train_abag_pool if yaml_train_abag_pool is not None else []
            num_abag = _yaml_spec.num_abag if _yaml_spec.num_abag is not None else len(abag_all)
            num_abag = min(num_abag, len(abag_all))
            training_list = random.sample(abag_all, k=num_abag) if num_abag > 0 else []
            random.shuffle(training_list)
            valid_all = list(yaml_valid_abag_pool) if yaml_valid_abag_pool is not None else []
            num_valid = (
                _yaml_spec.num_valid_abag
                if _yaml_spec.num_valid_abag is not None
                else len(valid_all)
            )
            num_valid = min(num_valid, len(valid_all))
            validation_list = (
                random.sample(valid_all, k=num_valid) if num_valid > 0 else []
            )
            if epoch_idx == 1 and local_rank == 0:
                print(
                    f'[YAML finetune] abag: {num_abag}, total: {len(training_list)}, '
                    f'valid: {len(validation_list)}/{len(valid_all)}'
                )
        else:
            training_list, validation_list = set_data('train', 0, 1462, epoch_idx, args)
            if excluded_pdbs:
                training_list, n_excl_train = _apply_pdb_exclusion(training_list, excluded_pdbs)
                validation_list, n_excl_valid = _apply_pdb_exclusion(validation_list, excluded_pdbs)
                if local_rank == 0 and (n_excl_train or n_excl_valid):
                    print(f'[list exclusion fallback] removed train={n_excl_train}, valid={n_excl_valid}')

        train_dataloader = data_module.train_dataloader(training_list, args.decoytype, dataset_config=dataset_config)
        if data_module.ds_train is not None and hasattr(data_module.ds_train, 'set_epoch'):
            data_module.ds_train.set_epoch(epoch_idx)

        # ── Training ──
        model.train()
        pre_trained.eval()
        epoch_loss = run_epoch_dpo(model, pre_trained, loss_fn, train_dataloader,
                                   epoch_idx, grad_scaler, optimizer, local_rank,
                                   callbacks, is_train=True, args=args,
                                   tier_dpo=tier_dpo, pair_cfg=pair_cfg,
                                   ord_loss_fn=ord_loss_fn)
        if dist.is_initialized():
            epoch_loss = reduce_epoch_loss(epoch_loss, world_size)
        if local_rank == 0:
            report_epoch_loss(epoch_loss, epoch_idx, mode='train', out_tag=args.param_name)
            if tier_dpo is not None:
                print(f'  [TierDPO train] eject={epoch_loss["dpo_eject_loss"]:.4f} '
                      f'top={epoch_loss["dpo_top_loss"]:.4f} '
                      f'compact={epoch_loss["compactness_loss"]:.4f} '
                      f'n_eject={epoch_loss["n_eject_pairs"]:.1f} '
                      f'n_top={epoch_loss["n_top_pairs"]:.1f} '
                      f'gap_mean={epoch_loss["compactness_gap_mean"]:.3f} '
                      f'ord_viol={epoch_loss["compactness_ordering_viol"]:.3f} '
                      f'band_viol={epoch_loss["compactness_band_viol"]:.3f}')
                # exp6: the within-tier term and the gen-gen pair substrate. Printed
                # here (not just to wandb) because whether the intervention is really
                # firing is the first thing to check on any restart.
                if getattr(args, 'dpo_gen_only_pairs', False):
                    _cells = ' '.join(
                        f'{c}={epoch_loss.get("paircell_" + c.replace("-", ""), 0.0):.1f}'
                        for c in ('D-D', 'A-A', 'B-B', 'C-C'))
                    print(f'  [exp6 gen-gen] within={epoch_loss.get("dpo_within_loss", 0.0):.4f} '
                          f'n_within={epoch_loss.get("n_within_pairs", 0.0):.1f} '
                          f'gen_decoys={epoch_loss.get("gen_n_decoys", 0.0):.1f} '
                          f'pool={epoch_loss.get("gen_n_pool", 0.0):.1f} '
                          f'cand_pairs={epoch_loss.get("gen_n_candidate_pairs", 0.0):.0f} '
                          f'cells={epoch_loss.get("gen_n_cells_present", 0.0):.1f} | {_cells}')
        if args.wandb:
            current_lr = optimizer.param_groups[0]['lr']
            log_dict = {
                "train loss": epoch_loss['final_loss'],
                "learning_rate": current_lr,
                "epoch": epoch_idx,
            }
            if tier_dpo is not None:
                log_dict["train_dpo_eject_loss"] = epoch_loss['dpo_eject_loss']
                log_dict["train_dpo_top_loss"] = epoch_loss['dpo_top_loss']
                log_dict["train_n_eject_pairs"] = epoch_loss['n_eject_pairs']
                log_dict["train_n_top_pairs"] = epoch_loss['n_top_pairs']
                # lambda-weighted contributions (what actually enters the total loss)
                log_dict["loss/eject_weighted"] = epoch_loss['dpo_eject_weighted']
                log_dict["loss/top_weighted"] = epoch_loss['dpo_top_weighted']
                log_dict["loss/compactness_weighted"] = epoch_loss['compactness_weighted']
                log_dict["compactness/skip_rate_no_x_or_a"] = epoch_loss['compactness_skipped_no_x_or_a']
                log_dict["loss/compactness"] = epoch_loss['compactness_loss']
                log_dict["compactness/n_pairs_used"] = epoch_loss['compactness_n_pairs']
                log_dict["compactness/x_to_a_gap_mean"] = epoch_loss['compactness_gap_mean']
                log_dict["compactness/x_to_a_gap_max"] = epoch_loss['compactness_gap_max']
                log_dict["compactness/ordering_violation_rate"] = epoch_loss['compactness_ordering_viol']
                log_dict["compactness/band_violation_rate"] = epoch_loss['compactness_band_viol']
            if getattr(args, 'use_phase_config', False):
                log_dict["train/phase"] = epoch_loss['phase']
                log_dict["train/lambda_sml"] = epoch_loss['lambda_sml']
                log_dict["train/lambda_dpo_eject"] = epoch_loss['lambda_dpo_eject']
                log_dict["train/lambda_dpo_top"] = epoch_loss['lambda_dpo_top']
                log_dict["train/lambda_compactness"] = epoch_loss['lambda_compactness']
                log_dict["train/sampler_target_eject"] = epoch_loss['sampler_target_eject']
                log_dict["train/sampler_target_top"] = epoch_loss['sampler_target_top']
                log_dict["train/sampler_actual_eject"] = epoch_loss['sampler_actual_eject']
                log_dict["train/sampler_actual_top"] = epoch_loss['sampler_actual_top']
                log_dict["train/sampler_hard_d_pairs"] = epoch_loss['sampler_hard_d_pairs']
                log_dict["train/sampler_random_d_pairs"] = epoch_loss['sampler_random_d_pairs']
                log_dict["train/sampler_xtal_or_a_vs_c_pairs"] = epoch_loss['sampler_xtal_or_a_vs_c_pairs']
                log_dict["train/sampler_xtal_vs_b_pairs"] = epoch_loss['sampler_xtal_vs_b_pairs']
                log_dict["train/sampler_xtal_vs_c_pairs"] = epoch_loss['sampler_xtal_vs_c_pairs']
                log_dict["train/sampler_xtal_vs_a_pairs"] = epoch_loss['sampler_xtal_vs_a_pairs']
                log_dict["train/sampler_a_local_best_vs_b_pairs"] = epoch_loss['sampler_a_local_best_vs_b_pairs']
            for _tier in ('X', 'A', 'B', 'C', 'D'):
                _k = f'tier_n_{_tier}'
                if _k in epoch_loss:
                    log_dict[f'train/{_k}_mean'] = epoch_loss[_k]
            logger.log_metrics(log_dict)

        args.save_ckpt_path = result_dir / f'{args.param_name}_{epoch_idx}.pt'
        _ckpt_iv = max(1, int(getattr(args, 'ckpt_interval', 1) or 1))
        if epoch_idx % _ckpt_iv == 0:
            save_state(model, optimizer, epoch_idx, args.save_ckpt_path, callbacks)

        # ── Validation (every epoch) ──
        valid_dataloader = data_module.val_dataloader(validation_list, args.decoytype, dataset_config=dataset_config)
        if data_module.ds_val is not None and hasattr(data_module.ds_val, 'set_epoch'):
            data_module.ds_val.set_epoch(epoch_idx)
        model.eval()
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='None of the inputs have requires_grad=True')
            epoch_loss = run_epoch_dpo(model, pre_trained, loss_fn, valid_dataloader,
                                       epoch_idx, grad_scaler, optimizer, local_rank,
                                       callbacks, is_train=False, args=args,
                                       tier_dpo=tier_dpo, pair_cfg=pair_cfg,
                                       ord_loss_fn=ord_loss_fn)
        if dist.is_initialized():
            epoch_loss = reduce_epoch_loss(epoch_loss, world_size)
        if local_rank == 0:
            report_epoch_loss(epoch_loss, epoch_idx, mode='valid', out_tag=args.param_name)
            if tier_dpo is not None:
                print(f'  [TierDPO valid] eject={epoch_loss["dpo_eject_loss"]:.4f} '
                      f'top={epoch_loss["dpo_top_loss"]:.4f} '
                      f'compact={epoch_loss["compactness_loss"]:.4f} '
                      f'n_eject={epoch_loss["n_eject_pairs"]:.1f} '
                      f'n_top={epoch_loss["n_top_pairs"]:.1f} '
                      f'gap_mean={epoch_loss["compactness_gap_mean"]:.3f} '
                      f'ord_viol={epoch_loss["compactness_ordering_viol"]:.3f} '
                      f'band_viol={epoch_loss["compactness_band_viol"]:.3f}')
            if 'top1_h3_lddt_mean' in epoch_loss:
                def _pv(key):
                    v = epoch_loss.get(key)
                    if v is None:
                        return float('nan')
                    return v.item() if isinstance(v, torch.Tensor) else float(v)

                print(
                    f'  [H3 valid] top1_lddt={_pv("top1_h3_lddt_mean"):.3f} '
                    f'nonxtal_lddt={_pv("top1_nonxtal_h3_lddt_mean"):.3f} '
                    f'xtal_top1_rate={_pv("top1_is_xtal_rate"):.3f} '
                    f'regret={_pv("h3_lddt_regret_mean"):.3f} '
                    f'nonxtal_regret={_pv("h3_lddt_regret_nonxtal_mean"):.3f}'
                )
        if args.wandb:
            current_lr = optimizer.param_groups[0]['lr']

            def _metric_val(key):
                if key not in epoch_loss:
                    return None
                v = epoch_loss[key]
                return v.item() if isinstance(v, torch.Tensor) else v

            # Name the wandb keys after the ACTUAL metric: the internal
            # epoch_loss['top1_rmsd'] etc. hold the label metric (cdr_lddt when
            # label_metric=loop_lddt), NOT Angstrom RMSD. Suffix by label_metric so
            # the panel name matches the value (matches the pretrain rename above).
            _lm = str(getattr(args, 'label_metric', 'loop_rmsd')).lower()
            _suf = 'cdr_lddt' if 'lddt' in _lm else 'rmsd'
            log_dict = {
                "valid/loss": epoch_loss['final_loss'],
                "valid/best_rank": epoch_loss['best_rank'],
                f"valid/top1_{_suf}": epoch_loss['top1_rmsd'],
                f"valid/top3_{_suf}": epoch_loss['top3_rmsd'],
                f"valid/top5_{_suf}": epoch_loss['top5_rmsd'],
                f"valid/top10_{_suf}": epoch_loss['top10_rmsd'],
                f"valid/{_suf}_diff": epoch_loss['rmsd_diff'],
                "learning_rate": current_lr,
                "epoch": epoch_idx,
            }
            for wandb_key, epoch_key in (
                ("val/top1_h3_rmsd_mean", "top1_h3_rmsd_mean"),
                ("val/top1_h3_lddt_mean", "top1_h3_lddt_mean"),
                ("val/top1_h3_rmsd_median", "top1_h3_rmsd_median"),
                ("val/top1_h3_lddt_median", "top1_h3_lddt_median"),
                ("val/top1_is_xtal_rate", "top1_is_xtal_rate"),
                ("val/top1_nonxtal_h3_rmsd_mean", "top1_nonxtal_h3_rmsd_mean"),
                ("val/top1_nonxtal_h3_rmsd_median", "top1_nonxtal_h3_rmsd_median"),
                ("val/top1_nonxtal_h3_lddt_mean", "top1_nonxtal_h3_lddt_mean"),
                ("val/top1_nonxtal_h3_lddt_median", "top1_nonxtal_h3_lddt_median"),
                ("val/oracle_h3_lddt_mean", "oracle_h3_lddt_mean"),
                ("val/oracle_h3_lddt_median", "oracle_h3_lddt_median"),
                ("val/oracle_nonxtal_h3_lddt_mean", "oracle_nonxtal_h3_lddt_mean"),
                ("val/oracle_nonxtal_h3_lddt_median", "oracle_nonxtal_h3_lddt_median"),
                ("val/h3_lddt_regret_mean", "h3_lddt_regret_mean"),
                ("val/h3_lddt_regret_median", "h3_lddt_regret_median"),
                ("val/h3_lddt_regret_nonxtal_mean", "h3_lddt_regret_nonxtal_mean"),
                ("val/h3_lddt_regret_nonxtal_median", "h3_lddt_regret_nonxtal_median"),
                ("val/top1_h3_success_rmsd_le_2", "top1_h3_success_rmsd_le_2"),
                ("val/top1_nonxtal_success_rmsd_le_2", "top1_nonxtal_success_rmsd_le_2"),
                ("val/top1_h3_success_lddt_ge_0.80", "top1_h3_success_lddt_ge_0.80"),
                ("val/top1_h3_success_lddt_ge_0.90", "top1_h3_success_lddt_ge_0.90"),
                ("val/top1_h3_success_lddt_ge_0.95", "top1_h3_success_lddt_ge_0.95"),
                ("val/top1_nonxtal_success_lddt_ge_0.80", "top1_nonxtal_success_lddt_ge_0.80"),
                ("val/top1_nonxtal_success_lddt_ge_0.90", "top1_nonxtal_success_lddt_ge_0.90"),
            ):
                val = _metric_val(epoch_key)
                if val is not None:
                    log_dict[wandb_key] = val
            if tier_dpo is not None:
                log_dict["valid_dpo_eject_loss"] = epoch_loss['dpo_eject_loss']
                log_dict["valid_dpo_top_loss"] = epoch_loss['dpo_top_loss']
                log_dict["valid_n_eject_pairs"] = epoch_loss['n_eject_pairs']
                log_dict["valid_n_top_pairs"] = epoch_loss['n_top_pairs']
                log_dict["valid/loss_eject_weighted"] = epoch_loss['dpo_eject_weighted']
                log_dict["valid/loss_top_weighted"] = epoch_loss['dpo_top_weighted']
                log_dict["valid/loss_compactness_weighted"] = epoch_loss['compactness_weighted']
                log_dict["valid/loss_compactness"] = epoch_loss['compactness_loss']
                log_dict["valid/compactness_gap_mean"] = epoch_loss['compactness_gap_mean']
                log_dict["valid/compactness_ordering_viol"] = epoch_loss['compactness_ordering_viol']
                log_dict["valid/compactness_band_viol"] = epoch_loss['compactness_band_viol']
            if getattr(args, 'use_phase_config', False):
                log_dict["valid/phase"] = epoch_loss['phase']
                log_dict["valid/lambda_sml"] = epoch_loss['lambda_sml']
                log_dict["valid/lambda_dpo_eject"] = epoch_loss['lambda_dpo_eject']
                log_dict["valid/lambda_dpo_top"] = epoch_loss['lambda_dpo_top']
                log_dict["valid/lambda_compactness"] = epoch_loss['lambda_compactness']
                log_dict["valid/sampler_target_eject"] = epoch_loss['sampler_target_eject']
                log_dict["valid/sampler_target_top"] = epoch_loss['sampler_target_top']
                log_dict["valid/sampler_actual_eject"] = epoch_loss['sampler_actual_eject']
                log_dict["valid/sampler_actual_top"] = epoch_loss['sampler_actual_top']
            logger.log_metrics(log_dict)

        # v2 fixed-manifest validation (intrinsic + final on identical pool).
        # ALL ranks must enter (sharded + all_gather); rank-0-only would deadlock DDP.
        _run_manifest_pools(model, args, dataset_config, device, epoch_idx, logger)

        for callback in callbacks:
            callback.on_epoch_end()
    for callback in callbacks:
        callback.on_fit_end()



def inference(model: nn.Module,
              callbakcs: List[BaseCallback],
              logger: Logger,
              args):
    device = torch.cuda.current_device()
    model.to(device=device)
    world_size=dist.get_world_size() if dist.is_initialized() else 1
    data_module = HUDataModule()
    dataset_config = getattr(args, 'dataset_config', None)
    
    decoytype_lower = args.decoytype.lower() if args.decoytype else ''

    if dataset_config:
        from dataset.config import load_dataset_spec
        _infer_spec = load_dataset_spec(dataset_config)
        args.label_metric = _infer_spec.label_metric
        args.near_native_cutoff = _infer_spec.effective_near_native_cutoff()
        if args.pdb_list_pickle is None:
            raise ValueError(
                '--dataset_config inference currently requires --pdb_list_pickle '
                'to specify the target list'
            )
        pdb_list = _read_list_from_pkl(args.pdb_list_pickle)
        from dataset.inference_filter import (
            filter_pdb_list_for_yaml_inference,
            write_skipped_log,
        )
        n_before = len(pdb_list)
        pdb_list, skipped_targets = filter_pdb_list_for_yaml_inference(
            pdb_list, dataset_config, epoch=1, verify_models=False,
        )
        if skipped_targets:
            log_name = f"skipped_{args.decoytype or 'inference'}_{pathlib.Path(args.pdb_list_pickle).stem}.log"
            skip_log = pathlib.Path('/home/sujin/projects/h3-loop-modeling/script/log') / log_name
            write_skipped_log(
                skipped_targets,
                skip_log,
                dataset_config=dataset_config,
                decoytype=args.decoytype or '',
            )
            if get_local_rank() == 0:
                print(f"\n[WARNING] Skipped {len(skipped_targets)}/{n_before} targets "
                      f"(no decoy pickle / no models). Log: {skip_log}")
        logging.info(f'Inference dataset_config: {dataset_config}')
        logging.info(f'Inference pdb_list_pickle: {args.pdb_list_pickle}')
        logging.info(f'Inference target count: {len(pdb_list)} (from {n_before} in list)')
        if not pdb_list:
            raise ValueError(
                f'No targets left after filtering for {dataset_config}. '
                f'See skipped log under script/log/.'
            )
        test_dataloader = data_module.test_dataloader(
            pdb_list,
            datatype='AbAg',
            pickle_pdb=False,
            decoytype=args.decoytype,
            dataset_config=dataset_config,
        )
    else:
        ### Target pickle with ag_local_rmsd from pdb2dict_combined.py output
        # Use args.pdb_list_pickle if provided, otherwise use default
        if args.pdb_list_pickle is not None:
            pdb_list_pickle = args.pdb_list_pickle
        else:
            if decoytype_lower == 'psh':
                pdb_list_pickle = '/home/sujin/DB/h3-loop-modeling/ab_ag/2_after210930/0_info/info.pkl'
            else:
                pdb_list_pickle = '/home/sujin/DB/h3-loop-modeling/ab_ag/0_igfold_197/0_info/info.pkl'
        
        # Use args.db_dir if provided, otherwise set based on decoytype
        if args.db_dir is not None:
            db_dir = args.db_dir
        else:
            # Default db_dir paths based on decoytype
            if decoytype_lower.startswith('boltz2'):
                db_dir = '/home/sujin/DB/h3-loop-modeling/ab_ag/0_igfold_197/33_boltz2_pdb2dict'
            elif decoytype_lower.startswith('af3'):
                db_dir = '/home/sujin/DB/h3-loop-modeling/ab_ag/0_igfold_197/15_af3_pdb2dict'
            elif decoytype_lower.startswith('igfold'):
                db_dir = '/home/sujin/DB/h3-loop-modeling/ab_ag/igfold_ordered/decoy'
            elif decoytype_lower == 'psh':
                db_dir = '/home/sujin/DB/h3-loop-modeling/ab_ag/2_after210930/17_psh_pdb2dict'
            else:
                db_dir = '/home/sujin/DB/h3-loop-modeling/ab_ag/0_igfold_197/15_af3_pdb2dict'
        
        # Determine use_subdirectory based on decoytype if not explicitly set
        # boltz2, af3, psh use flat structure (no subdirectory), igfold uses subdirectory
        use_subdirectory = args.use_subdirectory
        if not use_subdirectory:
            # Keep as-is (already False)
            pass
        elif decoytype_lower in ('boltz2', 'af3', 'psh'):
            use_subdirectory = False
        
        logging.info(f'Inference decoytype: {args.decoytype}')
        logging.info(f'Inference db_dir: {db_dir}')
        logging.info(f'Inference pdb_list_pickle: {pdb_list_pickle}')
        logging.info(f'Inference use_subdirectory: {use_subdirectory}')
        
        test_dataloader = data_module.test_dataloader(
            pdb_list_pickle,
            pickle_pdb=True,
            decoytype=args.decoytype,
            use_subdirectory=use_subdirectory,
            db_dir=db_dir
        )
    
    ### Legacy paths (commented for reference)
    '''
    # Old IgFold path with subdirectories
    pdb_list_pickle = '/home/sujin/DB/h3-loop-modeling/ab_ag/igfold_ordered/igfold-list-test-4.pkl'
    test_dataloader = data_module.test_dataloader(pdb_list_pickle,pickle_pdb=True,decoytype=args.decoytype,use_subdirectory=True)
    
    # Old graph pickle mode (no ag_local_rmsd)
    test_dataloader = data_module.test_dataloader(pdb_list,pickle_pdb=False,decoytype=args.decoytype)
    '''
    
    # test_dataloader = data_module.test_dataloader(test_set=test_list,datatype='igfold') #datatype=igfold for default inference dataset (FALC+PertMD)
    
    model.eval()
    info_dict = {}
    loss_fn = Sujin_loss()
    training = False
    inference_mode = True
    show_progress = (get_local_rank() == 0)
    source = (args.decoytype or '').lower()
    with torch.no_grad():
        for i, batch in tqdm(
            enumerate(test_dataloader),
            total=len(test_dataloader),
            unit='batch',
            desc='Inference',
            disable=(not show_progress),
        ):
            pdb = batch[2]
            batched_graph, rmsd_s = to_cuda(batch[0:2])
            # Split DGL batched graph into individual graphs
            graphs = dgl.unbatch(batched_graph)
            ranking_list = None
            if len(batch) >= 4 and batch[3] is not None:
                x = batch[3]
                if isinstance(x, torch.Tensor):
                    ranking_list = x.detach().cpu().tolist()
                elif isinstance(x, (list, tuple)):
                    ranking_list = list(x)
                else:
                    ranking_list = [int(x)]
            ag_local_s = None
            if len(batch) >= 5 and batch[4] is not None:
                ag_local_s = batch[4].to(rmsd_s.device)
            decoy_meta = batch[5] if len(batch) >= 6 else None
            eval_metrics = _extract_eval_metrics(batch)
            split_size = 50
            if len(rmsd_s) > split_size:
                # Split batched_graph and rmsd_s into chunks
                all_out = []
                all_rmsd = []
                all_rank = []
                all_ag_local = []
                num_splits = (len(rmsd_s) + split_size - 1) // split_size
                for j in range(num_splits):
                    start_idx = j * split_size
                    end_idx = min((j + 1) * split_size, len(rmsd_s))
                    sub_graphs = graphs[start_idx:end_idx]
                    batched_graph_split = dgl.batch(sub_graphs).to(device)
                    rmsd_s_split = rmsd_s[start_idx:end_idx]
                    rank_split = None if ranking_list is None else ranking_list[start_idx:end_idx]
                    ag_split = None if ag_local_s is None else ag_local_s[start_idx:end_idx]
                    pred_split = model(batched_graph_split)
                    out_split = pred_split['out']
                    all_out.append(out_split)
                    all_rmsd.append(rmsd_s_split)
                    if rank_split is not None:
                        all_rank.append(rank_split)
                    if ag_split is not None:
                        all_ag_local.append(ag_split)

                final_out = torch.cat(all_out, dim=0)
                final_rmsd = torch.cat(all_rmsd, dim=0)
                final_rank = None if not all_rank else sum(all_rank, [])
                final_ag_local = torch.cat(all_ag_local, dim=0) if all_ag_local else None
                device_ = final_out.device
                loss, loss_dic = loss_fn(pdb, final_rmsd, final_out, training, device_, inference_mode, rmsd_cutoff=2.0, loss_type=args.loss_type, label_metric=(getattr(args, 'label_metric', None) or 'loop_rmsd'))
                _add_eval_top1_metrics(
                    loss_dic,
                    eval_metrics,
                    final_out,
                    getattr(args, 'label_metric', None) or 'loop_rmsd',
                )
                loss_dic['pred'] = final_out
                loss_dic['h3_rmsd'] = final_rmsd
                loss_dic['rank'] = final_rank
                for metric_name in ('loop_rmsd', 'loop_lddt'):
                    metric_tensor = _metric_tensor(eval_metrics, metric_name, device_)
                    if metric_tensor is not None and metric_tensor.numel() == final_out.numel():
                        loss_dic[metric_name] = metric_tensor.detach().cpu().tolist()
                if final_ag_local is not None:
                    loss_dic['ag_local_rmsd'] = final_ag_local.detach().cpu().tolist()
                info_dict[pdb] = loss_dic
            else:
                # Use ranking_list directly, do not normalize
                pred = model(batched_graph)
                device_ = pred['out'].device
                loss, loss_dic = loss_fn(pdb, rmsd_s, pred['out'], training, device_, inference_mode, rmsd_cutoff=2.0, loss_type=args.loss_type, label_metric=(getattr(args, 'label_metric', None) or 'loop_rmsd'))
                _add_eval_top1_metrics(
                    loss_dic,
                    eval_metrics,
                    pred['out'],
                    getattr(args, 'label_metric', None) or 'loop_rmsd',
                )
                loss_dic['pred'] = pred['out']
                loss_dic['h3_rmsd'] = rmsd_s
                loss_dic['rank'] = ranking_list
                for metric_name in ('loop_rmsd', 'loop_lddt'):
                    metric_tensor = _metric_tensor(eval_metrics, metric_name, device_)
                    if metric_tensor is not None and metric_tensor.numel() == pred['out'].numel():
                        loss_dic[metric_name] = metric_tensor.detach().cpu().tolist()
                if ag_local_s is not None:
                    loss_dic['ag_local_rmsd'] = ag_local_s.detach().cpu().tolist()
                info_dict[pdb] = loss_dic

            # Attach source and decoy identity metadata
            info_dict[pdb]['source'] = source
            if decoy_meta is not None:
                info_dict[pdb]['decoys'] = decoy_meta

            # Rename rmsd_diff -> rmsd_regret
            if 'rmsd_diff' in info_dict[pdb]:
                info_dict[pdb]['rmsd_regret'] = info_dict[pdb].pop('rmsd_diff')

            if show_progress:
                agl = info_dict[pdb].get('ag_local_rmsd')
                if agl is not None:
                    fin = [float(x) for x in agl if x == x]
                    if fin:
                        logging.info(
                            "inference %s: ag_local_rmsd min=%.4f max=%.4f (n=%d)",
                            pdb, min(fin), max(fin), len(agl),
                        )
                    else:
                        logging.info(
                            "inference %s: ag_local_rmsd all non-finite (n=%d)",
                            pdb, len(agl),
                        )
    info_dict = to_cpu(info_dict)
    inference_dir = _inference_output_dir(args)
    inference_dir.mkdir(parents=True, exist_ok=True)
    with open(inference_dir / f'{args.param_name}.info', 'wb') as fp:
        pickle.dump(info_dict, fp)
    return None
    

def print_parameters_count(model):
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of trainable parameters: {num_params_trainable}')


if __name__ == '__main__':
    is_distributed = init_distributed()
    local_rank = get_local_rank()
    args = PARSER.parse_args()
    # Bridge --label_metric / --near_native_cutoff to every load_dataset_spec()
    # call (train.py's spec AND the dataset's internally-loaded spec) via env.
    if getattr(args, 'label_metric', None):
        os.environ['CDR_LABEL_METRIC'] = args.label_metric
    if getattr(args, 'near_native_cutoff', None) is not None:
        os.environ['CDR_NEAR_NATIVE_CUTOFF'] = str(args.near_native_cutoff)
    # Graph-cache config -> env, inherited by forked DataLoader workers (MyDataset
    # reads these). 'read' serves graphs from a prebuilt cache with per-item
    # fallback to on-the-fly; 'off' (default) leaves the pipeline unchanged.
    if getattr(args, 'graph_cache_mode', 'off') == 'read':
        os.environ['CDR_GRAPH_CACHE_MODE'] = 'read'
        if getattr(args, 'graph_cache_dir', None):
            os.environ['CDR_GRAPH_CACHE_DIR'] = args.graph_cache_dir
        if getattr(args, 'graph_cache_no_verify_struct', False):
            os.environ['CDR_GRAPH_CACHE_VERIFY_STRUCT'] = '0'
        logging.info('graph cache: READ mode, dir=%s (per-item fallback to on-the-fly)',
                     getattr(args, 'graph_cache_dir', None))
    logging.getLogger().setLevel(logging.CRITICAL if local_rank != 0 or args.silent else logging.INFO)
    logging.info("====== Sujin's Loop Decoy ReRanking =======")
    logging.info('|            Training procedure           |')
    logging.info('===========================================')
    if args.seed is not None:
        logging.info(f'Using seed {args.seed}')
        seed_everything(args.seed)
    logger = LoggerCollection([WandbLogger(
                                project='se3-transformer',
                                name=args.param_name,
                                save_dir=args.log_dir,
                                id=args.wandb_id if args.wandb_id else None,
                                ) if args.wandb
    else DLLogger(save_dir=args.log_dir,filename=args.dllogger_name)])
    datamodule = HUDataModule(**vars(args))
    if args.all_atom:
        node_l1_dim = MAX_NUM_ATOM
        
    else:
        node_l1_dim = 4
    if args.wandb and get_local_rank() == 0:
        run_id = logger.loggers[0].experiment.id
        logging.info(f'W&B run ID: {run_id}')
    if args.run_type == 'finetune':
        args.nodewise_score = True

    print('[using all atom] ',args.all_atom)
    print('[nodewise score] ',args.nodewise_score)
    if args.all_atom:
        model = Sujin_with_SE3_allatom(
            fiber_in = Fiber({0: args.embedded_node_dim,1:node_l1_dim}),
            fiber_out = Fiber({0: args.num_degrees * args.num_channels,1:20}),
            fiber_edge = Fiber({0: args.embedded_edge_dim,1:1}),
            tensor_cores = using_tensor_cores(args.amp),
            **vars(args)
        )
    else:
        model = Sujin_with_SE3(
                fiber_in=Fiber({0: args.embedded_node_dim,1:node_l1_dim}),
                fiber_out=Fiber({0: args.num_degrees * args.num_channels,1:20}),###MUST
                fiber_edge=Fiber({0: args.embedded_edge_dim,1:1}),
                use_nodewise_score=args.nodewise_score,
                tensor_cores=using_tensor_cores(args.amp),  # use Tensor Cores more effectively
                **vars(args)
        )
    callbacks =[ QM9LRSchedulerCallback(logger, epochs=args.epochs)]
    print_parameters_count(model)
    
    logger.log_hyperparams(vars(args))
    # v2: record fixed-validation manifest identity in W&B config (path/hash/counts).
    _man_cfg = {}
    for _pool, _p in (('multisource', getattr(args, 'val_manifest_multisource', None)),
                      ('boltz2', getattr(args, 'val_manifest_boltz2', None))):
        if _p and os.path.exists(_p):
            try:
                import json as _json
                with open(_p) as _f:
                    _mm = _json.load(_f)
                _man_cfg[f'val_manifest_{_pool}_path'] = str(_p)
                _man_cfg[f'val_manifest_{_pool}_hash'] = _mm.get('hash')
                _man_cfg[f'val_manifest_{_pool}_n_targets'] = _mm.get('n_targets')
                _man_cfg[f'val_manifest_{_pool}_n_decoys'] = _mm.get('n_decoys')
            except Exception as _e:
                logging.warning('[manifest cfg] %s: %s', _pool, _e)
    if _man_cfg:
        logger.log_hyperparams(_man_cfg)
    if args.wandb and get_local_rank() == 0:
        logger.save_run_files(
            dataset_config=args.dataset_config,
            run_script=getattr(args, 'run_script', None),
        )

    if get_local_rank() == 0:
        print('se3 config loss ',se3_config.loss.weight)
    increase_l2_fetch_granularity()
    ###

    if args.run_type=='train':
        train(model,callbacks, logger, args)
        logging.info('Training finished successfully')
    elif args.run_type=='finetune':
        finetune(model, callbacks, logger, args)
        logging.info('Fine-tuning finished successfully')
    elif args.run_type=='inference':
        trained_model_dir = f'{args.save_model_path}'
        
        checkpoint = torch.load(trained_model_dir, map_location={'cuda:0': f'cuda:{get_local_rank()}'})
        missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
        allowed_missing = {
            'ord_head.weight', 'ord_head.bias',
            'interface_head.weight', 'final_head.weight',  # v2 multi-head (absent in v0/v1 ckpts)
        }
        bad_missing = [key for key in missing if key not in allowed_missing]
        if bad_missing or unexpected:
            raise RuntimeError(
                f'Incompatible inference checkpoint: missing={missing}, unexpected={unexpected}. '
                f'Only missing {sorted(allowed_missing)} is allowed.'
            )
        if get_local_rank() == 0:
            print(f'[inference] loaded checkpoint: {trained_model_dir}')
            if missing:
                print(f'[inference] missing keys: {missing}')


        inference(
            model,
            callbacks,
            logger,
            args
        )
    
        logging.info('===========================================')
        logging.info('|            Inference Finished           |')
        logging.info('===========================================')
