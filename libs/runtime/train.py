import logging
import pathlib
from typing import List
import sys,pickle
sys.path.insert(0,'/home/sujin/projects/h3-loop-modeling/libs/')
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from tqdm import tqdm
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
    TOP1_H3_VAL_LIST_KEYS,
)
from runtime.phase_config import get_phase_config
from runtime.pair_sampling import (
    PairSamplingConfig, build_training_pairs, build_decoy_tiers, detect_xtal_mask,
)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_descriptor')
import os,random
from runtime.constants import *
import traceback
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:256')
import copy
import warnings


import dgl

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

def run_epoch(model,dataloader, epoch_idx, grad_scaler, optimizer, local_rank,
        callbacks,is_train,args):
    epoch_loss = initialize_epoch_loss()
    ##
    if is_train: train_tag='train';training=True;inference=False
    else:train_tag='valid';training=False;inference=False
    ##
    loss_fn=Sujin_loss()
    info_dict={}
    
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), unit='batch',
                         #desc=f'{train_tag} Epoch {epoch_idx}', disable=(True)):
                         desc=f'{train_tag} Epoch {epoch_idx}', disable=(local_rank != 0)):
        batched_graph, rmsd_s = to_cuda(batch[0:2])
        pdb = batch[2]
        node_info = _batch_node_info(batched_graph)
        if _should_skip_oversized_batch(batched_graph, pdb, epoch_idx, i, train_tag, args, local_rank):
            del batched_graph, rmsd_s
            torch.cuda.empty_cache()
            continue
        try:
            for callback in callbacks:
                callback.on_batch_start()
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(batched_graph)
                device=pred['out'].device

                # TODO: Add loss function for nodewise score
                if args.nodewise_score:
                    nodewise_score = pred['nodewise_score']
                    print('nodewise_score ',len(nodewise_score), nodewise_score[0].shape)

                loss,loss_dic=loss_fn(pdb,rmsd_s,pred['out'],training,device,inference,rmsd_cutoff=2.0,loss_type=args.loss_type)
                # Detach GPU tensors in loss_dic to prevent holding computation graph
                info_dict[pdb] = {k: (v.detach().cpu().item() if isinstance(v, torch.Tensor) else v) for k, v in loss_dic.items()}
                loss = loss/args.accumulate_grad_batches
            ###
            epoch_loss=update_epoch_loss(epoch_loss,loss_dic)
            if is_train:
                def _zero_dummy_pretrain():
                    d = pred['out'].sum() * 0.0
                    ol = pred.get('ord_logits')
                    return d + ol.sum() * 0.0 if ol is not None else d

                if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
                    loss = _zero_dummy_pretrain()
                elif torch.isnan(loss):
                    print('save nan pred and rmsd for that target as pickle')
                    nan_path =  f'/home/sujin/projects/h3-loop-modeling/libs/results/{args.param_name}/nan/'
                    if not os.path.exists(nan_path):
                        os.makedirs(f'{nan_path}')
                    with open(f'{nan_path}/{pdb}.dat','wb')as fp:
                        pickle.dump([pred['out'].detach().cpu(),rmsd_s.detach().cpu()],fp)
                    loss = _zero_dummy_pretrain()
                grad_scaler.scale(loss).backward()
                # Free GPU memory from this batch before the next optimizer step
                del batched_graph, pred, loss_dic
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
                del batched_graph, pred, loss, loss_dic
                torch.cuda.empty_cache()
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
    with torch.no_grad():
        epoch_loss=finalize_epoch_loss(epoch_loss)
    with open('/home/sujin/projects/h3-loop-modeling/libs/results/%s/%s.%i.%i.info'%(args.param_name,train_tag,epoch_idx,local_rank,),'wb')as fp:
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
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank
                #find_unused_parameters=True
                )
        model._set_static_graph()
    model.train()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999),weight_decay=args.weight_decay)

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
    result_dir = f'/home/sujin/projects/h3-loop-modeling/libs/results/{args.param_name}'
    os.makedirs(result_dir, exist_ok=True)

    # get checkpoint model
    if epoch_start == 0 and args.save_model_path and os.path.exists(args.save_model_path):
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
        _libs_dir = '/home/sujin/projects/h3-loop-modeling/libs/'
        if _libs_dir not in sys.path:
            sys.path.insert(0, _libs_dir)
        from dataset.config import load_dataset_spec
        _yaml_spec = load_dataset_spec(dataset_config)

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

    for epoch_idx in range(epoch_start+1, 1000):
        # ── Build training / validation lists ──
        if _yaml_spec and (_yaml_spec.train_list or getattr(_yaml_spec, "train_list_from_pkl", "")):
            # YAML-driven: read abag + gp lists, random sample like set_data()
            random.seed(epoch_idx)
            abag_all = yaml_train_abag_pool if yaml_train_abag_pool is not None else []
            num_abag = _yaml_spec.num_abag if _yaml_spec.num_abag is not None else len(abag_all)
            num_abag = min(num_abag, len(abag_all))
            training_list = random.sample(abag_all, k=num_abag) if num_abag > 0 else []
            if _yaml_spec.gp_list:
                gp_all = _read_list_file(_yaml_spec.gp_list)
                num_gp = _yaml_spec.num_gp if _yaml_spec.num_gp is not None else len(gp_all)
                num_gp = min(num_gp, len(gp_all))
                training_list += random.sample(gp_all, k=num_gp)
            random.shuffle(training_list)
            if epoch_idx == epoch_start + 1 and get_local_rank() == 0:
                print(f'[YAML] abag: {num_abag}/{len(abag_all)}, gp: {num_gp if _yaml_spec.gp_list else 0}/{len(gp_all) if _yaml_spec.gp_list else 0}, total: {len(training_list)}')
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
        args.save_ckpt_path=f'/home/sujin/projects/h3-loop-modeling/libs/results/{args.param_name}/%s_%i.pt'%(args.param_name,epoch_idx);args.ckpt_interval=2
        if epoch_idx %args.ckpt_interval==0:
            save_state(model, optimizer, epoch_idx, args.save_ckpt_path, callbacks)
        if args.wandb:
            current_lr = optimizer.param_groups[0]['lr']
            logger.log_metrics({"train loss":epoch_loss['final_loss'],
                "learning_rate": current_lr,
                "epoch":epoch_idx})
        
        #### Validation
        if epoch_idx %2 ==0:
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
                logger.log_metrics({"valid loss":epoch_loss['final_loss'],
                    "best rank":epoch_loss['best_rank'],
                    "rmsd_diff":epoch_loss['rmsd_diff'],
                    "top1_rmsd":epoch_loss["top1_rmsd"],
                    "top3_rmsd":epoch_loss["top3_rmsd"],
                    "top5_rmsd":epoch_loss["top5_rmsd"],
                    "top10_rmsd":epoch_loss["top10_rmsd"],
                    "learning_rate": current_lr,
                    "epoch":epoch_idx})
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
    for _k in ('tier_n_X', 'tier_n_A', 'tier_n_B', 'tier_n_C', 'tier_n_D'):
        epoch_loss[_k] = []
    train_tag = 'train' if is_train else 'valid'
    is_training = is_train
    info_dict = {}
    phase_cfg = _get_enabled_phase_config(args)

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), unit='batch',
                         desc=f'{train_tag} Epoch {epoch_idx}', disable=(local_rank != 0)):
        batched_graph, rmsd_s = to_cuda(batch[0:2])
        pdb = batch[2]
        device = rmsd_s.device
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
                    pi['out'], rmsd_s, pdb, device, is_training,
                    return_components=True,
                )
            else:
                base_loss, out = loss_fn.forward(
                    pi_ns, ref_ns,
                    pi['out'], rmsd_s, pdb, device, is_training,
                )
                loss_components = None
            del ref_ns

            # ── Tier-based DPO losses (step 2) ──
            if tier_dpo is not None and pair_cfg is not None:
                pi_out = pi['out'].squeeze()
                ref_out = ref_out_raw.squeeze()
                del ref_out_raw
                is_xtal = detect_xtal_mask(rmsd_s)
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
                with torch.no_grad():
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
                    )
                eject_loss, n_eject = tier_dpo.dpo_eject_loss(pi_out, ref_out, pairs, device)
                top_loss, n_top = tier_dpo.dpo_top_loss(pi_out, ref_out, pairs, device)
                del ref_out, pairs
                tier_counts = pair_summary.get('tier_counts', {})
                for _tier in ('X', 'A', 'B', 'C', 'D'):
                    out[f'tier_n_{_tier}'] = float(tier_counts.get(_tier, 0))
                # ── step 3: compactness band loss (skip when no Tier A) ──
                _lambda_compact = phase_cfg.lambda_compactness if phase_cfg is not None else getattr(args, 'lambda_compactness', 0.0)
                _tiers_c = pair_summary.get('tiers', {})
                if _lambda_compact > 0.0 and _tiers_c.get('A'):
                    _delta_a = getattr(args, 'delta_A', 1.0)
                    compact_loss, compact_metrics = tier_dpo.compactness_loss(
                        pi_out, _tiers_c['X'], _tiers_c['A'], _delta_a, device)
                else:
                    compact_loss = torch.tensor(0.0, device=device)
                    compact_metrics = {'n_pairs_used': 0, 'x_to_a_gap_mean': 0.0,
                                       'x_to_a_gap_max': 0.0, 'ordering_violation_rate': 0.0,
                                       'band_violation_rate': 0.0}
                if phase_cfg is not None and loss_components is not None:
                    loss = (
                        phase_cfg.lambda_sml * loss_components['loss_sml']
                        + phase_cfg.lambda_dpo_eject * eject_loss
                        + phase_cfg.lambda_dpo_top * top_loss
                        + phase_cfg.lambda_compactness * compact_loss
                    )
                else:
                    loss = (
                        base_loss
                        + args.lambda_dpo_eject * eject_loss
                        + args.lambda_dpo_top * top_loss
                        + _lambda_compact * compact_loss
                    )
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
                    _scores = pi['out'].squeeze()
                    _rmsd = rmsd_s.squeeze() if rmsd_s.dim() > 1 else rmsd_s
                    _lddt = (
                        h3_lddt_s.squeeze()
                        if h3_lddt_s is not None and h3_lddt_s.dim() > 1
                        else h3_lddt_s
                    )
                    out.update(compute_top1_h3_validation_metrics(_rmsd, _scores, _lddt))

            loss = _apply_ord_aux_loss(
                loss, out, pi, h3_lddt_s, h3_loop_len, ord_loss_fn, args, device,
            )
            out['final_loss'] = loss.detach()
            info_dict[pdb] = {
                k: (v.detach().cpu().item() if isinstance(v, torch.Tensor) else v)
                for k, v in out.items()
            }
            loss = loss / args.accumulate_grad_batches

            epoch_loss = update_epoch_loss(epoch_loss, out)

            if is_train:
                def _zero_dummy():
                    """Dummy loss touching both out and ord_head for DDP static_graph."""
                    d = pi['out'].sum() * 0.0
                    ol = pi.get('ord_logits')
                    return d + ol.sum() * 0.0 if ol is not None else d

                if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
                    loss = _zero_dummy()
                elif torch.isnan(loss):
                    nan_path = f'/home/sujin/projects/h3-loop-modeling/libs/results/{args.param_name}/nan/'
                    os.makedirs(nan_path, exist_ok=True)
                    with open(f'{nan_path}/{pdb}.dat', 'wb') as fp:
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
    with open('/home/sujin/projects/h3-loop-modeling/libs/results/%s/%s.%i.%i.info' % (
              args.param_name, train_tag, epoch_idx, local_rank), 'wb') as fp:
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

    # Load reference-policy checkpoint.
    # Priority: save_model_path (explicit reference) -> load_ckpt_path (resume-only fallback).
    reference_ckpt_path = None
    if args.save_model_path and os.path.exists(args.save_model_path):
        reference_ckpt_path = args.save_model_path
    elif args.load_ckpt_path and os.path.exists(args.load_ckpt_path):
        reference_ckpt_path = args.load_ckpt_path
        if local_rank == 0:
            print('[finetune] save_model_path not set; using load_ckpt_path as reference model')
    else:
        raise FileNotFoundError(
            f'No valid reference checkpoint found. '
            f'save_model_path={args.save_model_path}, load_ckpt_path={args.load_ckpt_path}'
        )

    checkpoint = torch.load(reference_ckpt_path, map_location={'cuda:0': f'cuda:{local_rank}'})
    missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
    if local_rank == 0:
        print(f'[finetune] loaded reference checkpoint: {reference_ckpt_path}')
        if missing:
            print(f'[finetune] missing keys (expected for ord_head): {missing}')
        if unexpected:
            print(f'[finetune] unexpected keys: {unexpected}')

    pre_trained = copy.deepcopy(model)
    pre_trained.half()
    model.to(device=device)
    pre_trained.to(device=device)
    for param in pre_trained.parameters():
        param.requires_grad = False
    torch.cuda.empty_cache()
    if local_rank == 0:
        print('[finetune] reference model converted to FP16 (saves ~50% weight memory)')

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999), weight_decay=args.weight_decay)
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
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        model._set_static_graph()
        if local_rank == 0:
            print('[finetune] DDP static_graph enabled', flush=True)

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
        _libs_dir = '/home/sujin/projects/h3-loop-modeling/libs/'
        if _libs_dir not in sys.path:
            sys.path.insert(0, _libs_dir)
        from dataset.config import load_dataset_spec
        _yaml_spec = load_dataset_spec(dataset_config)
        n_pairs = max(16, _yaml_spec.n_decoy // 4)
        if local_rank == 0:
            print(f'[finetune] YAML dataset config loaded: {dataset_config}')
            print(f'[finetune] n_decoy={_yaml_spec.n_decoy}, n_pairs={n_pairs}')

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
    result_dir = f'/home/sujin/projects/h3-loop-modeling/libs/results/{args.param_name}'
    os.makedirs(result_dir, exist_ok=True)

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

    for epoch_idx in range(epoch_start + 1, num_epochs + 1):
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

        args.save_ckpt_path = f'{result_dir}/{args.param_name}_{epoch_idx}.pt'
        args.ckpt_interval = 1
        if epoch_idx % args.ckpt_interval == 0:
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

            log_dict = {
                "valid loss": epoch_loss['final_loss'],
                "best rank": epoch_loss['best_rank'],
                "rmsd_diff": epoch_loss['rmsd_diff'],
                "top1_rmsd": epoch_loss['top1_rmsd'],
                "top3_rmsd": epoch_loss['top3_rmsd'],
                "top5_rmsd": epoch_loss['top5_rmsd'],
                "top10_rmsd": epoch_loss['top10_rmsd'],
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
                loss, loss_dic = loss_fn(pdb, final_rmsd, final_out, training, device_, inference_mode, rmsd_cutoff=2.0, loss_type=args.loss_type)
                loss_dic['pred'] = final_out
                loss_dic['h3_rmsd'] = final_rmsd
                loss_dic['rank'] = final_rank
                if final_ag_local is not None:
                    loss_dic['ag_local_rmsd'] = final_ag_local.detach().cpu().tolist()
                info_dict[pdb] = loss_dic
            else:
                # Use ranking_list directly, do not normalize
                pred = model(batched_graph)
                device_ = pred['out'].device
                loss, loss_dic = loss_fn(pdb, rmsd_s, pred['out'], training, device_, inference_mode, rmsd_cutoff=2.0, loss_type=args.loss_type)
                loss_dic['pred'] = pred['out']
                loss_dic['h3_rmsd'] = rmsd_s
                loss_dic['rank'] = ranking_list
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
    with open(f'/home/sujin/projects/h3-loop-modeling/libs/inference/{args.param_name}.info', 'wb') as fp:
        pickle.dump(info_dict, fp)
    return None
    

def print_parameters_count(model):
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of trainable parameters: {num_params_trainable}')


if __name__ == '__main__':
    is_distributed = init_distributed()
    local_rank = get_local_rank()
    args = PARSER.parse_args()
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
        # result_dir = '/home/sujin/projects/h3-loop-modeling/libs/results'
        result_dir = '/home/dykim/h3_loop_modeling/results/finetune'
        trained_model_dir = f'{args.save_model_path}'
        
        checkpoint = torch.load(trained_model_dir, map_location={'cuda:0': f'cuda:{get_local_rank()}'})
        model.load_state_dict(checkpoint['state_dict'])
        
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999),weight_decay=args.weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


        inference(
            model,
            callbacks,
            logger,
            args
        )
    
        logging.info('===========================================')
        logging.info('|            Inference Finished           |')
        logging.info('===========================================')
