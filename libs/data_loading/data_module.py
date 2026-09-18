import torch.distributed as dist
from abc import ABC
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from data_loading.graph_generation_from_target import (
    generate_graphs_from_target,
    create_dictionary_from_model,
    build_edge_mask,
    build_graph,
    decoy_identity_from_model,
    _get_model_metric,
)
from runtime.utils import get_local_rank
from runtime.arguments import PARSER
from data_loading.pdb2dict import Target, Model
import dgl
import sys
import io
import pickle,torch,random,os,glob,time
from pathlib import Path
import logging as _logging
from evaluation.loop_metrics import compute_loop_metrics_from_structures

# Dedicated module logger. Its level defaults to WARNING so that the per-target
# diagnostic INFO logs in _getitem_yaml do not spam training output, while the
# root logger stays at INFO for train.py progress logs. Override with e.g.
#   logging.getLogger("data_loading.data_module").setLevel(logging.INFO)
_dm_logger = _logging.getLogger(__name__)
_dm_logger.setLevel(_logging.WARNING)

DB_DIR = '/home/sujin/DB/h3-loop-modeling'

# args = PARSER.parse_args()
from runtime.arguments import PARSER
...
def _running_in_notebook():
    return 'ipykernel' in sys.modules or any(
        a == '-f' or a.startswith('--f=') for a in sys.argv[1:]
    )

def _init_module_args():
    if _running_in_notebook():
        return PARSER.parse_args([])
    return PARSER.parse_args()

def set_runtime_args(parsed_args):
    global args
    args = parsed_args

args = _init_module_args()


class _TargetUnpickler(pickle.Unpickler):
    """Redirect pickled module references to data_loading.pdb2dict classes.

    Pickle files created by running pdb2dict_combined.py (or pdb2dict.py)
    directly store classes under __main__.  This unpickler maps those
    references to the canonical data_loading.pdb2dict module so they can
    be loaded from any entry-point script.
    """
    _REDIRECT_MODULES = frozenset({
        "__main__", "pdb2dict_combined",
        "pdb2dict_ag", "pdb2dict_pipeline",
    })
    _CLASS_MAP = {"Target": Target, "Model": Model}

    def find_class(self, module: str, name: str):
        if module in self._REDIRECT_MODULES and name in self._CLASS_MAP:
            return self._CLASS_MAP[name]
        return super().find_class(module, name)


def _load_target_pickle(path: str):
    """Load a Target pickle with module-path redirection."""
    with open(path, 'rb') as f:
        return _TargetUnpickler(f).load()


def _timed_load_target_pickle(path: str):
    """Load a Target pickle and split file-read vs unpickle time."""
    t0 = time.perf_counter()
    with open(path, 'rb') as f:
        payload = f.read()
    t1 = time.perf_counter()
    obj = _TargetUnpickler(io.BytesIO(payload)).load()
    t2 = time.perf_counter()
    return obj, t1 - t0, t2 - t1


class _GraphPickleUnpickler(pickle.Unpickler):
    """DGL 1.x graph pickles reference DGLHeteroGraph; DGL 2.x uses DGLGraph."""

    _DGL_MODULES = frozenset({
        "dgl", "dgl.graph", "dgl.heterograph", "dgl.backend",
    })

    def find_class(self, module: str, name: str):
        if name in ("DGLHeteroGraph", "DGLGraph") and module in self._DGL_MODULES:
            return dgl.DGLGraph
        return super().find_class(module, name)


def _load_graph_pickle(path: str):
    """Load a graph-list pickle saved with an older DGL version."""
    with open(path, "rb") as f:
        return _GraphPickleUnpickler(f).load()


def _model_loop_label(model, spec=None):
    task_scope = getattr(spec, "task_scope", "full_cdr") if spec is not None else "full_cdr"
    label_metric = getattr(spec, "label_metric", "loop_rmsd") if spec is not None else "loop_rmsd"
    value = _get_model_metric(model, label_metric, task_scope=task_scope)
    if torch.isfinite(torch.tensor(value)):
        return value
    if task_scope == "h3":
        return _get_model_metric(model, "h3_rmsd", task_scope="h3")
    return float("nan")


def _loop_ranges_from_spec(spec):
    gb = getattr(spec, "graph_build", None)
    raw_ranges = getattr(gb, "cdr_ranges", None)
    if not raw_ranges:
        return None
    return {
        chain: tuple((int(start), int(end)) for start, end in ranges)
        for chain, ranges in raw_ranges.items()
    }


def _model_loop_label_from_target(model, target, spec=None):
    value = _model_loop_label(model, spec)
    if torch.isfinite(torch.tensor(value)):
        return value

    native_structure = getattr(target, "gt_structure", None)
    model_structure = getattr(model, "md_structure", None)
    if native_structure is None or model_structure is None:
        return value

    try:
        loop_ranges = _loop_ranges_from_spec(spec)
        kwargs = {"loop_ranges": loop_ranges} if loop_ranges else {}
        metrics = compute_loop_metrics_from_structures(
            native_structure,
            model_structure,
            **kwargs,
        )
    except Exception as exc:
        _logging.debug("Failed to compute on-the-fly loop metrics: %s", exc)
        return value

    model.loop_rmsd = metrics.loop_rmsd
    model.loop_lddt = metrics.loop_lddt
    for cdr_name, cdr_value in metrics.per_cdr_rmsd.items():
        setattr(model, f"{cdr_name.lower()}_rmsd", cdr_value)
    for cdr_name, cdr_value in metrics.per_cdr_lddt.items():
        setattr(model, f"{cdr_name.lower()}_lddt", cdr_value)
    return _model_loop_label(model, spec)


def _graph_build_cdr_ranges(gb):
    return getattr(gb, "cdr_ranges", None)


def graph_build_param_dict(spec, gb):
    """Canonical dict of every graph-affecting build param.

    Shared by the cache builder (``preprocess/build_graph_cache.py``) and the
    read-time staleness check so both agree on what defines a cache generation.
    ``dist_cutoff`` is the wide *envelope* the cache is (to be) built at; the
    exact-match subset (see ``graph_pack.BUILD_SIG_EXACT_KEYS``) feeds the
    build_sig, ``dist_cutoff`` is the coverage dimension checked separately.
    """
    cr = _graph_build_cdr_ranges(gb)
    return {
        'dist_cutoff': float(getattr(gb, 'dist_cutoff_center', 10.0)),
        'max_neighbors': int(getattr(gb, 'max_neighbors', 0)),
        'use_all_atom': bool(getattr(gb, 'use_all_atom', False)),
        'cdr_context_cutoff': float(getattr(gb, 'cdr_context_cutoff', 15.0)),
        'max_context_residues': int(getattr(gb, 'max_context_residues', 120)),
        'h3_range': list(getattr(gb, 'h3_range', (95, 102))),
        'cdr_ranges': (None if cr is None else
                       {k: list(v) for k, v in dict(cr).items()}) if not isinstance(cr, (list, tuple))
                      else [list(x) for x in cr],
        'task_scope': str(getattr(spec, 'task_scope', 'full_cdr')),
    }

# ── Lazy imports for YAML-driven pipeline (avoid import errors when not used) ──
_dataset_pkg_loaded = False
def _ensure_dataset_pkg():
    global _dataset_pkg_loaded
    if _dataset_pkg_loaded:
        return
    # Ensure libs/ is on sys.path so "dataset" package is importable
    _libs = str(Path(__file__).resolve().parent.parent)
    if _libs not in sys.path:
        sys.path.insert(0, _libs)
    _dataset_pkg_loaded = True


def consider_decoy_distr(pdb, rmsd, n_sel_decoy=100, n_near_native=20, rmsd_cutoff=2.0):
    print('### CONSIDER DECOY DISTRIBUTION ###')
    selected_decoy=[]; decoy_rmsd=[]
    near_native_idx=[]; non_native_idx=[]
    for i in range(len(rmsd)):
        if rmsd[i]<=rmsd_cutoff:
            near_native_idx.append(i)
        else:
            non_native_idx.append(i)
    if len(near_native_idx)<n_near_native:
        print(f'near native decoy is less than {n_near_native} for {pdb}: {len(near_native_idx)}')
        selected_decoy = random.sample(near_native_idx, k=len(near_native_idx))
    else:
        selected_decoy = random.sample(near_native_idx, k=n_near_native)
    selected_decoy.extend(random.sample(non_native_idx, k=n_sel_decoy-n_near_native))
    random.shuffle(selected_decoy)
    decoy_rmsd = [rmsd[i] for i in selected_decoy]

    return selected_decoy, decoy_rmsd

def graph_from_index(pdb, index_list, graph_list):
    sel_graph_list = []
    for i in range(len(index_list)):
        sel_graph_list.append(graph_list[index_list[i]])
    return dgl.batch(sel_graph_list)


# Interface metrics require an antibody-antigen COMPLEX; they are undefined for an
# apo target and must be NaN there, not a "perfect" constant.
_INTERFACE_METRICS = ("fnat", "dockq")


def _is_interface_metric(metric_name):
    lm = str(metric_name).lower()
    return any(k in lm for k in _INTERFACE_METRICS)


def _pdb_id_is_holo(pdb_id):
    """True when *pdb_id* names an antibody-antigen complex.

    Target ids follow ``{pdb}_{Hchain}_{Lchain}[_{antigen…}]`` — a 4th field means
    an antigen is present. Validated against every target in the precomputed
    interface parquets (2203 targets: 1440 holo / 763 apo, **0 disagreements**
    between this rule and "the store holds a finite fnat for this target").
    """
    return len(str(pdb_id).split("_")) >= 4



def _get_dataloader(dataset: Dataset, shuffle: bool, **kwargs) -> DataLoader:
    # Classic or distributed dataloader depending on the context
    sampler = DistributedSampler(dataset, shuffle=shuffle) if dist.is_initialized() else None
    return DataLoader(dataset, shuffle=(shuffle and sampler is None), sampler=sampler, drop_last=True,**kwargs)
def merge_save(pdb_tag,fn1,fn2,fn3,fn4):
    def read_pickle(fn):
        with open(fn,'rb')as fp:
            dat=pickle.load(fp)
        return dat
    dic={}
    dic['graph_falc_fn']=read_pickle(fn1)
    dic['graph_pertmd_fn']=read_pickle(fn2)
    dic['rmsd_falc_fn']=read_pickle(fn3)
    dic['rmsd_pertmd_fn']=read_pickle(fn4)
    with open('/store/hu_tmp/sujin_graph/%s.dat'%pdb_tag,'wb')as fp:
        pickle.dump(dic,fp)


class DataModule(ABC):
    """ Abstract DataModule. Children must define self.ds_{train | val | test}. """

    def __init__(self, **dataloader_kwargs):
        super().__init__()
        if get_local_rank() == 0:
            self.prepare_data()

        # Wait until rank zero has prepared the data (download, preprocessing, ...)
        if dist.is_initialized():
            dist.barrier(device_ids=[get_local_rank()])

        self.dataloader_kwargs = {'pin_memory': True, 'persistent_workers': False,
                                  **dataloader_kwargs}
        self.ds_train, self.ds_val, self.ds_test = None, None, None

    def prepare_data(self):
        """ Method called only once per node. Put here any downloading or preprocessing """
        pass

    def train_dataloader(self, train_set, decoytype=None, dataset_config=None) -> DataLoader:
        self.ds_train = MyDataset(train_set, is_train=True, decoytype=decoytype, dataset_config=dataset_config)
        return _get_dataloader(self.ds_train, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self, val_set, decoytype=None, dataset_config=None) -> DataLoader:
        self.ds_val = MyDataset(val_set, is_train=False, decoytype=decoytype, dataset_config=dataset_config)
        return _get_dataloader(self.ds_val, shuffle=False, **self.dataloader_kwargs)

    def test_dataloader(self,test_set,datatype=None,pickle_pdb=False,decoytype=None,use_subdirectory=True,db_dir=None,dataset_config=None) -> DataLoader:
        self.datatype=datatype
        if pickle_pdb:
            # Use provided db_dir or default to legacy path
            if db_dir is None:
                db_dir = Path(DB_DIR)/'ab_ag'/'igfold_ordered'/'decoy'
            else:
                db_dir = Path(db_dir)
            self.ds_test = IgFoldTestSet(pdb_list_file=Path(test_set), db_dir=db_dir, decoy_type=decoytype, use_subdirectory=use_subdirectory)
        else:
            self.ds_test = MyDataset(
                test_set,
                is_train=False,
                datatype=self.datatype,
                decoytype=decoytype,
                dataset_config=dataset_config,
            )
        return _get_dataloader(self.ds_test, shuffle=False, **self.dataloader_kwargs)

dir_path='/home/sujin/practice/prac_dl/gat/dat/pickle_new'

class MyDataset(Dataset):
    def __init__(self, inp_dat, is_train=True, datatype=None, decoytype=None, dataset_config=None):
        self.inp_dat = inp_dat  # list of pdb ids
        self.is_train = is_train
        self.datatype = datatype
        self.decoytype = decoytype
        # ── YAML-driven pipeline (optional) ──
        self._ds_spec = None
        self._registry = None
        self._epoch = 1
        self._metric_store = None
        # v2 fixed-validation manifest: {target_id: {(source, norm_seed, sample), ...}}.
        # When set (via load_val_manifest), _getitem_yaml selects exactly these
        # decoys for validation, bypassing the tier sampler / DockQ gate / source
        # weights so the eval pool is identical across phase/epoch/head.
        self._manifest_keys = None
        self._manifest_meta = None
        # ── graph cache (Layer-B indexed containers; see libs/data_loading/graph_pack.py)
        # Mode/dir come from env (set by train.py from --graph_cache_mode/-dir) or
        # the YAML spec.graph_cache; default off. In 'read' mode a valid pack for a
        # (target, source) supplies graphs (and label keys) without loading the
        # pdb2dict pickle; a missing/stale pack falls back to on-the-fly per item.
        self._graph_cache_mode = 'off'
        self._graph_cache_dir = None
        self._graph_cache_verify_struct = True
        self._graph_cache_mmap = True
        self._pack_handles = {}        # (sname, target) -> GraphPack | None (per worker)
        self._pack_lru = []            # LRU order of open handle keys
        self._expected_build_sig = None
        self._cache_disabled_reason = None
        if dataset_config is not None:
            _ensure_dataset_pkg()
            from dataset.config import load_dataset_spec
            from dataset.source_registry import SourceRegistry
            self._ds_spec = load_dataset_spec(dataset_config)
            self._registry = SourceRegistry(self._ds_spec)
            _logging.info('MyDataset: YAML config loaded from %s', dataset_config)
            pm = getattr(self._ds_spec, 'precomputed_metrics', None)
            if pm is not None and pm.enabled:
                from dataset.precomputed_metrics import PrecomputedMetricStore
                self._metric_store = PrecomputedMetricStore(
                    root=pm.root,
                    source_subdirs=pm.sources,
                    metrics_filename=pm.metrics_filename,
                )
                _logging.debug(
                    'MyDataset: precomputed metrics enabled (root=%s, require=%s)',
                    pm.root, pm.require,
                )
            # graph-cache config: env overrides YAML spec.graph_cache
            gc = getattr(self._ds_spec, 'graph_cache', None)
            self._graph_cache_mode = str(
                os.environ.get('CDR_GRAPH_CACHE_MODE',
                               getattr(gc, 'mode', None) if gc else None) or 'off').lower()
            self._graph_cache_dir = (
                os.environ.get('CDR_GRAPH_CACHE_DIR')
                or (getattr(gc, 'dir', None) if gc else None))
            if os.environ.get('CDR_GRAPH_CACHE_VERIFY_STRUCT') is not None:
                self._graph_cache_verify_struct = (
                    os.environ['CDR_GRAPH_CACHE_VERIFY_STRUCT'] not in ('0', 'false', 'False'))
            if self._graph_cache_mode == 'read' and self._graph_cache_dir:
                self._init_graph_cache()

    def _init_graph_cache(self):
        """Resolve the expected build_sig for the current run (once per worker).

        The loader is self-sufficient: it recomputes the build_sig from the live
        graph-build params + graph-code hash and compares it against each pack's
        stored sig, so a param/code change is *detected* and falls back to
        on-the-fly rather than silently reading stale graphs.
        """
        try:
            from data_loading.graph_pack import compute_build_sig, compute_code_sig
            gb = self._ds_spec.graph_build
            params = graph_build_param_dict(self._ds_spec, gb)
            self._expected_build_sig = compute_build_sig(params, compute_code_sig())
            self._graph_cache_params = params
            _logging.info('MyDataset: graph cache READ dir=%s expected_build_sig=%s',
                          self._graph_cache_dir, self._expected_build_sig)
        except Exception as e:
            self._cache_disabled_reason = f'init failed: {e}'
            self._graph_cache_mode = 'off'
            _logging.warning('MyDataset: graph cache disabled (%s)', self._cache_disabled_reason)

    def _get_pack(self, sname, resolved_target):
        """Return a validated GraphPack for (source, target) or None (fallback).

        Never raises: any failure (missing file, bad sig, stale struct) returns
        None so the caller builds that item on the fly. Handles are cached per
        worker with a small LRU bound to avoid fd/mmap growth.
        """
        if self._graph_cache_mode != 'read' or not self._graph_cache_dir:
            return None
        key = (sname, str(resolved_target))
        if key in self._pack_handles:
            self._pack_lru.remove(key); self._pack_lru.append(key)
            return self._pack_handles[key]
        pack = None
        try:
            from data_loading.graph_pack import GraphPack, compute_struct_sig
            path = os.path.join(self._graph_cache_dir, sname, f'{resolved_target}.gpk')
            if os.path.exists(path):
                pk = GraphPack(path, mode='mmap' if self._graph_cache_mmap else 'pread')
                ok = True
                if self._expected_build_sig and pk.build_sig != self._expected_build_sig:
                    _logging.warning('graph cache: build_sig mismatch %s/%s (pack=%s exp=%s) -> on-the-fly',
                                     sname, resolved_target, pk.build_sig, self._expected_build_sig)
                    ok = False
                if ok and self._graph_cache_verify_struct and pk.struct_sig is not None:
                    src_files = (pk.header.get('meta') or {}).get('source_files') or []
                    if src_files and compute_struct_sig(src_files, pk.keys) != pk.struct_sig:
                        _logging.warning('graph cache: struct_sig stale %s/%s -> on-the-fly',
                                         sname, resolved_target)
                        ok = False
                if ok:
                    pack = pk
                else:
                    pk.close()
        except Exception as e:
            _logging.warning('graph cache: open failed %s/%s (%s) -> on-the-fly',
                             sname, resolved_target, e)
            pack = None
        # insert (even None: caches the negative result so we don't re-stat every epoch... but
        # workers are per-epoch, so None is fine to cache within an epoch)
        self._pack_handles[key] = pack
        self._pack_lru.append(key)
        while len(self._pack_lru) > 32:
            old = self._pack_lru.pop(0)
            oldpk = self._pack_handles.pop(old, None)
            if oldpk is not None:
                oldpk.close()
        return pack

    def set_epoch(self, epoch: int):
        """Called at the start of each epoch so schedulable params update."""
        self._epoch = epoch

    def load_val_manifest(self, path):
        """Load a fixed validation manifest (see preprocess/build_val_manifest.py).

        Returns meta dict {version, hash, pool, n_targets, n_decoys, gate}. Also
        sets self.inp_dat to the manifest target list so iteration is over the
        fixed pool. Decoys are keyed by (source, seed, sample), never by index.
        """
        import json as _json
        with open(path) as f:
            man = _json.load(f)

        def _ns(s):
            return None if s is None else int(s)
        self._manifest_keys = {
            str(tid): {(str(src), _ns(seed), int(samp)) for src, seed, samp in cands}
            for tid, cands in man["targets"].items()
        }
        self._manifest_meta = {k: man.get(k) for k in
                               ("version", "hash", "pool", "n_targets", "n_decoys", "gate")}
        self._manifest_meta["path"] = str(path)
        self.inp_dat = list(self._manifest_keys.keys())
        return self._manifest_meta

    def __len__(self):
        return len(self.inp_dat)
    
    def use_fixed_decoy(self, rmsd, pdb, decoy_dic):
        decoy_idx = decoy_dic[pdb]
        graph_set = []; decoy_rmsd=[]
        for i in range(len(decoy_idx)):
            graph_set.append(self.graph_list[decoy_idx[i]])
            decoy_rmsd.append(rmsd[decoy_idx[i]])
        return dgl.batch(graph_set), decoy_idx, decoy_rmsd
        
    # Select the decoy indicies without binning
    # Forced to contain at least one decoy with RMSD higher than the cutoff among selected 64 decoys
    def select_wo_binning(self,rmsd,pdb,n_decoy=64):
        sel_idx = []; sel_rmsd = []
        total_idx = list(range(len(rmsd)))
        sel_idx = random.sample(total_idx, k=n_decoy)
        graph_list64 = []
        try:
            for i in range(n_decoy):
                graph_list64.append(self.graph_list[sel_idx[i]])
                sel_rmsd.append(rmsd[sel_idx[i]])
        except Exception as e:
            print('error occured ', e)
            print(pdb)
            print('len graph_list ',len(self.graph_list), len(self.graph_list[0]))
            print('sel idx ',len(sel_idx),sel_idx)
            print('i ',i)
            print('\n')
        return dgl.batch(graph_list64), sel_idx, sel_rmsd
    
    # near-native decoy (RMSD<=2) included in selected decoy set
    # also non-native decoy (RMSD>2) should be included in selected decoy set
    def select_wo_binning_rmsd(self,rmsd,pdb,n_decoy=64):
        sel_idx=[]; sel_rmsd=[]
        total_idx=list(range(len(rmsd)))
        # 'nnd' for near-native decoys
        nnd_idx2=[i for i in total_idx if rmsd[i]<=2.0] # RMSD<2인 decoy
        nnd_idx1=[i for i in total_idx if rmsd[i]<=1.0]
        non_nat_idx2=[i for i in total_idx if rmsd[i]>2.0] # RMSD>2인 decoy
        '''if not (len(nnd_idx2)==len(total_idx)): # RMSD<2인 decoy가 전체 decoy가 아닐 때 -> non-native가 하나라도 존재
            num_sel_nnd=min(16,len(nnd_idx2))
            sel_nnd_16 = random.sample(nnd_idx2, k=num_sel_nnd)
            sel_non_1 = random.sample([i for i in total_idx if i not in nnd_idx2], k=1)
            sel_remain = random.sample([i for i in total_idx if i not in sel_nnd_16], k=n_decoy-num_sel_nnd-1)
            sel_idx = sel_nnd_16 + sel_non_1 + sel_remain
        else:
            sel_idx=random.sample(total_idx,k=n_decoy) # 전부 near-native만 있음'''
        ###
        assert len(nnd_idx2)!=0
        assert len(non_nat_idx2)!=0
        num_sel_near = min(16, len(nnd_idx2))
        num_sel_non = min(16, len(non_nat_idx2))
        sel_near_16 = random.sample(nnd_idx2, k=num_sel_near)
        sel_non_16 = random.sample(non_nat_idx2, k=num_sel_non)
        sel_remain = random.sample([i for i in total_idx if i not in sel_near_16+sel_non_16], k=n_decoy-num_sel_near-num_sel_non)
        sel_idx = sel_near_16 + sel_non_16 + sel_remain

        random.shuffle(sel_idx)
        graph_list64 = []
        
        try:
            for i in range(n_decoy):
                graph_list64.append(self.graph_list[sel_idx[i]])
                sel_rmsd.append(rmsd[sel_idx[i]])
        except Exception as e:
            print('error occured ', e)
            print(pdb)
            print('len graph_list ',len(self.graph_list),len(nnd_idx1),len(nnd_idx2),len(total_idx),"!!!!!!!!!!!!!",tag)
            print('sel idx ',len(sel_idx),sel_idx)
            print('i ',i)
            print('\n')
        return dgl.batch(graph_list64), sel_idx, sel_rmsd
    
    def select_all_decoys(self,rmsd,pdb,n_decoy=None):
        # For inference mode, do not select decoys randomly
        sel_idx = []; sel_rmsd = []
        sel_idx = list(range(len(rmsd))) # sel_idx = total_idx(in ramdom sampling)
        graph_list_total = []
        try:
            for i in range(len(rmsd)):
                graph_list_total.append(self.graph_list[sel_idx[i]])
                sel_rmsd.append(rmsd[sel_idx[i]])
        except Exception as e:
            print('error occured ', e)
            print(pdb)
            print('len graph_list ',len(self.graph_list))
            print('sel idx ',len(sel_idx),sel_idx)
            print('i ',i)
            print('\n') 
            sys.exit()
        if n_decoy:
            graph_list_total=graph_list_total[:n_decoy]
            sel_idx=sel_idx[:n_decoy]
            sel_rmsd=sel_rmsd[:n_decoy]
        return dgl.batch(graph_list_total), sel_idx, sel_rmsd

    def select_nnd(self,rmsd,pdb):
        sel_idx = [i for i in range(len(rmsd)) if rmsd[i]<=2.0][:100]
        sel_rmsd = [rmsd[i] for i in sel_idx]
        graph_list_nnd = []
        if len(sel_idx)==0:
            print('No near-native decoy for ',pdb)
            sys.exit()
        try:
            for i in range(len(sel_idx)):
                graph_list_nnd.append(self.graph_list[sel_idx[i]])
        except Exception as e:
            print('error occured ', e)
            print(pdb)
            print('len graph_list ',len(self.graph_list))
            print('sel idx ',len(sel_idx),sel_idx)
            print('i ',i)
            print('\n')
            sys.exit()
        return dgl.batch(graph_list_nnd), sel_idx, sel_rmsd
    
    def select_near_non(self, rmsd):
        sel_near_idx = [i for i in range(len(rmsd)) if rmsd[i]<=2.0]
        sel_non_idx = [i for i in range(len(rmsd)) if rmsd[i]>2.0]
        sel_near_16 = random.sample(sel_near_idx, k=min(16, len(sel_near_idx)))
        sel_non_48 = random.sample(sel_non_idx, k=min(48, len(sel_non_idx)))
        sel_idx = sel_near_16 + sel_non_48
        sel_rmsd = [rmsd[i] for i in sel_idx]
        graph_list = [self.graph_list[i] for i in sel_idx]
        return dgl.batch(graph_list), sel_idx, sel_rmsd

    def __getitem__(self,index):
        # self.datatype=None
        pdb = self.inp_dat[index] 

        # ── YAML-exclusive path: skip ALL hardcoded logic below ──
        # When dataset_config (YAML) is provided, the entire dataset is
        # controlled by YAML — sources, paths, RMSD filtering, mixing.
        if self._ds_spec is not None:
            return self._getitem_yaml(pdb)

        # ── Original hardcoded path (backward compat, runs when no YAML) ──
        n_decoy=64
        ulr_len=-1
        if self.datatype is None:
            if len(pdb)==4:
                self.datatype='GP'
            else:
                self.datatype='AbAg'
            
        graph_fn = []
        rmsd_fn = []
        def for_db(cond):
            tmp = cond.split('-')
            cond = f'pertmd_{tmp[0]}_c300'
            if tmp[-1]=='256':
                cond += '_256'
            return cond
                
        if args.all_atom:
            if self.datatype=='GP':
                path = f'{DB_DIR}/general_protein'
            elif self.datatype=='AbAg':
                path = f'{DB_DIR}/ab_ag/DB_final_train'
                
            graph_fn.append(f'{path}/graph-all-atom/1-falc/{pdb}.dat')
            graph_fn.append(f'{path}/graph-all-atom/2-pertmd-t1000-32/{pdb}.dat')
            rmsd_fn.append(f'{path}/decoy/{pdb}/{pdb}_fp.rmsd')
            
            if self.datatype=='AbAg':
                cond = 't500-64'
                graph_fn.append(f'{path}/graph-all-atom/3-pertmd-{cond}/{pdb}.dat')
                rmsd_fn.append(f'{path}/decoy/{pdb}/{for_db(cond)}/{pdb}_p.rmsd')
            
                with open(f'{path}/decoy/{pdb}/{pdb}.ulr') as ulr_file:
                    for line in ulr_file:
                        line = line.strip()
                        tmp = line.split(' ')[3]
                        ulr_len = int(tmp.split('-')[1])-int(tmp.split('-')[0])+1
                        if ulr_len>11:
                            cond = 't3000-64'
                            graph_fn.append(f'{path}/graph-all-atom/4-pertmd-{cond}/{pdb}.dat')
                            rmsd_fn.append(f'{path}/decoy/{pdb}/{for_db(cond)}/{pdb}_p.rmsd')
                        if ulr_len>18:
                            cond = 't3000-256'
                            graph_fn.append(f'{path}/graph-all-atom/5-pertmd-{cond}/{pdb}.dat')
                            rmsd_fn.append(f'{path}/decoy/{pdb}/{for_db(cond)}/{pdb}_p.rmsd')
            
        else: # when using original SE3 graph model (non all-atom)
            if self.datatype=='GP':
                graph_fn = [f'{DB_DIR}/general_protein/graph/se3/{pdb}.dat']
                rmsd_fn = [f'{DB_DIR}/general_protein/decoy/{pdb}/{pdb}_fp.rmsd']
                
            elif self.datatype=='AbAg':
                if args.run_type=='inference':
                    db_dir = 'DB_final_test_67'
                    n_decoy=1032
                else:
                    db_dir = 'DB_final_train'
                    
                graph_fn = [f'{DB_DIR}/ab_ag/{db_dir}/graph/se3/{pdb}.dat']
                rmsd_fn = [f'{DB_DIR}/ab_ag/{db_dir}/decoy/{pdb}/{pdb}_fp.rmsd']
                if self.decoytype=='fp_extended':
                    # TODO : Get decoy type information from dataloader using ULR (H3 loop length)
                    path_list=f'{DB_DIR}/ab_ag/{db_dir}/graph'
                    # Add t500_c300 for all targets
                    name = 't500_c300'
                    graph_fn.append(f'{path_list}/pertMD_{name}/{pdb}.dat')
                    rmsd_fn.append(f'{DB_DIR}/ab_ag/{db_dir}/decoy/{pdb}/pertmd_{name}/{pdb}_p.rmsd')
                    
                    with open(f'{DB_DIR}/ab_ag/{db_dir}/decoy/{pdb}/{pdb}.ulr') as ulr_file:
                        for line in ulr_file:
                            line = line.strip()
                            tmp = line.split(' ')[3]
                            ulr_len = int(tmp.split('-')[1])-int(tmp.split('-')[0])+1
                            if ulr_len>11:
                                name = 't3000_c300'
                                graph_fn.append(f'{path_list}/pertMD_{name}/{pdb}.dat')
                                rmsd_fn.append(f'{DB_DIR}/ab_ag/{db_dir}/decoy/{pdb}/pertmd_{name}/{pdb}_p.rmsd')
                            if ulr_len>18:
                                name = 't3000_c300_256'
                                graph_fn.append(f'{path_list}/pertMD_{name}/{pdb}.dat')
                                rmsd_fn.append(f'{DB_DIR}/ab_ag/{db_dir}/decoy/{pdb}/pertmd_{name}/{pdb}_p.rmsd')
                
            elif self.datatype=='igfold':
                if args.run_type=='inference':
                    db_dir = 'igfold_ordered'
                    n_decoy=1032+32+64+64+256
                graph_path = f'{DB_DIR}/ab_ag/{db_dir}/graph'
                decoy_path = f'{DB_DIR}/ab_ag/{db_dir}/decoy'
                # graph_fn = [f'{DB_DIR}/ab_ag/{db_dir}/graph/se3/{pdb}.dat']
                # rmsd_fn = [f'{DB_DIR}/ab_ag/{db_dir}/decoy/{pdb}/{pdb}_fp.rmsd']
                graph_fn = [f'{graph_path}/se3/{pdb}.dat', \
                        f'{graph_path}/3-pertmd-t500-64/{pdb}.dat', 
                        f'{graph_path}/4-pertmd-t3000-64/{pdb}.dat', \
                        f'{graph_path}/5-pertmd-t3000-256/{pdb}.dat' \
                    ]
                rmsd_fn = [f'{decoy_path}/{pdb}/{pdb}_fp.rmsd', \
                        f'{decoy_path}/{pdb}/pertmd_t500_c300/{pdb}_p.rmsd', \
                        f'{decoy_path}/{pdb}/pertmd_t3000_c300/{pdb}_p.rmsd', \
                        f'{decoy_path}/{pdb}/pertmd_t3000_c300_256/{pdb}_p.rmsd' \
                    ]
            
            elif self.datatype=='igfold_commat':
                if args.run_type=='inference':
                    db_dir = 'igfold_ordered'
                    n_decoy=32
                graph_fn = [f'{DB_DIR}/ab_ag/{db_dir}/graph/se3_commat/{pdb}.dat']
                rmsd_fn = [f'{DB_DIR}/ab_ag/{db_dir}/decoy/{pdb}/ComMat/{pdb}_c.rmsd']
            
            elif self.datatype=='igfold_4':
                if args.run_type=='inference':
                    db_dir = 'igfold'
                    n_decoy=4
                graph_fn = [f'{DB_DIR}/ab_ag/{db_dir}/graph/igfold_4/{pdb}.dat']
                rmsd_fn = [f'{DB_DIR}/ab_ag/{db_dir}/igfold_4/{pdb}/{pdb}_fp.rmsd']
            
            if not args.run_type=='inference':
                self.datatype=None
        
            
        self.graph_list=[]; rmsd_s=torch.Tensor([])
        for graph_file in graph_fn:
            try:
                graph_list_tmp = _load_graph_pickle(graph_file)
                self.graph_list.extend(graph_list_tmp)
            except FileNotFoundError:
                dir_path, fname = os.path.split(graph_file)
                prefix, ext = os.path.splitext(fname)
                pattern = os.path.join(dir_path, f"{prefix}*{ext}")
                matches = glob.glob(pattern)
                if matches:
                    graph_list_tmp = _load_graph_pickle(matches[0])
                else:
                    raise FileNotFoundError(f"Graph file {graph_file} not found and no matching files found in {dir_path}.")
                self.graph_list.extend(graph_list_tmp)
        for rmsd_file in rmsd_fn:
            if not os.path.exists(rmsd_file):
                continue
            try:
                with open(rmsd_file,'rb')as fp:
                    rmsd=pickle.load(fp)
                    if not isinstance(rmsd, torch.Tensor):
                        rmsd = torch.Tensor(rmsd)
                    rmsd_s=torch.cat([rmsd_s,rmsd],dim=0)
            except FileNotFoundError:
                dir_path, fname = os.path.split(rmsd_file)
                prefix, ext = os.path.splitext(fname)
                pattern = os.path.join(dir_path, f"{prefix}*{ext}")
                matches = glob.glob(pattern)
                if matches:
                    with open(matches[0], 'rb') as fp:
                        rmsd = pickle.load(fp)
                        if not isinstance(rmsd, torch.Tensor):
                            rmsd = torch.Tensor(rmsd)
                        rmsd_s=torch.cat([rmsd_s,rmsd],dim=0)
                else:
                    raise FileNotFoundError(f"RMSD file {rmsd_file} not found and no matching files found in {dir_path}.")
        rmsd=torch.Tensor(rmsd_s)
        
        
        
        # ── YAML pipeline override (LEGACY — unreachable when _ds_spec is set,
        #    because __getitem__ returns early via _getitem_yaml above) ──
        if self._ds_spec is not None and self.is_train:
            extra_graphs, extra_rmsds = self._load_graphs_yaml(pdb)
            if extra_graphs:
                self.graph_list.extend(extra_graphs)
                if not isinstance(extra_rmsds, torch.Tensor):
                    extra_rmsds = torch.Tensor(extra_rmsds)
                rmsd = torch.cat([rmsd, extra_rmsds], dim=0)

        if args.run_type=='inference':
            graph_set, decoy_idx, decoy_rmsd = self.select_all_decoys(rmsd, pdb) # falc 1000개에 대해서만 하고싶으면 n_decoy=1000
            # if selected decoy indices exsits, load them
            # with open(f'/home/sujin/h3-loop-modeling/libs/se3_transformer/inference/1205_test_sel_idx_nnd_over1.dat','rb') as f:
            #     decoy_dic = pickle.load(f)
            # graph_set, decoy_idx, decoy_rmsd = self.use_fixed_decoy(rmsd, pdb, decoy_dic)
        elif args.sel_nnd_only:
            graph_set, decoy_idx, decoy_rmsd = self.select_nnd(rmsd, pdb)
        elif args.sel_near_non:
            graph_set, decoy_idx, decoy_rmsd = self.select_near_non(rmsd)
        else:
            graph_set, decoy_idx, decoy_rmsd = self.select_wo_binning_rmsd(rmsd, pdb, n_decoy)
        
        # Free the full graph list; only the selected batch matters now
        self.graph_list = []
        return graph_set, torch.Tensor(decoy_rmsd), pdb

    # ──────────────────────────────────────────────────────────────
    # YAML-driven graph loading (called from __getitem__ when config is set)
    # ──────────────────────────────────────────────────────────────
    def _load_graphs_yaml(self, pdb_id: str):
        """Load additional decoy graphs from YAML-configured sources.

        Returns (list[DGLGraph], list[float])  – may be empty.
        """
        _ensure_dataset_pkg()
        from dataset.conversion import convert_and_cache
        from dataset.mixing import sample_mix

        spec = self._ds_spec
        epoch = self._epoch
        candidates = self._registry.get_candidates(pdb_id, epoch)
        if not candidates:
            return [], []

        # ── Load raw graphs per source ──
        per_source_graphs = {}   # source_name -> (graphs, rmsds)
        for cand in candidates:
            sname = cand.source_name
            if sname in per_source_graphs:
                continue  # already processed

            if cand.file_type == 'graph_pickle':
                graphs, rmsds = self._read_graph_pickle(cand.graph_path, cand.rmsd_path)
            elif cand.file_type == 'target_model_pickle':
                gb = spec.graph_build
                cache_root = spec.cache_root or os.path.join(spec.db_root, '_graph_cache')
                gp, rp = convert_and_cache(
                    cand.target_model_path, cache_root, sname, pdb_id,
                    dist_cutoff_center=gb.dist_cutoff_center,
                    random_range=gb.random_range,
                    max_neighbors=gb.max_neighbors,
                    use_all_atom=gb.use_all_atom,
                    h3_range=tuple(gb.h3_range),
                    cdr_ranges=_graph_build_cdr_ranges(gb),
                    task_scope=spec.task_scope,
                    label_metric=spec.label_metric,
                    cdr_context_cutoff=gb.cdr_context_cutoff,
                    max_context_residues=gb.max_context_residues,
                    graph_crop_debug=gb.graph_crop_debug,
                )
                if gp is None:
                    continue
                graphs, rmsds = self._read_graph_pickle(gp, rp)
            else:
                continue

            if not graphs:
                continue

            per_source_graphs[sname] = (graphs, rmsds)

        # ── Mixing ──
        counts = {s: len(g) for s, (g, _) in per_source_graphs.items()}
        rng = random.Random(spec.seed + epoch + hash(pdb_id) % 10000)
        allocation = sample_mix(counts, epoch, spec, rng)

        merged_g, merged_r = [], []
        for sname, n_sel in allocation.items():
            gs, rs = per_source_graphs[sname]
            if n_sel >= len(gs):
                merged_g.extend(gs)
                merged_r.extend(rs)
            else:
                idxs = rng.sample(range(len(gs)), k=n_sel)
                for i in idxs:
                    merged_g.append(gs[i])
                    merged_r.append(rs[i])

        return merged_g, merged_r

    @staticmethod
    def _read_graph_pickle(graph_path, rmsd_path):
        """Read a graph pickle and optional rmsd pickle."""
        graphs, rmsds = [], []
        if graph_path and os.path.exists(graph_path):
            graphs = _load_graph_pickle(graph_path)
        if rmsd_path and os.path.exists(rmsd_path):
            with open(rmsd_path, 'rb') as fp:
                r = pickle.load(fp)
                if isinstance(r, torch.Tensor):
                    rmsds = r.tolist()
                elif isinstance(r, list):
                    rmsds = r
                else:
                    rmsds = list(r)
        elif graphs:
            # No rmsd file → fill with NaN
            rmsds = [float('nan')] * len(graphs)
        return graphs, rmsds

    # ──────────────────────────────────────────────────────────────
    # YAML-exclusive loading path  (replaces hardcoded logic)
    # ──────────────────────────────────────────────────────────────
    def _labels_for_models(self, target_obj, spec, source_name, resolved_pdb_id, metric_name=None):
        """Return per-model loop labels (training targets) for *target_obj*.

        When a precomputed-metric store is configured and has data for
        *source_name*, labels are looked up from the parquet by decoy identity
        ``(target_id, seed, sample)`` instead of being recomputed from
        structures.  Falls back to on-the-fly computation when:
          - no store is configured, or
          - the source has no precomputed parquet, or
          - the store is in non-require mode and a decoy lookup misses.
        """
        models = target_obj.models
        label_metric = metric_name or getattr(spec, "label_metric", "loop_rmsd")
        if str(source_name).lower() == "xtal":
            lm = str(label_metric).lower()
            # The native IS the reference, so every metric takes its perfect value.
            # Only the rmsd family is lower-is-better (perfect = 0); dockq / lddt /
            # fnat are all higher-is-better (perfect = 1). Keyed on "rmsd" rather
            # than enumerating the good metrics, because the old enumeration
            # ("dockq" or "lddt" -> 1.0, else 0.0) silently sent **fnat** down the
            # rmsd branch and labelled the native complex fnat=0.0 — i.e. the best
            # possible structure was taught to the interface head as the worst.
            #
            # Interface metrics do not EXIST for an apo target: with no antigen
            # there are no native contacts, so "perfect contact recovery" is
            # meaningless. Emitting 1.0 there used to make every apo target carry
            # exactly one finite-fnat decoy, which (a) fed the interface objective
            # a vacuous absolute target and (b) since the regression gradient
            # scales as tau/sqrt(n), let those n=1 targets contribute 8x the
            # gradient of a normal n=64 target. NaN drops them out cleanly.
            if _is_interface_metric(lm) and not _pdb_id_is_holo(resolved_pdb_id):
                return [float("nan") for _ in models]
            value = 0.0 if "rmsd" in lm else 1.0
            return [value for _ in models]

        store = self._metric_store
        pm = getattr(spec, "precomputed_metrics", None)
        if (
            store is None
            or (pm is not None and pm.sources and source_name not in pm.sources)
            or not store.has_source(source_name)
        ):
            return [_model_loop_label_from_target(m, target_obj, spec) for m in models]

        from dataset.precomputed_metrics import decoy_identity_for_lookup, metric_column

        require = bool(pm.require) if pm is not None else True
        column = metric_column(
            label_metric,
            getattr(spec, "task_scope", "full_cdr"),
        )

        labels = []
        n_found = 0
        for pos, model in enumerate(models):
            seed, sample = decoy_identity_for_lookup(model, pos)
            value = store.lookup(source_name, resolved_pdb_id, seed, sample, column)
            if value == value:  # finite
                n_found += 1
            elif not require and column in ("cdr_rmsd", "cdr_lddt"):
                # On-the-fly fallback only for the primary loop label; the
                # h3_lddt / dockq eval channels must stay NaN on a miss (a loop
                # value there would corrupt H3 / CAPRI validation).
                value = _model_loop_label_from_target(model, target_obj, spec)
            labels.append(value)
        _logging.debug(
            "_getitem_yaml: precomputed labels %s/%s found=%d/%d column=%s",
            source_name, resolved_pdb_id, n_found, len(models), column,
        )
        return labels

    def _eval_metrics_for_models(self, target_obj, spec, source_name, resolved_pdb_id):
        """Return direction-neutral eval labels for validation/inference output.

        Includes h3_lddt and dockq so validation can report the top-1 decoy's
        H3 lDDT and CAPRI class (dockq is NaN for apo targets / sources without
        a dockq parquet, and is used validation-only, never in the loss).
        """
        return {
            "loop_rmsd": self._labels_for_models(
                target_obj, spec, source_name, resolved_pdb_id, metric_name="loop_rmsd",
            ),
            "loop_lddt": self._labels_for_models(
                target_obj, spec, source_name, resolved_pdb_id, metric_name="loop_lddt",
            ),
            "h3_lddt": self._labels_for_models(
                target_obj, spec, source_name, resolved_pdb_id, metric_name="h3_lddt",
            ),
            "dockq": self._labels_for_models(
                target_obj, spec, source_name, resolved_pdb_id, metric_name="dockq",
            ),
            "fnat": self._labels_for_models(
                target_obj, spec, source_name, resolved_pdb_id, metric_name="fnat",
            ),
        }

    @staticmethod
    def _read_rmsd_only(rmsd_path):
        """Load only RMSD values from a pickle file (lightweight, no graph loading)."""
        if not rmsd_path or not os.path.exists(rmsd_path):
            return []
        try:
            with open(rmsd_path, 'rb') as fp:
                r = pickle.load(fp)
            if isinstance(r, torch.Tensor):
                return r.tolist()
            elif isinstance(r, list):
                return r
            else:
                return list(r)
        except Exception:
            return []

    def _getitem_yaml(self, pdb_id, _depth=0):
        """YAML-exclusive loading: RMSD-first → filter → mix → load selected graphs.

        When dataset_config (YAML) is set, this method replaces ALL hardcoded
        path logic in __getitem__.  The pipeline:
          1. Resolve candidate sources from SourceRegistry
          2. Load RMSD pickles only (lightweight) for each source
          3. Training/valid: apply configurable RMSD range filter.
             Inference: no index filtering (all decoys kept for scoring; filter offline).
          4. Run source-weighted mixing to allocate n_decoy across sources
          5. Load graph pickles ONLY for allocated sources, extracting selected indices
          6. Return (batched_graph, rmsd_tensor, pdb_id, extra[, ag_local_tensor/meta/metrics in inference])

        Memory optimization: graph files for excluded sources are never loaded.
        Within allocated sources, only selected indices are kept; the rest are freed.
        """
        _ensure_dataset_pkg()
        from dataset.conversion import convert_and_cache
        from dataset.metrics import load_ag_local_rmsd

        spec = self._ds_spec
        epoch = self._epoch
        gb = spec.graph_build
        inference_yaml = (getattr(args, 'run_type', None) == 'inference')
        # v2 fixed-validation manifest mode: select exact (source,seed,sample)
        # decoys from the manifest, bypassing tier sampler / gate / source weights.
        manifest_mode = (self._manifest_keys is not None) and (not inference_yaml)
        manifest_target = self._manifest_keys.get(str(pdb_id)) if manifest_mode else None

        # Graph cache is usable unless this run needs on-the-fly H3 lDDT labels
        # (finetune + ord-aux), which require the structure (compute_q) and so
        # cannot be served from a pack. When conflicting, fall back to on-the-fly.
        _cache_h3_conflict = (getattr(args, 'run_type', None) == 'finetune'
                              and getattr(args, 'use_ord_aux_loss', False))
        cache_enabled = (self._graph_cache_mode == 'read'
                         and self._graph_cache_dir is not None
                         and not _cache_h3_conflict)

        def _mnorm_seed(s):
            if s is None or (isinstance(s, float) and s != s) or s == -1:
                return None
            return int(s)
        profile_data_loading = bool(
            getattr(spec, "profile_data_loading", False)
            or getattr(gb, "graph_crop_debug", False)
        )
        data_profile = {
            "read_structure_time": 0.0,
            "parse_structure_time": 0.0,
            "cdr_crop_time": 0.0,
            "node_feature_time": 0.0,
            "edge_feature_time": 0.0,
            "dgl_graph_build_time": 0.0,
            "metric_lookup_time": 0.0,
            "total_getitem_time": 0.0,
        }
        getitem_t0 = time.perf_counter()

        def _profile_add(key, value):
            if profile_data_loading:
                data_profile[key] = data_profile.get(key, 0.0) + float(value)

        def _load_target_for_getitem(path):
            if not profile_data_loading:
                return _load_target_pickle(path)
            obj, read_t, parse_t = _timed_load_target_pickle(path)
            _profile_add("read_structure_time", read_t)
            _profile_add("parse_structure_time", parse_t)
            return obj

        def _log_data_profile(graph_set=None, n_graphs=0):
            if not profile_data_loading:
                return
            data_profile["total_getitem_time"] = time.perf_counter() - getitem_t0
            total_nodes = int(graph_set.num_nodes()) if graph_set is not None else 0
            total_edges = int(graph_set.num_edges()) if graph_set is not None else 0
            msg = (
                "[DATA_PROFILE] "
                f"target={pdb_id} avg_over=1 "
                f"read_structure_time={data_profile['read_structure_time']:.4f} "
                f"parse_structure_time={data_profile['parse_structure_time']:.4f} "
                f"cdr_crop_time={data_profile['cdr_crop_time']:.4f} "
                f"node_feature_time={data_profile['node_feature_time']:.4f} "
                f"edge_feature_time={data_profile['edge_feature_time']:.4f} "
                f"dgl_graph_build_time={data_profile['dgl_graph_build_time']:.4f} "
                f"metric_lookup_time={data_profile['metric_lookup_time']:.4f} "
                f"total_getitem_time={data_profile['total_getitem_time']:.4f} "
                f"n_graphs={int(n_graphs)} "
                f"total_nodes={total_nodes} "
                f"total_edges={total_edges}"
            )
            _logging.info(msg)

        # ── Phase 1: Resolve candidates and load RMSD only ──
        candidates = self._registry.get_candidates(pdb_id, epoch)
        if not candidates:
            _logging.warning("_getitem_yaml: no candidates for %s", pdb_id)
            raise FileNotFoundError(f"No YAML sources available for {pdb_id}")

        # per_source: sname → (graph_path, labels, valid_indices, target_pickle_path,
        #                      ag_local_full, eval_metrics_full)
        #   ag_local_full: one float per decoy (model attr or metrics file); NaN if unknown.
        #   eval_metrics_full: loop_rmsd/loop_lddt lists used only for validation/inference reporting.
        #   graph_path is set for graph_pickle and cached target_model_pickle.
        #   target_pickle_path is set for on-the-fly target_model_pickle (graph_path = None).
        per_source = {}
        per_source_gated_out: dict = {}
        # manifest support: per-source decoy identity keys + resolved id, aligned
        # with the flat decoy index used in Phase 3.
        per_source_keys = {}
        per_source_resolved = {}
        # graph-cache: sname -> open GraphPack whose decoys back this source's
        # per_source arrays (index i == pack decoy i). Phase 3 reads graphs here.
        per_source_pack = {}

        def _ag_local_from_model(m):
            v = getattr(m, "ag_local_rmsd", float("nan"))
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")

        for cand in candidates:
            sname = cand.source_name
            if sname in per_source:
                continue

            graph_path = cand.graph_path
            rmsd_path = cand.rmsd_path
            metrics_path = cand.metrics_path
            metrics_format = cand.metrics_format
            target_pickle_path = None   # set only for on-the-fly sources
            ag_local_full = None
            eval_metrics_full = None

            # For target_model_pickle sources:
            if cand.file_type == "target_model_pickle":
                # ── graph-cache read: if a valid pack exists, take labels from the
                # store keyed by the pack's decoy keys (no pdb2dict pickle load) and
                # defer graph reads to Phase 3. Any miss/stale → _pack None → on-the-fly.
                # Only store-backed sources are cacheable (labels come from the
                # parquet by key); Xtal and any non-store source stay on-the-fly.
                _pm = getattr(spec, "precomputed_metrics", None)
                _store_backed = (self._metric_store is not None
                                 and (_pm is None or not _pm.sources or sname in _pm.sources)
                                 and self._metric_store.has_source(sname))
                _pack = (self._get_pack(sname, cand.pdb_id)
                         if (cache_enabled and gb.on_the_fly and _store_backed) else None)
                if _pack is not None:
                    try:
                        from dataset.precomputed_metrics import metric_column as _metric_column
                        _keys = [tuple(k) for k in _pack.keys]   # (source, seed, sample)
                        _store = self._metric_store
                        _ts = getattr(spec, "task_scope", "full_cdr")
                        def _cv(_metric):
                            _col = _metric_column(_metric, _ts)
                            return [_store.lookup(sname, cand.pdb_id, k[1], k[2], _col) for k in _keys]
                        all_rmsds = _cv(getattr(spec, "label_metric", "loop_lddt"))
                        eval_metrics_full = {
                            "loop_rmsd": _cv("loop_rmsd"),
                            "loop_lddt": _cv("loop_lddt"),
                            "h3_lddt": _cv("h3_lddt"),
                            "dockq": _cv("dockq"),
                            "fnat": _cv("fnat"),
                        }
                        ag_local_full = [float("nan")] * len(_keys)
                        per_source_pack[sname] = _pack
                        per_source_keys[sname] = [(_mnorm_seed(k[1]), int(k[2])) for k in _keys]
                        per_source_resolved[sname] = cand.pdb_id
                        target_pickle_path = None
                        graph_path = None
                        rmsd_path = None
                    except Exception as e:
                        _logging.warning("graph cache read-prep failed %s/%s (%s) -> on-the-fly",
                                         sname, cand.pdb_id, e)
                        _pack = None
                        per_source_pack.pop(sname, None)

                if _pack is not None:
                    pass   # labels+graphs supplied by the cache above / Phase 3
                elif gb.on_the_fly:
                    # On-the-fly mode: load Target pickle for RMSD only,
                    # defer graph generation to Phase 3.
                    try:
                        target_obj = _load_target_for_getitem(cand.target_model_path)
                        t_metric = time.perf_counter()
                        all_rmsds = self._labels_for_models(
                            target_obj, spec, sname, cand.pdb_id,
                        )
                        eval_metrics_full = self._eval_metrics_for_models(
                            target_obj, spec, sname, cand.pdb_id,
                        )
                        ag_local_full = [_ag_local_from_model(m) for m in target_obj.models]
                        # Identity keys (seed, sample) aligned to model index — used
                        # for manifest selection; same identities the store is keyed by.
                        if manifest_mode:
                            from dataset.precomputed_metrics import decoy_identity_for_lookup
                            _cand_keys = []
                            for _pos, _m in enumerate(target_obj.models):
                                _sd, _sm = decoy_identity_for_lookup(_m, _pos)
                                _cand_keys.append((_mnorm_seed(_sd), int(_sm)))
                            per_source_keys[sname] = _cand_keys
                            per_source_resolved[sname] = cand.pdb_id
                        _profile_add("metric_lookup_time", time.perf_counter() - t_metric)
                        target_pickle_path = cand.target_model_path
                        graph_path = None
                        rmsd_path = None
                        del target_obj  # free — we'll reload in Phase 3
                    except Exception as e:
                        _logging.warning("Failed to load Target pickle %s: %s", cand.target_model_path, e)
                        continue
                else:
                    # Cached mode: convert & cache (original behaviour)
                    cache_root = spec.cache_root or os.path.join(spec.db_root, "_graph_cache")
                    gp, rp = convert_and_cache(
                        cand.target_model_path,
                        cache_root,
                        sname,
                        pdb_id,
                        dist_cutoff_center=gb.dist_cutoff_center,
                        random_range=gb.random_range,
                        max_neighbors=gb.max_neighbors,
                        use_all_atom=gb.use_all_atom,
                        h3_range=tuple(gb.h3_range),
                        cdr_ranges=_graph_build_cdr_ranges(gb),
                        task_scope=spec.task_scope,
                        label_metric=spec.label_metric,
                        cdr_context_cutoff=gb.cdr_context_cutoff,
                        max_context_residues=gb.max_context_residues,
                        graph_crop_debug=gb.graph_crop_debug,
                    )
                    if gp is None:
                        continue
                    graph_path = gp
                    rmsd_path = rp

            # Load RMSD only (small file) — skip for on-the-fly (already loaded
            # above) and for graph-cache sources (labels already taken from the
            # store by pack key; this block is for legacy graph_pickle sources).
            if target_pickle_path is None and sname not in per_source_pack:
                t_metric = time.perf_counter()
                all_rmsds = self._read_rmsd_only(rmsd_path)
                ag_local_full = [float("nan")] * len(all_rmsds)
                if metrics_path:
                    ag_map = load_ag_local_rmsd(metrics_path, metrics_format)
                    for i in range(len(all_rmsds)):
                        ag_local_full[i] = float(ag_map.get(i, float("nan")))
                # If a precomputed store covers this graph_pickle source (e.g. GP
                # ULR loop lDDT), use those labels by decoy index instead of the
                # legacy .rmsd, so the label matches label_metric (cdr_lddt,
                # higher-is-better). Decoy index == graph .dat order == parquet
                # `sample` (seed=None). GP has no antigen -> dockq stays NaN.
                store = self._metric_store
                pm = getattr(spec, "precomputed_metrics", None)
                if (
                    store is not None
                    and (pm is None or not pm.sources or sname in pm.sources)
                    and store.has_source(sname)
                ):
                    from dataset.precomputed_metrics import metric_column
                    n = len(all_rmsds)
                    ts = getattr(spec, "task_scope", "full_cdr")
                    def _vals(metric_name):
                        col = metric_column(metric_name, ts)
                        return [store.lookup(sname, cand.pdb_id, None, i, col) for i in range(n)]
                    lddt = _vals(getattr(spec, "label_metric", "loop_lddt"))
                    all_rmsds = lddt
                    eval_metrics_full = {
                        "loop_rmsd": _vals("loop_rmsd"),
                        "loop_lddt": _vals("loop_lddt"),
                        "h3_lddt": _vals("h3_lddt"),
                        "dockq": [float("nan")] * n,
                        "fnat": _vals("fnat"),
                    }
                _profile_add("metric_lookup_time", time.perf_counter() - t_metric)
            if eval_metrics_full is None:
                eval_metrics_full = {getattr(spec, "label_metric", "loop_rmsd"): list(all_rmsds)}
            if not all_rmsds:
                continue
            if ag_local_full is None or len(ag_local_full) != len(all_rmsds):
                ag_local_full = [float("nan")] * len(all_rmsds)

            # Default identity keys for non-on-the-fly sources (graph_pickle):
            # seed=None, sample=flat index (matches GP precompute convention).
            if manifest_mode and sname not in per_source_keys:
                per_source_keys[sname] = [(None, i) for i in range(len(all_rmsds))]
                per_source_resolved[sname] = cand.pdb_id

            valid_indices = list(range(len(all_rmsds)))

            # Drop decoys with no usable label (NaN). This is required when
            # precomputed metrics are used (a missing parquet row → NaN label),
            # and is harmless otherwise (NaN labels are unusable for training).
            if self._metric_store is not None:
                n_before = len(valid_indices)
                valid_indices = [i for i in valid_indices if all_rmsds[i] == all_rmsds[i]]
                n_dropped = n_before - len(valid_indices)
                if n_dropped:
                    _logging.warning(
                        "_getitem_yaml: %s/%s dropped %d/%d decoys without precomputed metric",
                        sname, pdb_id, n_dropped, n_before,
                    )

            # Training/validation: RMSD range filter.
            # Inference: keep all decoys (filters are for offline analysis only).
            # Manifest mode: gate/filter were already applied at manifest build, so
            # keep the full index range and let Phase 2 select by identity key.
            if not inference_yaml and not manifest_mode:
                # Configurable RMSD range filter. The [min_rmsd, max_rmsd] band is
                # RMSD-specific (Å); skip it entirely for higher-is-better labels
                # (loop_lddt) where the values live in [0, 1] and the band is
                # meaningless.
                rf = spec.rmsd_filter
                if rf is not None and not spec.label_higher_is_better():
                    valid_indices = [
                        i for i in valid_indices
                        if rf.min_rmsd <= all_rmsds[i] <= rf.max_rmsd
                    ]

                # v1 DockQ gate (training AND validation): keep holo decoys with
                # DockQ >= spec.dockq_gate; apo decoys (NaN DockQ) always kept.
                gate = getattr(spec, "dockq_gate", None)
                if gate is not None and eval_metrics_full is not None:
                    dqs = eval_metrics_full.get("dockq", [])
                    def _keep_dockq(i, _dqs=dqs, _g=float(gate)):
                        v = _dqs[i] if i < len(_dqs) else float("nan")
                        return (v != v) or (v >= _g)   # NaN (apo) -> keep; else gate
                    n_before = len(valid_indices)
                    _kept = [i for i in valid_indices if _keep_dockq(i)]
                    # exp1: the DockQ gate removes essentially every low-fnat decoy
                    # (measured: 0.8 % of Boltz2_s10n10 [0,.1) survives), so the
                    # low absolute-fnat tiers are absent from the gated pool and a
                    # tier top-up drawing from it would be a no-op. Remember what
                    # the gate dropped so the top-up can reach the target's ORIGINAL
                    # candidate pool — still that target's own decoys, never another's.
                    if (int(getattr(args, "fnat_tier_topup", 0) or 0) > 0
                            or int(getattr(args, "joint_cell_topup", 0) or 0) > 0):
                        _kept_set = set(_kept)
                        per_source_gated_out[sname] = [
                            i for i in valid_indices if i not in _kept_set]
                    valid_indices = _kept
                    n_dropped = n_before - len(valid_indices)
                    if n_dropped:
                        _logging.debug(
                            "_getitem_yaml: %s/%s DockQ-gate(>=%.2f) dropped %d/%d decoys",
                            sname, pdb_id, float(gate), n_dropped, n_before,
                        )

                # exp8 fnat gate (training pool): keep holo decoys with
                # fnat > spec.fnat_gate; apo/GP decoys (NaN fnat) always kept.
                #
                # This is the SAME rule the exp8 SML applies inside the loss, moved
                # up into candidate selection so the ~9 % of decoys that could only
                # ever be discarded are not loaded and forwarded first. It also makes
                # the training pool agree with the fnat>0.5-gated validation
                # manifests, which is the point: train and valid should share one
                # definition of "usable pose".
                #
                # Strict >, matching the project-wide "fnat>0.5" filter and the loss.
                _fgate = getattr(spec, "fnat_gate", None)
                if _fgate is not None and eval_metrics_full is not None:
                    _fns = eval_metrics_full.get("fnat", [])
                    def _keep_fnat(i, _f=_fns, _g=float(_fgate)):
                        v = _f[i] if i < len(_f) else float("nan")
                        return (v != v) or (v > _g)    # NaN (apo/GP) -> keep
                    _nb = len(valid_indices)
                    valid_indices = [i for i in valid_indices if _keep_fnat(i)]
                    if _nb - len(valid_indices):
                        _logging.debug(
                            "_getitem_yaml: %s/%s fnat-gate(>%.2f) dropped %d/%d decoys",
                            sname, pdb_id, float(_fgate), _nb - len(valid_indices), _nb,
                        )

            if valid_indices or manifest_mode:
                per_source[sname] = (
                    graph_path,
                    all_rmsds,
                    valid_indices,
                    target_pickle_path,
                    ag_local_full,
                    eval_metrics_full,
                )

        if not per_source:
            if self._metric_store is not None:
                reason = "no precomputed metric rows for this target (require=true)"
            elif inference_yaml:
                reason = "no decoys available"
            elif spec.rmsd_filter is not None:
                reason = "no decoys left after RMSD filtering"
            else:
                reason = "no decoys available"
            # In training/validation, skip this target and try another one rather
            # than crashing the whole run (targets missing from the precomputed
            # parquet, etc.). Bounded retry guards against infinite recursion.
            if not inference_yaml and len(self.inp_dat) > 1 and _depth < 50:
                _logging.warning(
                    "_getitem_yaml: %s -> %s; skipping to another target (retry %d)",
                    pdb_id, reason, _depth + 1,
                )
                alt_idx = random.randint(0, len(self.inp_dat) - 1)
                return self._getitem_yaml(self.inp_dat[alt_idx], _depth + 1)
            raise FileNotFoundError(f"No loadable decoys for {pdb_id} ({reason})")

        # ── Phase 2: Select decoys to materialize ──
        # Training/validation keeps the existing tier-based sampler.
        # In inference, we materialize every valid decoy from the YAML sources.
        _XTAL_RMSD_THR = 0.01
        _XTAL_LDDT_THR = 0.99
        higher_is_better = spec.label_higher_is_better()
        near_cut = spec.effective_near_native_cutoff()

        def _is_xtal(v):
            """Native/crystal decoy: RMSD ~ 0 or lDDT ~ 1 depending on metric."""
            return (v >= _XTAL_LDDT_THR) if higher_is_better else (v < _XTAL_RMSD_THR)

        def _is_near_native(v):
            return (v >= near_cut) if higher_is_better else (v <= near_cut)

        rng = random.Random(spec.seed + epoch + hash(pdb_id) % 10000)

        if manifest_mode:
            # Select exactly the manifest decoys by identity key (source,seed,sample).
            source_selected = {}
            for sname in per_source:
                rid = per_source_resolved.get(sname, str(pdb_id))
                mset = self._manifest_keys.get(str(rid))
                if not mset:
                    continue
                keys = per_source_keys.get(sname, [])
                sel = [i for i, k in enumerate(keys) if (sname,) + k in mset]
                if sel:
                    source_selected[sname] = sel
            if not source_selected:
                raise FileNotFoundError(
                    f"No manifest decoys resolved for {pdb_id} "
                    f"(resolved={per_source_resolved})"
                )
        elif inference_yaml:
            source_selected = {
                sname: list(valid_indices)
                for sname, (_, _, valid_indices, _, _, _) in per_source.items()
            }
        else:
            # Step 1: Xtal 1 fixed. Step 2: A≤8, B≤24, C≤16, D≤16 by tier. Step 3–4: fill rest from pool.
            # Tier boundaries are direction-aware:
            #   loop_rmsd (lower better): A≤0.8, B≤1.5, C≤2.0, D>2.0  (Å)
            #   loop_lddt (higher better): A≥0.90, B≥0.80, C≥0.70, D<0.70
            if higher_is_better:
                _LDDT_A, _LDDT_B, _LDDT_C = 0.90, 0.80, 0.70
            else:
                _TIER_A, _TIER_B, _TIER_C = 0.8, 1.5, 2.0
            # Tier quotas are phase-scheduled (v2). Pretrain uses the balanced
            # default; the Phase-C DPO finetune (use_phase_config) selects C1/C2/C3
            # quotas to match the eject->balanced->top DPO schedule.
            try:
                from runtime.phase_config import get_tier_quota
                _qkey = "pretrain"
                if getattr(args, "run_type", None) == "finetune" and getattr(args, "use_phase_config", False):
                    _qkey = f"C{int(getattr(args, 'current_phase', 1))}"
                _q = get_tier_quota(_qkey)
                _Q_A, _Q_B, _Q_C, _Q_D = _q["A"], _q["B"], _q["C"], _q["D"]
            except Exception:
                _Q_A, _Q_B, _Q_C, _Q_D = 8, 24, 16, 16
            n_decoy_target = spec.n_decoy

            # Build flat pool: (pool_index, sname, idx, rmsd)
            # `pool_fnat` is index-aligned with `pool`; kept as a separate list
            # because pool entries are unpacked positionally elsewhere.
            pool: list = []
            pool_fnat: list = []
            topup_only_ii: set = set()
            for sname, (_, all_rmsds, valid_indices, _, _, _emf) in per_source.items():
                _fl = (_emf or {}).get("fnat", [])
                for idx in valid_indices:
                    rmsd = all_rmsds[idx]
                    pool.append((sname, idx, rmsd))
                    pool_fnat.append(_fl[idx] if idx < len(_fl) else float("nan"))
            # gate-dropped candidates of THIS target, reachable only by the Step-5
            # fnat-tier top-up (excluded from the cdr_lddt tier quotas below).
            for sname, _dropped in per_source_gated_out.items():
                if sname not in per_source:
                    continue
                _, all_rmsds, _, _, _, _emf = per_source[sname]
                _fl = (_emf or {}).get("fnat", [])
                for idx in _dropped:
                    topup_only_ii.add(len(pool))
                    pool.append((sname, idx, all_rmsds[idx]))
                    pool_fnat.append(_fl[idx] if idx < len(_fl) else float("nan"))

            if not pool:
                raise FileNotFoundError(f"No decoys in pool for {pdb_id}")

            # Step 4 (v2): source scheduling. Each source has an epoch-dependent
            # weight (SourceSpec.weight, possibly a ScheduleSpec). Within each tier
            # we draw decoys with probability proportional to their source weight,
            # so method-mixing schedules take effect while tier quotas (and the
            # near/non balance they guarantee) are preserved.
            _src_w = {}
            for _sn in per_source:
                _sp = spec.sources.get(_sn) if getattr(spec, "sources", None) else None
                try:
                    _src_w[_sn] = float(_sp.effective_weight(epoch)) if _sp is not None else 1.0
                except Exception:
                    _src_w[_sn] = 1.0

            def _wsample(ii_list, k, _rng):
                """Weighted sampling without replacement by source weight
                (Efraimidis-Spirakis). Falls back to all items when k >= n."""
                if k <= 0 or not ii_list:
                    return []
                if k >= len(ii_list):
                    return list(ii_list)
                keyed = []
                for _ii in ii_list:
                    _w = _src_w.get(pool[_ii][0], 1.0)
                    if _w <= 0.0:
                        continue
                    keyed.append((_rng.random() ** (1.0 / _w), _ii))
                keyed.sort(reverse=True)
                return [_ii for _, _ii in keyed[:k]]

            # Classify by tier (X = xtal/native, then A/B/C/D by quality)
            def _tier(r):
                if _is_xtal(r):
                    return "X"
                if higher_is_better:
                    if r >= _LDDT_A:
                        return "A"
                    if r >= _LDDT_B:
                        return "B"
                    if r >= _LDDT_C:
                        return "C"
                    return "D"
                if r <= _TIER_A:
                    return "A"
                if r <= _TIER_B:
                    return "B"
                if r <= _TIER_C:
                    return "C"
                return "D"

            tier_to_ii: dict = {"X": [], "A": [], "B": [], "C": [], "D": []}
            for i, (sname, idx, rmsd) in enumerate(pool):
                if i in topup_only_ii:
                    continue          # gate-dropped: Step 5 only, never the quotas
                tier_to_ii[_tier(rmsd)].append(i)

            selected_ii = set()
            # Step 1: crystal (native) inclusion is probability-gated and annealed
            # over training via spec.xtal_gate_prob (ScheduleSpec resolved at this
            # epoch). Early epochs keep the native as a positive anchor (prob ~1.0);
            # final-stage epochs drop it (prob ~0.0) so the model must discriminate
            # among model-generated decoys. When excluded, tier-X is also held out
            # of the random fill below so no crystal sneaks in.
            xtal_prob = spec.effective_xtal_prob(epoch)
            include_xtal = bool(tier_to_ii["X"]) and (rng.random() < xtal_prob)
            if include_xtal:
                selected_ii.add(rng.choice(tier_to_ii["X"]))
            # Step 2: quota per tier (without replacement).
            # tier-X (native) counts toward the A quota, so an included crystal
            # reduces the number of A-tier decoys drawn by one.
            _n_x_sel = 1 if include_xtal else 0
            for tier_key, quota in [("A", max(0, _Q_A - _n_x_sel)), ("B", _Q_B), ("C", _Q_C), ("D", _Q_D)]:
                available = [i for i in tier_to_ii[tier_key] if i not in selected_ii]
                k = min(quota, len(available))
                if k > 0:
                    for ii in _wsample(available, k, rng):
                        selected_ii.add(ii)
            # Step 3–4: remaining from pool (never pull extra / un-gated crystals)
            xtal_ii = set(tier_to_ii["X"])
            remaining = n_decoy_target - len(selected_ii)
            unselected_ii = [
                i for i in range(len(pool))
                if i not in selected_ii and i not in xtal_ii and i not in topup_only_ii
            ]
            if remaining > 0 and unselected_ii:
                k = min(remaining, len(unselected_ii))
                for ii in _wsample(unselected_ii, k, rng):
                    selected_ii.add(ii)

            # ── exp4-1 "cell importance sampler" ────────────────────────────
            # A different regime from Step 5b: no gate, no top-up. ALL slots are
            # drawn from the ungated pool with a per-cell importance weight
            #     w_cell = ((p_AF3 + eps) / (p_train_ungated + eps)) ** beta
            # over the joint (cdr_lddt bin, fnat bin) grid, WITHOUT replacement.
            # p_AF3 is AF3's own DECOY distribution, not its pair-cell one.
            # Offline simulation over all 1,298 targets picked beta=0.5: it cuts
            # JS(train || AF3) by 35 % and lands the AF3-like cell at 21.56 %
            # against AF3's own 21.87 %, while the smallest source's share RISES
            # (8.66 % -> 9.53 %). beta>=0.75 starts eroding source diversity.
            _cw_path = getattr(args, "cell_importance_weights", "") or ""
            if _cw_path and pool_fnat:
                global _CELL_W_CACHE
                try:
                    _CELL_W_CACHE
                except NameError:
                    _CELL_W_CACHE = {}
                _cw = _CELL_W_CACHE.get(_cw_path)
                if _cw is None:
                    import json as _json
                    with open(_cw_path) as _fh:
                        _cw = _json.load(_fh)
                    _cw['_l'] = list(_cw['l_edges'])
                    _cw['_f'] = list(_cw['f_edges'])
                    _cw['_nf'] = len(_cw['_f']) + 1
                    _CELL_W_CACHE[_cw_path] = _cw
                _wt = _cw['cell_weight']; _le = _cw['_l']; _fe = _cw['_f']; _nf = _cw['_nf']

                def _bin(v, edges):
                    b = 0
                    for e in edges:
                        if v < e:
                            return b
                        b += 1
                    return b

                # every candidate of this target, gated and gate-dropped alike
                _all_ii = list(range(len(pool)))
                _w = []
                for _i in _all_ii:
                    _l, _f = pool[_i][2], pool_fnat[_i]
                    if _l != _l or _f != _f:
                        _w.append(1.0)          # no interface label: neutral weight
                    else:
                        _w.append(float(_wt[_bin(_l, _le) * _nf + _bin(_f, _fe)]))
                # Gumbel top-k == weighted sampling without replacement
                _k = min(n_decoy_target, len(_all_ii))
                _keyed = []
                for _i, _wi in zip(_all_ii, _w):
                    if _wi <= 0:
                        continue
                    _keyed.append((rng.random() ** (1.0 / _wi), _i))
                _keyed.sort(reverse=True)
                selected_ii = {_i for _, _i in _keyed[:_k]}
                # this branch replaces the whole tier selection; skip the top-ups
                _jt = 0

            # ── Step 5b (exp4 "gate_rescue"): joint cdr_lddt x fnat cell top-up ──
            # exp1's Step 5 only asked "is this absolute fnat tier missing?", which
            # is too coarse: what the model has never seen is the JOINT cell
            # "loop geometry is fine BUT the binding pose is wrong". Measured, the
            # DockQ 0.49 gate deletes 98.8 % of exactly that cell (29,283 raw
            # Boltz2 decoys over 573 targets -> 341 over 6 targets), so the
            # information exists and is simply filtered out one step before
            # training.
            #
            # So reserve a fixed number of slots for targeted decoys drawn from
            # this target's OWN pre-gate pool, filled by priority:
            #   P1  cdr_lddt >= 0.8 and fnat <  0.3   AF3-like hard negative
            #   P2  0.6 <= cdr_lddt < 0.8, fnat < 0.3
            #   P3  cdr_lddt >= 0.8 and fnat >= 0.7   only if the target has no
            #                                         positive anchor already
            # Empty cells are NOT back-filled by sampling with replacement; the
            # unused slots simply return to the natural gated sampler.
            _jt = 0 if (_cw_path and pool_fnat) else int(getattr(args, "joint_cell_topup", 0) or 0)
            if _jt > 0 and pool_fnat:
                _q1 = int(getattr(args, "joint_topup_p1", 8) or 0)
                _q2 = int(getattr(args, "joint_topup_p2", 4) or 0)
                _q3 = int(getattr(args, "joint_topup_p3", 4) or 0)
                _lddt_hi, _lddt_mid = 0.8, 0.6
                _fnat_lo, _fnat_hi = 0.3, 0.7

                def _lab(i):
                    return pool[i][2], pool_fnat[i]      # (cdr_lddt, fnat)

                def _cell(i, lo, hi, flo, fhi):
                    l, f = _lab(i)
                    if l != l or f != f:
                        return False
                    return (lo <= l < hi) and (flo <= f < fhi)

                # does the CURRENT selection already contain a positive anchor?
                _has_anchor = any(
                    _cell(i, _lddt_hi, 2.0, _fnat_hi, 2.0) for i in selected_ii)
                _cands = {
                    'P1': [i for i in topup_only_ii
                           if i not in selected_ii and _cell(i, _lddt_hi, 2.0, -1.0, _fnat_lo)],
                    'P2': [i for i in topup_only_ii
                           if i not in selected_ii and _cell(i, _lddt_mid, _lddt_hi, -1.0, _fnat_lo)],
                    'P3': ([] if _has_anchor else
                           [i for i in topup_only_ii
                            if i not in selected_ii and _cell(i, _lddt_hi, 2.0, _fnat_hi, 2.0)]),
                }
                _quota = {'P1': _q1, 'P2': _q2, 'P3': _q3}
                # unused P3 slots roll into P1, the cell that matters most
                if not _cands['P3']:
                    _quota['P1'] += _quota.pop('P3'); _cands.pop('P3')
                _protected = set(tier_to_ii["X"]) & selected_ii
                _added = 0
                for _pk in ('P1', 'P2', 'P3'):
                    if _pk not in _cands or _quota.get(_pk, 0) <= 0 or not _cands[_pk]:
                        continue
                    _k = min(_quota[_pk], len(_cands[_pk]), max(0, _jt - _added))
                    for _a in _wsample(_cands[_pk], _k, rng):
                        # evict a natural pick to keep the batch size fixed: take it
                        # from the largest fnat tier, never the native, never the
                        # sole member of its tier, and never another targeted decoy
                        _sel_by_f = {}
                        for _i in selected_ii:
                            _f = pool_fnat[_i]
                            if _f == _f:
                                _sel_by_f.setdefault(min(int(_f / 0.2), 4), []).append(_i)
                        _big = max((k for k, v in _sel_by_f.items() if len(v) > 1),
                                   key=lambda k: len(_sel_by_f[k]), default=None)
                        if _big is None:
                            break
                        _drop = [i for i in _sel_by_f[_big]
                                 if i not in _protected and i not in topup_only_ii]
                        if not _drop:
                            break
                        selected_ii.discard(rng.choice(_drop))
                        selected_ii.add(_a)
                        _added += 1

            # ── Step 5 (exp1): absolute-fnat-tier top-up ─────────────────────
            # The tier quotas above are keyed on cdr_lddt, and high cdr_lddt is
            # strongly correlated with the native-perturbation sources, so the
            # selection collapses onto the top fnat tiers (measured: 68% in
            # [.7,1], 29% in [.5,.7), ~0% below .3). An absolute-tier balanced
            # pair budget cannot balance tiers the batch never contains.
            #
            # So: keep the cdr_lddt selection as the base, and for every absolute
            # fnat tier that EXISTS IN THIS TARGET'S OWN CANDIDATE POOL but is
            # absent from the selection, force in a couple of its decoys, evicting
            # the same number from the most over-represented tier. Tiers the
            # target genuinely lacks are never invented, and nothing is ever
            # pulled from another target.
            _topup = int(getattr(args, "fnat_tier_topup", 0) or 0)
            if _jt > 0:
                _topup = 0        # joint-cell top-up supersedes the tier-only one
            if _topup > 0 and pool_fnat:
                # SINGLE source of truth for the tier edges: the sampler must fill
                # exactly the tiers the loss later balances over. Duplicating the
                # tuple here would let the two drift apart silently.
                from runtime.sujin_loss import FNAT_TIER_EDGES as _edges

                def _ftier(v):
                    if v != v:                      # NaN -> no interface tier
                        return None
                    for _k, _e in enumerate(_edges):
                        if v < _e:
                            return _k
                    return len(_edges)

                _pool_t = [_ftier(v) for v in pool_fnat]
                _avail = {}
                for _i, _tt in enumerate(_pool_t):
                    if _tt is not None:
                        _avail.setdefault(_tt, []).append(_i)
                if _avail:
                    _sel_t = {}
                    for _i in selected_ii:
                        _tt = _pool_t[_i]
                        if _tt is not None:
                            _sel_t.setdefault(_tt, []).append(_i)
                    _missing = sorted(set(_avail) - set(_sel_t))
                    _protected = set(tier_to_ii["X"]) & selected_ii
                    for _mt in _missing:
                        _cand = [i for i in _avail[_mt] if i not in selected_ii]
                        if not _cand:
                            continue
                        _add = _wsample(_cand, min(_topup, len(_cand)), rng)
                        for _a in _add:
                            # evict from the currently largest fnat tier, never the
                            # native and never a tier's last remaining member
                            _big = max(
                                (k for k, v in _sel_t.items() if len(v) > 1),
                                key=lambda k: len(_sel_t[k]), default=None)
                            if _big is None:
                                break
                            _drop = [i for i in _sel_t[_big] if i not in _protected]
                            if not _drop:
                                break
                            _d = rng.choice(_drop)
                            selected_ii.discard(_d)
                            _sel_t[_big].remove(_d)
                            selected_ii.add(_a)
                            _sel_t.setdefault(_mt, []).append(_a)

            # Map selected pool indices → source_selected[sname] = [idx, ...]
            source_selected = {}
            for ii in selected_ii:
                sname, idx, _ = pool[ii]
                source_selected.setdefault(sname, []).append(idx)

        # ── Phase 3: Load only the selected graphs ──

        merged_graphs = []
        merged_rmsds = []
        merged_rankings = [] if inference_yaml else None
        merged_ag_local = [] if inference_yaml else None
        merged_decoy_meta = [] if inference_yaml else None
        merged_eval_metrics = {"loop_rmsd": [], "loop_lddt": [], "h3_lddt": [], "dockq": [], "fnat": []}
        # v2: per-decoy source type {0=holo, 1=apo, 2=gp}. GP is decided by the
        # source's `domain` field; antibody sources are apo vs holo by finite DockQ.
        merged_source_type = []
        # v2 tier-balanced pair sampling needs the GENERATION source of each decoy
        # (Boltz2 vs ComMat vs PertMD …), which source_type (holo/apo/gp) does not
        # carry. Ids are indices into the spec's sorted source list, so they are
        # stable across targets, epochs and ranks.
        merged_source_id = []
        _src_order = sorted(getattr(spec, "sources", {}) or {})
        _src_index = {s: i for i, s in enumerate(_src_order)}

        def _source_type_code(_sname, _dockq_val):
            _src = spec.sources.get(_sname) if getattr(spec, "sources", None) else None
            if _src is not None and getattr(_src, "domain", "antibody") == "general_protein":
                return 2.0  # gp
            return 0.0 if (_dockq_val == _dockq_val) else 1.0  # finite DockQ -> holo, NaN -> apo

        # On-the-fly H3 lDDT is only consumed by the ordinal aux loss. Tier-DPO
        # pair sampling does NOT use it (build_training_pairs discards h3_lddt), so
        # do not trigger the heavy `benchmark` dependency just for tier-DPO/Phase-C.
        # Precomputed h3_lddt (eval_metrics) still drives H3 validation reporting.
        use_h3_lddt = (
            not inference_yaml
            and getattr(args, 'run_type', None) == 'finetune'
            and getattr(args, 'use_ord_aux_loss', False)
        )
        merged_h3_lddt = [] if use_h3_lddt else None
        native_cache = None
        if use_h3_lddt:
            try:
                from dataset.h3_lddt_onthefly import (
                    build_native_cache_from_gt_structure,
                    compute_q,
                    resolve_native_pickle_path,
                )
            except ImportError as _e:
                _logging.warning(
                    "_getitem_yaml: on-the-fly H3 lDDT unavailable (%s); "
                    "disabling ord-aux on-the-fly labels for this run", _e,
                )
                use_h3_lddt = False
                merged_h3_lddt = None
        if use_h3_lddt:
            native_pkl = resolve_native_pickle_path(per_source)
            if native_pkl:
                try:
                    nat_target = _load_target_for_getitem(native_pkl)
                    if nat_target.gt_structure is not None:
                        native_cache = build_native_cache_from_gt_structure(
                            nat_target.gt_structure,
                            pdb_id,
                            tuple(gb.h3_range),
                        )
                    del nat_target
                except Exception:
                    _logging.exception(
                        "_getitem_yaml: native cache failed for %s (%s)",
                        pdb_id, native_pkl,
                    )

        for sname, selected in source_selected.items():
            graph_path, all_rmsds, valid_indices, target_pkl, ag_local_full, eval_metrics_full = per_source[sname]

            if sname in per_source_pack:
                # ── graph-cache read: slice the selected decoys straight from the
                # indexed container (no pickle load, no build). Narrow the wide
                # envelope back to the training cutoff at read time; reproduce
                # random_range jitter per decoy in training, fixed otherwise.
                pk = per_source_pack[sname]
                sel_sorted = sorted(selected)
                _env_cut = float(pk.envelope.get('dist_cutoff', gb.dist_cutoff_center))
                _center = float(gb.dist_cutoff_center)
                if (not inference_yaml) and (not manifest_mode) and gb.random_range > 0:
                    _edge_cutoff = [min(_env_cut, _center + random.uniform(-gb.random_range, gb.random_range))
                                    for _ in sel_sorted]
                else:
                    _edge_cutoff = min(_env_cut, _center)
                try:
                    _graphs = pk.read_graphs(sel_sorted, edge_cutoff=_edge_cutoff)
                except Exception:
                    _logging.exception("graph cache read failed for %s/%s", sname, pdb_id)
                    _graphs = []
                for _gi, idx in enumerate(sel_sorted):
                    if _gi >= len(_graphs):
                        break
                    merged_graphs.append(_graphs[_gi])
                    merged_rmsds.append(all_rmsds[idx])
                    for metric_name in merged_eval_metrics:
                        vals = eval_metrics_full.get(metric_name, [])
                        merged_eval_metrics[metric_name].append(
                            vals[idx] if idx < len(vals) else float("nan"))
                    _dq_l = eval_metrics_full.get("dockq", [])
                    merged_source_id.append(float(_src_index.get(sname, -1)))
                    merged_source_type.append(_source_type_code(
                        sname, _dq_l[idx] if idx < len(_dq_l) else float("nan")))
                    if merged_h3_lddt is not None:
                        # finetune+ord-aux disables the cache (see cache_enabled), so
                        # this path is only reached when merged_h3_lddt is None; guard
                        # defensively with the store's H3 value.
                        _h3 = eval_metrics_full.get("h3_lddt", [])
                        merged_h3_lddt.append(_h3[idx] if idx < len(_h3) else float("nan"))
                    if inference_yaml:
                        _k = pk.keys[idx] if idx < len(pk.keys) else (sname, None, idx)
                        merged_rankings.append(int(_k[2]) if _k[2] is not None else int(idx))
                        alr = ag_local_full[idx] if idx < len(ag_local_full) else float("nan")
                        merged_ag_local.append(float(alr))
                        meta = {"file": f"model_{_k[2]}.pkl", "seed": _k[1],
                                "sample": _k[2], "source": sname}
                        for metric_name in merged_eval_metrics:
                            vals = eval_metrics_full.get(metric_name, [])
                            meta[metric_name] = vals[idx] if idx < len(vals) else float("nan")
                        merged_decoy_meta.append(meta)
            elif target_pkl is not None:
                # On-the-fly: reload Target pickle and generate graphs for selected models only
                try:
                    target_obj = _load_target_for_getitem(target_pkl)
                    for idx in selected:
                        if idx < len(target_obj.models):
                            model = target_obj.models[idx]
                            dic = create_dictionary_from_model(
                                model,
                                use_all_atom=gb.use_all_atom,
                                h3_range=tuple(gb.h3_range),
                                cdr_ranges=_graph_build_cdr_ranges(gb),
                                task_scope=spec.task_scope,
                                cdr_context_cutoff=gb.cdr_context_cutoff,
                                max_context_residues=gb.max_context_residues,
                                graph_crop_debug=gb.graph_crop_debug,
                                target_id=pdb_id,
                                profile_timings=data_profile if profile_data_loading else None,
                            )
                            dist_cutoff = gb.dist_cutoff_center
                            # Manifest (fixed-eval) mode keeps the cutoff deterministic
                            # so the same decoy yields the same graph every epoch —
                            # otherwise the "fixed" validation curve carries graph noise.
                            if not inference_yaml and not manifest_mode and gb.random_range > 0:
                                dist_cutoff += random.uniform(-gb.random_range, gb.random_range)
                            t_edge_mask = time.perf_counter()
                            dic = build_edge_mask(dic, dist_cut_off=dist_cutoff)
                            _profile_add("edge_feature_time", time.perf_counter() - t_edge_mask)
                            g = build_graph(
                                dic,
                                use_all_atom=gb.use_all_atom,
                                dist_cut_off=dist_cutoff,
                                max_neighbors=gb.max_neighbors,
                                profile_timings=data_profile if profile_data_loading else None,
                            )
                            del dic
                            merged_graphs.append(g)
                            merged_rmsds.append(all_rmsds[idx])
                            for metric_name in merged_eval_metrics:
                                vals = eval_metrics_full.get(metric_name, [])
                                merged_eval_metrics[metric_name].append(
                                    vals[idx] if idx < len(vals) else float("nan")
                                )
                            _dq_l = eval_metrics_full.get("dockq", [])
                            merged_source_id.append(float(_src_index.get(sname, -1)))
                            merged_source_type.append(_source_type_code(
                                sname, _dq_l[idx] if idx < len(_dq_l) else float("nan")))
                            if merged_h3_lddt is not None:
                                merged_h3_lddt.append(
                                    compute_q(sname, model, model.md_structure, native_cache)
                                )
                            if inference_yaml:
                                rank = getattr(model, 'ranking', idx)
                                if rank == -1:
                                    rank = None
                                merged_rankings.append(int(rank) if rank is not None else int(idx))
                                alr = ag_local_full[idx] if idx < len(ag_local_full) else float("nan")
                                merged_ag_local.append(float(alr))
                                meta = decoy_identity_from_model(model, idx)
                                meta["source"] = sname
                                for metric_name in merged_eval_metrics:
                                    vals = eval_metrics_full.get(metric_name, [])
                                    meta[metric_name] = vals[idx] if idx < len(vals) else float("nan")
                                merged_decoy_meta.append(meta)
                    del target_obj
                except Exception:
                    _logging.exception("On-the-fly graph generation failed for %s/%s", sname, pdb_id)
            else:
                # Graph pickle: load and extract selected indices
                if graph_path and os.path.exists(graph_path):
                    all_graphs = _load_graph_pickle(graph_path)
                    for idx in selected:
                        if idx < len(all_graphs):
                            merged_graphs.append(all_graphs[idx])
                            merged_rmsds.append(all_rmsds[idx])
                            for metric_name in merged_eval_metrics:
                                vals = eval_metrics_full.get(metric_name, [])
                                merged_eval_metrics[metric_name].append(
                                    vals[idx] if idx < len(vals) else float("nan")
                                )
                            _dq_l = eval_metrics_full.get("dockq", [])
                            merged_source_id.append(float(_src_index.get(sname, -1)))
                            merged_source_type.append(_source_type_code(
                                sname, _dq_l[idx] if idx < len(_dq_l) else float("nan")))
                            if merged_h3_lddt is not None:
                                merged_h3_lddt.append(float("nan"))
                            if inference_yaml:
                                merged_rankings.append(int(idx))
                                alr = ag_local_full[idx] if idx < len(ag_local_full) else float("nan")
                                merged_ag_local.append(float(alr))
                                meta = {"file": f"model_{idx}.pkl", "seed": None, "sample": idx, "source": sname}
                                for metric_name in merged_eval_metrics:
                                    vals = eval_metrics_full.get(metric_name, [])
                                    meta[metric_name] = vals[idx] if idx < len(vals) else float("nan")
                                merged_decoy_meta.append(meta)
                    del all_graphs  # free memory immediately

        if not merged_graphs:
            raise FileNotFoundError(f"No graphs loaded for {pdb_id}")

        # total non-xtal pool size (before del) — used by priority_A (top-2% of full pool)
        total_non_xtal_pool = sum(
            sum(1 for i in vi if not _is_xtal(all_rmsds[i]))
            for _, all_rmsds, vi, _, _, _ in per_source.values()
        )

        # Free intermediate data structures no longer needed
        del per_source, source_selected

        # ── Phase 4: Final trim & diagnostics ──
        # Tier-based selection normally yields ≤ n_decoy; trim only if we overshoot.
        # Near/non-native split is direction-aware (uses near_native_cutoff).
        _MIN_PER_CLASS = 16
        n_total = len(merged_graphs)
        n_decoy = spec.n_decoy

        near_idx = [i for i in range(n_total) if _is_near_native(merged_rmsds[i])]
        non_idx  = [i for i in range(n_total) if not _is_near_native(merged_rmsds[i])]

        if not near_idx or not non_idx:
            min_r = min(merged_rmsds) if merged_rmsds else float('nan')
            max_r = max(merged_rmsds) if merged_rmsds else float('nan')
            cnt = len(merged_rmsds)
            if not near_idx:
                _logging.warning("_getitem_yaml: %s has NO near-native decoys; count=%d, label=[%.3f,%.3f]",
                                 pdb_id, cnt, min_r, max_r)
            if not non_idx:
                _logging.warning("_getitem_yaml: %s has NO non-native decoys; count=%d, label=[%.3f,%.3f]",
                                 pdb_id, cnt, min_r, max_r)

        if (not inference_yaml) and (not manifest_mode) and n_total > n_decoy:
            # Trim to n_decoy, preserving class balance
            if near_idx and non_idx:
                n_near = min(_MIN_PER_CLASS, len(near_idx), max(1, n_decoy // 2))
                n_non = min(_MIN_PER_CLASS, len(non_idx), n_decoy - n_near)
                sel_near = rng.sample(near_idx, k=n_near)
                sel_non  = rng.sample(non_idx,  k=n_non)
                guaranteed = set(sel_near + sel_non)
                pool = [i for i in range(n_total) if i not in guaranteed]
                n_rest = min(n_decoy - len(guaranteed), len(pool))
                sel_rest = rng.sample(pool, k=n_rest) if n_rest > 0 else []
                final_idx = list(guaranteed) + sel_rest
            else:
                final_idx = rng.sample(range(n_total), k=n_decoy)
            rng.shuffle(final_idx)
            final_idx = final_idx[:n_decoy]
            merged_graphs = [merged_graphs[i] for i in final_idx]
            merged_rmsds  = [merged_rmsds[i]  for i in final_idx]
            for metric_name in merged_eval_metrics:
                merged_eval_metrics[metric_name] = [
                    merged_eval_metrics[metric_name][i] for i in final_idx
                ]
            if merged_h3_lddt is not None:
                merged_h3_lddt = [merged_h3_lddt[i] for i in final_idx]
            if merged_source_type:
                merged_source_type = [merged_source_type[i] for i in final_idx]
            if merged_source_id:
                merged_source_id = [merged_source_id[i] for i in final_idx]

        try:
            graph_set = dgl.batch(merged_graphs)
        except Exception as e:
            _logging.warning(
                "_getitem_yaml: %s — dgl.batch failed (schema mismatch "
                "between graph_pickle / target_model_pickle sources), "
                "skipping this target: %s", pdb_id, e,
            )
            del merged_graphs, merged_rmsds
            if len(self.inp_dat) > 1 and _depth < 50:
                alt_idx = random.randint(0, len(self.inp_dat) - 1)
                return self._getitem_yaml(self.inp_dat[alt_idx], _depth + 1)
            raise
        del merged_graphs
        rmsd_tensor = torch.tensor(merged_rmsds, dtype=torch.float32)
        del merged_rmsds
        finite_labels = rmsd_tensor[torch.isfinite(rmsd_tensor)]
        ulr_count = int(graph_set.ndata["ulr"].sum().item()) if "ulr" in graph_set.ndata else 0
        if ulr_count == 0:
            _logging.warning("_getitem_yaml: %s has zero CDR/ULR nodes in batched graph", pdb_id)
        if finite_labels.numel() > 0:
            _dm_logger.info(
                "_getitem_yaml: %s label_metric=%s task_scope=%s n=%d min=%.3f max=%.3f ulr_nodes=%d",
                pdb_id,
                getattr(spec, "label_metric", "loop_rmsd"),
                getattr(spec, "task_scope", "full_cdr"),
                int(finite_labels.numel()),
                float(finite_labels.min().item()),
                float(finite_labels.max().item()),
                ulr_count,
            )
        else:
            _logging.warning("_getitem_yaml: %s has no finite loop labels", pdb_id)

        _log_data_profile(graph_set=graph_set, n_graphs=len(rmsd_tensor))
        eval_metric_tensors = {
            name: torch.tensor(values, dtype=torch.float32)
            for name, values in merged_eval_metrics.items()
        }
        # v2 per-decoy source type {0=holo,1=apo,2=gp}; carried in the same
        # eval-metrics dict so it flows to run_epoch via _extract_eval_metrics.
        if merged_source_type and len(merged_source_type) == len(rmsd_tensor):
            eval_metric_tensors["source_type"] = torch.tensor(
                merged_source_type, dtype=torch.float32
            )
        if merged_source_id and len(merged_source_id) == len(rmsd_tensor):
            eval_metric_tensors["source_id"] = torch.tensor(
                merged_source_id, dtype=torch.float32
            )

        if inference_yaml:
            ag_local_tensor = torch.tensor(merged_ag_local, dtype=torch.float32)
            del merged_ag_local
            return (
                graph_set, rmsd_tensor, pdb_id, merged_rankings,
                ag_local_tensor, merged_decoy_meta, eval_metric_tensors,
            )
        if merged_h3_lddt is not None:
            h3_lddt_tensor = torch.tensor(merged_h3_lddt, dtype=torch.float32)
            h3_loop_len = float("nan")
            if native_cache is not None:
                h3_loop_len = float(native_cache.n_h3_residues)
            h3_loop_len_tensor = torch.tensor([h3_loop_len], dtype=torch.float32)
            return (
                graph_set, rmsd_tensor, pdb_id, total_non_xtal_pool,
                h3_lddt_tensor, h3_loop_len_tensor, eval_metric_tensors,
            )
        return graph_set, rmsd_tensor, pdb_id, total_non_xtal_pool, eval_metric_tensors

    def get_source_stats(self, pdb_id: str, pair_cfg=None):
        """Return per-source stats (path, exists, count, rmsd range) and full-pool tier counts.
        Uses same Phase-1 logic as _getitem_yaml (no graph loading). For use in debug/inspection.
        """
        _ensure_dataset_pkg()
        from dataset.conversion import convert_and_cache
        from runtime.pair_sampling import build_decoy_tiers, PairSamplingConfig

        spec = self._ds_spec
        epoch = self._epoch
        gb = spec.graph_build
        cfg = pair_cfg if pair_cfg is not None else PairSamplingConfig()
        _XTAL_RMSD_THR = 0.01

        candidates = self._registry.get_candidates(pdb_id, epoch)
        if not candidates:
            return {"per_source": [], "full_pool_tier_counts": {}}

        per_source_list = []
        all_rmsds_combined = []
        seen_sources = set()
        for cand in candidates:
            sname = cand.source_name
            if sname in seen_sources:
                continue
            seen_sources.add(sname)
            graph_path = cand.graph_path
            rmsd_path = cand.rmsd_path
            metrics_path = cand.metrics_path
            metrics_format = cand.metrics_format
            target_pickle_path = None
            path_for_display = None
            exists = False

            if cand.file_type == "target_model_pickle":
                path_for_display = cand.target_model_path
                exists = path_for_display and os.path.exists(path_for_display)
                all_rmsds = []
                if exists:
                    try:
                        target_obj = _load_target_pickle(cand.target_model_path)
                        all_rmsds = [_model_loop_label(m, spec) for m in target_obj.models]
                        del target_obj
                    except Exception:
                        pass
            else:
                path_for_display = graph_path
                exists = graph_path and os.path.exists(graph_path)
                all_rmsds = self._read_rmsd_only(rmsd_path) if rmsd_path else []

            if not all_rmsds:
                per_source_list.append({
                    "name": sname,
                    "path": path_for_display or "(none)",
                    "exists": exists,
                    "count": 0,
                    "rmsd_min": None,
                    "rmsd_max": None,
                })
                continue

            valid_indices = list(range(len(all_rmsds)))
            rf = spec.rmsd_filter
            if rf is not None:
                valid_indices = [i for i in valid_indices if rf.min_rmsd <= all_rmsds[i] <= rf.max_rmsd]

            rmsd_vals = [all_rmsds[i] for i in valid_indices]
            count = len(valid_indices)
            rmsd_min = min(rmsd_vals) if rmsd_vals else None
            rmsd_max = max(rmsd_vals) if rmsd_vals else None
            per_source_list.append({
                "name": sname,
                "path": path_for_display or "(none)",
                "exists": exists,
                "count": count,
                "rmsd_min": rmsd_min,
                "rmsd_max": rmsd_max,
            })
            for i in valid_indices:
                all_rmsds_combined.append(all_rmsds[i])

        full_pool_tier_counts = {}
        if all_rmsds_combined:
            rmsds_t = torch.tensor(all_rmsds_combined, dtype=torch.float32)
            is_xtal = rmsds_t < _XTAL_RMSD_THR
            tiers = build_decoy_tiers(rmsds_t, is_xtal, cfg)
            full_pool_tier_counts = {k: len(v) for k, v in tiers.items()}

        return {"per_source": per_source_list, "full_pool_tier_counts": full_pool_tier_counts}


class IgFoldTestSet(Dataset):
    # Class-level storage for skipped pdbs across all instances
    _skipped_pdbs: list = []
    _log_file_written: bool = False
    
    def __init__(self, pdb_list_file: Path, db_dir: Path, decoy_type: str, use_subdirectory: bool = True):
        """
        Dataset for the IgFold test set using Target objects stored as pickle files.

        Parameters:
            pdb_list_file (Path): Path to a pickle file containing:
                                 - list of PDB IDs, OR
                                 - dict with 'list' key containing PDB IDs, OR 
                                 - dict mapping PDB IDs to info
            db_dir (Path): Base directory where Target pickles are stored.
            decoy_type (str): Type of decoy (e.g., 'af3', 'abb2', 'igfold4_local_opt').
            use_subdirectory (bool): If True, expects db_dir/<pdb_id>/<decoy_type>.pkl
                                    If False, expects db_dir/<pdb_id>.pkl (default: True)
        """
        with open(pdb_list_file, 'rb') as f:
            loaded_data = pickle.load(f)
        
        # Handle different pickle formats
        if isinstance(loaded_data, dict):
            if 'list' in loaded_data:
                # Format: {'list': [pdb_id1, pdb_id2, ...]}
                raw_pdb_list = loaded_data['list']
                self.pdb_dic = {pdb: {} for pdb in raw_pdb_list}
            else:
                # Format: {pdb_id: {...}, ...}
                self.pdb_dic = loaded_data
                raw_pdb_list = list(self.pdb_dic.keys())
        elif isinstance(loaded_data, list):
            # Format: [pdb_id1, pdb_id2, ...]
            raw_pdb_list = loaded_data
            self.pdb_dic = {pdb: {} for pdb in raw_pdb_list}
        else:
            raise ValueError(f"Unsupported pdb_list format: {type(loaded_data)}")
        
        self.db_dir = db_dir
        self.decoy_type = decoy_type
        self.use_subdirectory = use_subdirectory
        
        # Filter out pdbs without model files and track skipped ones
        IgFoldTestSet._skipped_pdbs = []  # Reset for each new dataset
        IgFoldTestSet._log_file_written = False
        self.pdb_list = []
        
        for pdb_id in raw_pdb_list:
            target_pickle = self._get_target_pickle_path(pdb_id)
            if target_pickle is not None and target_pickle.exists():
                # Check if target has models
                try:
                    target = _load_target_pickle(str(target_pickle))
                    if hasattr(target, 'models') and len(target.models) > 0:
                        self.pdb_list.append(pdb_id)
                    else:
                        IgFoldTestSet._skipped_pdbs.append((pdb_id, 'no_models'))
                except Exception as e:
                    IgFoldTestSet._skipped_pdbs.append((pdb_id, f'load_error: {str(e)}'))
            else:
                IgFoldTestSet._skipped_pdbs.append((pdb_id, 'file_not_found'))
        
        # Log skipped pdbs
        if IgFoldTestSet._skipped_pdbs:
            self._write_skipped_log()
        
        super().__init__()
    
    def _get_target_pickle_path(self, pdb_id: str) -> Path:
        """Get the path to target pickle file."""
        if self.use_subdirectory:
            decoy_type = pdb_id if self.decoy_type == 'abb2' else self.decoy_type
            return self.db_dir / pdb_id / f'{decoy_type}.pkl'
        else:
            target_pickle = self.db_dir / f'{pdb_id}.pkl'
            if target_pickle.exists():
                return target_pickle
            # Try pattern matching for new DB paths
            import glob as glob_module
            pattern = str(self.db_dir / f'{pdb_id}_*.pkl')
            matches = sorted(glob_module.glob(pattern))
            if matches:
                return Path(matches[0])
            return target_pickle  # Return default path even if not found
    
    def _write_skipped_log(self):
        """Write skipped pdbs to log file."""
        if IgFoldTestSet._log_file_written:
            return
        
        log_path = self.db_dir / 'skipped_pdbs.log'
        with open(log_path, 'w') as f:
            f.write(f"# Skipped {len(IgFoldTestSet._skipped_pdbs)} pdb ids during inference\n")
            f.write(f"# decoy_type: {self.decoy_type}\n")
            f.write(f"# db_dir: {self.db_dir}\n\n")
            for pdb_id, reason in IgFoldTestSet._skipped_pdbs:
                f.write(f"{pdb_id}\t{reason}\n")
        
        print(f"\n[WARNING] Skipped {len(IgFoldTestSet._skipped_pdbs)} pdb ids (no models or file not found)")
        print(f"Skipped log saved to: {log_path}")
        if len(IgFoldTestSet._skipped_pdbs) <= 10:
            for pdb_id, reason in IgFoldTestSet._skipped_pdbs:
                print(f"  - {pdb_id}: {reason}")
        else:
            for pdb_id, reason in IgFoldTestSet._skipped_pdbs[:5]:
                print(f"  - {pdb_id}: {reason}")
            print(f"  ... and {len(IgFoldTestSet._skipped_pdbs) - 5} more")
        
        IgFoldTestSet._log_file_written = True

    def __len__(self):
        return len(self.pdb_list)
    
    def __getitem__(self, index):
        pdb_id = self.pdb_list[index]
        
        # Get target pickle path (already validated in __init__)
        target_pickle = self._get_target_pickle_path(pdb_id)
        target = _load_target_pickle(str(target_pickle))

        graphs, rmsds, ranks, decoy_meta = generate_graphs_from_target(target, random_range=args.dist_range, use_all_atom=args.all_atom)
        
        if len(graphs) == 0:
            raise ValueError(f"No graphs generated for {pdb_id}. Target has {len(target.models)} models.")
        
        batched_graph = dgl.batch(graphs)
        rmsd_tensor = torch.tensor(rmsds, dtype=torch.float32)
        
        return batched_graph, rmsd_tensor, pdb_id, ranks, None, decoy_meta

        



class HUDataModule(DataModule):
    def __init__(self,
                #  train_set:str,
                #  val_set:str,
                 batch_size: int = 1,
                 num_workers: int = 8,
                 **kwargs):
        super().__init__(batch_size=batch_size, num_workers=num_workers, collate_fn=self._collate)
        # train_set=read_set_path(train_set)
        # val_set=read_set_path(val_set)

        # self.ds_train=MyDataset(train_set,is_train=True,datatype='GP')
        # self.ds_val=MyDataset(val_set,is_train=False,datatype='GP')
        # datatype: GP, AbAg, HUloop

    def _collate(self, samples):
        sample = samples[0]
        batched_graph=sample[0]
        rmsd_s=sample[1]
        pdb=sample[2]
        rank_s = sample[3] if len(sample) > 3 else None
        ag_local_s = sample[4] if len(sample) > 4 else None
        decoy_meta = sample[5] if len(sample) > 5 else None
        eval_metrics = sample[-1] if isinstance(sample[-1], dict) else None
        if decoy_meta is not None:
            if eval_metrics is not None:
                return batched_graph, rmsd_s, pdb, rank_s, ag_local_s, decoy_meta, eval_metrics
            return batched_graph, rmsd_s, pdb, rank_s, ag_local_s, decoy_meta
        h3_lddt_s = None
        h3_loop_len_s = None
        if len(sample) > 4 and isinstance(sample[4], torch.Tensor):
            if sample[4].numel() > 1:
                h3_lddt_s = sample[4]
                if len(sample) > 5 and isinstance(sample[5], torch.Tensor):
                    h3_loop_len_s = sample[5]
        if h3_lddt_s is not None:
            if eval_metrics is not None:
                return batched_graph, rmsd_s, pdb, rank_s, h3_lddt_s, h3_loop_len_s, eval_metrics
            return batched_graph, rmsd_s, pdb, rank_s, h3_lddt_s, h3_loop_len_s
        if ag_local_s is None:
            return batched_graph, rmsd_s, pdb, rank_s
        return batched_graph, rmsd_s, pdb, rank_s, ag_local_s


def read_set_path(set_fn):
    memo=[]
    with open(set_fn)as fp:
        for line in fp:
            memo.append('%s'%(line.strip('\n')))
    return memo
