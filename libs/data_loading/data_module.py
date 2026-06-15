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
        if dataset_config is not None:
            _ensure_dataset_pkg()
            from dataset.config import load_dataset_spec
            from dataset.source_registry import SourceRegistry
            self._ds_spec = load_dataset_spec(dataset_config)
            self._registry = SourceRegistry(self._ds_spec)
            _logging.info('MyDataset: YAML config loaded from %s', dataset_config)

    def set_epoch(self, epoch: int):
        """Called at the start of each epoch so schedulable params update."""
        self._epoch = epoch

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

    def _getitem_yaml(self, pdb_id):
        """YAML-exclusive loading: RMSD-first → filter → mix → load selected graphs.

        When dataset_config (YAML) is set, this method replaces ALL hardcoded
        path logic in __getitem__.  The pipeline:
          1. Resolve candidate sources from SourceRegistry
          2. Load RMSD pickles only (lightweight) for each source
          3. Training/valid: apply configurable RMSD range filter.
             Inference: no index filtering (all decoys kept for scoring; filter offline).
          4. Run source-weighted mixing to allocate n_decoy across sources
          5. Load graph pickles ONLY for allocated sources, extracting selected indices
          6. Return (batched_graph, rmsd_tensor, pdb_id, extra[, ag_local_tensor in inference])

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

        # per_source: sname → (graph_path, all_rmsds, valid_indices, target_pickle_path, ag_local_full)
        #   ag_local_full: one float per decoy (model attr or metrics file); NaN if unknown.
        #   graph_path is set for graph_pickle and cached target_model_pickle.
        #   target_pickle_path is set for on-the-fly target_model_pickle (graph_path = None).
        per_source = {}

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

            # For target_model_pickle sources:
            if cand.file_type == "target_model_pickle":
                if gb.on_the_fly:
                    # On-the-fly mode: load Target pickle for RMSD only,
                    # defer graph generation to Phase 3.
                    try:
                        target_obj = _load_target_for_getitem(cand.target_model_path)
                        t_metric = time.perf_counter()
                        all_rmsds = [_model_loop_label_from_target(m, target_obj, spec) for m in target_obj.models]
                        ag_local_full = [_ag_local_from_model(m) for m in target_obj.models]
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

            # Load RMSD only (small file) — skip for on-the-fly (already loaded above)
            if target_pickle_path is None:
                t_metric = time.perf_counter()
                all_rmsds = self._read_rmsd_only(rmsd_path)
                ag_local_full = [float("nan")] * len(all_rmsds)
                if metrics_path:
                    ag_map = load_ag_local_rmsd(metrics_path, metrics_format)
                    for i in range(len(all_rmsds)):
                        ag_local_full[i] = float(ag_map.get(i, float("nan")))
                _profile_add("metric_lookup_time", time.perf_counter() - t_metric)
            if not all_rmsds:
                continue
            if ag_local_full is None or len(ag_local_full) != len(all_rmsds):
                ag_local_full = [float("nan")] * len(all_rmsds)

            valid_indices = list(range(len(all_rmsds)))

            # Training/validation: RMSD range filter.
            # Inference: keep all decoys (filters are for offline analysis only).
            if not inference_yaml:
                # Configurable RMSD range filter
                rf = spec.rmsd_filter
                if rf is not None:
                    valid_indices = [
                        i for i in valid_indices
                        if rf.min_rmsd <= all_rmsds[i] <= rf.max_rmsd
                    ]

            if valid_indices:
                per_source[sname] = (graph_path, all_rmsds, valid_indices, target_pickle_path, ag_local_full)

        if not per_source:
            raise FileNotFoundError(
                f"No loadable decoys for {pdb_id}"
                + ("" if inference_yaml else " after RMSD/quality filtering")
            )

        # ── Phase 2: Select decoys to materialize ──
        # Training/validation keeps the existing tier-based sampler.
        # In inference, we materialize every valid decoy from the YAML sources.
        _XTAL_RMSD_THR = 0.01
        rng = random.Random(spec.seed + epoch + hash(pdb_id) % 10000)

        if inference_yaml:
            source_selected = {
                sname: list(valid_indices)
                for sname, (_, _, valid_indices, _, _) in per_source.items()
            }
        else:
            # Step 1: Xtal 1 fixed. Step 2: A≤8, B≤24, C≤16, D≤16 by tier. Step 3–4: fill rest from pool.
            _TIER_A, _TIER_B, _TIER_C = 0.8, 1.5, 2.0
            _Q_A, _Q_B, _Q_C, _Q_D = 8, 24, 16, 16
            n_decoy_target = spec.n_decoy

            # Build flat pool: (pool_index, sname, idx, rmsd)
            pool: list = []
            for sname, (_, all_rmsds, valid_indices, _, _) in per_source.items():
                for idx in valid_indices:
                    rmsd = all_rmsds[idx]
                    pool.append((sname, idx, rmsd))

            if not pool:
                raise FileNotFoundError(f"No decoys in pool for {pdb_id}")

            # Classify by tier (X = xtal, then A/B/C/D by RMSD)
            def _tier(r):
                if r < _XTAL_RMSD_THR:
                    return "X"
                if r <= _TIER_A:
                    return "A"
                if r <= _TIER_B:
                    return "B"
                if r <= _TIER_C:
                    return "C"
                return "D"

            tier_to_ii: dict = {"X": [], "A": [], "B": [], "C": [], "D": []}
            for i, (sname, idx, rmsd) in enumerate(pool):
                tier_to_ii[_tier(rmsd)].append(i)

            selected_ii = set()
            # Step 1: 1 Xtal
            if tier_to_ii["X"]:
                selected_ii.add(rng.choice(tier_to_ii["X"]))
            # Step 2: quota per tier (without replacement)
            for tier_key, quota in [("A", _Q_A), ("B", _Q_B), ("C", _Q_C), ("D", _Q_D)]:
                available = [i for i in tier_to_ii[tier_key] if i not in selected_ii]
                k = min(quota, len(available))
                if k > 0:
                    for ii in rng.sample(available, k):
                        selected_ii.add(ii)
            # Step 3–4: remaining from pool
            remaining = n_decoy_target - len(selected_ii)
            unselected_ii = [i for i in range(len(pool)) if i not in selected_ii]
            if remaining > 0 and unselected_ii:
                k = min(remaining, len(unselected_ii))
                for ii in rng.sample(unselected_ii, k):
                    selected_ii.add(ii)

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
        use_h3_lddt = (
            not inference_yaml
            and getattr(args, 'run_type', None) == 'finetune'
            and (
                getattr(args, 'use_ord_aux_loss', False)
                or getattr(args, 'use_tier_dpo', False)
            )
        )
        merged_h3_lddt = [] if use_h3_lddt else None
        native_cache = None
        if use_h3_lddt:
            from dataset.h3_lddt_onthefly import (
                build_native_cache_from_gt_structure,
                compute_q,
                resolve_native_pickle_path,
            )
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
            graph_path, all_rmsds, valid_indices, target_pkl, ag_local_full = per_source[sname]

            if target_pkl is not None:
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
                            if not inference_yaml and gb.random_range > 0:
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
                                merged_decoy_meta.append(decoy_identity_from_model(model, idx))
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
                            if merged_h3_lddt is not None:
                                merged_h3_lddt.append(float("nan"))
                            if inference_yaml:
                                merged_rankings.append(int(idx))
                                alr = ag_local_full[idx] if idx < len(ag_local_full) else float("nan")
                                merged_ag_local.append(float(alr))
                                merged_decoy_meta.append(
                                    {"file": f"model_{idx}.pkl", "seed": None, "sample": idx}
                                )
                    del all_graphs  # free memory immediately

        if not merged_graphs:
            raise FileNotFoundError(f"No graphs loaded for {pdb_id}")

        # total non-xtal pool size (before del) — used by priority_A (top-2% of full pool)
        _XTAL_RMSD_THR = 0.01
        total_non_xtal_pool = sum(
            sum(1 for i in vi if all_rmsds[i] >= _XTAL_RMSD_THR)
            for _, all_rmsds, vi, _, _ in per_source.values()
        )

        # Free intermediate data structures no longer needed
        del per_source, source_selected

        # ── Phase 4: Final trim & diagnostics ──
        # Tier-based selection normally yields ≤ n_decoy; trim only if we overshoot.
        _BALANCE_CUTOFF = 2.0
        _MIN_PER_CLASS = 16
        n_total = len(merged_graphs)
        n_decoy = spec.n_decoy
        rmsd_cutoff = _BALANCE_CUTOFF

        near_idx = [i for i in range(n_total) if merged_rmsds[i] <= rmsd_cutoff]
        non_idx  = [i for i in range(n_total) if merged_rmsds[i] > rmsd_cutoff]

        if not near_idx or not non_idx:
            min_r = min(merged_rmsds) if merged_rmsds else float('nan')
            max_r = max(merged_rmsds) if merged_rmsds else float('nan')
            cnt = len(merged_rmsds)
            if not near_idx:
                _logging.warning("_getitem_yaml: %s has NO near-native decoys; count=%d, rmsd=[%.3f,%.3f]",
                                 pdb_id, cnt, min_r, max_r)
            if not non_idx:
                _logging.warning("_getitem_yaml: %s has NO non-native decoys; count=%d, rmsd=[%.3f,%.3f]",
                                 pdb_id, cnt, min_r, max_r)

        if (not inference_yaml) and n_total > n_decoy:
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
            if merged_h3_lddt is not None:
                merged_h3_lddt = [merged_h3_lddt[i] for i in final_idx]

        try:
            graph_set = dgl.batch(merged_graphs)
        except Exception as e:
            _logging.warning(
                "_getitem_yaml: %s — dgl.batch failed (schema mismatch "
                "between graph_pickle / target_model_pickle sources), "
                "skipping this target: %s", pdb_id, e,
            )
            del merged_graphs, merged_rmsds
            alt_idx = random.randint(0, len(self.inp_dat) - 1)
            return self.__getitem__(alt_idx)
        del merged_graphs
        rmsd_tensor = torch.tensor(merged_rmsds, dtype=torch.float32)
        del merged_rmsds
        finite_labels = rmsd_tensor[torch.isfinite(rmsd_tensor)]
        ulr_count = int(graph_set.ndata["ulr"].sum().item()) if "ulr" in graph_set.ndata else 0
        if ulr_count == 0:
            _logging.warning("_getitem_yaml: %s has zero CDR/ULR nodes in batched graph", pdb_id)
        if finite_labels.numel() > 0:
            _logging.info(
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

        if inference_yaml:
            ag_local_tensor = torch.tensor(merged_ag_local, dtype=torch.float32)
            del merged_ag_local
            return graph_set, rmsd_tensor, pdb_id, merged_rankings, ag_local_tensor, merged_decoy_meta
        if merged_h3_lddt is not None:
            h3_lddt_tensor = torch.tensor(merged_h3_lddt, dtype=torch.float32)
            h3_loop_len = float("nan")
            if native_cache is not None:
                h3_loop_len = float(native_cache.n_h3_residues)
            h3_loop_len_tensor = torch.tensor([h3_loop_len], dtype=torch.float32)
            return (
                graph_set, rmsd_tensor, pdb_id, total_non_xtal_pool,
                h3_lddt_tensor, h3_loop_len_tensor,
            )
        return graph_set, rmsd_tensor, pdb_id, total_non_xtal_pool

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
        batched_graph=samples[0][0]
        rmsd_s=samples[0][1]
        pdb=samples[0][2]
        rank_s = samples[0][3] if len(samples[0]) > 3 else None
        ag_local_s = samples[0][4] if len(samples[0]) > 4 else None
        decoy_meta = samples[0][5] if len(samples[0]) > 5 else None
        if decoy_meta is not None:
            return batched_graph, rmsd_s, pdb, rank_s, ag_local_s, decoy_meta
        h3_lddt_s = None
        h3_loop_len_s = None
        if len(samples[0]) > 4 and isinstance(samples[0][4], torch.Tensor):
            if samples[0][4].numel() > 1:
                h3_lddt_s = samples[0][4]
                if len(samples[0]) > 5 and isinstance(samples[0][5], torch.Tensor):
                    h3_loop_len_s = samples[0][5]
        if h3_lddt_s is not None:
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
