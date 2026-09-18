import torch,os,dgl
from config import config as se3_config
import numpy as np
def concat_dic(dic1,dic2):# two dictionary has same key and all value is Tensor object
    dic={}
    for key in dic1.keys():
        dic[key]=torch.cat([dic1[key],dic2[key]],dim=0)
    return dic
def normalize_vector(x):
    y=torch.norm(x,dim=-1)
    y=y**(-1)
    y=y.unsqueeze(-1)
    return x*y
def cast_global_coord(ndata):
    a=torch.zeros_like(ndata['l1'])
    a=torch.cat([a[:,0:1,:],ndata['l1']],dim=-2)#CA N C CB O
    a=a+ndata['pos'].unsqueeze(-2)
    return a
def get_pairwise_dist(graph):
    crd=cast_global_coord(graph.ndata)#N,5,3
    edges=graph.edges()
    src_crd=crd[edges[0]][:,:,None,:]#N,5,1,3
    dest_crd=crd[edges[1]][:,None,:,:]#N,1,5,3
    tmp=dest_crd-src_crd
    tmp=torch.norm(tmp,dim=-1)#N,5,5
    tmp=tmp.view(-1,25)
    return tmp
def clone_dic(dic,skip=[],expansion_template=None):
    out_dic={}
    for key in dic.keys():
        if key in skip:
            continue
        out_dic[key]=dic[key].clone().detach()
    return out_dic

def gen_bin_center(min_center,no_bin,bin_width,device='cuda'):
    return (torch.arange(no_bin)*bin_width+min_center).to(device=device)
def bin_index(value,bin_center_s):
    value=value.unsqueeze(-1)
    delta=torch.abs(value-bin_center_s)
    delta=torch.argmin(delta,dim=-1)
    return delta
def initialize_epoch_loss():
    epoch_loss={}
    for key in se3_config.loss.weight.keys():
        epoch_loss[key]=[]
    return epoch_loss
def update_epoch_loss(epoch_loss,single_loss):
    for key in single_loss.keys():
        v = single_loss[key]
        if key not in epoch_loss:
            epoch_loss[key]=[]
        epoch_loss[key].append(v.detach().cpu().item() if isinstance(v, torch.Tensor) else v)
    return epoch_loss
def epoch_loss_be_tensor(epoch_loss):
    for key in single_loss.keys():
        epoch_loss[key]=torch.Tensor(epoch_loss[key])
    return epoch_loss
def finalize_epoch_loss(epoch_loss):
    for key in epoch_loss.keys():
        #if key in ['topk_acc','n_count']:
        #    epoch_loss[key]=torch.Tensor(epoch_loss[key]).sum(dim=0).cuda()
        #else:
        epoch_loss[key]=torch.Tensor(epoch_loss[key]).mean(dim=0).cuda()
    return epoch_loss
def report_epoch_loss(epoch_loss,epoch_idx,mode,out_tag=None):
    if out_tag==None:
        out_tag='default'
    if os.path.exists('%s.%s.log'%(out_tag,mode)):
        with open('%s.%s.log'%(out_tag,mode)) as fp:
            wrt=fp.readlines()
    else:
        header='%10i '%epoch_idx
        for key in se3_config.loss.weight.keys():
            header+='%10s '%key
        header+='\n'
        wrt=[header]
    sen='%10i '%epoch_idx
    for key in se3_config.loss.weight.keys():
        if mode =='train':
            if not key == 'final_loss':
                continue
        sen+='%10.3f '%epoch_loss[key].tolist()
    sen+='\n'
    wrt.append(sen)
    with open('%s.%s.log'%(out_tag,mode),'wt') as fp:
        fp.writelines(wrt)
def _aggregate_finite(vals, prefix, out):
    """Add mean/median for finite values to *out*."""
    arr = np.array(vals, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        out[f'{prefix}_mean'] = float('nan')
        out[f'{prefix}_median'] = float('nan')
        return
    out[f'{prefix}_mean'] = float(np.mean(finite))
    out[f'{prefix}_median'] = float(np.median(finite))


def aggregate_top1_h3_val_metrics(epoch_loss, list_keys=None):
    """Gather per-target top1 H3 metrics across DDP ranks; return summary scalars.

    Uses one ``all_gather_object`` for all per-target lists (validation only).
    """
    import torch.distributed as dist
    from runtime.sujin_loss import TOP1_H3_VAL_LIST_KEYS

    keys = list_keys or TOP1_H3_VAL_LIST_KEYS
    local = {k: list(epoch_loss.get(k, [])) for k in keys}
    if dist.is_initialized():
        world_size = dist.get_world_size()
        all_local = [None] * world_size
        dist.all_gather_object(all_local, local)
        gathered = {k: [] for k in keys}
        for rank_data in all_local:
            if not rank_data:
                continue
            for k in keys:
                gathered[k].extend(rank_data.get(k, []))
    else:
        gathered = local

    out = {}
    _aggregate_finite(gathered.get('top1_h3_rmsd', []), 'top1_h3_rmsd', out)
    if not np.isnan(out.get('top1_h3_rmsd_mean', float('nan'))):
        finite_rmsd = np.array(gathered['top1_h3_rmsd'], dtype=float)
        finite_rmsd = finite_rmsd[np.isfinite(finite_rmsd)]
        out['top1_h3_success_rmsd_le_2'] = float(np.mean(finite_rmsd <= 2.0))

    _aggregate_finite(gathered.get('top1_h3_lddt', []), 'top1_h3_lddt', out)
    finite_lddt = np.array(gathered.get('top1_h3_lddt', []), dtype=float)
    finite_lddt = finite_lddt[np.isfinite(finite_lddt)]
    if len(finite_lddt) > 0:
        out['top1_h3_success_lddt_ge_0.80'] = float(np.mean(finite_lddt >= 0.80))
        out['top1_h3_success_lddt_ge_0.90'] = float(np.mean(finite_lddt >= 0.90))
        out['top1_h3_success_lddt_ge_0.95'] = float(np.mean(finite_lddt >= 0.95))
    else:
        out['top1_h3_success_lddt_ge_0.80'] = float('nan')
        out['top1_h3_success_lddt_ge_0.90'] = float('nan')
        out['top1_h3_success_lddt_ge_0.95'] = float('nan')

    _aggregate_finite(gathered.get('oracle_h3_lddt', []), 'oracle_h3_lddt', out)
    _aggregate_finite(gathered.get('h3_lddt_regret', []), 'h3_lddt_regret', out)

    xtal_flags = np.array(gathered.get('top1_is_xtal', []), dtype=float)
    if len(xtal_flags) > 0:
        out['top1_is_xtal_rate'] = float(np.mean(xtal_flags))

    _aggregate_finite(gathered.get('top1_nonxtal_h3_rmsd', []), 'top1_nonxtal_h3_rmsd', out)
    finite_nx_rmsd = np.array(gathered.get('top1_nonxtal_h3_rmsd', []), dtype=float)
    finite_nx_rmsd = finite_nx_rmsd[np.isfinite(finite_nx_rmsd)]
    if len(finite_nx_rmsd) > 0:
        out['top1_nonxtal_success_rmsd_le_2'] = float(np.mean(finite_nx_rmsd <= 2.0))
    else:
        out['top1_nonxtal_success_rmsd_le_2'] = float('nan')

    _aggregate_finite(gathered.get('top1_nonxtal_h3_lddt', []), 'top1_nonxtal_h3_lddt', out)
    finite_nx_lddt = np.array(gathered.get('top1_nonxtal_h3_lddt', []), dtype=float)
    finite_nx_lddt = finite_nx_lddt[np.isfinite(finite_nx_lddt)]
    if len(finite_nx_lddt) > 0:
        out['top1_nonxtal_success_lddt_ge_0.80'] = float(np.mean(finite_nx_lddt >= 0.80))
        out['top1_nonxtal_success_lddt_ge_0.90'] = float(np.mean(finite_nx_lddt >= 0.90))
    else:
        out['top1_nonxtal_success_lddt_ge_0.80'] = float('nan')
        out['top1_nonxtal_success_lddt_ge_0.90'] = float('nan')

    _aggregate_finite(gathered.get('oracle_nonxtal_h3_lddt', []), 'oracle_nonxtal_h3_lddt', out)
    _aggregate_finite(gathered.get('h3_lddt_regret_nonxtal', []), 'h3_lddt_regret_nonxtal', out)

    return out


def reduce_epoch_loss(epoch_loss,world_size):
    # Only all-reduce keys present on EVERY rank, iterated in a deterministic
    # (sorted) order. Some validation keys are emitted conditionally (e.g.
    # holo-only CAPRI / finite-only H3), so ranks can hold different keysets;
    # reducing per-rank key order would mismatch the collective and deadlock.
    import torch.distributed as dist
    local_keys = set(epoch_loss.keys())
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_keys)
    common = set(gathered[0])
    for g in gathered[1:]:
        common &= set(g)
    for loss_key in sorted(common):
        v = epoch_loss[loss_key]
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(float(v), device='cuda')
        dist.all_reduce(v)
        epoch_loss[loss_key] = v / world_size
    return epoch_loss


    



