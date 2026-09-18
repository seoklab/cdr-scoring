import torch, os, sys
import pickle
import logging as _logging
from torch import nn, distributed
import torch.nn.functional as F
import torch.utils.data as data

import numpy as np
import math
import time
import collections
import argparse
import random

import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax


class LossFunction():
    def __init__(self):
        pass

    def forward(self,lddt_decoy,MLscore):
        lddt_decoy = lddt_decoy.squeeze(dim=0)
        MLscore = MLscore.squeeze(dim=1)
        diff = lddt_decoy - MLscore
        mse = (torch.pow(diff,2)/(lddt_decoy.size(0)))*64 # torch shape (64)
        loss = torch.sum(mse)
        out_ranked,rank_ml = torch.unique(MLscore,sorted=True,return_inverse=True)
        lddt_ranked,rank_ans = torch.unique(lddt_decoy,sorted=True,return_inverse=True)
        for i in range(len(rank_ans)):
            if rank_ans[i]==0:
                answer=i
                break
        rank_target = rank_ml[answer]

        return loss,rank_target
    
    
def _ranking_top1_orig_idx(rmsd_decoy, MLscore):
    """Original decoy index of the top1 model-scored decoy (matches Ranking)."""
    rmsd_nr_idx = []
    rmsd = []
    for i in range(len(rmsd_decoy)):
        if rmsd_decoy[i] not in rmsd:
            rmsd.append(rmsd_decoy[i].tolist())
            rmsd_nr_idx.append(i)
    ml_dedup = MLscore[rmsd_nr_idx]
    _, idx_ml = torch.sort(ml_dedup.clone().detach(), descending=False)
    return int(rmsd_nr_idx[idx_ml[0].item()])


def _lddt_metrics_for_top1(h3_lddt_decoy, orig_idx, oracle_lddt=float('nan')):
    """Per-target lDDT success flags and regret for a chosen top1 index."""
    out = {
        'top1_h3_lddt': float('nan'),
        'top1_h3_success_lddt_ge_0.80': float('nan'),
        'top1_h3_success_lddt_ge_0.90': float('nan'),
        'top1_h3_success_lddt_ge_0.95': float('nan'),
        'h3_lddt_regret': float('nan'),
    }
    if h3_lddt_decoy is None or orig_idx is None:
        return out
    lddt = float(h3_lddt_decoy[orig_idx].detach().cpu().item())
    out['top1_h3_lddt'] = lddt
    if math.isfinite(lddt):
        out['top1_h3_success_lddt_ge_0.80'] = float(lddt >= 0.80)
        out['top1_h3_success_lddt_ge_0.90'] = float(lddt >= 0.90)
        out['top1_h3_success_lddt_ge_0.95'] = float(lddt >= 0.95)
        if math.isfinite(oracle_lddt):
            out['h3_lddt_regret'] = float(oracle_lddt - lddt)
    return out


def compute_top1_h3_validation_metrics(
    rmsd_decoy, ml_score, h3_lddt_decoy=None, xtal_rmsd_thr=0.01,
):
    """Per-target top1 / oracle H3 metrics for all decoys and non-xtal-only ranking."""
    _, _, topn_rmsd = Ranking(rmsd_decoy, ml_score)
    top1_rmsd = float(topn_rmsd['top1_rmsd'])
    orig_idx = _ranking_top1_orig_idx(rmsd_decoy, ml_score)
    top1_is_xtal = float(rmsd_decoy[orig_idx].item() < xtal_rmsd_thr)

    oracle_all = float('nan')
    oracle_nonxtal = float('nan')
    if h3_lddt_decoy is not None and len(h3_lddt_decoy) == len(rmsd_decoy):
        finite_all = [
            float(v) for v in h3_lddt_decoy.detach().cpu().tolist()
            if math.isfinite(float(v))
        ]
        if finite_all:
            oracle_all = max(finite_all)
        is_xtal = rmsd_decoy < xtal_rmsd_thr
        finite_nx = [
            float(h3_lddt_decoy[i].item())
            for i in range(len(h3_lddt_decoy))
            if not bool(is_xtal[i].item()) and math.isfinite(float(h3_lddt_decoy[i].item()))
        ]
        if finite_nx:
            oracle_nonxtal = max(finite_nx)

    metrics = {
        'top1_h3_rmsd': top1_rmsd,
        'top1_h3_success_rmsd_le_2': float(top1_rmsd <= 2.0),
        'top1_is_xtal': top1_is_xtal,
        'oracle_h3_lddt': oracle_all,
        'top1_nonxtal_h3_rmsd': float('nan'),
        'top1_nonxtal_success_rmsd_le_2': float('nan'),
        'oracle_nonxtal_h3_lddt': oracle_nonxtal,
        'top1_nonxtal_h3_lddt': float('nan'),
        'top1_nonxtal_success_lddt_ge_0.80': float('nan'),
        'top1_nonxtal_success_lddt_ge_0.90': float('nan'),
        'h3_lddt_regret_nonxtal': float('nan'),
        'top1_h3_lddt': float('nan'),
        'h3_lddt_regret': float('nan'),
        'top1_h3_success_lddt_ge_0.80': float('nan'),
        'top1_h3_success_lddt_ge_0.90': float('nan'),
        'top1_h3_success_lddt_ge_0.95': float('nan'),
    }
    metrics.update(_lddt_metrics_for_top1(h3_lddt_decoy, orig_idx, oracle_all))

    is_xtal = rmsd_decoy < xtal_rmsd_thr
    non_xtal_idx = torch.where(~is_xtal)[0]
    if len(non_xtal_idx) > 0:
        rmsd_nx = rmsd_decoy[non_xtal_idx]
        score_nx = ml_score[non_xtal_idx]
        _, _, topn_nx = Ranking(rmsd_nx, score_nx)
        top1_nx_rmsd = float(topn_nx['top1_rmsd'])
        metrics['top1_nonxtal_h3_rmsd'] = top1_nx_rmsd
        metrics['top1_nonxtal_success_rmsd_le_2'] = float(top1_nx_rmsd <= 2.0)
        if h3_lddt_decoy is not None:
            lddt_nx = h3_lddt_decoy[non_xtal_idx]
            orig_nx_local = _ranking_top1_orig_idx(rmsd_nx, score_nx)
            nx_metrics = _lddt_metrics_for_top1(lddt_nx, orig_nx_local, oracle_nonxtal)
            metrics['top1_nonxtal_h3_lddt'] = nx_metrics['top1_h3_lddt']
            metrics['top1_nonxtal_success_lddt_ge_0.80'] = nx_metrics['top1_h3_success_lddt_ge_0.80']
            metrics['top1_nonxtal_success_lddt_ge_0.90'] = nx_metrics['top1_h3_success_lddt_ge_0.90']
            metrics['h3_lddt_regret_nonxtal'] = nx_metrics['h3_lddt_regret']

    return metrics


# Keys gathered per-target during validation (single all_gather_object batch).
TOP1_H3_VAL_LIST_KEYS = (
    'top1_h3_rmsd', 'top1_h3_lddt', 'oracle_h3_lddt', 'h3_lddt_regret',
    'top1_h3_success_rmsd_le_2', 'top1_h3_success_lddt_ge_0.80',
    'top1_h3_success_lddt_ge_0.90', 'top1_h3_success_lddt_ge_0.95',
    'top1_is_xtal',
    'top1_nonxtal_h3_rmsd', 'top1_nonxtal_h3_lddt', 'oracle_nonxtal_h3_lddt',
    'h3_lddt_regret_nonxtal', 'top1_nonxtal_success_rmsd_le_2',
    'top1_nonxtal_success_lddt_ge_0.80', 'top1_nonxtal_success_lddt_ge_0.90',
)


def Ranking(rmsd_decoy, MLscore, higher_is_better=False):
    """Rank decoys by model score and report the label of the top-ranked ones.

    Parameters
    ----------
    rmsd_decoy : per-decoy label values (RMSD or lDDT).
    MLscore    : model scores.
    higher_is_better : direction of BOTH the label and the model score.
        - False (RMSD): best label = min, best model score = min (ascending).
        - True  (lDDT): best label = max, best model score = max (descending).

    The returned dict keeps the legacy key names ('top1_rmsd', ...) for logging
    compatibility, but the values are in the units of the active label metric.
    """
    rmsd_nr_idx = []
    rmsd = []
    for i in range(len(rmsd_decoy)):
        if rmsd_decoy[i] not in rmsd:
            rmsd.append(rmsd_decoy[i].tolist())
            rmsd_nr_idx.append(i)
    rmsd_decoy = torch.tensor(rmsd)
    MLscore = MLscore[rmsd_nr_idx]

    # Best (oracle) decoy by label direction.
    rmsd_sorted, idx_ans = torch.sort(rmsd_decoy, descending=higher_is_better)
    best_idx = idx_ans[0].item()
    best_rmsd = rmsd_sorted[0].item()

    # Model ranking: best-scoring first (direction matches the metric).
    ml_sorted, idx_ml = torch.sort(MLscore.clone().detach(), descending=higher_is_better)
    MLscore_ans = MLscore[best_idx]
    best_pred_rank = ml_sorted.tolist().index(MLscore_ans) + 1

    # set same device with top10_indices
    device = idx_ml.device
    rmsd_decoy = rmsd_decoy.to(device)
    # "best label among top-k model-ranked decoys": max for lDDT, min for RMSD.
    _reduce = (lambda t: t.max()) if higher_is_better else (lambda t: t.min())
    top10_indices = idx_ml[:10]
    top1_rmsd = rmsd_decoy[top10_indices[0]]
    top3_rmsd = _reduce(torch.index_select(rmsd_decoy, 0, index=top10_indices[:3]))
    top5_rmsd = _reduce(torch.index_select(rmsd_decoy, 0, index=top10_indices[:5]))
    top10_rmsd = _reduce(torch.index_select(rmsd_decoy, 0, index=top10_indices[:10]))
    topn_rmsd = {'top1_rmsd':top1_rmsd.item(), 'top3_rmsd':top3_rmsd.item(), 'top5_rmsd':top5_rmsd.item(), 'top10_rmsd':top10_rmsd.item()}

    top1_idx = idx_ml[0].item()

    diff_rmsd = abs(top1_rmsd.item()-best_rmsd)


    return best_pred_rank, diff_rmsd, topn_rmsd
    
    
class LossFunction_pairwise_rank():
    def __init__(self):
        pass
    
    def forward(self, lddt_decoy, MLscore, training):
        lddt_decoy = lddt_decoy.squeeze(dim=0)
        # MLscore = MLscore.squeeze(dim=1)
        mask = torch.triu(torch.ones(64,64),diagonal=1).to(bool).to(device)
        
        lddt_decoy_2 = lddt_decoy.repeat(64,1)
        lddt_decoy_1 = torch.transpose(lddt_decoy_2,0,1)
        lddt_diff = lddt_decoy_1 - lddt_decoy_2
        lddt_sign = torch.masked_select(torch.sign(lddt_diff),mask) # 64*63/2 = 2016

        MLscore_2 = torch.masked_select(MLscore.repeat(64,1),mask)
        MLscore_1 = torch.masked_select(torch.transpose(MLscore.repeat(64,1),0,1),mask)

        lossfunction = nn.MarginRankingLoss(margin=1.0)
        loss = lossfunction(MLscore_1, MLscore_2, lddt_sign)

        
        if not training:
            best_rank, diff_lddt = Ranking(lddt_decoy, MLscore)
        if training:
            best_rank = []; diff_lddt = []
        
        return loss, best_rank, diff_lddt
        
class LossFunction_rank():
    def __init__(self):
        pass

    def forward(self,lddt,out):
        lddt = lddt.squeeze(0)
        out = out.squeeze(1)

        lddt_id = torch.sort(lddt)[1]
        idx = torch.arange(0,64,dtype=torch.float,requires_grad=True).to(device)
        rank_lddt = torch.take(idx,lddt_id)

        out_id = torch.sort(out)[1]
        rank_out = torch.take(idx,out_id)

        diff = rank_lddt-rank_out
        mse = (torch.pow(diff,2)/(lddt.size(0)))*64
        loss=torch.sum(mse)
        for i in range(len(rank_lddt)):
            if rank_lddt[i]==0:
                answer=i
                break
        rank_target = rank_out[answer] # target이 예측한 실제 1등의 등수
    
        return loss,rank_target

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=16, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        
    def forward(self, target, pred, isTraining, device, inference=False):
        pred_log_softmax = pred.log_softmax(dim=self.dim)
        pred_softmax = pred.softmax(dim=self.dim)
 
        target_label = torch.floor(target*16)
        target_label = target_label.type(torch.LongTensor).to(device)
        decoy_length = len(target_label)
        target_label_neighbor = torch.zeros(decoy_length,16)

        # 'target_label_neighbor' returns 2 if target label is the first or the end of the bins.
        # It also returns 1 if the class is the neighbor of the target class.
        for i in range(len(target_label)):
            idx_2 = int(target_label[i])
            if idx_2 == 16:
                target_label[i] = 15
                idx_2 = int(target_label[i])
                
            if idx_2 == 0:
                target_label_neighbor[i,1] = 2
            elif idx_2 == 15:
                target_label_neighbor[i,14] = 2
            else:
                target_label_neighbor[i,idx_2-1] = 1
                target_label_neighbor[i,idx_2+1] = 1

        true_distr = torch.zeros_like(pred_log_softmax)
        true_distr[target_label_neighbor==2] = self.smoothing
        true_distr[target_label_neighbor==1] = self.smoothing/2
        true_distr.scatter_(1,target_label.view(-1,1), self.confidence)
        
        if inference:
            CEloss=0
        else:
            CEloss = torch.mean(torch.sum(-true_distr * pred_log_softmax, dim=self.dim))
        bin_median = ((torch.arange(self.cls)*2 + 1)/32).to(device)
        pLDDT = torch.inner(pred_softmax, bin_median)
        
        
        if not isTraining:
            best_rank, diff_lddt = Ranking(target,pLDDT)
        else:
            best_rank = []; diff_lddt= []

            
        return CEloss, best_rank, diff_lddt, pLDDT
    
class Cee_5A(nn.Module):
    def __init__(self, classes=32, smoothing=0.1, dim=-1, cutoff=5.0):
        super(Cee_5A, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.cutoff = cutoff
        
    def forward(self, pdb, target, pred, isTraining, device):
        pred_log_softmax = pred.log_softmax(dim=self.dim) #[num_decoys] #모델 예측값
        pred_softmax = pred.softmax(dim=self.dim)
 
        target_label = torch.floor(target*self.cls/self.cutoff)
        decoy_length = len(target_label)
        for i in range(decoy_length):
            if target_label[i]>=self.cls:
                target_label[i]=self.cls-1 # rmsd 5이상인 것들 전부 마지막 bin으로 처리 (31)
        target_label = target_label.type(torch.LongTensor).to(device)
        target_label_neighbor = torch.zeros(len(target_label),self.cls) # (64,32)

        # 'target_label_neighbor' returns 2 if target label is the first or the end of the bins.
        # It also returns 1 if the class is the neighbor of the target class.
        for i in range(len(target_label)):
            idx_2 = int(target_label[i])
            if idx_2 == self.cls:
                target_label[i] = self.cls-1
                idx_2 = int(target_label[i])
                
            if idx_2 == 0:
                target_label_neighbor[i,1] = 2
            elif idx_2 == self.cls-1:
                # target_label_neighbor[i,self.cls-2] = 2
                continue #rmsd 기준에서는 5이상인 것들 묶어서 마지막 bin에 대해서는 smoothing 없음
                # 대신 마지막에 0.9 confidence만 가짐
            else:
                target_label_neighbor[i,idx_2-1] = 1
                target_label_neighbor[i,idx_2+1] = 1
 
        true_distr = torch.zeros_like(pred_log_softmax)
        true_distr[target_label_neighbor==2] = self.smoothing
        true_distr[target_label_neighbor==1] = self.smoothing/2
        true_distr.scatter_(1,target_label.view(-1,1), self.confidence)
        
        CEloss = torch.mean(torch.sum(-true_distr * pred_log_softmax, dim=self.dim))
        bin_median = ((torch.arange(self.cls)*2 + 1)/(self.cls*2)).to(device)
        bin_median[self.cls-1]=10 # 마지막 bin으로 예측한 경우에는 가중치 높게 줌
        pRMSD = torch.inner(pred_softmax, bin_median) * (self.cls-1)/self.cutoff
        
        
        if not isTraining:
            best_rank, diff_lddt, top1_rmsd = Ranking(target,pRMSD)
        else:
            best_rank = []; diff_lddt= []
            
        return CEloss, best_rank, diff_lddt, pRMSD, top1_rmsd
    
        
class SoftMarginLoss_lddt(nn.Module):
    def __init__(self):
        super(SoftMarginLoss, self).__init__()
    
    def forward(self,target,pred,isTraining,lddt_cutoff=0.9):
        # Default lDDT cutoff of near-native vs. non-native is 0.9
        mask_nonNative = target.lt(lddt_cutoff) # lDDT less than the cutoff
        
        # if there is no near-native decoys, return 0
        # print(pred[~mask_nonNative].shape)
        if pred[~mask_nonNative].size(0)==0:
            return 0
        
        epsilon = 1e-5
        # Average score of near-native decoys
        avg_nearNative = torch.mean(pred[~mask_nonNative])
        # diff_near_non = -1*(pred[mask_nonNative] - avg_nearNative) # Original sign
        diff_near_non = (pred[mask_nonNative] - avg_nearNative) # Reverse sign
        loss = torch.nn.functional.softplus(diff_near_non)
        loss = loss.mean()+1e-5
        
        if loss.isnan():
            return 0
        
        # if not isTraining:
        #     best_rank, diff_lddt = Ranking(target,pred)
        # else:
        #     best_rank = []; diff_lddt = []

        # return loss, best_rank, diff_lddt
        return loss
    # loss, best_rank, diff_lddt, (pLDDT) from LossFunction
    
class SoftMarginLoss_rmsd(nn.Module):
    def __init__(self):
        super(SoftMarginLoss_rmsd, self).__init__()
    
    def forward(self,pdb,target,pred,isTraining,inference=False,rmsd_cutoff=2.0,withCEE=False,higher_is_better=False,
                fnat=None, tier_lddt=None, tier_fnat=None, tier_lddt_low=None, stats=None,
                fnat_gate=None, near_cut=None, non_cut=None,
                source_id=None, demote_ids=None, demote_max_frac=0.5):
        # ``higher_is_better`` selects the metric direction:
        #   False (loop_rmsd): near-native = label < cutoff, and lower model score = better.
        #   True  (loop_lddt): near-native = label >= cutoff, and higher model score = better.
        #
        # exp8 (fnat_gate + near_cut/non_cut): the pose filter and the loop cutoffs are
        # separated, because they answer different questions.
        #
        #   step 1  DROP  finite fnat <= fnat_gate            (0.5)
        #           a decoy docked to the wrong place is not a usable answer, so it
        #           should not be teaching the loop score either. apo/GP have no fnat
        #           and are kept -- their filter is undefined, not passed.
        #   step 2  near  cdr_lddt >= near_cut                (0.90)
        #           SKIP  non_cut < cdr_lddt < near_cut       (0.85-0.90 dead band)
        #           non   cdr_lddt <= non_cut                 (0.85)
        #
        # Why the non class stops at 0.85 rather than 0.80: after the fnat gate, only
        # 4.2 decoys/target below 0.80 survive from sources OTHER than PertMD_all, and
        # 64 % of targets have none at all -- so a 0.80 cutoff makes "non-native" mean
        # "PertMD_all" (88 % of the class) and the model learns to detect an Fv
        # distortion instead of a bad loop. At 0.85 the same figures are 12.4/target
        # and 27 %.
        #
        # demote_ids caps how much of the non class ONE source may occupy
        # (demote_max_frac, default 0.5): keep = min(n_demote, max(1, n_other)). The
        # max(1, ...) keeps a target alive when the capped source is its only negative
        # -- with a hard min(n_demote, n_other) the class empties for 27 % of holo
        # targets while the global share barely moves (42.2 % vs 42.9 %).
        #
        # exp6b/exp6all: when tier_lddt/tier_fnat are given, the near/non split is made
        # on BOTH axes and the confusable middle class is dropped from the loss:
        #
        #   T1 near   cdr_lddt >= tier_lddt  and  fnat >= tier_fnat
        #   T2 SKIP   cdr_lddt >= tier_lddt  and  fnat <  tier_fnat   (loop right, pose wrong)
        #   T3 non    cdr_lddt <  tier_lddt  and  fnat <  tier_fnat
        #   T4 SKIP   cdr_lddt <  tier_lddt  and  fnat >= tier_fnat   (3.9 % of AF3)
        #
        # Why T2 is skipped rather than pushed down: it sits at cdr_lddt ~0.85, so a
        # hard margin against it fights the cdr_lddt signal itself and is unstable near
        # the threshold (22 % of decoys sit within 0.05 of the cdr_lddt boundary, vs
        # 6 % near the fnat boundary). T1-vs-T2 is left to the DPO `top` term later.
        #
        # Targets with no finite fnat at all (apo, GP) fall back to the 1-D rule, so
        # those domains keep training unchanged.
        if inference:
            loss = 0
        else:
            # ── exp8: fnat-gated 1-D SML with a dead band and a per-source cap ──
            if near_cut is not None and non_cut is not None and higher_is_better:
                keep = torch.ones_like(target, dtype=torch.bool)
                _n_gated = 0
                if fnat_gate is not None and fnat is not None:
                    _f = fnat.reshape(-1).to(target.device).float()
                    if _f.numel() == target.numel():
                        _bad = torch.isfinite(_f) & _f.le(float(fnat_gate))
                        keep = ~_bad
                        _n_gated = int(_bad.sum())
                mask_near = keep & target.ge(float(near_cut))
                mask_nonNative = keep & target.le(float(non_cut))
                # dead band counted BEFORE the cap, so sml_n_skipped means the band
                # and sml_n_demote_dropped means the cap -- never a mix of the two.
                _n_band = int((keep & ~mask_near & ~mask_nonNative).sum())
                # cap one source's share of the non class
                _n_demoted_dropped = 0
                if demote_ids and source_id is not None:
                    _sid = source_id.reshape(-1).to(target.device).float()
                    if _sid.numel() == target.numel():
                        _is_dem = torch.zeros_like(mask_nonNative)
                        for _d in demote_ids:
                            _is_dem |= (_sid.round().long() == int(_d))
                        _dem = mask_nonNative & _is_dem
                        _oth = mask_nonNative & ~_is_dem
                        _n_dem, _n_oth = int(_dem.sum()), int(_oth.sum())
                        # ratio r = keep/(keep+other) <= demote_max_frac
                        _r = float(demote_max_frac)
                        _allow = int(_n_oth * _r / max(1e-9, 1.0 - _r)) if _r < 1.0 else _n_dem
                        _allow = max(1, _allow) if _n_dem > 0 else 0
                        if _n_dem > _allow:
                            # deterministic: keep the WORST-scoring-label ones (lowest
                            # cdr_lddt) so the hardest negatives survive the cap
                            _idx = torch.nonzero(_dem, as_tuple=False).reshape(-1)
                            _order = torch.argsort(target[_idx])          # ascending label
                            _drop = _idx[_order[_allow:]]
                            mask_nonNative = mask_nonNative.clone()
                            mask_nonNative[_drop] = False
                            _n_demoted_dropped = int(_n_dem - _allow)
                if stats is not None:
                    stats['sml_n_T1'] = float(mask_near.sum())
                    stats['sml_n_T3'] = float(mask_nonNative.sum())
                    stats['sml_n_skipped'] = float(_n_band)
                    stats['sml_tier_fallback'] = 0.0
                    stats['sml_n_fnat_gated'] = float(_n_gated)
                    stats['sml_n_demote_dropped'] = float(_n_demoted_dropped)
                if mask_near.sum() == 0 or mask_nonNative.sum() == 0:
                    zero = torch.tensor(0.0, device=pred.device)
                    return zero, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                epsilon = 1e-5
                avg_nearNative = torch.mean(pred[mask_near])
                loss = torch.nn.functional.softplus(pred[mask_nonNative] - avg_nearNative)
                loss = loss.mean() + epsilon
                if loss.isnan():
                    zero = torch.tensor(0.0, device=pred.device)
                    return zero, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                if isTraining or withCEE:
                    return loss, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                best_rank, diff_rmsd, topn = Ranking(target, pred, higher_is_better=True)
                return (loss, best_rank, diff_rmsd, topn['top1_rmsd'],
                        topn['top3_rmsd'], topn['top5_rmsd'], topn['top10_rmsd'])
            _use_tier = (tier_lddt is not None and tier_fnat is not None
                         and fnat is not None and higher_is_better)
            _fallback = False
            if _use_tier:
                _f = fnat.reshape(-1).to(target.device).float()
                _ok = torch.isfinite(_f)
                if not bool(_ok.any()):
                    _use_tier, _fallback = False, True
            if _use_tier:
                hi_l = target.ge(float(tier_lddt))
                hi_f = _f.ge(float(tier_fnat)) & _ok
                lo_f = _f.lt(float(tier_fnat)) & _ok
                mask_near = hi_l & hi_f                       # T1
                mask_nonNative = (~hi_l) & lo_f               # T3
                if stats is not None:
                    stats['sml_n_T1'] = float(mask_near.sum())
                    stats['sml_n_T3'] = float(mask_nonNative.sum())
                    stats['sml_n_skipped'] = float((~mask_near & ~mask_nonNative).sum())
                    stats['sml_tier_fallback'] = 0.0
                if mask_near.sum() == 0 or mask_nonNative.sum() == 0:
                    zero = torch.tensor(0.0, device=pred.device)
                    return zero, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                epsilon = 1e-5
                avg_nearNative = torch.mean(pred[mask_near])
                loss = torch.nn.functional.softplus(pred[mask_nonNative] - avg_nearNative)
                loss = loss.mean() + epsilon
                if loss.isnan():
                    zero = torch.tensor(0.0, device=pred.device)
                    return zero, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                if isTraining or withCEE:
                    return loss, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                best_rank, diff_rmsd, topn = Ranking(target, pred, higher_is_better=True)
                return (loss, best_rank, diff_rmsd, topn['top1_rmsd'],
                        topn['top3_rmsd'], topn['top5_rmsd'], topn['top10_rmsd'])
            # 1-D fallback for targets with no fnat (apo antibody, GP). Mirrors the
            # 2-D rule by leaving a DEAD BAND instead of splitting at a single point:
            #
            #   near  cdr_lddt >= tier_lddt          (0.80)
            #   skip  tier_lddt_low <= l < tier_lddt (0.70-0.80) -- the 1-D analogue of T2
            #   non   cdr_lddt <  tier_lddt_low      (0.70)
            #
            # Without the band a decoy at 0.79 is a negative and 0.81 a positive, and
            # 22 % of decoys sit within 0.05 of 0.80. The lower edge is 0.70 because
            # apo coverage falls off a cliff below it: targets able to form both
            # classes are 97.5 % at 0.70 but 34.7 % at 0.65 (GP: 84.4 % / 73.1 %).
            if _fallback and tier_lddt is not None and tier_lddt_low is not None and higher_is_better:
                mask_near = target.ge(float(tier_lddt))
                mask_nonNative = target.lt(float(tier_lddt_low))
                if stats is not None:
                    stats['sml_n_T1'] = float(mask_near.sum())
                    stats['sml_n_T3'] = float(mask_nonNative.sum())
                    stats['sml_n_skipped'] = float((~mask_near & ~mask_nonNative).sum())
                    stats['sml_tier_fallback'] = 1.0
                if mask_near.sum() == 0 or mask_nonNative.sum() == 0:
                    zero = torch.tensor(0.0, device=pred.device)
                    return zero, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                epsilon = 1e-5
                avg_nearNative = torch.mean(pred[mask_near])
                loss = torch.nn.functional.softplus(pred[mask_nonNative] - avg_nearNative)
                loss = loss.mean() + epsilon
                if loss.isnan():
                    zero = torch.tensor(0.0, device=pred.device)
                    return zero, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                if isTraining or withCEE:
                    return loss, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                best_rank, diff_rmsd, topn = Ranking(target, pred, higher_is_better=True)
                return (loss, best_rank, diff_rmsd, topn['top1_rmsd'],
                        topn['top3_rmsd'], topn['top5_rmsd'], topn['top10_rmsd'])
            if stats is not None:
                stats['sml_n_T1'] = 0.0
                stats['sml_n_T3'] = 0.0
                stats['sml_n_skipped'] = 0.0
                stats['sml_tier_fallback'] = 1.0 if _fallback else 0.0
            # near/non-native split by metric direction (cutoff = near_native_cutoff)
            if higher_is_better:
                mask_nonNative = target.lt(rmsd_cutoff)  # lDDT below cutoff => non-native
            else:
                mask_nonNative = target.ge(rmsd_cutoff)   # RMSD >= cutoff => non-native

            # if there is no near-native or no non-native decoys, log stats and return zero loss tensor
            if pred[~mask_nonNative].size(0)==0 or pred[mask_nonNative].size(0)==0:
                min_r = float(torch.min(target))
                max_r = float(torch.max(target))
                cnt = target.numel()
                present = 'non-native' if pred[~mask_nonNative].size(0)==0 else 'near-native'
                _logging.warning(
                    "%s: only %s decoys (count=%d) label range [%.3f, %.3f]",
                    pdb, present, cnt, min_r, max_r
                )
                # return tensor so subsequent code stays consistent
                zero = torch.tensor(0.0, device=pred.device)
                return zero, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            epsilon = 1e-5
            # Average score of near-native decoys
            avg_nearNative = torch.mean(pred[~mask_nonNative])
            # Push non-native scores away from the near-native average in the
            # "worse" direction. RMSD: non-native should score HIGHER (sign -1);
            # lDDT: non-native should score LOWER  (sign +1).
            sign = 1.0 if higher_is_better else -1.0
            diff_near_non = sign*(pred[mask_nonNative] - avg_nearNative)
            loss = torch.nn.functional.softplus(diff_near_non)
            loss = loss.mean() + epsilon
            if loss.isnan():
                print(pdb, loss)
                zero = torch.tensor(0.0, device=pred.device)
                return zero, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if isTraining or withCEE:
            best_rank = 0.0; diff_rmsd = 0.0
            top1_rmsd=0.0; top3_rmsd=0.0; top5_rmsd=0.0; top10_rmsd=0.0
        else:
            best_rank, diff_rmsd, topn_rmsd = Ranking(target,pred,higher_is_better=higher_is_better)
            top1_rmsd = topn_rmsd['top1_rmsd']
            top3_rmsd = topn_rmsd['top3_rmsd']
            top5_rmsd = topn_rmsd['top5_rmsd']
            top10_rmsd = topn_rmsd['top10_rmsd']
        
        return loss, best_rank, diff_rmsd, top1_rmsd, top3_rmsd, top5_rmsd, top10_rmsd


class InterfaceSoftRankLoss(nn.Module):
    """Within-target soft RankNet on the interface head against fnat.

    Replaces the earlier absolute-cutoff soft-margin formulation, which was a
    zero-loss no-op in 70-80 % of batches: fnat has low WITHIN-target variance,
    so a fixed near/non-native boundary rarely puts both classes inside one
    target (only 21-28 % of Boltz2_s10n10 targets at any cutoff 0.5-0.9), and
    the sparse surviving signal destabilised training. A pairwise formulation
    guarantees contrast in every target with >= 2 valid decoys.

    For every unique unordered pair (i, j) of valid decoys inside ONE target:

        delta_s = s_i - s_j                       (raw interface logits, no sigmoid)
        q_ij    = sigmoid((fnat_i - fnat_j) / tau_fnat)     soft preference target
        w_ij    = |2 * q_ij - 1|                            pair confidence
        pair    = w_ij * BCEWithLogits(delta_s, q_ij)

    ``q_ij`` is a continuous target in [0, 1]; BCEWithLogitsLoss accepts that
    directly and is numerically stable (log-sum-exp internally). ``w_ij`` is 0
    when fnat_i == fnat_j, so ties contribute nothing.

    Normalisation is deliberately by PAIR COUNT, not by the weight sum: a target
    whose decoys all share nearly the same fnat then produces a small gradient
    instead of being renormalised back up to full strength. Per-target losses
    are then averaged with equal weight, matching the target-wise convention
    used everywhere else in this codebase.

    Only decoys with finite fnat AND finite prediction participate (apo / GP
    decoys carry NaN fnat and drop out here). Targets left with < 2 valid decoys
    are skipped entirely.

    ``group_ids`` optionally partitions the input into several targets; with
    ``None`` (the live path, where one batch == one target) all decoys are
    treated as a single target.

    Returns ``(loss, stats)`` where ``stats`` is a plain-float dict for logging:
    ``num_rank_targets``, ``num_rank_pairs``, ``mean_abs_fnat_pair_delta``,
    ``mean_pair_confidence``.
    """

    # dead_zone > 0 drops pairs with |dfnat| <= dead_zone before averaging;
    # 0.0 keeps the original behaviour. NOTE Af3MatchedTierPairLoss(power=0) is
    # NOT equivalent: there the mass is split evenly across CELLS (= full tier
    # equalisation), here it is split evenly across surviving PAIRS.
    def __init__(self, tau_fnat=0.1, dead_zone=0.0):
        super(InterfaceSoftRankLoss, self).__init__()
        self.tau_fnat = float(tau_fnat)
        self.dead_zone = float(dead_zone)

    def forward(self, fnat, iface_pred, group_ids=None, tau_fnat=None):
        tau = float(self.tau_fnat if tau_fnat is None else tau_fnat)
        if tau <= 0:
            raise ValueError(f'tau_fnat must be > 0 (got {tau})')
        fnat = fnat.reshape(-1).float()
        iface_pred = iface_pred.reshape(-1)
        device = iface_pred.device
        zero_stats = {'num_rank_targets': 0.0, 'num_rank_pairs': 0.0,
                      'mean_abs_fnat_pair_delta': 0.0, 'mean_pair_confidence': 0.0,
                      'frac_pairs_kept': 0.0}
        if fnat.numel() != iface_pred.numel():
            return torch.zeros((), device=device), dict(zero_stats)

        finite = torch.isfinite(fnat) & torch.isfinite(iface_pred)
        if int(finite.sum()) < 2:
            return torch.zeros((), device=device), dict(zero_stats)

        if group_ids is None:
            groups = [finite]
        else:
            gids = torch.as_tensor(group_ids, device=device).reshape(-1)
            groups = [finite & (gids == g) for g in torch.unique(gids)]

        target_losses = []
        n_pairs_total = 0
        delta_sum = 0.0
        conf_sum = 0.0
        kept_frac = []
        for mask in groups:
            n = int(mask.sum())
            # a single decoy has no pair -> no ranking signal for this target
            if n < 2:
                continue
            f = fnat[mask]
            s = iface_pred[mask].float()
            ii, jj = torch.triu_indices(n, n, offset=1, device=device)
            df = f[ii] - f[jj]
            n_all = ii.numel()
            if self.dead_zone > 0:
                keep = df.abs() > self.dead_zone
                kept_frac.append(float(keep.float().mean()))
                if not bool(keep.any()):
                    continue
                ii, jj, df = ii[keep], jj[keep], df[keep]
            else:
                kept_frac.append(1.0)
            delta_s = s[ii] - s[jj]
            q = torch.sigmoid(df / tau)
            w = (2.0 * q - 1.0).abs()
            pair_bce = F.binary_cross_entropy_with_logits(delta_s, q, reduction='none')
            # mean over the pair COUNT (not the weight sum) on purpose
            target_losses.append((w * pair_bce).mean())
            n_pairs_total += ii.numel()
            delta_sum += float(df.abs().sum())
            conf_sum += float(w.sum())

        if not target_losses:
            return torch.zeros((), device=device), dict(zero_stats)

        loss = torch.stack(target_losses).mean()
        stats = {
            'num_rank_targets': float(len(target_losses)),
            'num_rank_pairs': float(n_pairs_total),
            'mean_abs_fnat_pair_delta': delta_sum / max(n_pairs_total, 1),
            'mean_pair_confidence': conf_sum / max(n_pairs_total, 1),
            'frac_pairs_kept': float(np.mean(kept_frac)) if kept_frac else 0.0,
        }
        return loss, stats


# Absolute fnat tier edges. Tiers are ABSOLUTE (not per-target quantiles) so the
# same fnat always lands in the same tier across targets and sources.
FNAT_TIER_EDGES = (0.1, 0.3, 0.5, 0.7)      # -> [0,.1) [.1,.3) [.3,.5) [.5,.7) [.7,1]


# Empirical (tier_i, tier_j) pair frequency of the AF3 TEST pool, measured on
# 142 holo targets x 100 candidates = 702,900 same-target pairs, AFTER the
# |delta fnat| > FNAT_DEAD_ZONE filter (so it describes the pairs actually
# trained on, not the raw pool). Upper triangular; row = lower tier index.
# Source: analyze/analyze_af3_pair_distribution.py
AF3_PAIR_CELL_P = (
    (0.131930, 0.214493, 0.066236, 0.053593, 0.132585),
    (0.000000, 0.025894, 0.034478, 0.028418, 0.035627),
    (0.000000, 0.000000, 0.013999, 0.024436, 0.024519),
    (0.000000, 0.000000, 0.000000, 0.031333, 0.042098),
    (0.000000, 0.000000, 0.000000, 0.000000, 0.140362),
)
FNAT_DEAD_ZONE = 0.05


class Af3MatchedTierPairLoss(nn.Module):
    """exp3: dead-zone filtered, AF3-matched tier-cell weighted soft RankNet.

    Two changes over :class:`TierBalancedSoftRankLoss`, both motivated by the
    measured AF3 test pair structure:

    **1. fnat dead-zone filter.** Pairs with ``|fnat_i - fnat_j| <= dead_zone``
    are dropped outright. On AF3 they are **70.9 %** of all same-target pairs and
    the within-tier median |delta fnat| is exactly 0.000 — they carry no ordering
    information, yet the soft target still hands them w_ij ~ 0.24 at tau=0.1, so
    numerically they drown the 29 % that do inform. Dropping them also flattens
    the cell distribution on its own: the two dominant cells fall from 68.6 % of
    all pairs to 27.2 %.

    **2. AF3-matched cell weighting, softened.** Cell mass is set to
    ``p_cell ** -power`` (default power=0.5, i.e. ``1/sqrt(p_cell)``) using the
    FIXED AF3 frequency table, then normalised over the cells the target actually
    has; within a cell the mass is uniform over its surviving pairs. Full
    equalisation (power=1) amplifies the rarest cell 59x and is what
    :class:`TierBalancedSoftRankLoss` did — it lost 0.027 fnat on the multisource
    valid pool. With the dead-zone filter applied first, power=0.5 amplifies only
    **3.9x**, so rare-but-informative cells are lifted without being allowed to
    dominate. power=0 reduces to plain soft RankNet on the surviving pairs.

    The per-pair objective is unchanged: continuous ``q_ij = sigmoid(dfnat/tau)``,
    confidence ``w_ij = |2q-1|``, ``w * BCEWithLogits(s_i - s_j, q)``. There is no
    hard tier classification term and no cross-target pair.

    Unlike exp1 this has **no same/cross-source axis** — exp3 is purely about tier
    cells, so the source split is left out to keep the change single-variable.
    """

    def __init__(self, tau_fnat=0.1, tier_edges=FNAT_TIER_EDGES,
                 dead_zone=FNAT_DEAD_ZONE, power=0.5, cell_p=AF3_PAIR_CELL_P):
        super(Af3MatchedTierPairLoss, self).__init__()
        self.tau_fnat = float(tau_fnat)
        self.tier_edges = tuple(float(x) for x in tier_edges)
        self.dead_zone = float(dead_zone)
        self.power = float(power)
        n = len(tier_edges) + 1
        pm = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                pm[i, j] = float(cell_p[i][j])
        # symmetrise so lookup by (min,max) is safe regardless of table layout
        pm = np.maximum(pm, pm.T)
        self._cell_p = pm
        # cell mass BEFORE per-target normalisation; 0 stays 0 (cell never seen)
        with np.errstate(divide='ignore'):
            cw = np.where(pm > 0, np.power(np.maximum(pm, 1e-12), -self.power), 0.0)
        self._cell_w = cw

    def forward(self, fnat, iface_pred, group_ids=None, tau_fnat=None):
        tau = float(self.tau_fnat if tau_fnat is None else tau_fnat)
        if tau <= 0:
            raise ValueError(f'tau_fnat must be > 0 (got {tau})')
        fnat = fnat.reshape(-1).float()
        iface_pred = iface_pred.reshape(-1)
        device = iface_pred.device
        zero = {'num_rank_targets': 0.0, 'num_rank_pairs': 0.0,
                'mean_abs_fnat_pair_delta': 0.0, 'mean_pair_confidence': 0.0,
                'frac_pairs_kept': 0.0, 'n_cells_used': 0.0,
                'frac_inter_tier': 0.0, 'n_targets_all_dead': 0.0}
        if fnat.numel() != iface_pred.numel():
            return torch.zeros((), device=device), dict(zero)
        finite = torch.isfinite(fnat) & torch.isfinite(iface_pred)
        if int(finite.sum()) < 2:
            return torch.zeros((), device=device), dict(zero)

        if group_ids is None:
            groups = [finite]
        else:
            g = torch.as_tensor(group_ids, device=device).reshape(-1)
            groups = [finite & (g == u) for u in torch.unique(g)]

        edges_t = torch.tensor(self.tier_edges, device=device)
        cw = torch.as_tensor(self._cell_w, device=device, dtype=torch.float32)

        losses, n_pairs, n_cells, kept_frac = [], 0, 0, []
        d_sum = c_sum = inter_w = 0.0
        n_all_dead = 0
        for mask in groups:
            n = int(mask.sum())
            if n < 2:
                continue
            f = fnat[mask]
            s = iface_pred[mask].float()
            ii, jj = torch.triu_indices(n, n, offset=1, device=device)
            df = f[ii] - f[jj]
            keep = df.abs() > self.dead_zone
            kept_frac.append(float(keep.float().mean()))
            if not bool(keep.any()):
                n_all_dead += 1          # every pair uninformative: nothing to learn
                continue
            ii, jj, df = ii[keep], jj[keep], df[keep]
            delta_s = s[ii] - s[jj]
            q = torch.sigmoid(df / tau)
            w_conf = (2.0 * q - 1.0).abs()
            bce = F.binary_cross_entropy_with_logits(delta_s, q, reduction='none')

            tier = torch.bucketize(f, edges_t)
            ti, tj = tier[ii], tier[jj]
            lo = torch.minimum(ti, tj)
            hi = torch.maximum(ti, tj)
            base = cw[lo, hi]                       # AF3-matched cell mass
            key = lo * 100 + hi
            uniq, inv = torch.unique(key, return_inverse=True)
            cnt = torch.bincount(inv, minlength=uniq.numel()).float()
            # split each cell's mass evenly over that cell's surviving pairs
            pw = base / cnt[inv].clamp(min=1.0)
            tot = pw.sum()
            if float(tot) <= 0:
                continue
            pw = pw / tot
            losses.append((pw * w_conf * bce).sum())
            n_pairs += int(keep.sum())
            n_cells += int(uniq.numel())
            d_sum += float((df.abs() * pw).sum())
            c_sum += float((w_conf * pw).sum())
            inter_w += float(((ti != tj).float() * pw).sum())

        if not losses:
            z = dict(zero)
            z['n_targets_all_dead'] = float(n_all_dead)
            z['frac_pairs_kept'] = float(np.mean(kept_frac)) if kept_frac else 0.0
            return torch.zeros((), device=device), z

        k = len(losses)
        loss = torch.stack(losses).mean()
        return loss, {
            'num_rank_targets': float(k),
            'num_rank_pairs': float(n_pairs),
            'mean_abs_fnat_pair_delta': d_sum / k,
            'mean_pair_confidence': c_sum / k,
            'frac_pairs_kept': float(np.mean(kept_frac)) if kept_frac else 0.0,
            'n_cells_used': n_cells / k,
            'frac_inter_tier': inter_w / k,
            'n_targets_all_dead': float(n_all_dead),
        }


class TierBalancedSoftRankLoss(nn.Module):
    """Absolute-tier balanced within-target Soft RankNet on fnat.

    Same per-pair objective as :class:`InterfaceSoftRankLoss` — the target stays
    the CONTINUOUS soft preference

        q_ij = sigmoid((fnat_i - fnat_j) / tau),  w_ij = |2 q_ij - 1|,
        pair = w_ij * BCEWithLogits(s_i - s_j, q_ij)

    and there is **no hard tier-classification term**. Tiers only decide how much
    each pair COUNTS, correcting the fact that uniform pair enumeration is
    dominated by whatever tier/source combination happens to be frequent.

    Each pair is labelled on two axes, both computed within one target:
      * inter-tier vs within-tier  (absolute fnat tiers, ``FNAT_TIER_EDGES``)
      * same-source vs cross-source (generation source of the two decoys)
    giving four cells, allocated ``inter_frac x same_frac`` etc. — with the
    defaults 0.75/0.75 that is 56.25 / 18.75 / 18.75 / 6.25 %. Inside the two
    inter-tier cells the budget is further split EQUALLY across the distinct
    unordered tier-pair combinations present, which is the "tier balanced" part:
    a (tier0,tier4) pair then carries the same total weight as the far more
    common (tier3,tier4) pair.

    Cells or tier-combos that a target cannot supply get their share
    redistributed proportionally over the ones it can, so a target is never
    dropped for being unbalanced (measured on real training batches: 5 % of
    targets have no inter-tier pair at all, and the two lowest tiers are almost
    unpopulated).

    ``mode``:
      * ``'weight'`` (default) — keep every pair and scale it by
        quota_cell / count_cell. Deterministic, zero sampling variance, and the
        realised loss composition equals the budget exactly.
      * ``'sample'`` — draw ``pairs_per_target`` pairs according to the quota.
        Matches a literal reading of "pair budget" but adds gradient noise.

    All pairs are within a single target in both modes; cross-target pairs are
    never formed.

    ``source_id`` is a per-decoy integer identifying the generation source. When
    it is None every pair counts as same-source and the source axis collapses.
    """

    def __init__(self, tau_fnat=0.1, tier_edges=FNAT_TIER_EDGES,
                 inter_tier_frac=0.75, same_source_frac=0.75,
                 mode='weight', pairs_per_target=512, seed=0):
        super(TierBalancedSoftRankLoss, self).__init__()
        self.tau_fnat = float(tau_fnat)
        self.tier_edges = tuple(float(x) for x in tier_edges)
        self.inter_tier_frac = float(inter_tier_frac)
        self.same_source_frac = float(same_source_frac)
        if mode not in ('weight', 'sample'):
            raise ValueError(f"mode must be 'weight' or 'sample' (got {mode})")
        self.mode = mode
        self.pairs_per_target = int(pairs_per_target)
        self._gen = np.random.default_rng(int(seed))

    # ── helpers ──────────────────────────────────────────────────────────────
    def _tier_of(self, f):
        return torch.bucketize(f, torch.tensor(self.tier_edges, device=f.device))

    def _cell_quota(self):
        it, ss = self.inter_tier_frac, self.same_source_frac
        return {
            ('inter', 'same'): it * ss,
            ('inter', 'cross'): it * (1.0 - ss),
            ('within', 'same'): (1.0 - it) * ss,
            ('within', 'cross'): (1.0 - it) * (1.0 - ss),
        }

    @staticmethod
    def _redistribute(quota, available):
        """Move the share of empty groups onto the non-empty ones, proportionally."""
        live = {k: v for k, v in quota.items() if available.get(k, 0) > 0}
        tot = sum(live.values())
        if tot <= 0:
            return {}
        return {k: v / tot for k, v in live.items()}

    # ── main ─────────────────────────────────────────────────────────────────
    def forward(self, fnat, iface_pred, source_id=None, group_ids=None, tau_fnat=None):
        tau = float(self.tau_fnat if tau_fnat is None else tau_fnat)
        if tau <= 0:
            raise ValueError(f'tau_fnat must be > 0 (got {tau})')
        fnat = fnat.reshape(-1).float()
        iface_pred = iface_pred.reshape(-1)
        device = iface_pred.device
        zero = {
            'num_rank_targets': 0.0, 'num_rank_pairs': 0.0,
            'mean_abs_fnat_pair_delta': 0.0, 'mean_pair_confidence': 0.0,
            'frac_inter_tier': 0.0, 'frac_cross_source': 0.0,
            'n_tier_combos': 0.0, 'frac_quota_redistributed': 0.0,
        }
        if fnat.numel() != iface_pred.numel():
            return torch.zeros((), device=device), dict(zero)

        finite = torch.isfinite(fnat) & torch.isfinite(iface_pred)
        if int(finite.sum()) < 2:
            return torch.zeros((), device=device), dict(zero)

        if source_id is None:
            sid_all = torch.zeros_like(fnat, dtype=torch.long)
        else:
            sid_all = torch.as_tensor(source_id, device=device).reshape(-1).long()
            if sid_all.numel() != fnat.numel():
                sid_all = torch.zeros_like(fnat, dtype=torch.long)

        if group_ids is None:
            groups = [finite]
        else:
            g = torch.as_tensor(group_ids, device=device).reshape(-1)
            groups = [finite & (g == u) for u in torch.unique(g)]

        base_quota = self._cell_quota()
        target_losses = []
        n_pairs_tot = 0
        d_sum = c_sum = 0.0
        inter_w = cross_w = 0.0
        combo_tot = 0
        redist_tot = 0.0
        n_used = 0

        for mask in groups:
            n = int(mask.sum())
            if n < 2:
                continue
            f = fnat[mask]
            s = iface_pred[mask].float()
            sid = sid_all[mask]
            tier = self._tier_of(f)

            ii, jj = torch.triu_indices(n, n, offset=1, device=device)
            df = f[ii] - f[jj]
            delta_s = s[ii] - s[jj]
            q = torch.sigmoid(df / tau)
            w_conf = (2.0 * q - 1.0).abs()
            pair_bce = F.binary_cross_entropy_with_logits(delta_s, q, reduction='none')

            ti, tj = tier[ii], tier[jj]
            is_inter = (ti != tj)
            is_cross = (sid[ii] != sid[jj])

            # group key per pair: cell + (for inter-tier) the tier-combo, so the
            # inter-tier budget can be spread evenly over combos.
            lo = torch.minimum(ti, tj)
            hi = torch.maximum(ti, tj)
            combo = lo * 100 + hi                       # unique per unordered tier pair
            keys = []
            for axis_inter, axis_cross, m in (
                ('inter', 'same', is_inter & ~is_cross),
                ('inter', 'cross', is_inter & is_cross),
                ('within', 'same', ~is_inter & ~is_cross),
                ('within', 'cross', ~is_inter & is_cross),
            ):
                keys.append(((axis_inter, axis_cross), m))

            # counts per (cell) and per (cell, tier-combo)
            avail_cell = {k: int(m.sum()) for k, m in keys}
            quota_cell = self._redistribute(base_quota, avail_cell)
            if not quota_cell:
                continue
            redist_tot += 1.0 - sum(base_quota[k] for k in quota_cell)

            pw = torch.zeros_like(pair_bce)
            n_combo_here = 0
            for k, m in keys:
                share = quota_cell.get(k, 0.0)
                if share <= 0 or avail_cell[k] == 0:
                    continue
                if k[0] == 'inter':
                    # split this cell's share EVENLY across the tier-combos it holds
                    cvals = torch.unique(combo[m])
                    n_combo_here = max(n_combo_here, int(cvals.numel()))
                    per = share / float(cvals.numel())
                    for cv in cvals:
                        mm = m & (combo == cv)
                        cnt = int(mm.sum())
                        if cnt:
                            pw[mm] = per / cnt
                else:
                    pw[m] = share / float(avail_cell[k])

            if self.mode == 'sample':
                p = (pw / pw.sum()).detach().cpu().numpy().astype('float64')
                p = p / p.sum()
                k = min(self.pairs_per_target, int((p > 0).sum()))
                idx = self._gen.choice(len(p), size=k, replace=False, p=p)
                sel = torch.zeros_like(pw, dtype=torch.bool)
                sel[torch.as_tensor(idx, device=device)] = True
                loss_t = (w_conf[sel] * pair_bce[sel]).mean()
                used = sel
                eff_w = torch.ones(int(sel.sum()), device=device) / max(int(sel.sum()), 1)
            else:
                # weighted mean; pw already sums to 1 over the pairs of this target
                loss_t = (pw * w_conf * pair_bce).sum()
                used = pw > 0
                eff_w = pw[used]

            target_losses.append(loss_t)
            n_used += 1
            npair = int(used.sum())
            n_pairs_tot += npair
            d_sum += float((df[used].abs() * eff_w).sum())
            c_sum += float((w_conf[used] * eff_w).sum())
            inter_w += float((is_inter[used].float() * eff_w).sum())
            cross_w += float((is_cross[used].float() * eff_w).sum())
            combo_tot += n_combo_here

        if not target_losses:
            return torch.zeros((), device=device), dict(zero)

        loss = torch.stack(target_losses).mean()
        stats = {
            'num_rank_targets': float(n_used),
            'num_rank_pairs': float(n_pairs_tot),
            # these are budget-weighted means, i.e. what the loss actually sees
            'mean_abs_fnat_pair_delta': d_sum / max(n_used, 1),
            'mean_pair_confidence': c_sum / max(n_used, 1),
            'frac_inter_tier': inter_w / max(n_used, 1),
            'frac_cross_source': cross_w / max(n_used, 1),
            'n_tier_combos': combo_tot / max(n_used, 1),
            'frac_quota_redistributed': redist_tot / max(n_used, 1),
        }
        return loss, stats


class CdrMatchedFnatContrastLoss(nn.Module):
    """exp5 "gate_tier_pair": CDR-matched, fnat-contrastive pair selection.

    Same continuous objective as the other soft-rank variants — the pair target
    stays ``q_ij = sigmoid((fnat_i - fnat_j)/tau)`` with confidence
    ``w_ij = |2q-1|`` and no hard tier label. Tiers are used ONLY to choose which
    pairs get budget.

    The pair the interface head must get right is one where the two decoys look
    equally good LOCALLY but differ completely at the interface, e.g.

        A: cdr_lddt 0.86, fnat 0.82
        B: cdr_lddt 0.84, fnat 0.08

    Any head trained on cdr_lddt is indifferent between those two, so this is
    exactly the supervision that makes the interface head carry information the
    intrinsic head does not. A pair qualifies as CONDITIONAL when, inside one
    target,

        |dfnat|      >= fnat_contrast_min          (default 0.20)
      AND ( |dcdr_lddt| <= lddt_match_tol          (default 0.05)
            OR the two sit in the same cdr_lddt bin
            OR in adjacent bins, when allow_adjacent_bin )

    Budget: ``cond_frac`` of the loss mass (default 0.60) goes to conditional
    pairs, the rest to ordinary informative pairs (|dfnat| > dead_zone). Targets
    with no conditional pair put everything on the ordinary set — nothing is
    fabricated.

    **Source-shortcut guard.** Conditional pairs are taken from the SAME source
    first; cross-source conditional pairs are only added when fewer than
    ``min_cond_pairs`` same-source ones exist. Without this the model can learn
    "looks like Boltz2 -> low, looks like PertMD -> high" instead of judging the
    interface, since the sources sit in very different fnat ranges. The strongest
    supervision is a high-cdr_lddt/high-fnat vs high-cdr_lddt/low-fnat pair from
    the same Boltz2 target.
    """

    def __init__(self, tau_fnat=0.1, dead_zone=FNAT_DEAD_ZONE,
                 lddt_match_tol=0.05, fnat_contrast_min=0.2, cond_frac=0.60,
                 lddt_bin_edges=(0.7, 0.8, 0.9), allow_adjacent_bin=True,
                 prefer_same_source=True, min_cond_pairs=8):
        super(CdrMatchedFnatContrastLoss, self).__init__()
        self.tau_fnat = float(tau_fnat)
        self.dead_zone = float(dead_zone)
        self.lddt_match_tol = float(lddt_match_tol)
        self.fnat_contrast_min = float(fnat_contrast_min)
        self.cond_frac = float(cond_frac)
        self.lddt_bin_edges = tuple(float(x) for x in lddt_bin_edges)
        self.allow_adjacent_bin = bool(allow_adjacent_bin)
        self.prefer_same_source = bool(prefer_same_source)
        self.min_cond_pairs = int(min_cond_pairs)

    def forward(self, fnat, iface_pred, cdr_lddt, source_id=None,
                group_ids=None, tau_fnat=None):
        tau = float(self.tau_fnat if tau_fnat is None else tau_fnat)
        if tau <= 0:
            raise ValueError(f'tau_fnat must be > 0 (got {tau})')
        fnat = fnat.reshape(-1).float()
        iface_pred = iface_pred.reshape(-1)
        lddt = torch.as_tensor(cdr_lddt).reshape(-1).float().to(iface_pred.device)
        device = iface_pred.device
        zero = {'num_rank_targets': 0.0, 'num_rank_pairs': 0.0,
                'mean_abs_fnat_pair_delta': 0.0, 'mean_pair_confidence': 0.0,
                'n_cond_pairs': 0.0, 'frac_cond_mass': 0.0,
                'frac_cond_same_source': 0.0, 'mean_abs_lddt_delta_cond': 0.0,
                'n_targets_no_cond': 0.0}
        if not (fnat.numel() == iface_pred.numel() == lddt.numel()):
            return torch.zeros((), device=device), dict(zero)
        finite = torch.isfinite(fnat) & torch.isfinite(iface_pred) & torch.isfinite(lddt)
        if int(finite.sum()) < 2:
            return torch.zeros((), device=device), dict(zero)

        if source_id is None:
            sid_all = torch.zeros_like(fnat, dtype=torch.long)
        else:
            sid_all = torch.as_tensor(source_id, device=device).reshape(-1).long()
            if sid_all.numel() != fnat.numel():
                sid_all = torch.zeros_like(fnat, dtype=torch.long)

        if group_ids is None:
            groups = [finite]
        else:
            gg = torch.as_tensor(group_ids, device=device).reshape(-1)
            groups = [finite & (gg == u) for u in torch.unique(gg)]

        edges = torch.tensor(self.lddt_bin_edges, device=device)
        losses, n_pairs, n_cond_tot = [], 0, 0
        d_sum = c_sum = cond_mass_sum = cond_same_sum = lddt_d_sum = 0.0
        n_no_cond = 0
        for mask in groups:
            n = int(mask.sum())
            if n < 2:
                continue
            f, s, l = fnat[mask], iface_pred[mask].float(), lddt[mask]
            sid = sid_all[mask]
            ii, jj = torch.triu_indices(n, n, offset=1, device=device)
            df = f[ii] - f[jj]
            dl = (l[ii] - l[jj]).abs()
            delta_s = s[ii] - s[jj]
            q = torch.sigmoid(df / tau)
            w_conf = (2.0 * q - 1.0).abs()
            bce = F.binary_cross_entropy_with_logits(delta_s, q, reduction='none')

            informative = df.abs() > self.dead_zone
            if not bool(informative.any()):
                continue
            lb = torch.bucketize(l, edges)
            dbin = (lb[ii] - lb[jj]).abs()
            lddt_close = (dl <= self.lddt_match_tol) | (dbin == 0)
            if self.allow_adjacent_bin:
                lddt_close = lddt_close | (dbin == 1)
            cond = informative & lddt_close & (df.abs() >= self.fnat_contrast_min)
            same_src = sid[ii] == sid[jj]
            cond_same = cond & same_src
            # same source first; widen to cross-source only if too few
            if self.prefer_same_source and int(cond_same.sum()) >= self.min_cond_pairs:
                cond_use = cond_same
            else:
                cond_use = cond

            pw = torch.zeros_like(bce)
            n_cond = int(cond_use.sum())
            if n_cond > 0:
                pw[cond_use] = self.cond_frac / n_cond
                rest = informative & ~cond_use
                if bool(rest.any()):
                    pw[rest] = (1.0 - self.cond_frac) / int(rest.sum())
                else:
                    pw[cond_use] = 1.0 / n_cond      # nothing else to spend on
            else:
                n_no_cond += 1
                pw[informative] = 1.0 / int(informative.sum())
            tot = pw.sum()
            if float(tot) <= 0:
                continue
            pw = pw / tot

            losses.append((pw * w_conf * bce).sum())
            used = pw > 0
            n_pairs += int(used.sum())
            n_cond_tot += n_cond
            d_sum += float((df[used].abs() * pw[used]).sum())
            c_sum += float((w_conf[used] * pw[used]).sum())
            cond_mass_sum += float(pw[cond_use].sum()) if n_cond else 0.0
            # fraction of the conditional MASS that landed on same-source pairs
            # (not merely how many were available) — this is what guards against
            # the source shortcut, so it is what must be logged.
            if n_cond:
                _m_all = float(pw[cond_use].sum())
                _m_same = float(pw[cond_use & same_src].sum())
                cond_same_sum += (_m_same / _m_all) if _m_all > 0 else 0.0
            lddt_d_sum += float(dl[cond_use].mean()) if n_cond else 0.0

        if not losses:
            z = dict(zero); z['n_targets_no_cond'] = float(n_no_cond)
            return torch.zeros((), device=device), z
        k = len(losses)
        return torch.stack(losses).mean(), {
            'num_rank_targets': float(k),
            'num_rank_pairs': float(n_pairs),
            'mean_abs_fnat_pair_delta': d_sum / k,
            'mean_pair_confidence': c_sum / k,
            'n_cond_pairs': n_cond_tot / k,
            'frac_cond_mass': cond_mass_sum / k,
            'frac_cond_same_source': cond_same_sum / k,
            'mean_abs_lddt_delta_cond': lddt_d_sum / k,
            'n_targets_no_cond': float(n_no_cond),
        }


class InterfaceFnatRegLoss(nn.Module):
    """Target-balanced SmoothL1 regression of (tau * interface logit) onto fnat.

    Auxiliary to :class:`InterfaceSoftRankLoss`. The rank loss only ever sees
    s_i - s_j, so it fixes the head's SLOPE but leaves its offset free and its
    output unreadable as a quality estimate. This term pins the offset:

        fnat_pred = tau_fnat * s          (LINEAR, no squashing)
        per-target loss = mean_over_decoys( SmoothL1(fnat_pred, fnat, beta) )
        loss            = mean_over_targets( per-target loss )

    **Why linear and not sigmoid(s).** The rank loss with temperature tau drives
    delta_s toward logit(q_ij) = (fnat_i - fnat_j)/tau, i.e. it is already
    learning s ~ fnat/tau + c. Multiplying by the SAME tau therefore expresses
    the prediction in exactly the units the rank loss is already working in, so
    the two terms agree on scale by construction and the aux only has to fix c.
    A sigmoid instead imposes a second, incompatible scale and — measured on the
    production model at init — saturates immediately: s had mean +18.4 / -31.9
    depending on seed, giving sigmoid'(s) of 1e-5 (healthy is 0.25), which makes
    the regression gradient vanish before it can move anything. Linear scaling
    has constant derivative tau, so it cannot saturate.

    The prediction is intentionally NOT clamped inside the loss: clamping would
    zero the gradient exactly for the decoys that are most wrong. Clamping is
    applied only to the logged ``mean_fnat_pred`` for interpretability.

    Averaging inside a target first and only then across targets keeps a target
    with 100 decoys from dominating one with 8 (same convention as the rank loss
    and as validation).

    ``beta`` is the SmoothL1 transition point; with fnat in [0,1] the PyTorch
    default beta=1.0 would keep the loss in its quadratic branch for any error
    the prediction can plausibly make, so a smaller beta (default 0.1) is what
    actually buys the linear/robust branch.

    Only finite-fnat decoys participate (apo/GP are NaN and drop out); targets
    left with no valid decoy are skipped.

    Returns ``(loss, stats)`` with ``num_reg_targets``, ``num_reg_decoys``,
    ``mean_abs_fnat_err`` (unclamped, matches the loss), ``mean_fnat_pred``
    (clamped to [0,1] for readability) and ``frac_pred_out_of_range``.
    """

    def __init__(self, beta=0.1, tau_fnat=0.1):
        super(InterfaceFnatRegLoss, self).__init__()
        self.beta = float(beta)
        self.tau_fnat = float(tau_fnat)

    def forward(self, fnat, iface_pred, group_ids=None, beta=None, tau_fnat=None):
        b = float(self.beta if beta is None else beta)
        tau = float(self.tau_fnat if tau_fnat is None else tau_fnat)
        if b <= 0:
            raise ValueError(f'fnat_reg_beta must be > 0 (got {b})')
        if tau <= 0:
            raise ValueError(f'tau_fnat must be > 0 (got {tau})')
        fnat = fnat.reshape(-1).float()
        iface_pred = iface_pred.reshape(-1)
        device = iface_pred.device
        zero_stats = {'num_reg_targets': 0.0, 'num_reg_decoys': 0.0,
                      'mean_abs_fnat_err': 0.0, 'mean_fnat_pred': 0.0,
                      'frac_pred_out_of_range': 0.0}
        if fnat.numel() != iface_pred.numel():
            return torch.zeros((), device=device), dict(zero_stats)

        finite = torch.isfinite(fnat) & torch.isfinite(iface_pred)
        if not bool(finite.any()):
            return torch.zeros((), device=device), dict(zero_stats)

        if group_ids is None:
            groups = [finite]
        else:
            gids = torch.as_tensor(group_ids, device=device).reshape(-1)
            groups = [finite & (gids == g) for g in torch.unique(gids)]

        target_losses = []
        n_decoys_total = 0
        err_sum = 0.0
        pred_sum = 0.0
        oor_sum = 0
        for mask in groups:
            n = int(mask.sum())
            if n < 1:
                continue
            f = fnat[mask]
            # same units the rank loss already works in; no squashing, no clamp
            p = tau * iface_pred[mask].float()
            # mean over this target's decoys, so per-target weight is equal
            target_losses.append(F.smooth_l1_loss(p, f, reduction='mean', beta=b))
            pd = p.detach()
            n_decoys_total += n
            err_sum += float((pd - f).abs().sum())
            pred_sum += float(pd.clamp(0.0, 1.0).sum())   # clamp for LOGGING only
            oor_sum += int(((pd < 0.0) | (pd > 1.0)).sum())

        if not target_losses:
            return torch.zeros((), device=device), dict(zero_stats)

        loss = torch.stack(target_losses).mean()
        stats = {
            'num_reg_targets': float(len(target_losses)),
            'num_reg_decoys': float(n_decoys_total),
            'mean_abs_fnat_err': err_sum / max(n_decoys_total, 1),
            'mean_fnat_pred': pred_sum / max(n_decoys_total, 1),
            'frac_pred_out_of_range': oor_sum / max(n_decoys_total, 1),
        }
        return loss, stats


def grad_scale_wrt(component, wrt):
    """L2 norm of d(component)/d(wrt), as a plain float, for logging only.

    Used to compare how hard each interface component pulls on the SAME head
    output, which is the only fair comparison point (comparing raw loss values
    across a BCE and a SmoothL1 says nothing about relative gradient pressure).
    Cheap: the subgraph from the head output to each component loss is a handful
    of elementwise ops, NOT the SE(3) backbone. Never raises.
    """
    if not isinstance(component, torch.Tensor) or not component.requires_grad:
        return 0.0
    try:
        g = torch.autograd.grad(component, wrt, retain_graph=True,
                                create_graph=False, allow_unused=True)[0]
    except Exception:
        return 0.0
    if g is None:
        return 0.0
    return float(g.detach().float().norm())


class FinalLoss(nn.Module):
    def __init__(self):
        super(FinalLoss, self).__init__()

    def forward(self,pdb,target,pred,isTraining,device,inference=False,rmsd_cutoff=2.0,loss_type='sml',label_metric='loop_rmsd',
                fnat=None, tier_lddt=None, tier_fnat=None, tier_lddt_low=None,
                fnat_gate=None, near_cut=None, non_cut=None,
                source_id=None, demote_ids=None, demote_max_frac=0.5):
        # ``label_metric`` picks the metric direction. loop_lddt is higher-is-better,
        # which flips the SML sign, the near/non-native split, and the ranking order.
        higher_is_better = 'lddt' in str(label_metric).lower()
        sml = SoftMarginLoss_rmsd()
        # cee = LabelSmoothingLoss(classes=16)
        cee = Cee_5A()

        # final_loss = loss_sml*1 + loss_cee
        # if inference:
        #     loss_cee, best_rank, lddt_diff, pLDDT = cee(target,pred,isTraining,device,inference)
        #     final_loss = loss_cee
        #     return final_loss, best_rank, lddt_diff, pLDDT
        # else: # For training and validation
            
        out_dic={}

        # if torch.isnan(pred).any() or torch.isnan(target).any():
        #     print(f"NaN detected in inputs for {pdb}: pred {pred}, target {target}")
        #     return torch.tensor(0.0).to(pred.device), [0.0]*6  # Return a default value and avoid unpacking error
            
        _tier_stats = {}
        if loss_type=='sml':
            try:
                final_loss, best_rank, rmsd_diff, top1_rmsd, top3_rmsd, top5_rmsd, top10_rmsd = sml(
                    pdb,target,pred,isTraining,inference,rmsd_cutoff,higher_is_better=higher_is_better,
                    fnat=fnat, tier_lddt=tier_lddt, tier_fnat=tier_fnat,
                    tier_lddt_low=tier_lddt_low, stats=_tier_stats,
                    fnat_gate=fnat_gate, near_cut=near_cut, non_cut=non_cut,
                    source_id=source_id, demote_ids=demote_ids,
                    demote_max_frac=demote_max_frac)
            except Exception as e:
                print('\n###### problem #####')
                print(pdb)
                print(e)
                print('\n')
                sys.exit()
            pRMSD=pred
        elif loss_type=='cee':
            final_loss, best_rank, rmsd_diff, pRMSD, top1_rmsd = cee(pdb,target,pred,isTraining,device)
        elif loss_type=='sml+cee':  
            pred_sml = pred[:,0]
            pred_cee = pred[:,1:]
            loss_sml, best_rank, rmsd_diff,top1_rmsd = sml(pdb,target,pred_sml,isTraining,inference,rmsd_cutoff,withCEE=True,higher_is_better=higher_is_better)
            loss_cee, best_rank, rmsd_diff, pRMSD, top1_rmsd = cee(pdb,target,pred_cee,isTraining,device)
            alpha=5 # for sml
            final_loss = (alpha/(alpha+1))*loss_sml+(1/(alpha+1))*loss_cee # Weight 5 can be changed
            
        with torch.no_grad():
            out_dic['final_loss']=final_loss
            out_dic['best_rank']=best_rank
            out_dic['rmsd_diff']=rmsd_diff
            out_dic['top1_rmsd']=top1_rmsd
            out_dic['top3_rmsd']=top3_rmsd
            out_dic['top5_rmsd']=top5_rmsd
            out_dic['top10_rmsd']=top10_rmsd
            # exp6b/exp6all: emit the 2-D tier counts on EVERY step (zeros when the
            # tier path is off) so all DDP ranks share the keyset.
            for _k in ('sml_n_T1', 'sml_n_T3', 'sml_n_skipped', 'sml_tier_fallback',
                       'sml_n_fnat_gated', 'sml_n_demote_dropped'):
                out_dic[_k] = float(_tier_stats.get(_k, 0.0))

        return final_loss, out_dic
    
class PreferenceLoss:
    def __init__(self):
        
        pass

    def sample_preference(self, pred, rmsd) -> dict:
        chosen_list = []
        rejected_list = []
        num_elements = rmsd.numel()

        for _ in range(num_elements):
            idx1, idx2 = random.sample(range(num_elements), 2)
                                                
            if rmsd[idx1] <= rmsd[idx2]:
                chosen, rejected = pred['out'][idx1], pred['out'][idx2]

            else:
                chosen, rejected = pred['out'][idx2], pred['out'][idx1]

            chosen_list.append(chosen.unsqueeze(0))
            rejected_list.append(rejected.unsqueeze(0))
        
        chosen_tensor = torch.cat(chosen_list)
        rejected_tensor = torch.cat(rejected_list)
        preference = {'chosen': chosen_tensor, 'rejected': rejected_tensor}

        return preference
    
    def pairwise_preference(self, pred, rmsd) -> dict:
        num_elements = rmsd.numel()
        idx1, idx2 = random.sample(range(num_elements), 2)
                                                
        if rmsd[idx1] <= rmsd[idx2]:
            chosen, rejected = pred['out'][idx1], pred['out'][idx2]

        else:
            chosen, rejected = pred['out'][idx2], pred['out'][idx1]

        preference = {'chosen': chosen.unsqueeze(0), 'rejected': rejected.unsqueeze(0)}

        return preference
    
    def exp_loss(self, preference):
        chosen = preference['chosen']
        rejected = preference['rejected']
        diff = chosen - rejected
        diff[diff > 100] = 100
        losses = torch.exp(diff)
        loss = torch.mean(losses)

        return loss
    
    def SiLU_loss(self, preference):
        chosen = preference['chosen']
        rejected = preference['rejected']
        diff = chosen - rejected
        diff[diff > 10] = 10
        losses = F.silu(diff)
        loss = torch.mean(losses)

        return loss

    def ReLU_loss(self, preference):
        chosen = preference['chosen']
        rejected = preference['rejected']
        diff = chosen - rejected
        diff[diff > 10] = 10
        losses = F.relu(diff)
        loss = torch.mean(losses)

        return loss
    
    def exp_1_loss(self, preference):
        chosen = preference['chosen']
        rejected = preference['rejected']
        diff = chosen - rejected
        diff[diff > 100] = 100
        diff[diff < 0] = 0
        losses = torch.exp(diff) - 1
        loss = torch.mean(losses)

        return loss
    
    def PairwiseLogisticLoss(self, preference):
        chosen = preference['chosen']
        rejected = preference['rejected']
        score = chosen - rejected
        score = torch.mean(score)
        loss = torch.log(1 + torch.exp(score))

        return loss
    
    def forward(self, pred, rmsd, device):
        out_dic = {}
        # preference = self.sample_preference(pred, rmsd)
        preference = self.pairwise_preference(pred, rmsd)
        loss = self.PairwiseLogisticLoss(preference).to(device)
        
        best_rank, diff_rmsd, topn_rmsd = Ranking(rmsd, pred['out'])
        
        with torch.no_grad():
            out_dic['final_loss'] = loss
            out_dic['best_rank'] = best_rank
            out_dic['rmsd_diff'] = diff_rmsd
            out_dic['top1_rmsd'] = topn_rmsd['top1_rmsd']
            out_dic['top3_rmsd'] = topn_rmsd['top3_rmsd']
            out_dic['top5_rmsd'] = topn_rmsd['top5_rmsd']
            out_dic['top10_rmsd'] = topn_rmsd['top10_rmsd']

        return loss, out_dic

class RankingLoss:
    def __init__(self) -> None:
        
        pass
        
    def forward(self, pred, rmsd, device):
        out_dic = {}
        top_one_pred = torch.softmax(pred['out'], dim=0)
        top_one_rmsd = torch.softmax(rmsd, dim=0)
        # loss = - torch.sum(top_one_rmsd * torch.log(top_one_pred)).to(device)
        loss = - F.kl_div(top_one_pred, top_one_rmsd, reduction='batchmean').to(device)
        
        best_rank, diff_rmsd, topn_rmsd = Ranking(rmsd, pred['out'])
        
        with torch.no_grad():
            out_dic['final_loss'] = loss
            out_dic['best_rank'] = best_rank
            out_dic['rmsd_diff'] = diff_rmsd
            out_dic['top1_rmsd'] = topn_rmsd['top1_rmsd']
            out_dic['top3_rmsd'] = topn_rmsd['top3_rmsd']
            out_dic['top5_rmsd'] = topn_rmsd['top5_rmsd']
            out_dic['top10_rmsd'] = topn_rmsd['top10_rmsd']
            
        return loss, out_dic

class TripletLoss:
    def __init__(self):
        
        pass
        
    def triplet_mining(self, pred, rmsd, rmsd_cutoff=2.0):
        mask_nonNative = rmsd.ge(rmsd_cutoff)

        nonNative = torch.where(mask_nonNative)[0]
        nearNative = torch.where(~mask_nonNative)[0]
        
        anchor = torch.argmin(rmsd).item()
        positive = nearNative[torch.randint(0, len(nearNative), (1,))].item()
        negative = nonNative[torch.randint(0, len(nonNative), (1,))].item()

        return anchor, positive, negative
    
    def forward(self, pred, rmsd, device):
        out_dic = {}
        anchor, positive, negative = self.triplet_mining(pred, rmsd)
        loss = F.triplet_margin_loss(pred['out'][anchor], pred['out'][positive], pred['out'][negative]).to(device)
        
        best_rank, diff_rmsd, topn_rmsd = Ranking(rmsd, pred['out'])
        
        with torch.no_grad():
            out_dic['final_loss'] = loss
            out_dic['best_rank'] = best_rank
            out_dic['rmsd_diff'] = diff_rmsd
            out_dic['top1_rmsd'] = topn_rmsd['top1_rmsd']
            out_dic['top3_rmsd'] = topn_rmsd['top3_rmsd']
            out_dic['top5_rmsd'] = topn_rmsd['top5_rmsd']
            out_dic['top10_rmsd'] = topn_rmsd['top10_rmsd']
            
        return loss, out_dic

class DPOLoss:
    def __init__(self, beta:float=0.1, label_smoothing:float=0.1, n_pairs:int=16):
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.n_pairs = n_pairs
        
    def log_prob_concat(self, score, rmsd):
        num_elements = rmsd.numel()
        chosen_log_probs = []
        rejected_log_probs = []

        for _ in range(self.n_pairs):
            idx1, idx2 = torch.randint(0, num_elements, (2,))
            
            if rmsd[idx1] <= rmsd[idx2]:
                chosen, rejected = - score['out'][idx1], - score['out'][idx2]

            else:
                chosen, rejected = - score['out'][idx2], - score['out'][idx1]

            pair = torch.stack([chosen, rejected])
            log_probs = F.log_softmax(pair, dim=0)
            chosen_log_probs.append(log_probs[0])
            rejected_log_probs.append(log_probs[1])

        preference = {
            'chosen': torch.stack(chosen_log_probs),
            'rejected': torch.stack(rejected_log_probs)
        }

        return preference
    
    def log_prob(self, score, rmsd):
        num_elements = rmsd.numel()
        chosen_log_probs = []
        rejected_log_probs = []
        
        for _ in range(num_elements):
            idx1, idx2 = torch.randint(0, num_elements, (2,))
            
            if rmsd[idx1] <= rmsd[idx2]:
                chosen, rejected = score[idx1], score[idx2]

            else:
                chosen, rejected = score[idx2], score[idx1]

            chosen_log_probs.append(chosen)
            rejected_log_probs.append(rejected)

        preference = {
            'chosen': torch.stack(chosen_log_probs),
            'rejected': torch.stack(rejected_log_probs)
        }

        return preference
    
    def nn_log_prob(self, score, rmsd):
        max_size = max(tensor.size(0) for tensor in score)
        mask_non = rmsd.ge(2.0)
        mask_near = ~mask_non
        idx_non = torch.nonzero(mask_non, as_tuple=True)[0]
        idx_near = torch.nonzero(mask_near, as_tuple=True)[0]
        if len(idx_non) == 0 or len(idx_near) == 0:
            return None
        chosen_log_probs = []
        rejected_log_probs = []
        for _ in range(self.n_pairs):
            ni = idx_non[torch.randint(0, len(idx_non), (1,))].item()
            ci = idx_near[torch.randint(0, len(idx_near), (1,))].item()
            chosen_log_probs.append(F.pad(score[ci], (0, max_size - score[ci].size(0))))
            rejected_log_probs.append(F.pad(score[ni], (0, max_size - score[ni].size(0))))
        preference = {
            'chosen': torch.stack(chosen_log_probs).squeeze(),
            'rejected': torch.stack(rejected_log_probs).squeeze()
        }
        return preference
        
    def dpo_loss(self, pi, ref, rmsd):
        pred_preference = self.nn_log_prob(pi, rmsd)
        ref_preference = self.nn_log_prob(ref, rmsd)
        if pred_preference is None or ref_preference is None:
            return pi[0].sum() * 0.0
        pred_ratio = pred_preference['chosen'] - pred_preference['rejected']
        del pred_preference
        ref_ratio = ref_preference['chosen'] - ref_preference['rejected']
        del ref_preference
        logits = pred_ratio - ref_ratio
        del pred_ratio, ref_ratio
        loss = - F.logsigmoid(self.beta * (logits)).mean()
        return loss

    def cdpo_loss(self, pred, ref, rmsd):
        pred_preference = self.log_prob_concat(pred, rmsd)
        ref_preference = self.log_prob_concat(ref, rmsd)
        pred_ratio = pred_preference['chosen'] - pred_preference['rejected']
        ref_ratio = ref_preference['chosen'] - ref_preference['rejected']
        logits = pred_ratio - ref_ratio
        loss = - F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing) - F.logsigmoid(-self.beta * logits) * self.label_smoothing

        return loss
    
    def ipo_loss(self, pi, ref, rmsd):
        pred_preference = self.log_prob(pi, rmsd)
        ref_preference = self.log_prob(ref, rmsd)
        pred_ratio = pred_preference['chosen'] - pred_preference['rejected']
        ref_ratio = ref_preference['chosen'] - ref_preference['rejected']
        logits = pred_ratio - ref_ratio
        loss = (logits - 1 / (2 * self.beta)) ** 2

        return loss
    
    def dpo_free_loss(self, pi, ref, rmsd):
        pred_preference = self.nn_log_prob(pi, rmsd)
        pred_ratio = pred_preference['chosen'] - pred_preference['rejected']
        logits = pred_ratio
        loss = - F.logsigmoid(self.beta * logits)
        return loss

    def forward(self, pi, ref, pred, rmsd, device):
        out_dic = {}
        loss, win_ratio, lose_ratio = self.dpo_loss(pi, ref, rmsd)
        loss = loss.to(device)
        best_rank, diff_rmsd, topn_rmsd = Ranking(rmsd, pred['out'])
        with torch.no_grad():
            out_dic['final_loss'] = loss
            out_dic['best_rank'] = best_rank
            out_dic['rmsd_diff'] = diff_rmsd
            out_dic['top1_rmsd'] = topn_rmsd['top1_rmsd']
            out_dic['top3_rmsd'] = topn_rmsd['top3_rmsd']
            out_dic['top5_rmsd'] = topn_rmsd['top5_rmsd']
            out_dic['top10_rmsd'] = topn_rmsd['top10_rmsd']
        return loss, out_dic

class TotalLoss:
    
    def __init__(self, n_pairs: int = 16):
        self.dpo_loss = DPOLoss(n_pairs=n_pairs)
        self.ndpo_loss = nDPOLoss()
        self.sml_loss = SoftMarginLoss_rmsd()
        self.weight = 10
    
    def forward(self, pi, ref, pred, rmsd, pdb, device, is_training, return_components=False,
                higher_is_better=False, near_native_cutoff=None,
                sml_kwargs=None, skip_legacy_dpo=False):
        # exp9: `sml_kwargs` threads exp8's SML rule (fnat gate, 0.90/0.85 cutoffs,
        # PertMD_all cap) into the DPO run's stabiliser, so the term holding the
        # model steady is the SAME objective it was pretrained on rather than the
        # legacy single-cutoff rule. `skip_legacy_dpo` drops the old nodewise DPO
        # term, which would otherwise add an uncontrolled third loss.
        out_dic = {}
        # Direction-aware SML: loop_lddt is higher-is-better (near-native cutoff 0.8),
        # loop_rmsd is lower-is-better (cutoff 2.0 Å). Flips near/non split + sign + ranking.
        _cut = near_native_cutoff if near_native_cutoff is not None else (0.8 if higher_is_better else 2.0)
        _smlkw = dict(sml_kwargs or {})
        _stats = {}
        loss_sml, best_rank, rmsd_diff, top1_rmsd, top3_rmsd, top5_rmsd, top10_rmsd = self.sml_loss(
            pdb, rmsd, pred, is_training, rmsd_cutoff=_cut, higher_is_better=higher_is_better,
            stats=_stats, **_smlkw)
        # Phase/tier path only uses SML; skip legacy near-vs-non DPO (crashes when all RMSD < 2Å).
        if return_components or skip_legacy_dpo:
            loss_dpo = loss_sml.new_zeros(())
        else:
            loss_dpo = self.dpo_loss.dpo_loss(pi, ref, rmsd)
        loss = loss_dpo + self.weight*loss_sml
        loss = loss.to(device)
        with torch.no_grad():
            out_dic['final_loss'] = loss
            out_dic['loss_sml'] = loss_sml.detach()
            out_dic['loss_dpo_base'] = loss_dpo.detach()
            out_dic['best_rank'] = best_rank
            out_dic['rmsd_diff'] = rmsd_diff
            out_dic['top1_rmsd'] = top1_rmsd
            out_dic['top3_rmsd'] = top3_rmsd
            out_dic['top5_rmsd'] = top5_rmsd
            out_dic['top10_rmsd'] = top10_rmsd
            for _k in ('sml_n_T1', 'sml_n_T3', 'sml_n_skipped', 'sml_tier_fallback',
                       'sml_n_fnat_gated', 'sml_n_demote_dropped'):
                out_dic[_k] = float(_stats.get(_k, 0.0))
        if return_components:
            return loss, out_dic, {
                'loss_sml': loss_sml,
                'loss_dpo_base': loss_dpo,
            }
        return loss, out_dic

class nDPOLoss:
    
    def __init__(self, beta:float=0.1):
        self.beta = beta
        
    def log_prob(self, pred, rmsd):
        score = [torch.log_softmax(-tensor, dim=0) for tensor in pred]
        idx1, idx2 = torch.randint(0, len(rmsd), (2,))
        if rmsd[idx1] <= rmsd[idx2]:
            winner, loser = score[idx1], score[idx2]
        else:
            winner, loser = score[idx2], score[idx1]
        return winner - loser
    
    def log_probs(self, pred, rmsd):
        score = [torch.log_softmax(-tensor, dim=0) for tensor in pred]
        log_probs = []
        for _ in range(16):
            idx1, idx2 = torch.randint(0, len(rmsd), (2,))
            if rmsd[idx1] <= rmsd[idx2]:
                winner = score[idx1]
                loser = score[idx2]
            else:
                winner = score[idx2]
                loser = score[idx1]
            log_probs.append(winner - loser)
        return torch.stack(log_probs)
    
    def all_log_prob(self, pred, rmsd):
        score = [torch.log_softmax(-tensor, dim=0) for tensor in pred]
        near = torch.argmin(rmsd)
        non = torch.argmax(rmsd)
        winner = score[near]
        loser = score[non]
        return winner - loser

    def nn_log_prob(self, pred, rmsd):
        max_size = max(tensor.size(0) for tensor in pred)
        mask_non = rmsd.ge(2.0)
        mask_near = ~mask_non
        idx_non = torch.nonzero(mask_non, as_tuple=True)[0]
        idx_near = torch.nonzero(mask_near, as_tuple=True)[0]
        if len(idx_non) == 0 or len(idx_near) == 0:
            return None
        chosen_log_probs = []
        rejected_log_probs = []
        for _ in range(16):
            ni = idx_non[torch.randint(0, len(idx_non), (1,))].item()
            ci = idx_near[torch.randint(0, len(idx_near), (1,))].item()
            c = torch.log_softmax(-pred[ci], dim=0)
            r = torch.log_softmax(-pred[ni], dim=0)
            chosen_log_probs.append(F.pad(c, (0, max_size - c.size(0))))
            rejected_log_probs.append(F.pad(r, (0, max_size - r.size(0))))
        preference = {
            'chosen': torch.stack(chosen_log_probs).squeeze(),
            'rejected': torch.stack(rejected_log_probs).squeeze()
        }
        return preference
    
    def ndpo_loss(self, pi, ref, rmsd):
        pi_preference = self.nn_log_prob(pi, rmsd)
        ref_preference = self.nn_log_prob(ref, rmsd)
        if pi_preference is None or ref_preference is None:
            return pi[0].sum() * 0.0
        pi_ratio = pi_preference['chosen'] - pi_preference['rejected']
        del pi_preference
        ref_ratio = ref_preference['chosen'] - ref_preference['rejected']
        del ref_preference
        logits = pi_ratio - ref_ratio
        del pi_ratio, ref_ratio
        loss = -F.logsigmoid(self.beta * logits).mean()
        return loss
    
    def forward(self, pi, ref, pred, rmsd, device):
        out_dic = {}
        loss = self.ndpo_loss(pi, ref, rmsd).mean().to(device)
        best_rank, diff_rmsd, topn_rmsd = Ranking(rmsd, pred['out'])
        with torch.no_grad():
            out_dic['final_loss'] = loss
            out_dic['best_rank'] = best_rank
            out_dic['rmsd_diff'] = diff_rmsd
            out_dic['top1_rmsd'] = topn_rmsd['top1_rmsd']
            out_dic['top3_rmsd'] = topn_rmsd['top3_rmsd']
            out_dic['top5_rmsd'] = topn_rmsd['top5_rmsd']
            out_dic['top10_rmsd'] = topn_rmsd['top10_rmsd']
        return loss, out_dic
    
class LDPOLoss:
    def __init__(self, beta:float=0.5):
        self.beta = beta
        
    def log_prob(self, score, rmsd):
        num_elements = rmsd.numel()
        chosen_log_probs = []
        rejected_log_probs = []
        true_log_probs = []
        false_log_probs = []

        for _ in range(16):
            idx1, idx2 = torch.randint(0, num_elements, (2,))
                
            if score[idx1] >= score[idx2]:
                chosen, rejected = score[idx1], score[idx2]

            else:
                chosen, rejected = score[idx2], score[idx1]
                
            score_pair = torch.stack([chosen, rejected])
            print(score_pair)
                
            if rmsd[idx1] <= rmsd[idx2]:
                true, false = rmsd[idx1], rmsd[idx2]
            
            else:
                true, false = rmsd[idx2], rmsd[idx1]
                
            rmsd_pair = torch.stack([true, false])
            print(rmsd_pair)
                
            pred_log_probs = F.softmax(score_pair, dim=0)
            rmsd_log_probs = F.softmax(rmsd_pair, dim=0)
            
            chosen_log_probs.append(pred_log_probs[0])
            rejected_log_probs.append(pred_log_probs[1])
            
            true_log_probs.append(rmsd_log_probs[0])
            false_log_probs.append(rmsd_log_probs[1])

        pred_preference = {
            'chosen': torch.stack(chosen_log_probs),
            'rejected': torch.stack(rejected_log_probs)
        }
        
        rmsd_preference = {
            'true': torch.stack(true_log_probs),
            'false': torch.stack(false_log_probs)
        }
        
        print(pred_preference)
        print(rmsd_preference)

        return pred_preference, rmsd_preference
    
    def prob_dist(self, score, rmsd):
        num_elements = rmsd.numel()
        chosen_probs = []
        rejected_probs = []
        true_probs = []
        false_probs = []
        
        for _ in range(64):
            idx1, idx2 = torch.randint(0, num_elements, (2,))
            
            if score[idx1] <= score[idx2]:
                chosen, rejected = score[idx1], score[idx2]
            
            else:
                chosen, rejected = score[idx2], score[idx1]
            
            chosen_probs.append(chosen)
            rejected_probs.append(rejected)
            
            if rmsd[idx1] <= rmsd[idx2]:
                true, false = rmsd[idx1], rmsd[idx2]
                
            else:
                true, false = rmsd[idx2], rmsd[idx1]
                
            true_probs.append(true)
            false_probs.append(false)
            
        score_preference = {
            'chosen': torch.stack(chosen_probs),
            'rejected': torch.stack(rejected_probs)
        }
        
        rmsd_preference = {
            'true': torch.stack(true_probs),
            'false': torch.stack(false_probs)
        }
        
        return score_preference, rmsd_preference

    def sample_pair_preference(self, score, rmsd):
        num_elements = rmsd.numel()
        score_preference = []
        rmsd_preference = []
        
        for _ in range(16):
            idx1, idx2 = torch.randint(0, num_elements, (2,))
            
            if score[idx1] <= score[idx2]:
                chosen, rejected = score[idx1], score[idx2]
                
            else:
                chosen, rejected = score[idx2], score[idx1]

            score_tensor = torch.stack([chosen, rejected])
            score_preference.append(F.softmax(score_tensor, dim=0))
            
            if rmsd[idx1] <= rmsd[idx2]:
                true, false = rmsd[idx1], rmsd[idx2]
                
            else:
                true, false = rmsd[idx2], rmsd[idx1]

            rmsd_tensor = torch.stack([true, false])
            rmsd_preference.append(F.softmax(rmsd_tensor, dim=0))
        
        score_preference = torch.stack(score_preference)
        rmsd_preference = torch.stack(rmsd_preference)

        return score_preference, rmsd_preference
    
    def kl_div_loss(self, score, rmsd):
        score_preference, rmsd_preference = self.sample_pair_preference(score, rmsd)
        loss = F.kl_div(score_preference, rmsd_preference, reduction='batchmean')

        return loss
    
    def kl_pair_loss(self, pred, rmsd_s):
        score_preference, rmsd_preference = self.sample_pair_preference(pred, rmsd_s)
        fn = torch.nn.KLDivLoss(reduction='batchmean')
        loss = fn(score_preference.log(), rmsd_preference)
        
        return loss
    
    def ldpo_loss(self, score, rmsd):
        score_preference, rmsd_preference = self.prob_dist(score, rmsd)
        score_ratio = score_preference['chosen'] - score_preference['rejected']
        rmsd_ratio = rmsd_preference['true'] - rmsd_preference['false']
        loss = F.kl_div(score_ratio, rmsd_ratio, log_target=True)
        
        return loss
    
    def forward(self, score, rmsd, pred, rmsd_s, device):
        out_dic = {}
        loss = self.kl_pair_loss(pred['out'], rmsd_s).mean().to(device)
        best_rank, diff_rmsd, topn_rmsd = Ranking(rmsd_s, pred['out'])
        
        with torch.no_grad():
            out_dic['final_loss'] = loss
            out_dic['best_rank'] = best_rank
            out_dic['rmsd_diff'] = diff_rmsd
            out_dic['top1_rmsd'] = topn_rmsd['top1_rmsd']
            out_dic['top3_rmsd'] = topn_rmsd['top3_rmsd']
            out_dic['top5_rmsd'] = topn_rmsd['top5_rmsd']
            out_dic['top10_rmsd'] = topn_rmsd['top10_rmsd']
            
        return loss, out_dic


class TierDPOLoss:
    """Tier-based DPO preference losses using pre-sampled pairs from pair_sampling.py.

    Score convention: **lower = better** (near-native decoys have lower scores).
    The loss pushes the policy model to widen the score gap between positive
    (near-native) and negative (non-native) decoys beyond the reference model's gap.

        loss = -log( sigma( beta * ( (neg_pi - pos_pi) - (neg_ref - pos_ref) ) ) )
                                       ^^^^^^^^^^^^       ^^^^^^^^^^^^
                                       policy margin      reference margin

    When ref_scores is None, falls back to reference-free logistic preference:
        loss = -log( sigma( beta * (neg_pi - pos_pi) ) )
    """

    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def _pairwise_dpo(self, pi_scores, ref_scores, pair_indices, device, higher_is_better=False):
        """Compute DPO loss for a list of (pos_idx, neg_idx) pairs.

        ``higher_is_better`` sets the score convention:
          - False (loop_rmsd, lower=better): correct margin = neg - pos.
          - True  (loop_lddt, higher=better): correct margin = pos - neg.
        Returns (loss, n_valid_pairs).
        """
        if not pair_indices:
            return torch.tensor(0.0, device=device), 0

        pos_idx = torch.tensor([p[0] for p in pair_indices], dtype=torch.long, device=device)
        neg_idx = torch.tensor([p[1] for p in pair_indices], dtype=torch.long, device=device)

        pi_pos = pi_scores[pos_idx]
        pi_neg = pi_scores[neg_idx]

        sign = -1.0 if higher_is_better else 1.0
        margin_pi = sign * (pi_neg - pi_pos)  # positive when model correctly ranks

        if ref_scores is not None:
            ref_pos = ref_scores[pos_idx].detach()
            ref_neg = ref_scores[neg_idx].detach()
            margin_ref = sign * (ref_neg - ref_pos)
            logits = margin_pi - margin_ref
        else:
            logits = margin_pi

        loss = -F.logsigmoid(self.beta * logits).mean()
        return loss, len(pair_indices)

    def dpo_eject_loss(self, pi_scores, ref_scores, pairs, device, higher_is_better=False):
        """Eject loss: separate near-native (X/A/B) from non-native (C/D)."""
        eject_pairs = [
            (p['positive_idx'], p['negative_idx'])
            for p in pairs if p['pair_type'] == 'eject'
        ]
        return self._pairwise_dpo(pi_scores, ref_scores, eject_pairs, device, higher_is_better)

    def dpo_top_loss(self, pi_scores, ref_scores, pairs, device, higher_is_better=False):
        """Top loss: ranking pressure within near-native pool (X/A vs B/C)."""
        top_pairs = [
            (p['positive_idx'], p['negative_idx'])
            for p in pairs if p['pair_type'] == 'top'
        ]
        return self._pairwise_dpo(pi_scores, ref_scores, top_pairs, device, higher_is_better)

    def dpo_within_loss(self, pi_scores, ref_scores, pairs, device, higher_is_better=False):
        """exp6: ranking pressure between two decoys in the SAME cdr_lddt tier.

        83.94 % of AF3 test pairs are within-tier, and the eject/top rule cannot
        produce a single one of them. This is also the regime the diagnosis found
        the model stuck at chance in (the |dcdr_lddt| 0.05-0.1 band, 49 % of
        informative pairs, 52 % accuracy), so it carries the whole intervention.
        """
        within_pairs = [
            (p['positive_idx'], p['negative_idx'])
            for p in pairs if p['pair_type'] == 'within'
        ]
        return self._pairwise_dpo(pi_scores, ref_scores, within_pairs, device, higher_is_better)

    def compactness_loss(self, pi_scores, xtal_indices, tier_a_indices, delta_a, device,
                         higher_is_better=False):
        """One-sided band loss: keep xtal-A score gap within [0, delta_A].

        Score convention: lower = better.
        For each A-tier decoy (score s_a) and mean xtal score s_x:
            gap_i  = s_a - s_x          (want 0 <= gap <= delta_A)
            L_i    = relu(-gap_i) + relu(gap_i - delta_A)
            loss   = mean(L_i)

        Ordering violation : gap < 0  (A scored better than xtal)
        Band violation     : gap > delta_A  (A too far from xtal)

        Returns (loss, metrics_dict).
        """
        if not xtal_indices or not tier_a_indices:
            zero = torch.tensor(0.0, device=device)
            return zero, {
                'n_pairs_used': 0,
                'x_to_a_gap_mean': 0.0,
                'x_to_a_gap_max': 0.0,
                'ordering_violation_rate': 0.0,
                'band_violation_rate': 0.0,
            }

        x_idx = torch.tensor(xtal_indices, dtype=torch.long, device=device)
        a_idx = torch.tensor(tier_a_indices, dtype=torch.long, device=device)

        s_x = pi_scores[x_idx].mean()   # scalar
        s_a = pi_scores[a_idx]          # [n_a]
        # Gap = how much worse A scores than xtal. lower=better -> worse = higher
        # score (s_a - s_x); higher=better -> worse = lower score (s_x - s_a).
        gap = (s_x - s_a) if higher_is_better else (s_a - s_x)  # want 0 <= gap <= delta_A

        loss_i = F.relu(-gap) + F.relu(gap - delta_a)
        loss = loss_i.mean()

        with torch.no_grad():
            gap_d = gap.detach()
            metrics = {
                'n_pairs_used': len(tier_a_indices),
                'x_to_a_gap_mean': gap_d.mean().item(),
                'x_to_a_gap_max': gap_d.max().item(),
                'ordering_violation_rate': (gap_d < 0).float().mean().item(),
                'band_violation_rate': (gap_d > delta_a).float().mean().item(),
            }
        return loss, metrics


class OrdinalH3LddtAuxLoss(nn.Module):
    """Auxiliary ordinal BCE + monotonic penalty on fixed H3 local lDDT cutoffs.

    Three logits predict P(lDDT >= cutoff) for cutoffs (p60, p80, p90).
    Monotonicity: p90 <= p80 <= p60.
    """

    def __init__(
        self,
        cutoff_mode: str = "fixed",
        cutoff_a: float = 0.6,
        cutoff_b: float = 0.8,
        cutoff_c: float = 0.9,
        rmsd_a: float = 2.0,
        rmsd_b: float = 1.5,
        rmsd_c: float = 0.8,
    ):
        super().__init__()
        self.cutoff_mode = cutoff_mode
        self.cutoff_60 = cutoff_a
        self.cutoff_80 = cutoff_b
        self.cutoff_90 = cutoff_c
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, ord_logits: torch.Tensor, q: torch.Tensor, loop_len=None):
        """
        Args:
            ord_logits: [B, 3] logits for (p60, p80, p90)
            q: [B] on-the-fly h3_local_lddt (NaN = masked out)

        Returns:
            loss_aux_ord, mono_penalty, n_valid (int)
        """
        del loop_len  # lDDT-only aux head; RMSD-derived cutoffs are not used.
        device = ord_logits.device
        valid = torch.isfinite(q)
        n_valid = int(valid.sum().item())
        zero = torch.tensor(0.0, device=device)

        if n_valid == 0:
            return zero, zero, 0

        logits = ord_logits[valid]
        q_v = q[valid]

        y60 = (q_v >= self.cutoff_60).float()
        y80 = (q_v >= self.cutoff_80).float()
        y90 = (q_v >= self.cutoff_90).float()

        loss60 = self.bce(logits[:, 0], y60)
        loss80 = self.bce(logits[:, 1], y80)
        loss90 = self.bce(logits[:, 2], y90)
        loss_aux_ord = (loss60 + loss80 + loss90).mean() / 3.0

        p60 = torch.sigmoid(logits[:, 0])
        p80 = torch.sigmoid(logits[:, 1])
        p90 = torch.sigmoid(logits[:, 2])
        mono_penalty = (F.relu(p90 - p80) + F.relu(p80 - p60)).mean()

        return loss_aux_ord, mono_penalty, n_valid