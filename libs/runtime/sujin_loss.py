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


def Ranking(rmsd_decoy,MLscore):
    rmsd_nr_idx = []
    rmsd = []
    for i in range(len(rmsd_decoy)):
        if rmsd_decoy[i] not in rmsd:
            rmsd.append(rmsd_decoy[i].tolist())
            rmsd_nr_idx.append(i)
    rmsd_decoy = torch.tensor(rmsd)
    MLscore = MLscore[rmsd_nr_idx]
    rmsd_ascend,idx_ans = torch.sort(rmsd_decoy,descending=False)
    best_idx = idx_ans[0].item()
    best_rmsd = rmsd_ascend[0].item()

    ml_ascend,idx_ml = torch.sort(MLscore.clone().detach(),descending=False)
    MLscore_ans = MLscore[best_idx]
    best_pred_rank = ml_ascend.tolist().index(MLscore_ans)+1
    
    # set same device with top10_indices
    device = idx_ml.device
    rmsd_decoy = rmsd_decoy.to(device)
    top10_indices = idx_ml[:10]
    top1_rmsd = rmsd_decoy[top10_indices[0]]
    top3_rmsd = torch.index_select(rmsd_decoy, 0, index=top10_indices[:3]).min()
    top5_rmsd = torch.index_select(rmsd_decoy, 0, index=top10_indices[:5]).min()
    top10_rmsd = torch.index_select(rmsd_decoy, 0, index=top10_indices[:10]).min()
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
    
    def forward(self,pdb,target,pred,isTraining,inference=False,rmsd_cutoff=2.0,withCEE=False):
        if inference:
            loss = 0
        else:
            # Default RMSD cutoff of near-native vs. non-native is 2.0
            mask_nonNative = target.ge(rmsd_cutoff) # RMSD greater than or equal to cutoff
            
            # if there is no near-native or no non-native decoys, log stats and return zero loss tensor
            if pred[~mask_nonNative].size(0)==0 or pred[mask_nonNative].size(0)==0:
                min_r = float(torch.min(target))
                max_r = float(torch.max(target))
                cnt = target.numel()
                present = 'non-native' if pred[~mask_nonNative].size(0)==0 else 'near-native'
                _logging.warning(
                    "%s: only %s decoys (count=%d) RMSD range [%.3f, %.3f]",
                    pdb, present, cnt, min_r, max_r
                )
                # return tensor so subsequent code stays consistent
                zero = torch.tensor(0.0, device=pred.device)
                return zero, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            epsilon = 1e-5
            # Average score of near-native decoys
            avg_nearNative = torch.mean(pred[~mask_nonNative])
            diff_near_non = -1*(pred[mask_nonNative] - avg_nearNative) # Original sign
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
            best_rank, diff_rmsd, topn_rmsd = Ranking(target,pred)
            top1_rmsd = topn_rmsd['top1_rmsd']
            top3_rmsd = topn_rmsd['top3_rmsd']
            top5_rmsd = topn_rmsd['top5_rmsd']
            top10_rmsd = topn_rmsd['top10_rmsd']
        
        return loss, best_rank, diff_rmsd, top1_rmsd, top3_rmsd, top5_rmsd, top10_rmsd


class FinalLoss(nn.Module):
    def __init__(self):
        super(FinalLoss, self).__init__()
        
    def forward(self,pdb,target,pred,isTraining,device,inference=False,rmsd_cutoff=2.0,loss_type='sml'):
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
            
        if loss_type=='sml':
            try:
                final_loss, best_rank, rmsd_diff, top1_rmsd, top3_rmsd, top5_rmsd, top10_rmsd = sml(pdb,target,pred,isTraining,inference,rmsd_cutoff)
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
            loss_sml, best_rank, rmsd_diff,top1_rmsd = sml(pdb,target,pred_sml,isTraining,inference,rmsd_cutoff,withCEE=True)
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
    
    def forward(self, pi, ref, pred, rmsd, pdb, device, is_training, return_components=False):
        out_dic = {}
        loss_sml, best_rank, rmsd_diff, top1_rmsd, top3_rmsd, top5_rmsd, top10_rmsd = self.sml_loss(pdb, rmsd, pred, is_training)
        # Phase/tier path only uses SML; skip legacy near-vs-non DPO (crashes when all RMSD < 2Å).
        if return_components:
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

    def _pairwise_dpo(self, pi_scores, ref_scores, pair_indices, device):
        """Compute DPO loss for a list of (pos_idx, neg_idx) pairs.

        Returns (loss, n_valid_pairs).
        """
        if not pair_indices:
            return torch.tensor(0.0, device=device), 0

        pos_idx = torch.tensor([p[0] for p in pair_indices], dtype=torch.long, device=device)
        neg_idx = torch.tensor([p[1] for p in pair_indices], dtype=torch.long, device=device)

        pi_pos = pi_scores[pos_idx]
        pi_neg = pi_scores[neg_idx]

        margin_pi = pi_neg - pi_pos  # positive when model correctly ranks

        if ref_scores is not None:
            ref_pos = ref_scores[pos_idx].detach()
            ref_neg = ref_scores[neg_idx].detach()
            margin_ref = ref_neg - ref_pos
            logits = margin_pi - margin_ref
        else:
            logits = margin_pi

        loss = -F.logsigmoid(self.beta * logits).mean()
        return loss, len(pair_indices)

    def dpo_eject_loss(self, pi_scores, ref_scores, pairs, device):
        """Eject loss: push near-native (X/A/B) scores below non-native (C/D)."""
        eject_pairs = [
            (p['positive_idx'], p['negative_idx'])
            for p in pairs if p['pair_type'] == 'eject'
        ]
        return self._pairwise_dpo(pi_scores, ref_scores, eject_pairs, device)

    def dpo_top_loss(self, pi_scores, ref_scores, pairs, device):
        """Top loss: ranking pressure within near-native pool (X/A vs B/C)."""
        top_pairs = [
            (p['positive_idx'], p['negative_idx'])
            for p in pairs if p['pair_type'] == 'top'
        ]
        return self._pairwise_dpo(pi_scores, ref_scores, top_pairs, device)

    def compactness_loss(self, pi_scores, xtal_indices, tier_a_indices, delta_a, device):
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
        gap = s_a - s_x                 # want 0 <= gap <= delta_A

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