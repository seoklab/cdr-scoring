import math
import random
import statistics

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ListNetInputError(ValueError):
    pass


def pad_groups(groups):
    width = max(int(g.numel()) for g in groups)
    out = groups[0].new_zeros((len(groups), width))
    mask = torch.zeros((len(groups), width), dtype=torch.bool, device=groups[0].device)
    for row, g in enumerate(groups):
        k = int(g.numel())
        out[row, :k] = g.reshape(-1)
        mask[row, :k] = True
    return out, mask


TEACHERS = ('fixed', 'adaptive_std')


def population_sd(labels, mask=None):
    q = labels.detach().double()
    if q.dim() == 1:
        q = q.unsqueeze(0)
    m = torch.ones_like(q, dtype=torch.bool) if mask is None else mask.bool().to(q.device).reshape(q.shape)
    n = m.sum(dim=1)
    mean = q.masked_fill(~m, 0.0).sum(dim=1) / n.clamp(min=1)
    var = (q - mean.unsqueeze(1)).masked_fill(~m, 0.0).pow(2).sum(dim=1) / n.clamp(min=1)
    return torch.where(n > 0, var.sqrt(), torch.full_like(var, float('nan')))


def adaptive_tau(labels, mask=None, floor=0.005):
    if not float(floor) > 0:
        raise ValueError(f'adaptive tau_q floor must be > 0 (got {floor})')
    sd = population_sd(labels, mask)
    return sd.clamp(min=float(floor)), sd


def listnet_loss(scores, labels, mask=None, tau_q=0.01, tau_s=1.0, names=None, label_range=None):
    tq_in = torch.as_tensor(tau_q, dtype=torch.float64).detach().reshape(-1)
    if not (bool(torch.isfinite(tq_in).all()) and bool((tq_in > 0).all()) and tau_s > 0):
        raise ValueError(f'listnet temperatures must be > 0 (tau_q={tau_q}, tau_s={tau_s})')
    with torch.cuda.amp.autocast(enabled=False):
        s = scores.float()
        q = labels.detach().float().to(s.device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        if s.dim() != 2 or s.shape != q.shape:
            raise ListNetInputError(
                f'score/label shape mismatch: scores {tuple(scores.shape)} vs labels {tuple(labels.shape)}')
        if tq_in.numel() not in (1, s.shape[0]):
            raise ListNetInputError(f'{tq_in.numel()} tau_q values for {s.shape[0]} target(s)')
        tq = tq_in.to(device=s.device, dtype=torch.float32).expand(s.shape[0]).unsqueeze(1)
        if mask is None:
            mask = torch.ones_like(s, dtype=torch.bool)
        else:
            mask = mask.bool().to(s.device)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if mask.shape != s.shape:
                raise ListNetInputError(
                    f'mask shape {tuple(mask.shape)} does not match scores {tuple(s.shape)}')

        ok = torch.isfinite(s) & torch.isfinite(q)
        if label_range is not None:
            ok = ok & (q >= float(label_range[0])) & (q <= float(label_range[1]))
        bad = mask & ~ok
        if bool(bad.any()):
            t, i = (int(v) for v in bad.nonzero()[0])
            name = names[t] if names is not None else t
            raise ListNetInputError(
                f'{int(bad.sum())} invalid value(s) among valid candidates; first: target={name} '
                f'candidate={i} score={float(s[t, i])} label={float(q[t, i])} '
                f'(label_range={label_range})')

        fill = ~mask
        n_valid = mask.sum(dim=1)
        use = n_valid >= 2
        # Rows that are all padding keep finite logits so their softmax is not NaN.
        neg = fill & (n_valid > 0).unsqueeze(1)
        q_logit = (q.masked_fill(fill, 0.0) / tq).masked_fill(neg, float('-inf'))
        s_logit = (s.masked_fill(fill, 0.0) / float(tau_s)).masked_fill(neg, float('-inf'))
        r = F.softmax(q_logit, dim=1).masked_fill(fill, 0.0).detach()
        log_r = F.log_softmax(q_logit, dim=1).masked_fill(fill, 0.0).detach()
        log_p = F.log_softmax(s_logit, dim=1).masked_fill(fill, 0.0)
        zero = s.new_zeros(())
        per_target = torch.where(use, -(r * log_p).sum(dim=1), zero)
        entropy = torch.where(use, -(r * log_r).sum(dim=1), zero)
    return {
        'loss_sum': per_target.sum(),
        'n_targets': int(use.sum()),
        'per_target': per_target,
        'use': use,
        'entropy': entropy,
        'r': r,
        'log_p': log_p,
    }


def target_mean_scale(n_valid_global, world_size=1):
    if n_valid_global <= 0:
        raise ValueError(f'no valid ListNet target in this step (n={n_valid_global})')
    return float(world_size) / float(n_valid_global)


def sample_group(indices, cap, key):
    indices = list(indices)
    if int(cap) < 2:
        raise ValueError(f'listnet candidate cap must be >= 2 (got {cap})')
    if len(indices) <= int(cap):
        return indices
    return random.Random(str(key)).sample(indices, int(cap))


def group_key(spec_seed, run_seed, epoch, target):
    return f'listnet|{spec_seed}|{run_seed}|{epoch}|{target}'


def epoch_order(targets, seed, epoch):
    order = list(targets)
    random.Random(f'listnet-order|{seed}|{epoch}').shuffle(order)
    return order


class GroupScan(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            return self.dataset[idx]
        except FileNotFoundError as e:
            return {'target': str(self.dataset.inp_dat[idx]), 'sources': {}, 'error': str(e)}


def scan_collate(sample):
    return sample


def summarize_scan(rows, cap):
    per_target, excluded = {}, {}
    n_total = n_nonfinite = n_valid_all = 0
    for row in rows:
        tgt = str(row['target'])
        src = {k: [int(x) for x in v] for k, v in (row.get('sources') or {}).items()}
        valid = sum(v[2] for v in src.values())
        n_total += sum(v[0] for v in src.values())
        n_nonfinite += sum(v[1] for v in src.values())
        n_valid_all += valid
        pooled = [float(x) for v in (row.get('q') or {}).values() for x in v]
        q_sd = float(population_sd(torch.tensor(pooled, dtype=torch.float64))[0]) if pooled else float('nan')
        n_fnat = sum(int(v) for v in (row.get('n_fnat_finite') or {}).values())
        per_target[tgt] = {'n_valid': valid, 'sources': src, 'q_sd': q_sd, 'n_fnat_finite': n_fnat,
                           'holo': n_fnat > 0}
        if valid < 2:
            excluded[tgt] = {'n_valid': valid, 'reason': row.get('error') or (
                'no candidate found' if not src else 'fewer than 2 valid candidates')}
    kept = [t for t in per_target if t not in excluded]
    sizes = [per_target[t]['n_valid'] for t in kept]
    summary = {
        'n_targets_in_split': len(per_target),
        'n_targets_valid': len(kept),
        'n_targets_excluded_lt2': len(excluded),
        'n_targets_without_any_candidate': sum(1 for t in excluded if not per_target[t]['sources']),
        'n_candidates_found': n_total,
        'n_candidates_dropped_nonfinite_label': n_nonfinite,
        'n_candidates_dropped_by_existing_filters': n_total - n_nonfinite - n_valid_all,
        'n_candidates_valid': n_valid_all,
        'n_candidates_valid_in_kept_targets': sum(sizes),
        'candidates_per_target_min': min(sizes) if sizes else 0,
        'candidates_per_target_median': float(statistics.median(sizes)) if sizes else 0.0,
        'candidates_per_target_mean': (sum(sizes) / len(sizes)) if sizes else 0.0,
        'candidates_per_target_max': max(sizes) if sizes else 0,
        'max_candidates_per_group': int(cap),
        'n_targets_above_cap': sum(1 for n in sizes if n > int(cap)),
        'n_targets_valid_holo': sum(1 for t in kept if per_target[t]['holo']),
        'n_targets_valid_apo': sum(1 for t in kept if not per_target[t]['holo']),
    }
    return summary, kept, excluded, per_target


TRAIN_SUBSETS = ('all', 'holo', 'apo')


def subset_groups(kept, per_target, subset):
    if subset not in TRAIN_SUBSETS:
        raise ValueError(f'unknown listnet train subset {subset!r} (expected one of {TRAIN_SUBSETS})')
    if subset == 'all':
        return list(kept)
    # holo = any valid candidate with a finite fnat, the rule the validation holo/apo split uses.
    mixed = [t for t in kept if 0 < per_target[t]['n_fnat_finite'] < per_target[t]['n_valid']]
    if mixed:
        raise ValueError(f'{len(mixed)} train group(s) mix candidates with and without fnat, so a '
                         f'holo/apo split is not clean; first: {mixed[:5]}')
    return [t for t in kept if per_target[t]['holo'] == (subset == 'holo')]


def teacher_tau(mode, tau_q=0.01, floor=0.005, q_sd=None):
    if mode == 'fixed':
        return float(tau_q)
    if mode == 'adaptive_std':
        if q_sd is None or not math.isfinite(float(q_sd)):
            raise ValueError(f'adaptive_std teacher needs a finite target q SD (got {q_sd})')
        return max(float(q_sd), float(floor))
    raise ValueError(f'unknown listnet teacher {mode!r} (expected one of {TEACHERS})')


def spread(values):
    v = [float(x) for x in values if math.isfinite(float(x))]
    if not v:
        return {'min': float('nan'), 'median': float('nan'), 'max': float('nan')}
    return {'min': min(v), 'median': float(statistics.median(v)), 'max': max(v)}


def val_teacher_stats(groups_q, groups_s, holo, mode, tau_q=0.01, floor=0.005, tau_s=1.0, names=None):
    q, mask = pad_groups([torch.as_tensor(g, dtype=torch.float64) for g in groups_q])
    s, _ = pad_groups([torch.as_tensor(g, dtype=torch.float64) for g in groups_s])
    if mode == 'adaptive_std':
        tau, sd = adaptive_tau(q, mask, floor)
    else:
        teacher_tau(mode, tau_q, floor)
        sd = population_sd(q, mask)
        tau = torch.full_like(sd, float(tau_q))
    tau = torch.where(torch.isfinite(tau), tau, torch.ones_like(tau))
    out = listnet_loss(s, q, mask, tau_q=tau, tau_s=tau_s, names=names, label_range=(0.0, 1.0))
    use = out['use']
    kl = (out['per_target'] - out['entropy']).double()
    hm = torch.as_tensor(holo, dtype=torch.bool)

    def mean(x, sel):
        return float(x[sel].mean()) if bool(sel.any()) else float('nan')
    t = spread(tau[use].tolist())
    return {
        'listnet_kl': mean(kl, use), 'listnet_ce': mean(out['per_target'].double(), use),
        'listnet_entropy': mean(out['entropy'].double(), use),
        'holo/listnet_kl': mean(kl, use & hm), 'apo/listnet_kl': mean(kl, use & ~hm),
        'n_kl_targets': float(use.sum()), 'n_kl_holo': float((use & hm).sum()),
        'n_kl_apo': float((use & ~hm).sum()),
        'tau_q_min': t['min'], 'tau_q_median': t['median'], 'tau_q_max': t['max'],
        'n_tau_q_at_floor': float((use & (sd <= float(floor))).sum()) if mode == 'adaptive_std' else 0.0,
    }
