from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


# ──────────────────────────────────────────────────────────────────────────
# AF3 test pair distribution over cdr_lddt tiers  (exp6 "gen-gen DPO")
#
# Measured on the 185 post-2021 AF3 test targets, 915,750 same-target pairs,
# with the tier boundaries below (analyze/af3_lddt_tier_pair_distribution.py).
# Percentages of all unordered same-target tier pairs.
#
# Two facts drive the exp6 sampler:
#   1. **83.94 % of AF3 pairs are WITHIN a tier** (D-D 29.23, A-A 21.41,
#      B-B 17.15, C-C 16.15).  The H3-DPO eject/top rule only ever builds
#      CROSS-tier pairs, so it covers just 13.91 % of the test distribution --
#      it trains the easy, well-separated comparisons and skips the regime the
#      model actually fails in.
#   2. **AF3 has no X tier at all** (0.00 %) while the generative train pool is
#      12.14 % X-something.  Crystal-quality decoys do not exist at test time,
#      so X is dropped rather than used as the positive anchor.
#
# Unlike exp3 (fnat tiers on the GATED multi-source pool, where 47.3 % of AF3
# mass sat in cells the pool lacked), every cell here is reachable: AF3 mass in
# cells with <1,000 available train pairs is 0.00 %.
# ──────────────────────────────────────────────────────────────────────────
LDDT_TIER_NAMES = ('X', 'A', 'B', 'C', 'D')
AF3_LDDT_TIER_PAIR_P: Dict[str, float] = {
    'D-D': 0.2923, 'A-A': 0.2141, 'B-B': 0.1715, 'C-C': 0.1615,
    'B-C': 0.0425, 'A-B': 0.0386, 'C-D': 0.0351, 'B-D': 0.0167,
    'A-D': 0.0140, 'A-C': 0.0137,
    # X-* is 0.00 % in AF3 and is therefore absent here on purpose.
}
# cross-tier cells keep the H3-DPO semantics so the existing eject/top losses and
# their lambdas still apply; same-tier cells are the new 'within' type.
_EJECT_CELLS = {'A-C', 'A-D', 'B-C', 'B-D'}      # near (A/B) vs far (C/D)
_TOP_CELLS = {'A-B', 'C-D'}                      # adjacent-tier ranking pressure


def af3_cell_name(t_i: str, t_j: str) -> str:
    """Canonical (order-independent) tier-pair cell key, e.g. ('C','A') -> 'A-C'."""
    order = {t: i for i, t in enumerate(LDDT_TIER_NAMES)}
    a, b = sorted((t_i, t_j), key=lambda t: order[t])
    return f'{a}-{b}'


def af3_cell_pair_type(cell: str) -> str:
    if cell in _EJECT_CELLS:
        return 'eject'
    if cell in _TOP_CELLS:
        return 'top'
    return 'within'


@dataclass
class PairSamplingConfig:
    total_pair_budget: int = 16
    use_lddt_tiers: bool = False
    xtal_rmsd_thr: float = 0.01
    tier_a_rmsd: float = 0.8
    tier_b_rmsd: float = 1.5
    tier_c_rmsd: float = 2.0
    # loop_lddt (higher-is-better) tier boundaries, mirroring the live sampler.
    xtal_lddt_thr: float = 0.99
    tier_a_lddt: float = 0.90
    tier_b_lddt: float = 0.80
    tier_c_lddt: float = 0.70


def detect_xtal_mask(rmsds, xtal_rmsd_thr: float = 0.01, higher_is_better: bool = False,
                     xtal_lddt_thr: float = 0.99):
    vals = torch.as_tensor(rmsds).detach()
    if higher_is_better:
        return vals >= float(xtal_lddt_thr)
    return vals < float(xtal_rmsd_thr)


def build_decoy_tiers(rmsds, is_xtal=None, cfg: Optional[PairSamplingConfig] = None,
                      higher_is_better: bool = False):
    cfg = cfg or PairSamplingConfig()
    rmsds = torch.as_tensor(rmsds).detach().float().cpu()
    if is_xtal is None:
        is_xtal = detect_xtal_mask(rmsds, cfg.xtal_rmsd_thr, higher_is_better, cfg.xtal_lddt_thr)
    else:
        is_xtal = torch.as_tensor(is_xtal).detach().bool().cpu()

    tiers: Dict[str, List[int]] = {key: [] for key in ("X", "A", "B", "C", "D")}
    for i, value in enumerate(rmsds.tolist()):
        if bool(is_xtal[i]):
            tiers["X"].append(i)
        elif higher_is_better:
            # lDDT: A>=0.90, B>=0.80, C>=0.70, D<0.70 (higher = better)
            if value >= cfg.tier_a_lddt:
                tiers["A"].append(i)
            elif value >= cfg.tier_b_lddt:
                tiers["B"].append(i)
            elif value >= cfg.tier_c_lddt:
                tiers["C"].append(i)
            else:
                tiers["D"].append(i)
        elif value <= cfg.tier_a_rmsd:
            tiers["A"].append(i)
        elif value <= cfg.tier_b_rmsd:
            tiers["B"].append(i)
        elif value <= cfg.tier_c_rmsd:
            tiers["C"].append(i)
        else:
            tiers["D"].append(i)
    return tiers


def _sample_pairs(positives, negatives, n_pairs, rng, pair_type, subtype):
    if not positives or not negatives or n_pairs <= 0:
        return []
    pairs = []
    for _ in range(int(n_pairs)):
        pairs.append(
            {
                "positive_idx": int(rng.choice(positives)),
                "negative_idx": int(rng.choice(negatives)),
                "pair_type": pair_type,
                "subtype": subtype,
            }
        )
    return pairs


def build_training_pairs(
    *,
    scores,
    rmsds,
    is_xtal,
    structure_id,
    phase,
    cfg: Optional[PairSamplingConfig],
    rng,
    total_non_xtal_pool_size=None,
    h3_lddt=None,
    higher_is_better=False,
):
    del scores, structure_id, phase, total_non_xtal_pool_size, h3_lddt
    cfg = cfg or PairSamplingConfig()
    tiers = build_decoy_tiers(rmsds, is_xtal, cfg, higher_is_better=higher_is_better)

    n_eject = cfg.total_pair_budget // 2
    n_top = cfg.total_pair_budget - n_eject

    near = tiers["X"] + tiers["A"] + tiers["B"]
    far = tiers["C"] + tiers["D"]
    top_pos = tiers["X"] + tiers["A"]
    top_neg = tiers["B"] + tiers["C"]

    pairs = []
    pairs.extend(_sample_pairs(near, far, n_eject, rng, "eject", "near_vs_far"))
    pairs.extend(_sample_pairs(top_pos, top_neg, n_top, rng, "top", "top_vs_mid"))

    eject_count = sum(1 for pair in pairs if pair["pair_type"] == "eject")
    top_count = sum(1 for pair in pairs if pair["pair_type"] == "top")
    summary = {
        "tiers": tiers,
        "tier_counts": {key: len(value) for key, value in tiers.items()},
        "target_budget": {"eject": n_eject, "top": n_top},
        "actual_pairs": {"eject": eject_count, "top": top_count},
        "eject_info": {"subtype_counts": {"near_vs_far": eject_count}},
        "top_info": {"subtype_counts": {"top_vs_mid": top_count}},
    }
    return pairs, summary


def build_af3_matched_pairs(
    *,
    rmsds,
    allowed_mask=None,
    cfg: Optional[PairSamplingConfig] = None,
    rng,
    higher_is_better: bool = True,
    min_delta: float = 0.02,
    cell_p: Optional[Dict[str, float]] = None,
    is_xtal=None,
):
    """exp6: same-target DPO pairs whose tier-cell mix matches the AF3 test set.

    Differs from :func:`build_training_pairs` in three ways:

    * **source restriction** -- ``allowed_mask`` (bool per decoy) keeps only
      generative decoys, so every pair is generative-vs-generative, the only
      comparison that resembles the deployment task.
    * **within-tier pairs** -- 83.94 % of AF3 pairs sit inside one tier; those are
      emitted with ``pair_type='within'``. The eject/top rule cannot make them.
    * **X tier dropped** -- AF3 contains no crystal-quality decoy, so X decoys are
      excluded from the pair pool entirely rather than serving as the anchor.

    The budget is split across cells in proportion to ``cell_p``, renormalised over
    the cells this particular batch can actually fill, so an absent cell donates
    its share to the others instead of silently shrinking the budget.

    ``min_delta`` is the label gap a pair must clear to be worth training on. It
    matters most for within-tier pairs, which are exactly the small-gap regime the
    ranking currently fails in; setting it to 0 admits ties.

    Returns ``(pairs, summary)`` with the same shapes the DPO loop already expects.
    """
    cfg = cfg or PairSamplingConfig()
    cell_p = dict(cell_p or AF3_LDDT_TIER_PAIR_P)
    vals = torch.as_tensor(rmsds).detach().float().cpu()
    n = vals.numel()

    tiers = build_decoy_tiers(vals, is_xtal, cfg, higher_is_better=higher_is_better)
    tier_of = {}
    for name, idxs in tiers.items():
        for i in idxs:
            tier_of[i] = name

    if allowed_mask is None:
        allowed = set(range(n))
    else:
        m = torch.as_tensor(allowed_mask).detach().bool().cpu()
        allowed = {i for i in range(n) if bool(m[i])}
    # AF3 has no X tier -> never pair against a crystal-quality decoy
    pool = sorted(i for i in allowed if tier_of.get(i, 'D') != 'X')

    v = vals.tolist()
    by_cell: Dict[str, List] = {}
    for a in range(len(pool)):
        for b in range(a + 1, len(pool)):
            i, j = pool[a], pool[b]
            d = v[i] - v[j]
            if abs(d) < min_delta:
                continue
            cell = af3_cell_name(tier_of.get(i, 'D'), tier_of.get(j, 'D'))
            if cell not in cell_p:
                continue
            # positive = better decoy under the metric direction
            pos, neg = (i, j) if (d > 0) == bool(higher_is_better) else (j, i)
            by_cell.setdefault(cell, []).append((pos, neg))

    budget = int(cfg.total_pair_budget)
    present = {c: p for c, p in cell_p.items() if by_cell.get(c)}
    pairs: List[Dict] = []
    cell_counts: Dict[str, int] = {}
    if present and budget > 0:
        tot = sum(present.values())
        # largest-remainder allocation so the budget is hit exactly
        raw = {c: budget * p / tot for c, p in present.items()}
        alloc = {c: int(x) for c, x in raw.items()}
        left = budget - sum(alloc.values())
        for c, _ in sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True):
            if left <= 0:
                break
            alloc[c] += 1
            left -= 1
        for cell, k in alloc.items():
            cand = by_cell[cell]
            if k <= 0 or not cand:
                continue
            picks = [cand[int(rng.integers(len(cand)))] for _ in range(k)] \
                if hasattr(rng, 'integers') else [cand[int(rng.choice(len(cand)))] for _ in range(k)]
            ptype = af3_cell_pair_type(cell)
            cell_counts[cell] = cell_counts.get(cell, 0) + len(picks)
            for pos, neg in picks:
                pairs.append({'positive_idx': int(pos), 'negative_idx': int(neg),
                              'pair_type': ptype, 'subtype': cell})

    counts = {t: sum(1 for p in pairs if p['pair_type'] == t)
              for t in ('eject', 'top', 'within')}
    summary = {
        'tiers': tiers,
        'tier_counts': {k: len(v_) for k, v_ in tiers.items()},
        'target_budget': {'eject': counts['eject'], 'top': counts['top']},
        'actual_pairs': counts,
        'eject_info': {'subtype_counts': {c: n_ for c, n_ in cell_counts.items()
                                          if af3_cell_pair_type(c) == 'eject'}},
        'top_info': {'subtype_counts': {c: n_ for c, n_ in cell_counts.items()
                                        if af3_cell_pair_type(c) == 'top'}},
        'af3_cell_counts': cell_counts,
        'n_allowed': len(allowed),
        'n_pool': len(pool),
        'n_candidate_pairs': sum(len(x) for x in by_cell.values()),
        'n_cells_present': len(present),
    }
    return pairs, summary


def build_top_region_pairs(
    *,
    rmsds,
    allowed_mask=None,
    cfg: Optional[PairSamplingConfig] = None,
    rng,
    higher_is_better: bool = True,
    region_cut: float = 0.85,
    band_lo: float = 0.90,
    min_delta: float = 0.05,
    min_delta_relaxed: float = 0.03,
    band_quota: float = 0.0,
    is_xtal=None,
):
    """exp9: DPO pairs drawn ONLY from the top region (both decoys >= region_cut).

    exp8's dead-band diagnostic showed the SML objective finishes its job in two
    epochs -- it separates "<=0.85" from ">=0.90" and then nothing inside the top
    changes for 35 more epochs (band spearman wanders in [-0.14, +0.32] with no
    trend). Test-time top-1 is decided almost entirely inside that frozen region,
    so this sampler puts every pair there and nowhere else.

    Direction: positive = the decoy CLOSER to the native basin (higher cdr_lddt).
    That is DPO-top in the sense that matters -- preference toward native, not a
    statement about how far apart the two decoys are.

    ``band_quota`` (0..1) reserves that share of the budget for BOUNDARY pairs,
    one decoy in [region_cut, band_lo) and the other >= band_lo. At 0.0 boundary
    pairs still appear, but only at whatever rate they occur naturally; the two
    exp9 variants differ in exactly this number.

    ``min_delta_relaxed`` is the fallback threshold for targets that cannot fill
    their budget at ``min_delta`` -- 81 % of train targets have a >=0.05 pair but
    92 % have a >=0.03 one, so relaxing rather than skipping keeps ~11 % more
    targets contributing.

    Xtal decoys are dropped: a crystal structure is not a candidate at test time,
    and pairing against it teaches a comparison that never occurs.

    Returns ``(pairs, summary)`` in the shape the DPO loop already expects, with
    every pair typed ``'within'`` so ``TierDPOLoss.dpo_within_loss`` consumes them.
    """
    cfg = cfg or PairSamplingConfig()
    vals = torch.as_tensor(rmsds).detach().float().cpu()
    n = vals.numel()
    tiers = build_decoy_tiers(vals, is_xtal, cfg, higher_is_better=higher_is_better)
    xtal = set(tiers.get('X', []))

    if allowed_mask is None:
        allowed = set(range(n))
    else:
        m = torch.as_tensor(allowed_mask).detach().bool().cpu()
        allowed = {i for i in range(n) if bool(m[i])}
    v = vals.tolist()
    # top region only; xtal excluded
    pool = sorted(i for i in allowed
                  if i not in xtal and v[i] == v[i] and v[i] >= float(region_cut))

    def _collect(thr):
        bnd, ins = [], []
        for a in range(len(pool)):
            for b in range(a + 1, len(pool)):
                i, j = pool[a], pool[b]
                d = v[i] - v[j]
                if abs(d) < thr:
                    continue
                pos, neg = (i, j) if (d > 0) == bool(higher_is_better) else (j, i)
                # boundary = straddles band_lo
                if (v[i] < band_lo) != (v[j] < band_lo):
                    bnd.append((pos, neg))
                else:
                    ins.append((pos, neg))
        return bnd, ins

    used_delta = float(min_delta)
    bnd, ins = _collect(used_delta)
    budget = int(cfg.total_pair_budget)
    if (len(bnd) + len(ins)) < budget and float(min_delta_relaxed) < used_delta:
        used_delta = float(min_delta_relaxed)
        bnd, ins = _collect(used_delta)

    def _pick(cand, k):
        if k <= 0 or not cand:
            return []
        if hasattr(rng, 'integers'):
            return [cand[int(rng.integers(len(cand)))] for _ in range(k)]
        return [cand[int(rng.choice(len(cand)))] for _ in range(k)]

    if float(band_quota) <= 0.0:
        # exp9-a: no boundary/inside distinction -- one pool, so boundary pairs
        # appear at whatever rate the target's own decoys produce.
        picks = _pick(bnd + ins, budget)
        n_bnd = 0          # subtype labelling below is only meaningful with a quota
    else:
        n_bnd_want = int(round(budget * float(band_quota)))
        picks = _pick(bnd, min(n_bnd_want, budget))
        n_bnd = len(picks)
        # remaining budget from the rest; if one bucket is empty the other absorbs it
        rest = ins if ins else bnd
        picks += _pick(rest, budget - n_bnd)

    pairs = [{'positive_idx': int(p), 'negative_idx': int(q),
              'pair_type': 'within',
              'subtype': 'boundary' if k < n_bnd else 'inside'}
             for k, (p, q) in enumerate(picks)]
    counts = {'eject': 0, 'top': 0, 'within': len(pairs)}
    summary = {
        'tiers': tiers,
        'tier_counts': {k: len(v_) for k, v_ in tiers.items()},
        'target_budget': {'eject': 0, 'top': 0},
        'actual_pairs': counts,
        'eject_info': {'subtype_counts': {}},
        'top_info': {'subtype_counts': {}},
        'af3_cell_counts': {},
        'n_allowed': len(allowed),
        'n_pool': len(pool),
        'n_candidate_pairs': len(bnd) + len(ins),
        'n_cells_present': 1 if pairs else 0,
        # exp9 diagnostics
        'top_used_delta': used_delta,
        'top_n_boundary_cand': len(bnd),
        'top_n_inside_cand': len(ins),
        'top_n_boundary_used': n_bnd,
    }
    return pairs, summary
