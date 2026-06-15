from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class PairSamplingConfig:
    total_pair_budget: int = 16
    use_lddt_tiers: bool = False
    xtal_rmsd_thr: float = 0.01
    tier_a_rmsd: float = 0.8
    tier_b_rmsd: float = 1.5
    tier_c_rmsd: float = 2.0


def detect_xtal_mask(rmsds, xtal_rmsd_thr: float = 0.01):
    return torch.as_tensor(rmsds).detach() < float(xtal_rmsd_thr)


def build_decoy_tiers(rmsds, is_xtal=None, cfg: Optional[PairSamplingConfig] = None):
    cfg = cfg or PairSamplingConfig()
    rmsds = torch.as_tensor(rmsds).detach().float().cpu()
    if is_xtal is None:
        is_xtal = rmsds < cfg.xtal_rmsd_thr
    else:
        is_xtal = torch.as_tensor(is_xtal).detach().bool().cpu()

    tiers: Dict[str, List[int]] = {key: [] for key in ("X", "A", "B", "C", "D")}
    for i, value in enumerate(rmsds.tolist()):
        if bool(is_xtal[i]):
            tiers["X"].append(i)
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
):
    del scores, structure_id, phase, total_non_xtal_pool_size, h3_lddt
    cfg = cfg or PairSamplingConfig()
    tiers = build_decoy_tiers(rmsds, is_xtal, cfg)

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
