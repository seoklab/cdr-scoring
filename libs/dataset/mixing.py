"""
dataset/mixing.py
=================
Within-target decoy mixing policy.

Pipeline (for a **single** target):
  Step 0 – candidate generation  (done by SourceRegistry)
  Step 1 – min-diversity fill    (1 decoy per source, iff ≥ threshold sources)
  Step 2 – weighted fill         (multinomial over source weights)
  Step 3 – per-source cap + max_fraction enforcement
"""
from __future__ import annotations

import math
import random as _random
from typing import Dict, List, Optional, Tuple

from .config import DatasetSpec, SourceSpec


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def sample_mix(
    per_source_counts: Dict[str, int],
    epoch: int,
    spec: DatasetSpec,
    rng: _random.Random | None = None,
) -> Dict[str, int]:
    """Decide how many decoys to draw from each source.

    Parameters
    ----------
    per_source_counts : {source_name: available_count}
        Number of decoys available *after* quality-filtering (Boltz2 cutoff,
        xtal gate, etc.).  Sources with 0 available are excluded.
    epoch : current epoch (used to resolve schedulable weights).
    spec  : DatasetSpec loaded from YAML.
    rng   : optional seeded Random instance for reproducibility.

    Returns
    -------
    {source_name: n_selected}   sum == min(spec.n_decoy, total_available)
    """
    if rng is None:
        rng = _random.Random()

    n_total = spec.n_decoy
    sources = spec.enabled_sources()

    # keep only sources that are actually available (count > 0)
    avail: Dict[str, int] = {
        name: cnt for name, cnt in per_source_counts.items()
        if cnt > 0 and name in sources
    }
    if not avail:
        return {}

    # ── Step 1: min diversity fill ──
    selected: Dict[str, int] = {name: 0 for name in avail}
    filled = 0
    if len(avail) >= spec.min_diversity_threshold:
        for name in avail:
            if avail[name] >= 1 and filled < n_total:
                selected[name] = 1
                filled += 1

    # ── Step 2: weighted fill (remaining slots) ──
    remaining = n_total - filled
    if remaining > 0:
        # Build weight vector for sources that still have capacity
        w_names: List[str] = []
        w_vals: List[float] = []
        w_caps: List[int] = []   # how many more can we draw from each

        # When only a single source is available, bypass max_fraction cap
        # so it can fill up to n_decoy (otherwise single-source targets
        # are unnecessarily under-sampled, causing missing near/non-native).
        single_source = len(avail) == 1

        for name in avail:
            src = sources[name]
            if single_source:
                cap_left = min(
                    avail[name] - selected[name],
                    src.per_target_cap - selected[name],
                )
            else:
                cap_left = min(
                    avail[name] - selected[name],
                    src.per_target_cap - selected[name],
                    _max_fraction_cap(n_total, src.max_fraction) - selected[name],
                )
            if cap_left <= 0:
                continue
            w = src.effective_weight(epoch)
            if w <= 0:
                continue
            w_names.append(name)
            w_vals.append(w)
            w_caps.append(cap_left)

        if w_names:
            selected = _weighted_fill(
                selected, remaining, w_names, w_vals, w_caps, rng
            )

    # Crystal (xtal) inclusion is probability-gated and annealed over training via
    # spec.xtal_gate_prob (a ScheduleSpec resolved at this epoch). Early epochs keep
    # the native crystal as a positive anchor (prob ~1.0); late epochs drop it (prob
    # ~0.0) so the model must discriminate among model-generated decoys.
    xtal_key = next((k for k in avail if k.lower() == "xtal"), None)
    if xtal_key is not None:
        xtal_prob = spec.effective_xtal_prob(epoch)
        include_xtal = rng.random() < xtal_prob
        if include_xtal:
            # Ensure exactly one crystal decoy is present (donate a slot if needed).
            if selected.get(xtal_key, 0) == 0 and sum(selected.values()) >= 1:
                donor = next(
                    (k for k, v in selected.items() if k.lower() != "xtal" and v >= 1),
                    None,
                )
                if donor is not None:
                    selected[donor] = selected[donor] - 1
                    selected[xtal_key] = 1
        else:
            # Drop any crystal picked by the diversity/weighted fill and hand its
            # slot(s) back to the largest-capacity non-xtal source.
            freed = selected.get(xtal_key, 0)
            if freed > 0:
                selected[xtal_key] = 0
                donor = max(
                    (k for k in selected if k.lower() != "xtal"),
                    key=lambda k: avail.get(k, 0) - selected.get(k, 0),
                    default=None,
                )
                if donor is not None:
                    add = min(freed, avail.get(donor, 0) - selected.get(donor, 0))
                    if add > 0:
                        selected[donor] = selected.get(donor, 0) + add

    # ensure total does not exceed available
    total_avail = sum(avail.values())
    if sum(selected.values()) > total_avail:
        # proportionally trim – very rare edge case
        factor = total_avail / sum(selected.values())
        selected = {k: max(1, int(v * factor)) for k, v in selected.items() if v > 0}

    return selected


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _max_fraction_cap(n_total: int, max_frac: float) -> int:
    return max(1, int(math.ceil(n_total * max_frac)))


def _weighted_fill(
    selected: Dict[str, int],
    remaining: int,
    names: List[str],
    weights: List[float],
    caps: List[int],
    rng: _random.Random,
) -> Dict[str, int]:
    """Fill *remaining* slots using multinomial sampling with caps."""
    for _ in range(remaining):
        if not names:
            break
        total_w = sum(weights)
        if total_w <= 0:
            break
        r = rng.random() * total_w
        cumulative = 0.0
        chosen_idx = len(names) - 1
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                chosen_idx = i
                break
        chosen = names[chosen_idx]
        selected[chosen] = selected.get(chosen, 0) + 1
        caps[chosen_idx] -= 1
        if caps[chosen_idx] <= 0:
            names.pop(chosen_idx)
            weights.pop(chosen_idx)
            caps.pop(chosen_idx)
    return selected
