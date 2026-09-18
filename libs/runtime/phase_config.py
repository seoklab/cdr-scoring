from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseConfig:
    phase_id: int
    eject_ratio: float
    top_ratio: float
    lambda_sml: float
    lambda_dpo_eject: float
    lambda_dpo_top: float
    lambda_compactness: float


# v2 Phase-C sub-phases (explicit; no auto epoch switching).
#   eject:top weighted ratio  C1=3:1  ->  C2=1:1  ->  C3=1:3
#   compactness weight        C1 > C2 > C3   (tighten xtal-A band early, relax later)
#   lambda_sml keeps the final head calibrated to cdr_lddt while DPO sharpens ranking.
_PHASES = {
    1: PhaseConfig(   # C1: eject-heavy (3:1), strong compactness
        phase_id=1,
        eject_ratio=0.75,
        top_ratio=0.25,
        lambda_sml=1.0,
        lambda_dpo_eject=0.75,
        lambda_dpo_top=0.25,
        lambda_compactness=1.0,
    ),
    2: PhaseConfig(   # C2: balanced (1:1), medium compactness
        phase_id=2,
        eject_ratio=0.5,
        top_ratio=0.5,
        lambda_sml=1.0,
        lambda_dpo_eject=0.5,
        lambda_dpo_top=0.5,
        lambda_compactness=0.5,
    ),
    3: PhaseConfig(   # C3: top-heavy (1:3), light compactness
        phase_id=3,
        eject_ratio=0.25,
        top_ratio=0.75,
        lambda_sml=1.0,
        lambda_dpo_eject=0.25,
        lambda_dpo_top=0.75,
        lambda_compactness=0.2,
    ),
}


def get_phase_config(phase_id: int) -> PhaseConfig:
    return _PHASES.get(int(phase_id), _PHASES[1])


# ──────────────────────────────────────────────────────────────
# v2 within-target decoy tier quotas (A/B/C/D; tier-X counts toward A).
# Pretrain uses the balanced default; Phase-C DPO finetune schedules the mix to
# match the eject -> balanced -> top DPO progression:
#   C1 (eject-heavy): more C/D for near-vs-far separation
#   C2 (balanced)
#   C3 (top-heavy):   more A/B for top-vs-mid ranking pressure
# ──────────────────────────────────────────────────────────────
_TIER_QUOTAS = {
    "pretrain": {"A": 8,  "B": 24, "C": 16, "D": 16},
    "C1":       {"A": 8,  "B": 20, "C": 16, "D": 20},
    "C2":       {"A": 12, "B": 24, "C": 16, "D": 12},
    "C3":       {"A": 16, "B": 24, "C": 16, "D": 8},
}


def get_tier_quota(key: str = "pretrain") -> dict:
    """Return a fresh {A,B,C,D} quota dict for the given phase key."""
    return dict(_TIER_QUOTAS.get(str(key), _TIER_QUOTAS["pretrain"]))
