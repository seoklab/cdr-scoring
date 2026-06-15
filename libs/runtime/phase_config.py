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


_PHASES = {
    1: PhaseConfig(
        phase_id=1,
        eject_ratio=0.7,
        top_ratio=0.3,
        lambda_sml=1.0,
        lambda_dpo_eject=1.0,
        lambda_dpo_top=0.5,
        lambda_compactness=0.0,
    ),
    2: PhaseConfig(
        phase_id=2,
        eject_ratio=0.5,
        top_ratio=0.5,
        lambda_sml=1.0,
        lambda_dpo_eject=1.0,
        lambda_dpo_top=1.0,
        lambda_compactness=0.0,
    ),
    3: PhaseConfig(
        phase_id=3,
        eject_ratio=0.4,
        top_ratio=0.6,
        lambda_sml=1.0,
        lambda_dpo_eject=1.0,
        lambda_dpo_top=1.0,
        lambda_compactness=0.0,
    ),
}


def get_phase_config(phase_id: int) -> PhaseConfig:
    return _PHASES.get(int(phase_id), _PHASES[1])
