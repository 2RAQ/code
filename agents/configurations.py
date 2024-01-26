from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThetaConfig:
    """soon"""

    dimensions: int
    init_upper: float
    init_lower: float
    n_thetas: int = 1


@dataclass
class EvalConfig:
    """soon"""

    timesteps: int = 0
    episodes: int = 1
    frequency: int = 1


DEFAULT_EVAL_CONFIG: EvalConfig = EvalConfig()
