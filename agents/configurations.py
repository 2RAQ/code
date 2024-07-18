from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThetaConfig:
    """A container holding the configuration for the estimator(s) of an agent.

    Attributes:
        dimensions: The number of features in the feature vector
        init_upper: The upper bound for the initialization of the theta vector
        init_lower: The lower bound for the initialization of the theta vector
        n_thetas: The number of theta vectors/estimates used in the agent
    """

    dimensions: int
    init_upper: float
    init_lower: float
    n_thetas: int = 1


@dataclass
class EvalConfig:
    """A container holding the configuration for the evaluation of an agent.

    Attributes:
        timesteps: The number of timesteps each evaluation epsisode should contain
        episodes: The number of episodes for which agent should be evaluated
        frequency: The number of training episodes between each evaluation

    """

    timesteps: int = 0
    episodes: int = 1
    frequency: int = 1


DEFAULT_EVAL_CONFIG: EvalConfig = EvalConfig()
