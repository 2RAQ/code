from __future__ import annotations

import copy

import gym
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")  # use CPU
# tf.config.run_functions_eagerly(True)  # for debugging

rng = np.random.default_rng(seed=1234)
tf.random.set_seed(int(rng.integers(2**32)))

env = gym.make("LunarLander-v2")
seed_env = int(rng.integers(2**32))
env.seed(seed_env)

from agents.configurations import EvalConfig
from agents.data_manager import (
    DataContainer,
    DataManager,
)
from agents.dqn_agents import (
    DDQAA,
    MMDQA,
    RRADQA,
    VanillaDQA,
)
from agents.parameter import (
    DecayByT,
    DecayByTSquared,
    WengEtAl2020MaxMin,
)

N_EXPERIMENTS: int = 100
N_EPISODES: int = 2000
THRESHOLD: int = 200
EVAL_CONFIG: EvalConfig = EvalConfig(1000, 100, 50)
GAMMA: float = 0.99
ALPHA: float = 0.0002
RHO_D_WEIGHT: int = 10000

agent_q = VanillaDQA(
    1,
    ALPHA,
    WengEtAl2020MaxMin(),
    GAMMA,
    copy.deepcopy(env),
    EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)
agent_dq = DDQAA(
    2,
    ALPHA,
    WengEtAl2020MaxMin(),
    GAMMA,
    copy.deepcopy(env),
    EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)
agent_mmq_n5 = MMDQA(
    5,
    ALPHA,
    WengEtAl2020MaxMin(),
    GAMMA,
    copy.deepcopy(env),
    EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)
agent_mmq_n10 = MMDQA(
    10,
    ALPHA,
    WengEtAl2020MaxMin(),
    GAMMA,
    copy.deepcopy(env),
    EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)
agent_rraq_rho25_n5 = RRADQA(
    5,
    ALPHA,
    WengEtAl2020MaxMin(),
    GAMMA,
    DecayByT(150, weight=RHO_D_WEIGHT),
    copy.deepcopy(env),
    EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)
agent_rraq_rho10_n5 = RRADQA(
    5,
    ALPHA,
    WengEtAl2020MaxMin(),
    GAMMA,
    DecayByTSquared(10, weight=RHO_D_WEIGHT),
    copy.deepcopy(env),
    EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)
agent_rraq_rho10_n10 = RRADQA(
    10,
    ALPHA,
    WengEtAl2020MaxMin(),
    GAMMA,
    DecayByTSquared(10, weight=RHO_D_WEIGHT),
    copy.deepcopy(env),
    EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)
agent_rraq_rho25_n10 = RRADQA(
    10,
    ALPHA,
    WengEtAl2020MaxMin(),
    GAMMA,
    DecayByTSquared(25, weight=RHO_D_WEIGHT),
    copy.deepcopy(env),
    EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)


steps_q = None
steps_dq = None
steps_mmq_n5 = None
steps_mmq_n10 = None
steps_rraq_rho25_n5 = None
steps_rraq_rho10_n5 = None
steps_rraq_rho10_n10 = None
steps_rraq_rho25_n10 = None

steps_q = agent_q.run_experiments(N_EXPERIMENTS, N_EPISODES, THRESHOLD)
steps_dq = agent_dq.run_experiments(N_EXPERIMENTS, N_EPISODES, THRESHOLD)
steps_mmq_n5 = agent_mmq_n5.run_experiments(N_EXPERIMENTS, N_EPISODES, THRESHOLD)
steps_mmq_n10 = agent_mmq_n10.run_experiments(N_EXPERIMENTS, N_EPISODES, THRESHOLD)
steps_rraq_rho25_n5 = agent_rraq_rho25_n5.run_experiments(
    N_EXPERIMENTS, N_EPISODES, THRESHOLD
)
steps_rraq_rho10_n5 = agent_rraq_rho10_n5.run_experiments(
    N_EXPERIMENTS, N_EPISODES, THRESHOLD
)
steps_rraq_rho10_n10 = agent_rraq_rho10_n5.run_experiments(
    N_EXPERIMENTS, N_EPISODES, THRESHOLD
)
steps_rraq_rho25_n10 = agent_rraq_rho25_n10.run_experiments(
    N_EXPERIMENTS, N_EPISODES, THRESHOLD
)


dm = DataManager()

if steps_q is not None:
    dm.add_raw_data(
        DataContainer(
            "q",
            steps_q,
            {
                "name": "Vanilla Q-Learning",
                "environment": "LunarLanger-v2",
                "env_seed": seed_env,
                "n_weights": 1,
                "alpha": ALPHA,
                "gamma": GAMMA,
                "n_episodes": N_EPISODES,
                "n_experiments": N_EXPERIMENTS,
                "eval_max_steps": EVAL_CONFIG.timesteps,
                "eval_episodes": EVAL_CONFIG.episodes,
                "eval_frequency": EVAL_CONFIG.frequency,
                "epsilon_decay": "WengEtAl2020MaxMin",
                "agent_seed": agent_q.seed,
            },
        )
    )
if steps_dq is not None:
    dm.add_raw_data(
        DataContainer(
            "dq",
            steps_dq,
            {
                "name": "Double Q-Learning",
                "environment": "LunarLanger-v2",
                "env_seed": seed_env,
                "n_weights": 2,
                "alpha": 2 * ALPHA,
                "gamma": GAMMA,
                "n_episodes": N_EPISODES,
                "n_experiments": N_EXPERIMENTS,
                "eval_max_steps": EVAL_CONFIG.timesteps,
                "eval_episodes": EVAL_CONFIG.episodes,
                "eval_frequency": EVAL_CONFIG.frequency,
                "epsilon_decay": "WengEtAl2020MaxMin",
                "agent_seed": agent_dq.seed,
            },
        )
    )
if steps_mmq_n5 is not None:
    dm.add_raw_data(
        DataContainer(
            "mmq_n5",
            steps_mmq_n5,
            {
                "name": "Maxmin Q-Learning",
                "environment": "LunarLanger-v2",
                "env_seed": seed_env,
                "n_weights": 5,
                "alpha": 5 * ALPHA,
                "gamma": GAMMA,
                "n_episodes": N_EPISODES,
                "n_experiments": N_EXPERIMENTS,
                "eval_max_steps": EVAL_CONFIG.timesteps,
                "eval_episodes": EVAL_CONFIG.episodes,
                "eval_frequency": EVAL_CONFIG.frequency,
                "epsilon-decay": "WengEtAl2020MaxMin",
                "agent_seed": agent_mmq_n5.seed,
            },
        )
    )
if steps_mmq_n10 is not None:
    dm.add_raw_data(
        DataContainer(
            "mmq_n10",
            steps_mmq_n10,
            {
                "name": "Maxmin Q-Learning",
                "environment": "LunarLanger-v2",
                "env_seed": seed_env,
                "n_weights": 10,
                "alpha": 10 * ALPHA,
                "gamma": GAMMA,
                "n_episodes": N_EPISODES,
                "n_experiments": N_EXPERIMENTS,
                "eval_max_steps": EVAL_CONFIG.timesteps,
                "eval_episodes": EVAL_CONFIG.episodes,
                "eval_frequency": EVAL_CONFIG.frequency,
                "epsilon-decay": "WengEtAl2020MaxMin",
                "agent_seed": agent_mmq_n10.seed,
            },
        )
    )
if steps_rraq_rho25_n5 is not None:
    dm.add_raw_data(
        DataContainer(
            "rraq_rho25_n5",
            steps_rraq_rho25_n5,
            {
                "name": "2RA Q-Learning",
                "environment": "LunarLanger-v2",
                "env_seed": seed_env,
                "n_weights": 5,
                "alpha": 5 * ALPHA,
                "rho": 150,
                "rho_decay_weight": RHO_D_WEIGHT,
                "rho_decay": f"rho * {RHO_D_WEIGHT} / (t + {RHO_D_WEIGHT})",
                "gamma": GAMMA,
                "n_episodes": N_EPISODES,
                "n_experiments": N_EXPERIMENTS,
                "eval_max_steps": EVAL_CONFIG.timesteps,
                "eval_episodes": EVAL_CONFIG.episodes,
                "eval_frequency": EVAL_CONFIG.frequency,
                "epsilon-decay": "WengEtAl2020MaxMin",
                "agent_seed": agent_rraq_rho25_n5.seed,
            },
        )
    )
if steps_rraq_rho10_n5 is not None:
    dm.add_raw_data(
        DataContainer(
            "rraq_rho10_n5",
            steps_rraq_rho10_n5,
            {
                "name": "2RA Q-Learning",
                "environment": "LunarLanger-v2",
                "env_seed": seed_env,
                "n_weights": 5,
                "alpha": 5 * ALPHA,
                "rho": 10,
                "rho_decay_weight": RHO_D_WEIGHT,
                "rho_decay": f"rho * {RHO_D_WEIGHT} / (t^2 + {RHO_D_WEIGHT})",
                "gamma": GAMMA,
                "n_episodes": N_EPISODES,
                "n_experiments": N_EXPERIMENTS,
                "eval_max_steps": EVAL_CONFIG.timesteps,
                "eval_episodes": EVAL_CONFIG.episodes,
                "eval_frequency": EVAL_CONFIG.frequency,
                "epsilon-decay": "WengEtAl2020MaxMin",
                "agent_seed": agent_rraq_rho10_n5.seed,
            },
        )
    )
if steps_rraq_rho10_n10 is not None:
    dm.add_raw_data(
        DataContainer(
            "rraq_rho10_n10",
            steps_rraq_rho10_n10,
            {
                "name": "2RA Q-Learning",
                "environment": "LunarLanger-v2",
                "env_seed": seed_env,
                "n_weights": 10,
                "alpha": 10 * ALPHA,
                "rho": 10,
                "rho_decay_weight": RHO_D_WEIGHT,
                "rho_decay": f"rho * {RHO_D_WEIGHT} / (t^2 + {RHO_D_WEIGHT})",
                "gamma": GAMMA,
                "n_episodes": N_EPISODES,
                "n_experiments": N_EXPERIMENTS,
                "eval_max_steps": EVAL_CONFIG.timesteps,
                "eval_episodes": EVAL_CONFIG.episodes,
                "eval_frequency": EVAL_CONFIG.frequency,
                "epsilon-decay": "WengEtAl2020MaxMin",
                "agent_seed": agent_rraq_rho10_n10.seed,
            },
        )
    )
if steps_rraq_rho25_n10 is not None:
    dm.add_raw_data(
        DataContainer(
            "rraq_rho25_n10",
            steps_rraq_rho25_n10,
            {
                "name": "2RA Q-Learning",
                "environment": "LunarLanger-v2",
                "env_seed": seed_env,
                "n_weights": 10,
                "alpha": 10 * ALPHA,
                "rho": 25,
                "rho_decay_weight": RHO_D_WEIGHT,
                "rho_decay": f"rho * {RHO_D_WEIGHT} / (t^2 + {RHO_D_WEIGHT})",
                "gamma": GAMMA,
                "n_episodes": N_EPISODES,
                "n_experiments": N_EXPERIMENTS,
                "eval_max_steps": EVAL_CONFIG.timesteps,
                "eval_episodes": EVAL_CONFIG.episodes,
                "eval_frequency": EVAL_CONFIG.frequency,
                "epsilon-decay": "WengEtAl2020MaxMin",
                "agent_seed": agent_rraq_rho25_n10.seed,
            },
        )
    )

dm.save_data("data/lunarlander")
