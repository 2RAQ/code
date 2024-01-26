from __future__ import annotations

from multiprocessing import Pool

import numpy as np
from absl import logging

from agents.configurations import EvalConfig, ThetaConfig
from agents.data_manager import DataContainer, DataManager
from agents.mse_agents import LFAAsDQAAL, LFAAsMMQA, LFAAsRQA, LFAAsVanillaQA
from agents.parameter import DecayByT, DecayByTSquared
from environments.make_features import MakeOneHot
from environments.random_env import RandomEnvironment
from environments.solver import LFAEnvSolver

# On M1 Mac
# import multiprocessing as mp
# mp.set_start_method("fork")

logging.set_verbosity(logging.DEBUG)


rng = np.random.default_rng(seed=1234)
seed_env = int(rng.integers(2**32))

THETA_SIZE: int = 30
THETA_INIT_L: float = 0.0
THETA_INIT_U: float = 0.0
REWARDS_Q: float = 0.01
REWARDS_P: float = 0.1
SAMPLES: int = 600000
EXPERIMENTS: int = 21
REPETITIONS: int = 100
EVAL_CONFIG: EvalConfig = EvalConfig()
GAMMA: float = 0.9
ALPHA: float = 0.01
ALPHA_D_WEIGHT: int = 100000
RHO_D_WEIGHT: int = 10000

features = MakeOneHot(10, 3)
solver = LFAEnvSolver(
    THETA_SIZE, features=features, gamma=GAMMA, alpha=DecayByT(0.2, 500)
)
env = RandomEnvironment(10, 3, q=REWARDS_Q, p=REWARDS_P, seed=seed_env, solver=solver)
env.gen_model_sequence(EXPERIMENTS)

agent_q = LFAAsVanillaQA(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U),
    alpha=DecayByT(ALPHA, weight=ALPHA_D_WEIGHT),
    gamma=GAMMA,
    env=env,
    features=features,
    eval_config=EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)
agent_dq = LFAAsDQAAL(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U),
    alpha=DecayByT(2 * ALPHA, weight=ALPHA_D_WEIGHT),
    gamma=GAMMA,
    env=env,
    features=features,
    eval_config=EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)
agent_mmq = LFAAsMMQA(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U, 10),
    alpha=DecayByT(10 * ALPHA, weight=ALPHA_D_WEIGHT),
    gamma=GAMMA,
    env=env,
    features=features,
    eval_config=EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)
agent_rraq_rho50_n10 = LFAAsRQA(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U, 10),
    alpha=DecayByT(10 * ALPHA, weight=ALPHA_D_WEIGHT),
    rho=DecayByTSquared(50, weight=RHO_D_WEIGHT),
    gamma=GAMMA,
    env=env,
    features=features,
    eval_config=EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)


with Pool(processes=4) as pool:
    q = pool.apply_async(
        agent_q.run_experiments, (SAMPLES, EXPERIMENTS, REPETITIONS, 0)
    )
    dq = pool.apply_async(
        agent_dq.run_experiments, (SAMPLES, EXPERIMENTS, REPETITIONS, 2)
    )
    mmq = pool.apply_async(
        agent_mmq.run_experiments, (SAMPLES, EXPERIMENTS, REPETITIONS, 4)
    )
    rraq_rho50_n10 = pool.apply_async(
        agent_rraq_rho50_n10.run_experiments, (SAMPLES, EXPERIMENTS, REPETITIONS, 18)
    )

    error_q, _ = q.get()
    error_dq, _ = dq.get()
    error_mmq, _ = mmq.get()
    error_rraq_rho50_n10, _ = rraq_rho50_n10.get()


dm = DataManager()

dm.add_raw_data(
    DataContainer(
        "q",
        error_q,
        {
            "name": "Vanilla Q-Learning",
            "environment": "Random environment",
            "n_states": 10,
            "n_actions": 3,
            "env_seed": seed_env,
            "rewards": f"{REWARDS_Q} * s^2 - {REWARDS_P} * a^2",
            "thetas": 1,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (t + {ALPHA_D_WEIGHT})",
            "rho": None,
            "rho_decay_weight": None,
            "rho_decay": None,
            "gamma": GAMMA,
            "n_samples": SAMPLES,
            "n_experiments": EXPERIMENTS,
            "n_repetitions": REPETITIONS,
            "features": "MakeOneHot",
            "agent_seed": agent_q.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "dq",
        error_dq,
        {
            "name": "Double Q-Learning",
            "environment": "Random environment",
            "n_states": 10,
            "n_actions": 3,
            "env_seed": seed_env,
            "rewards": f"{REWARDS_Q} * s^2 - {REWARDS_P} * a^2",
            "thetas": 2,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": 2 * ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (t + {ALPHA_D_WEIGHT})",
            "rho": None,
            "rho_decay_weight": None,
            "rho_decay": None,
            "gamma": GAMMA,
            "n_samples": SAMPLES,
            "n_experiments": EXPERIMENTS,
            "n_repetitions": REPETITIONS,
            "features": "MakeOneHot",
            "agent_seed": agent_dq.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "mmq",
        error_mmq,
        {
            "name": "MaxMin Q-Learning",
            "environment": "Random environment",
            "n_states": 10,
            "n_actions": 3,
            "env_seed": seed_env,
            "rewards": f"{REWARDS_Q} * s^2 - {REWARDS_P} * a^2",
            "thetas": 10,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": 10 * ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (t + {ALPHA_D_WEIGHT})",
            "rho": None,
            "rho_decay_weight": None,
            "rho_decay": None,
            "gamma": GAMMA,
            "n_samples": SAMPLES,
            "n_experiments": EXPERIMENTS,
            "n_repetitions": REPETITIONS,
            "features": "MakeOneHot",
            "agent_seed": agent_mmq.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "rraq_rho50_n10",
        error_rraq_rho50_n10,
        {
            "name": "2RA Q-Learning",
            "environment": "Random environment",
            "n_states": 10,
            "n_actions": 3,
            "env_seed": seed_env,
            "rewards": f"{REWARDS_Q} * s^2 - {REWARDS_P} * a^2",
            "thetas": 10,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": 10 * ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (t + {ALPHA_D_WEIGHT})",
            "rho": 50,
            "rho_decay_weight": RHO_D_WEIGHT,
            "rho_decay": f"rho * {RHO_D_WEIGHT} / (t^2 + {RHO_D_WEIGHT})",
            "gamma": GAMMA,
            "n_samples": SAMPLES,
            "n_experiments": EXPERIMENTS,
            "n_repetitions": REPETITIONS,
            "features": "MakeOneHot",
            "agent_seed": agent_rraq_rho50_n10.seed,
        },
    )
)


dm.save_data("data/random_environment")
