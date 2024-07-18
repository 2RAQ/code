from __future__ import annotations

from multiprocessing import Pool

from absl import logging
import numpy as np

from agents.configurations import (
    EvalConfig,
    ThetaConfig,
)
from agents.data_manager import (
    DataContainer,
    DataManager,
)
from agents.mse_agents import (
    LFAAsAverageQA,
    LFAAsDQAAL,
    LFAAsMMQA,
    LFAAsREDQA,
    LFAAsRQA,
    LFAAsVanillaQA,
    LFAAsVRQA,
)
from agents.parameter import (
    DecayByT,
    DecayByTSquared,
)
from environments.bairds_example import BairdsExample
from environments.make_features import BairdsFeatures
from environments.solver import LFAEnvSolver

# On M Mac
# import multiprocessing as mp
# mp.set_start_method("fork")

logging.set_verbosity(logging.DEBUG)


rng = np.random.default_rng(seed=1234)
seed_env = 1234

WORKERS: int = 17
THETA_SIZE: int = 12
THETA_INIT_L: float = 0.0
THETA_INIT_U: float = 2.0
REWARDS_L: float = -0.05
REWARDS_U: float = 0.05
SAMPLES: int = 2000000
EXPERIMENTS: int = 100
REPETITIONS: int = 1
EVAL_CONFIG: EvalConfig = EvalConfig()
GAMMA: float = 0.8
ALPHA: float = 0.01
ALPHA_D_WEIGHT: int = 100000
RHO_D_WEIGHT: int = 1000

features = BairdsFeatures(6)
solver = LFAEnvSolver(
    THETA_SIZE, features=features, gamma=GAMMA, alpha=DecayByT(0.3, 100)
)
env = BairdsExample(6, (REWARDS_L, REWARDS_U), seed=seed_env, solver=solver)

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
agent_rraq_rho0_n1 = LFAAsRQA(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U, 1),
    alpha=DecayByT(ALPHA, weight=ALPHA_D_WEIGHT),
    rho=DecayByTSquared(0, weight=RHO_D_WEIGHT),
    gamma=GAMMA,
    env=env,
    features=features,
    eval_config=EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)
agent_rraq_rho0_n10 = LFAAsRQA(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U, 10),
    alpha=DecayByT(10 * ALPHA, weight=ALPHA_D_WEIGHT),
    rho=DecayByTSquared(0, weight=RHO_D_WEIGHT),
    gamma=GAMMA,
    env=env,
    features=features,
    eval_config=EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)
agent_rraq_rho05_n10 = LFAAsRQA(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U, 10),
    alpha=DecayByT(10 * ALPHA, weight=ALPHA_D_WEIGHT),
    rho=DecayByTSquared(0.5, weight=RHO_D_WEIGHT),
    gamma=GAMMA,
    env=env,
    features=features,
    eval_config=EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)
agent_rraq_rho2_n10 = LFAAsRQA(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U, 10),
    alpha=DecayByT(10 * ALPHA, weight=ALPHA_D_WEIGHT),
    rho=DecayByTSquared(2, weight=RHO_D_WEIGHT),
    gamma=GAMMA,
    env=env,
    features=features,
    eval_config=EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)
agent_rraq_rho5_n10 = LFAAsRQA(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U, 10),
    alpha=DecayByT(10 * ALPHA, weight=ALPHA_D_WEIGHT),
    rho=DecayByTSquared(5, weight=RHO_D_WEIGHT),
    gamma=GAMMA,
    env=env,
    features=features,
    eval_config=EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)
agent_rraq_rho10_n10 = LFAAsRQA(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U, 10),
    alpha=DecayByT(10 * ALPHA, weight=ALPHA_D_WEIGHT),
    rho=DecayByTSquared(10, weight=RHO_D_WEIGHT),
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
agent_rraq_rho05_n1 = LFAAsRQA(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U, 1),
    alpha=DecayByT(ALPHA, weight=ALPHA_D_WEIGHT),
    rho=DecayByTSquared(0.5, weight=RHO_D_WEIGHT),
    gamma=GAMMA,
    env=env,
    features=features,
    eval_config=EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)
agent_rraq_rho05_n2 = LFAAsRQA(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U, 2),
    alpha=DecayByT(2 * ALPHA, weight=ALPHA_D_WEIGHT),
    rho=DecayByTSquared(0.5, weight=RHO_D_WEIGHT),
    gamma=GAMMA,
    env=env,
    features=features,
    eval_config=EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)
agent_rraq_rho05_n5 = LFAAsRQA(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U, 5),
    alpha=DecayByT(5 * ALPHA, weight=ALPHA_D_WEIGHT),
    rho=DecayByTSquared(0.5, weight=RHO_D_WEIGHT),
    gamma=GAMMA,
    env=env,
    features=features,
    eval_config=EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)
agent_rraq_rho05_n20 = LFAAsRQA(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U, 20),
    alpha=DecayByT(20 * ALPHA, weight=ALPHA_D_WEIGHT),
    rho=DecayByTSquared(0.5, weight=RHO_D_WEIGHT),
    gamma=GAMMA,
    env=env,
    features=features,
    eval_config=EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)
agent_redq = LFAAsREDQA(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U, 10),
    alpha=DecayByT(ALPHA, weight=ALPHA_D_WEIGHT),
    g=1,
    m=2,
    gamma=GAMMA,
    env=env,
    features=features,
    eval_config=EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)
agent_avq = LFAAsAverageQA(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U, 1),
    alpha=DecayByT(ALPHA, weight=ALPHA_D_WEIGHT),
    k=10,
    gamma=GAMMA,
    env=env,
    features=features,
    eval_config=EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)
agent_vrq = LFAAsVRQA(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U),
    alpha=DecayByT(ALPHA, weight=ALPHA_D_WEIGHT),
    gamma=GAMMA,
    d=10,
    env=env,
    features=features,
    eval_config=EVAL_CONFIG,
    seed=int(rng.integers(2**32)),
)


with Pool(processes=WORKERS) as pool:
    q = pool.apply_async(
        agent_q.run_experiments, (SAMPLES, EXPERIMENTS, REPETITIONS, 0)
    )
    dq = pool.apply_async(
        agent_dq.run_experiments, (SAMPLES, EXPERIMENTS, REPETITIONS, 2)
    )
    mmq = pool.apply_async(
        agent_mmq.run_experiments, (SAMPLES, EXPERIMENTS, REPETITIONS, 4)
    )
    rraq_rho0_n1 = pool.apply_async(
        agent_rraq_rho0_n1.run_experiments, (SAMPLES, EXPERIMENTS, REPETITIONS, 6)
    )
    rraq_rho0_n10 = pool.apply_async(
        agent_rraq_rho0_n10.run_experiments, (SAMPLES, EXPERIMENTS, REPETITIONS, 8)
    )
    rraq_rho05_n10 = pool.apply_async(
        agent_rraq_rho05_n10.run_experiments, (SAMPLES, EXPERIMENTS, REPETITIONS, 10)
    )
    rraq_rho2_n10 = pool.apply_async(
        agent_rraq_rho2_n10.run_experiments, (SAMPLES, EXPERIMENTS, REPETITIONS, 12)
    )
    rraq_rho5_n10 = pool.apply_async(
        agent_rraq_rho5_n10.run_experiments, (SAMPLES, EXPERIMENTS, REPETITIONS, 14)
    )
    rraq_rho10_n10 = pool.apply_async(
        agent_rraq_rho10_n10.run_experiments, (SAMPLES, EXPERIMENTS, REPETITIONS, 16)
    )
    rraq_rho50_n10 = pool.apply_async(
        agent_rraq_rho50_n10.run_experiments, (SAMPLES, EXPERIMENTS, REPETITIONS, 18)
    )
    rraq_rho05_n1 = pool.apply_async(
        agent_rraq_rho05_n1.run_experiments, (SAMPLES, EXPERIMENTS, REPETITIONS, 20)
    )
    rraq_rho05_n2 = pool.apply_async(
        agent_rraq_rho05_n2.run_experiments, (SAMPLES, EXPERIMENTS, REPETITIONS, 22)
    )
    rraq_rho05_n5 = pool.apply_async(
        agent_rraq_rho05_n5.run_experiments, (SAMPLES, EXPERIMENTS, REPETITIONS, 24)
    )
    rraq_rho05_n20 = pool.apply_async(
        agent_rraq_rho05_n20.run_experiments, (SAMPLES, EXPERIMENTS, REPETITIONS, 26)
    )
    redq = pool.apply_async(
        agent_redq.run_experiments, (SAMPLES, EXPERIMENTS, REPETITIONS, 28)
    )
    avq = pool.apply_async(
        agent_avq.run_experiments, (SAMPLES, EXPERIMENTS, REPETITIONS, 28)
    )
    vrq = pool.apply_async(
        agent_vrq.run_experiments, (SAMPLES, EXPERIMENTS, REPETITIONS, 12)
    )

    error_q, _ = q.get()
    error_dq, _ = dq.get()
    error_mmq, _ = mmq.get()
    error_rraq_rho0_n1, _ = rraq_rho0_n1.get()
    error_rraq_rho0_n10, _ = rraq_rho0_n10.get()
    error_rraq_rho05_n10, _ = rraq_rho05_n10.get()
    error_rraq_rho2_n10, _ = rraq_rho2_n10.get()
    error_rraq_rho5_n10, _ = rraq_rho5_n10.get()
    error_rraq_rho10_n10, _ = rraq_rho10_n10.get()
    error_rraq_rho50_n10, _ = rraq_rho50_n10.get()
    error_rraq_rho05_n1, _ = rraq_rho05_n1.get()
    error_rraq_rho05_n2, _ = rraq_rho05_n2.get()
    error_rraq_rho05_n5, _ = rraq_rho05_n5.get()
    error_rraq_rho05_n20, _ = rraq_rho05_n20.get()
    error_redq, _ = redq.get()
    error_avq, _ = avq.get()
    error_vrq, _ = vrq.get()


dm = DataManager()

dm.add_raw_data(
    DataContainer(
        "q",
        error_q,
        {
            "name": "Vanilla Q-Learning",
            "environment": "Baird's example",
            "n_states": 6,
            "n_actions": 2,
            "env_seed": seed_env,
            "reward_range": (REWARDS_L, REWARDS_U),
            "thetas": 1,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (t + {ALPHA_D_WEIGHT})",
            "gamma": GAMMA,
            "n_samples": SAMPLES,
            "n_experiments": EXPERIMENTS,
            "n_repetitions": REPETITIONS,
            "features": "BairdsFeatures",
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
            "environment": "Baird's example",
            "n_states": 6,
            "n_actions": 2,
            "env_seed": seed_env,
            "reward_range": (REWARDS_L, REWARDS_U),
            "thetas": 2,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": 2 * ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (t + {ALPHA_D_WEIGHT})",
            "gamma": GAMMA,
            "n_samples": SAMPLES,
            "n_experiments": EXPERIMENTS,
            "n_repetitions": REPETITIONS,
            "features": "BairdsFeatures",
            "agent_seed": agent_dq.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "mmq",
        error_mmq,
        {
            "name": "Maxmin Q-Learning",
            "environment": "Baird's example",
            "n_states": 6,
            "n_actions": 2,
            "env_seed": seed_env,
            "reward_range": (REWARDS_L, REWARDS_U),
            "thetas": 10,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": 10 * ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (t + {ALPHA_D_WEIGHT})",
            "gamma": GAMMA,
            "n_samples": SAMPLES,
            "n_experiments": EXPERIMENTS,
            "n_repetitions": REPETITIONS,
            "features": "BairdsFeatures",
            "agent_seed": agent_mmq.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "rraq_rho0_n1",
        error_rraq_rho0_n1,
        {
            "name": "2RA Q-Learning",
            "environment": "Baird's example",
            "n_states": 6,
            "n_actions": 2,
            "env_seed": seed_env,
            "reward_range": (REWARDS_L, REWARDS_U),
            "thetas": 1,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (t + {ALPHA_D_WEIGHT})",
            "rho": 0,
            "rho_decay_weight": RHO_D_WEIGHT,
            "rho_decay": f"rho * {RHO_D_WEIGHT} / (t^2 + {RHO_D_WEIGHT})",
            "gamma": GAMMA,
            "n_samples": SAMPLES,
            "n_experiments": EXPERIMENTS,
            "n_repetitions": REPETITIONS,
            "features": "BairdsFeatures",
            "agent_seed": agent_rraq_rho0_n1.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "rraq_rho0_n10",
        error_rraq_rho0_n10,
        {
            "name": "2RA Q-Learning",
            "environment": "Baird's example",
            "n_states": 6,
            "n_actions": 2,
            "env_seed": seed_env,
            "reward_range": (REWARDS_L, REWARDS_U),
            "thetas": 10,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": 10 * ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (t + {ALPHA_D_WEIGHT})",
            "rho": 0,
            "rho_decay_weight": RHO_D_WEIGHT,
            "rho_decay": f"rho * {RHO_D_WEIGHT} / (t^2 + {RHO_D_WEIGHT})",
            "gamma": GAMMA,
            "n_samples": SAMPLES,
            "n_experiments": EXPERIMENTS,
            "n_repetitions": REPETITIONS,
            "features": "BairdsFeatures",
            "agent_seed": agent_rraq_rho0_n10.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "rraq_rho05_n10",
        error_rraq_rho05_n10,
        {
            "name": "2RA Q-Learning",
            "environment": "Baird's example",
            "n_states": 6,
            "n_actions": 2,
            "env_seed": seed_env,
            "reward_range": (REWARDS_L, REWARDS_U),
            "thetas": 10,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": 10 * ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (t + {ALPHA_D_WEIGHT})",
            "rho": 0.5,
            "rho_decay_weight": RHO_D_WEIGHT,
            "rho_decay": f"rho * {RHO_D_WEIGHT} / (t^2 + {RHO_D_WEIGHT})",
            "gamma": GAMMA,
            "n_samples": SAMPLES,
            "n_experiments": EXPERIMENTS,
            "n_repetitions": REPETITIONS,
            "features": "BairdsFeatures",
            "agent_seed": agent_rraq_rho05_n10.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "rraq_rho2_n10",
        error_rraq_rho2_n10,
        {
            "name": "2RA Q-Learning",
            "environment": "Baird's example",
            "n_states": 6,
            "n_actions": 2,
            "env_seed": seed_env,
            "reward_range": (REWARDS_L, REWARDS_U),
            "thetas": 10,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": 10 * ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (t + {ALPHA_D_WEIGHT})",
            "rho": 2,
            "rho_decay_weight": RHO_D_WEIGHT,
            "rho_decay": f"rho * {RHO_D_WEIGHT} / (t^2 + {RHO_D_WEIGHT})",
            "gamma": GAMMA,
            "n_samples": SAMPLES,
            "n_experiments": EXPERIMENTS,
            "n_repetitions": REPETITIONS,
            "features": "BairdsFeatures",
            "agent_seed": agent_rraq_rho2_n10.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "rraq_rho5_n10",
        error_rraq_rho5_n10,
        {
            "name": "2RA Q-Learning",
            "environment": "Baird's example",
            "n_states": 6,
            "n_actions": 2,
            "env_seed": seed_env,
            "reward_range": (REWARDS_L, REWARDS_U),
            "thetas": 10,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": 10 * ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (t + {ALPHA_D_WEIGHT})",
            "rho": 5,
            "rho_decay_weight": RHO_D_WEIGHT,
            "rho_decay": f"rho * {RHO_D_WEIGHT} / (t^2 + {RHO_D_WEIGHT})",
            "gamma": GAMMA,
            "n_samples": SAMPLES,
            "n_experiments": EXPERIMENTS,
            "n_repetitions": REPETITIONS,
            "features": "BairdsFeatures",
            "agent_seed": agent_rraq_rho5_n10.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "rraq_rho10_n10",
        error_rraq_rho10_n10,
        {
            "name": "2RA Q-Learning",
            "environment": "Baird's example",
            "n_states": 6,
            "n_actions": 2,
            "env_seed": seed_env,
            "reward_range": (REWARDS_L, REWARDS_U),
            "thetas": 10,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": 10 * ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (t + {ALPHA_D_WEIGHT})",
            "rho": 10,
            "rho_decay_weight": RHO_D_WEIGHT,
            "rho_decay": f"rho * {RHO_D_WEIGHT} / (t^2 + {RHO_D_WEIGHT})",
            "gamma": GAMMA,
            "n_samples": SAMPLES,
            "n_experiments": EXPERIMENTS,
            "n_repetitions": REPETITIONS,
            "features": "BairdsFeatures",
            "agent_seed": agent_rraq_rho10_n10.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "rraq_rho50_n10",
        error_rraq_rho50_n10,
        {
            "name": "2RA Q-Learning",
            "environment": "Baird's example",
            "n_states": 6,
            "n_actions": 2,
            "env_seed": seed_env,
            "reward_range": (REWARDS_L, REWARDS_U),
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
            "features": "BairdsFeatures",
            "agent_seed": agent_rraq_rho50_n10.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "rraq_rho05_n1",
        error_rraq_rho05_n1,
        {
            "name": "2RA Q-Learning",
            "environment": "Baird's example",
            "n_states": 6,
            "n_actions": 2,
            "env_seed": seed_env,
            "reward_range": (REWARDS_L, REWARDS_U),
            "thetas": 1,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (t + {ALPHA_D_WEIGHT})",
            "rho": 0.5,
            "rho_decay_weight": RHO_D_WEIGHT,
            "rho_decay": f"rho * {RHO_D_WEIGHT} / (t^2 + {RHO_D_WEIGHT})",
            "gamma": GAMMA,
            "n_samples": SAMPLES,
            "n_experiments": EXPERIMENTS,
            "n_repetitions": REPETITIONS,
            "features": "BairdsFeatures",
            "agent_seed": agent_rraq_rho05_n1.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "rraq_rho05_n2",
        error_rraq_rho05_n2,
        {
            "name": "2RA Q-Learning",
            "environment": "Baird's example",
            "n_states": 6,
            "n_actions": 2,
            "env_seed": seed_env,
            "reward_range": (REWARDS_L, REWARDS_U),
            "thetas": 2,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": 2 * ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (t + {ALPHA_D_WEIGHT})",
            "rho": 0.5,
            "rho_decay_weight": RHO_D_WEIGHT,
            "rho_decay": f"rho * {RHO_D_WEIGHT} / (t^2 + {RHO_D_WEIGHT})",
            "gamma": GAMMA,
            "n_samples": SAMPLES,
            "n_experiments": EXPERIMENTS,
            "n_repetitions": REPETITIONS,
            "features": "BairdsFeatures",
            "agent_seed": agent_rraq_rho05_n2.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "rraq_rho05_n5",
        error_rraq_rho05_n5,
        {
            "name": "2RA Q-Learning",
            "environment": "Baird's example",
            "n_states": 6,
            "n_actions": 2,
            "env_seed": seed_env,
            "reward_range": (REWARDS_L, REWARDS_U),
            "thetas": 5,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": 5 * ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (t + {ALPHA_D_WEIGHT})",
            "rho": 0.5,
            "rho_decay_weight": RHO_D_WEIGHT,
            "rho_decay": f"rho * {RHO_D_WEIGHT} / (t^2 + {RHO_D_WEIGHT})",
            "gamma": GAMMA,
            "n_samples": SAMPLES,
            "n_experiments": EXPERIMENTS,
            "n_repetitions": REPETITIONS,
            "features": "BairdsFeatures",
            "agent_seed": agent_rraq_rho05_n5.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "rraq_rho05_n20",
        error_rraq_rho05_n20,
        {
            "name": "2RA Q-Learning",
            "environment": "Baird's example",
            "n_states": 6,
            "n_actions": 2,
            "env_seed": seed_env,
            "reward_range": (REWARDS_L, REWARDS_U),
            "thetas": 20,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": 20 * ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (t + {ALPHA_D_WEIGHT})",
            "rho": 0.5,
            "rho_decay_weight": RHO_D_WEIGHT,
            "rho_decay": f"rho * {RHO_D_WEIGHT} / (t^2 + {RHO_D_WEIGHT})",
            "gamma": GAMMA,
            "n_samples": SAMPLES,
            "n_experiments": EXPERIMENTS,
            "n_repetitions": REPETITIONS,
            "features": "BairdsFeatures",
            "agent_seed": agent_rraq_rho05_n20.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "redq",
        error_redq,
        {
            "name": "REDQ-Learning",
            "environment": "Baird's example",
            "n_states": 6,
            "n_actions": 2,
            "env_seed": seed_env,
            "reward_range": (REWARDS_L, REWARDS_U),
            "thetas": 10,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (t + {ALPHA_D_WEIGHT})",
            "G": 1,
            "M": 2,
            "gamma": GAMMA,
            "n_samples": SAMPLES,
            "n_experiments": EXPERIMENTS,
            "n_repetitions": REPETITIONS,
            "features": "BairdsFeatures",
            "agent_seed": agent_redq.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "avq",
        error_avq,
        {
            "name": "AverageDQN Q-Learning",
            "environment": "Baird's example",
            "n_states": 6,
            "n_actions": 2,
            "env_seed": seed_env,
            "reward_range": (REWARDS_L, REWARDS_U),
            "thetas": 1,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (t + {ALPHA_D_WEIGHT})",
            "K": 10,
            "gamma": GAMMA,
            "n_samples": SAMPLES,
            "n_experiments": EXPERIMENTS,
            "n_repetitions": REPETITIONS,
            "features": "BairdsFeatures",
            "agent_seed": agent_avq.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "vrq",
        error_vrq,
        {
            "name": "Variance Reduced Q-Learning",
            "environment": "Baird's example",
            "n_states": 6,
            "n_actions": 2,
            "env_seed": seed_env,
            "reward_range": (REWARDS_L, REWARDS_U),
            "thetas": 1,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (t + {ALPHA_D_WEIGHT})",
            "D": 10,
            "gamma": GAMMA,
            "n_samples": SAMPLES,
            "n_experiments": EXPERIMENTS,
            "n_repetitions": REPETITIONS,
            "features": "BairdsFeatures",
            "agent_seed": agent_vrq.seed,
        },
    )
)


dm.save_data("data/bairds_example")
