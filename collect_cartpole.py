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
from agents.parameter import (
    DecayByT,
    WengEtAl2020MaxMin,
)
from agents.threshold_agents import (
    LFASyAverageQA,
    LFASyDQAA,
    LFASyMMQA,
    LFASyRQA,
    LFASyVanillaQA,
)
from environments.gym_envs import IntegratedCartPole
from environments.make_features import DiscretizeCartpole

# On M Mac
# import multiprocessing as mp
# mp.set_start_method("fork")

logging.set_verbosity(logging.DEBUG)

rng = np.random.default_rng(seed=1234)

env = IntegratedCartPole()
seed_env = int(rng.integers(2**32))
env.set_seed(int(seed_env))
features = DiscretizeCartpole(env, buckets=[1, 1, 6, 12])

WORKERS: int = 5
THETA_SIZE: int = 144
THETA_INIT_L: float = 0.0
THETA_INIT_U: float = 0.0
N_EXPERIMENTS: int = 100
N_EPISODES: int = 1000
THRESHOLD: int = 195
EVAL_CONFIG: EvalConfig = EvalConfig(210, 100, 50)
GAMMA: float = 0.999
ALPHA: float = 0.4
ALPHA_D_WEIGHT: int = 100
RHO_D_WEIGHT: int = 10000

agent_q = LFASyVanillaQA(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U),
    alpha=DecayByT(ALPHA, weight=ALPHA_D_WEIGHT),
    epsilon=WengEtAl2020MaxMin(),
    gamma=GAMMA,
    env=env,
    features=features,
    eval_config=EVAL_CONFIG,
    max_steps=1000,
    seed=int(rng.integers(2**32)),
)
agent_dq = LFASyDQAA(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U),
    alpha=DecayByT(2 * ALPHA, weight=ALPHA_D_WEIGHT),
    epsilon=WengEtAl2020MaxMin(),
    gamma=GAMMA,
    env=env,
    features=features,
    eval_config=EVAL_CONFIG,
    max_steps=1000,
    seed=int(rng.integers(2**32)),
)
agent_mmq = LFASyMMQA(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U, 8),
    alpha=DecayByT(8 * ALPHA, weight=ALPHA_D_WEIGHT),
    epsilon=WengEtAl2020MaxMin(),
    gamma=GAMMA,
    env=env,
    features=features,
    eval_config=EVAL_CONFIG,
    max_steps=1000,
    seed=int(rng.integers(2**32)),
)
agent_rraq = LFASyRQA(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U, 8),
    alpha=DecayByT(8 * ALPHA, weight=ALPHA_D_WEIGHT),
    epsilon=WengEtAl2020MaxMin(),
    gamma=GAMMA,
    rho=DecayByT(150, RHO_D_WEIGHT),
    env=env,
    features=features,
    eval_config=EVAL_CONFIG,
    max_steps=1000,
    seed=int(rng.integers(2**32)),
)
agent_avq = LFASyAverageQA(
    theta_config=ThetaConfig(THETA_SIZE, THETA_INIT_L, THETA_INIT_U, 1),
    alpha=DecayByT(ALPHA, weight=ALPHA_D_WEIGHT),
    epsilon=WengEtAl2020MaxMin(),
    gamma=GAMMA,
    k=10,
    env=env,
    features=features,
    eval_config=EVAL_CONFIG,
    max_steps=1000,
    seed=int(rng.integers(2**32)),
)


with Pool(processes=WORKERS) as pool:
    q = pool.apply_async(
        agent_q.run_experiments, (N_EXPERIMENTS, N_EPISODES, THRESHOLD, 0)
    )
    dq = pool.apply_async(
        agent_dq.run_experiments, (N_EXPERIMENTS, N_EPISODES, THRESHOLD, 2)
    )
    mmq = pool.apply_async(
        agent_mmq.run_experiments, (N_EXPERIMENTS, N_EPISODES, THRESHOLD, 4)
    )
    rraq = pool.apply_async(
        agent_rraq.run_experiments, (N_EXPERIMENTS, N_EPISODES, THRESHOLD, 6)
    )
    avq = pool.apply_async(
        agent_avq.run_experiments, (N_EXPERIMENTS, N_EPISODES, THRESHOLD, 10)
    )

    steps_q = q.get()
    steps_dq = dq.get()
    steps_mmq = mmq.get()
    steps_rraq = rraq.get()
    steps_avq = avq.get()

dm = DataManager()


dm.add_raw_data(
    DataContainer(
        "q",
        steps_q,
        {
            "name": "Vanilla Q-Learning",
            "environment": "Cartpole-v0",
            "env_seed": seed_env,
            "thetas": 1,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (episode + {ALPHA_D_WEIGHT})",
            "gamma": GAMMA,
            "n_episodes": N_EPISODES,
            "n_experiments": N_EXPERIMENTS,
            "eval_max_steps": EVAL_CONFIG.timesteps,
            "eval_episodes": EVAL_CONFIG.episodes,
            "eval_frequency": EVAL_CONFIG.frequency,
            "features": "Tiled Cartpole Statespace",
            "epsilon-decay": "WengEtAl2020MaxMin",
            "agent_seed": agent_q.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "dq",
        steps_dq,
        {
            "name": "Double Q-Learning",
            "environment": "Cartpole-v0",
            "env_seed": seed_env,
            "thetas": 2,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": 2 * ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (episode + {ALPHA_D_WEIGHT})",
            "gamma": GAMMA,
            "n_episodes": N_EPISODES,
            "n_experiments": N_EXPERIMENTS,
            "eval_max_steps": EVAL_CONFIG.timesteps,
            "eval_episodes": EVAL_CONFIG.episodes,
            "eval_frequency": EVAL_CONFIG.frequency,
            "features": "Tiled Cartpole Statespace",
            "epsilon-decay": "WengEtAl2020MaxMin",
            "agent_seed": agent_dq.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "mmq",
        steps_mmq,
        {
            "name": "Maxmin Q-Learning",
            "environment": "Cartpole-v0",
            "env_seed": seed_env,
            "thetas": 8,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": 8 * ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (episode + {ALPHA_D_WEIGHT})",
            "gamma": GAMMA,
            "n_episodes": N_EPISODES,
            "n_experiments": N_EXPERIMENTS,
            "eval_max_steps": EVAL_CONFIG.timesteps,
            "eval_episodes": EVAL_CONFIG.episodes,
            "eval_frequency": EVAL_CONFIG.frequency,
            "features": "Tiled Cartpole Statespace",
            "epsilon-decay": "WengEtAl2020MaxMin",
            "agent_seed": agent_mmq.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "rraq",
        steps_rraq,
        {
            "name": "2RA Q-Learning",
            "environment": "Cartpole-v0",
            "env_seed": seed_env,
            "thetas": 8,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": 8 * ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (episode + {ALPHA_D_WEIGHT})",
            "rho": 150,
            "rho_decay_weight": RHO_D_WEIGHT,
            "rho_decay": f"rho * {RHO_D_WEIGHT} / (t + {RHO_D_WEIGHT})",
            "gamma": GAMMA,
            "n_episodes": N_EPISODES,
            "n_experiments": N_EXPERIMENTS,
            "eval_max_steps": EVAL_CONFIG.timesteps,
            "eval_episodes": EVAL_CONFIG.episodes,
            "eval_frequency": EVAL_CONFIG.frequency,
            "features": "Tiled Cartpole Statespace",
            "epsilon-decay": "WengEtAl2020MaxMin",
            "agent_seed": agent_rraq.seed,
        },
    )
)
dm.add_raw_data(
    DataContainer(
        "avq",
        steps_avq,
        {
            "name": "AverageDQn Q-Learning",
            "environment": "Cartpole-v0",
            "env_seed": seed_env,
            "thetas": 1,
            "theta_init": (THETA_INIT_L, THETA_INIT_U),
            "alpha": ALPHA,
            "alpha_decay_weight": ALPHA_D_WEIGHT,
            "alpha_decay": f"alpha * {ALPHA_D_WEIGHT} / (episode + {ALPHA_D_WEIGHT})",
            "K": 10,
            "gamma": GAMMA,
            "n_episodes": N_EPISODES,
            "n_experiments": N_EXPERIMENTS,
            "eval_max_steps": EVAL_CONFIG.timesteps,
            "eval_episodes": EVAL_CONFIG.episodes,
            "eval_frequency": EVAL_CONFIG.frequency,
            "features": "Tiled Cartpole Statespace",
            "epsilon-decay": "WengEtAl2020MaxMin",
            "agent_seed": agent_avq.seed,
        },
    )
)


dm.save_data("data/cartpole")
