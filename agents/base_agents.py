from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from absl import logging
from tqdm import tqdm

from environments.base_environment import Timestep
from environments.make_features import MakeFeatures

if TYPE_CHECKING:
    from agents.configurations import EvalConfig, ThetaConfig
    from agents.parameter import DecayParameter
    from environments.base_environment import ENV, Environment


class Agent:
    """soon"""

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        gamma: float,
        features: MakeFeatures,
        env: ENV | Environment,
        eval_config: EvalConfig,
        max_steps: int,
        seed: int,
    ) -> None:
        """soon"""
        self._rng: np.random._generator.Generator = np.random.default_rng(seed)
        self.alpha: DecayParameter = alpha
        self.gamma: float = gamma
        self.theta_config: ThetaConfig = theta_config
        self.n_thetas: np.ndarray = np.arange(theta_config.n_thetas)
        self.theta: np.ndarray = self._init_theta()
        self.eval_theta: np.ndarray = np.zeros(theta_config.n_thetas)
        self.features: MakeFeatures = features
        self.env: ENV | Environment = env
        self.eval: EvalConfig = eval_config
        self.seed: int = seed
        self.max_timesteps: int = max_steps

    def _init_theta(self) -> np.ndarray:
        """soon"""
        tc = self.theta_config
        return (tc.init_upper - tc.init_lower) * self._rng.random(
            (tc.dimensions, tc.n_thetas)
        ) + tc.init_lower

    @abstractmethod
    def _set_eval_theta(self) -> None:
        """soon"""
        pass

    def _reset_model(self) -> None:
        """soon"""
        self.theta = self._init_theta()

    def reset_agent(self, reset_rng: bool = False) -> None:
        """soon"""
        if reset_rng:
            self._set_seed(self.seed)
        self.env.reset()
        logging.info(
            f"Resetting {self.__class__.__name__} and {self.env.__class__.__name__}"
        )

    def _env_reset(self) -> np.ndarray:
        """soon"""
        return self.features.make_features(self.env.reset())

    def _set_seed(self, seed: None | int = None) -> None:
        """soon"""
        if not seed:
            seed = self._rng.integers(2**32)
        self._rng = np.random.default_rng(seed=seed)

    def _env_step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """soon"""
        next_state, reward, terminal = self.env.step(action)
        return self.features.make_features(next_state), reward, terminal

    def _timestep(self, state: np.ndarray) -> Timestep:
        """soon"""
        action = self._select_action(state)
        next_state, reward, terminal = self._env_step(action)
        return Timestep(
            state,
            action,
            reward,
            next_state,
            terminal,
        )

    @abstractmethod
    def _select_action(self, state: np.ndarray) -> int:
        """soon"""
        pass

    @abstractmethod
    def _evaluation_select_action(self, state: np.ndarray) -> int:
        """soon"""
        pass

    def _evaluation_timestep(self, state: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """soon"""
        action = self._evaluation_select_action(state)
        return self._env_step(action)

    def _evaluation_episode(self) -> float:
        """soon"""
        sum_rewards = 0.0
        state = self._env_reset()
        for _ in range(self.eval.timesteps):
            state, reward, terminal = self._evaluation_timestep(state)
            sum_rewards += reward
            if terminal:
                break
        return sum_rewards

    def evaluation(self) -> float:
        """soon"""
        self._set_eval_theta()
        rewards = np.zeros(self.eval.episodes)
        for t in range(self.eval.episodes):
            rewards[t] = self._evaluation_episode()
        return float(np.mean(rewards))

    @abstractmethod
    def _update_step(
        self,
        alpha: float,
        t: int,
        ts: Timestep,
    ) -> float | None:
        """soon"""
        pass


class MSEAgent(Agent):
    """This agent compares the learned estimators with the optimal estimator as
    obtained from the solver within each environment. The behavioural policy, which
    creates the trajectorie on which updates are performed are obtained from random
    uniform actions. This is possible since the synthetic MSE agents allow the
    entire state space to be visited by a random policy.
    """

    squared_errors: np.ndarray
    mse: np.ndarray
    se: np.ndarray
    mean_rewards: np.ndarray
    se_rewards: np.ndarray

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        gamma: float,
        features: MakeFeatures,
        env: Environment,
        eval_config: EvalConfig,
        max_steps: int,
        seed: int,
    ) -> None:
        """soon"""
        super().__init__(
            theta_config, alpha, gamma, features, env, eval_config, max_steps, seed
        )
        self.samples: list[Timestep] = []
        self.eval_rewards: np.ndarray = np.zeros(0)

    def _select_action(self, state: np.ndarray) -> int:
        """soon"""
        return self._rng.choice(self.env.actions)

    @abstractmethod
    def _squared_error(self) -> float:
        """soon"""
        pass

    def _gen_samples(self, n_samples: int) -> None:
        """soon"""
        self.samples.clear()
        state = self._env_reset()
        for n in range(n_samples):
            timestep = self._timestep(state)
            self.samples.append(timestep)
            state = timestep.next_state
            if timestep.terminal or (n + 1) % self.max_timesteps == 0:
                state = self._env_reset()

    def train_model(self, experiment: int, repetition: int) -> None:
        """soon"""
        self._reset_model()
        for t in range(len(self.samples)):
            self.squared_errors[experiment, repetition, t] = self._squared_error()
            alpha = self.alpha.decay(t)
            self._update_step(alpha, t, self.samples[t])
            if self.eval.timesteps > 0 and t % self.eval.frequency == 0:
                t_index = t // self.eval.frequency
                self.eval_rewards[experiment, repetition, t_index] = self.evaluation()
        self.squared_errors[experiment, repetition, -1] = self._squared_error()

    def experiment(
        self,
        n_samples: int,
        experiment: int,
        n_repetitions: int,
        pos: int,
    ) -> None:
        """Runs the experiment and sets new transition dynamics if available
        else it simply resets the environment and the next run is performed on
        the same transition dynamics.
        """
        for repetition in (
            repetitions := tqdm(range(n_repetitions), position=pos + 1, leave=False)
        ):
            repetitions.set_description(
                f"{self.__class__.__name__} Repetition for experiment {experiment}"
            )
            self._gen_samples(n_samples)
            self.train_model(experiment, repetition)

    def run_experiments(
        self, n_samples: int, n_experiments: int, n_repetitions: int = 1, pos: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """soon"""
        logging.debug(
            f"Agent {self.__class__.__name__} "
            f"running {n_experiments} experiements with {n_samples} samples "
            f" and {n_repetitions} repetitions each."
        )
        self.squared_errors = np.zeros((n_experiments, n_repetitions, n_samples + 1))
        if self.eval.timesteps > 0:
            self.eval_rewards = np.zeros(
                (n_experiments, n_repetitions, n_samples // self.eval.frequency)
            )
        for experiment in (
            experiments := tqdm(range(n_experiments), position=pos, leave=False)
        ):
            experiments.set_description(f"{self.__class__.__name__} Experiment")
            self.experiment(n_samples, experiment, n_repetitions, pos)
            self.env.set_new_transition_dynamics()
        return self.squared_errors, self.eval_rewards


class ThresholdAgent(Agent):
    """This agent needs exploration as not all states in the environment can be
    visited by random play. The trajectory on which the estimators are updates
    are therefore obtained on an epsilon-greedy policy based on the current
    estimator.
    """

    steps_to_threshold: np.ndarray

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        epsilon: DecayParameter,
        gamma: float,
        env: ENV | Environment,
        features: MakeFeatures,
        eval_config: EvalConfig,
        max_steps: int,
        seed: int,
    ) -> None:
        """soon"""
        super().__init__(
            theta_config, alpha, gamma, features, env, eval_config, max_steps, seed
        )
        self.epsilon: DecayParameter = epsilon
        self.env_actions: np.ndarray = np.arange(env.action_space.n)
        self.mean_rewards: list[list[float]] = []
        self.td_errors: list[list[float]] = []

    def _env_step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """soon"""
        next_state, reward, terminal, _ = self.env.step(action)
        return self.features.make_features(next_state), reward, terminal

    def _select_action(self, state: np.ndarray) -> int:
        """soon"""
        if self._rng.random() < self.epsilon.value:
            action = int(self._rng.choice(self.env_actions))
        else:
            self._set_eval_theta()
            action = self._evaluation_select_action(state)
        return action

    def _episode(self, e: int, t: int) -> tuple[int, float]:
        """soon"""
        state = self._env_reset()
        for _ in range(self.max_timesteps):
            ts = self._timestep(state)
            td_squared_error = self._update_step(self.alpha.decay(e), t, ts)
            t += 1
            state = ts.next_state
            if ts.terminal:
                break
        return t, td_squared_error  # type: ignore[return-value]

    def _experiment(self, experiment: int, episodes: int, threshold: int) -> int:
        """soon"""
        self._reset_model()
        t = 0
        for episode in range(episodes):
            t, td_squared_error = self._episode(episode, t)
            self.epsilon.decay(episode)
            self.td_errors[experiment].append(td_squared_error)
            if episode % self.eval.frequency == 0:
                eval_reward = self.evaluation()
                self.mean_rewards[experiment].append(eval_reward)
                if eval_reward >= threshold:
                    break
        return episode + 1

    def run_experiment(
        self,
        n_experiments: int,
        episodes_per_experiment: int,
        threshold: int,
        pos: int = 0,
    ) -> np.ndarray:
        """soon"""
        self.steps_to_threshold = np.zeros(n_experiments)
        for e in (experiments := tqdm(range(n_experiments), position=pos)):
            experiments.set_description(f"{self.__class__.__name__} Experiment")
            self.mean_rewards.append([])
            self.td_errors.append([])
            self.steps_to_threshold[e] = self._experiment(
                e, episodes_per_experiment, threshold
            )
        return self.steps_to_threshold
