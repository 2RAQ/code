from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from agents.base_agents import MSEAgent
from agents.configurations import DEFAULT_EVAL_CONFIG

if TYPE_CHECKING:
    from agents.configurations import EvalConfig, ThetaConfig
    from agents.parameter import DecayParameter
    from environments.base_environment import Environment, Timestep
    from environments.make_features import MakeFeatures


class LFAAsVanillaQA(MSEAgent):
    """Watkins' Q-learning.



    Attributes: Same as the MSEAgent base-class.
    """

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        gamma: float,
        env: Environment,
        features: MakeFeatures,
        eval_config: EvalConfig = DEFAULT_EVAL_CONFIG,
        max_steps: int = 10000,
        seed: int = 1234,
    ) -> None:
        theta_config.n_thetas = 1
        super().__init__(
            theta_config, alpha, gamma, features, env, eval_config, max_steps, seed
        )

    def _set_eval_theta(self) -> None:
        """soon"""
        self.eval_theta = self.theta[:, 0]

    def _evaluation_select_action(self, state: np.ndarray) -> int:
        """soon"""
        return int(np.argmax(state @ self.eval_theta))

    def _squared_error(self) -> float:
        """soon"""
        squared_error = np.linalg.norm(self.theta - self.env.solver.theta)
        return float(squared_error**2)

    def _update_step(
        self,
        alpha: float,
        t: int,
        ts: Timestep,
    ) -> None:
        """soon"""
        q_target = ts.reward
        current_value = ts.state[ts.action] @ self.theta
        if not ts.terminal:
            q_target += self.gamma * np.max(ts.next_state @ self.theta)
        q_update = q_target - current_value
        self.theta[:, 0] += alpha * q_update * ts.state[ts.action]


class LFAAsDQA(MSEAgent):
    """soon"""

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        gamma: float,
        env: Environment,
        features: MakeFeatures,
        eval_config: EvalConfig = DEFAULT_EVAL_CONFIG,
        max_steps: int = 10000,
        seed: int = 1234,
    ) -> None:
        """soon"""
        theta_config.n_thetas = 2
        super().__init__(
            theta_config, alpha, gamma, features, env, eval_config, max_steps, seed
        )

    def _set_eval_theta(self) -> None:
        """soon"""
        self.eval_theta = self.theta[:, 0]

    def _evaluation_select_action(self, state: np.ndarray) -> int:
        """soon"""
        return int(np.argmax(state @ self.eval_theta))

    def _squared_error(self) -> float:
        """soon"""
        theta_star = self.env.solver.theta
        squared_error = np.linalg.norm(self.theta[:, 0] - theta_star[:, 0])
        return float(squared_error**2)

    def _update_step(
        self,
        alpha: float,
        t: int,
        ts: Timestep,
    ) -> None:
        """soon"""
        beta = self._rng.choice(self.n_thetas)
        q_target = ts.reward
        current_value = ts.state[ts.action] @ self.theta[:, beta]
        if not ts.terminal:
            ns = ts.next_state
            max_action = np.argmax(ns @ self.theta[:, 1 - beta])
            q_target += self.gamma * (ns @ self.theta[:, beta])[max_action]
        q_update = q_target - current_value
        self.theta[:, beta] += alpha * q_update * ts.state[ts.action]


class LFAAsDQAAL(LFAAsDQA):
    """soon"""

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        gamma: float,
        env: Environment,
        features: MakeFeatures,
        eval_config: EvalConfig = DEFAULT_EVAL_CONFIG,
        max_steps: int = 10000,
        seed: int = 1234,
    ) -> None:
        """soon"""
        super().__init__(
            theta_config,
            alpha,
            gamma,
            env,
            features,
            eval_config,
            max_steps,
            seed,
        )

    def _set_eval_theta(self) -> None:
        """soon"""
        self.eval_theta = np.mean(self.theta, axis=1)

    def _evaluation_select_action(self, state: np.ndarray) -> int:
        """soon"""
        return int(np.argmax(state @ self.eval_theta))

    def _squared_error(self) -> float:
        """soon"""
        theta_star = self.env.solver.theta[:, 0]
        squared_error = np.linalg.norm(np.mean(self.theta, axis=1) - theta_star)
        return float(squared_error**2)


class LFAAsMMQA(MSEAgent):
    """soon"""

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        gamma: float,
        env: Environment,
        features: MakeFeatures,
        eval_config: EvalConfig = DEFAULT_EVAL_CONFIG,
        max_steps: int = 10000,
        seed: int = 1234,
    ) -> None:
        """soon"""
        super().__init__(
            theta_config, alpha, gamma, features, env, eval_config, max_steps, seed
        )

    def _set_eval_theta(self) -> None:
        """soon"""
        self.eval_theta = np.mean(self.theta, axis=1)

    def _evaluation_select_action(self, state: np.ndarray) -> int:
        """soon"""
        return int(np.argmax(state @ self.eval_theta))

    def _squared_error(self) -> float:
        """soon"""
        theta_star = self.env.solver.theta[:, 0]
        squared_error = np.linalg.norm(np.mean(self.theta, axis=1) - theta_star)
        return float(squared_error**2)

    def _next_state_value(
        self, state: np.ndarray, action: int, next_state: np.ndarray
    ) -> np.ndarray:
        """soon"""
        min_theta = np.argmin(state[action] @ self.theta)
        return np.max(next_state @ self.theta[:, min_theta])

    def _update_step(
        self,
        alpha: float,
        t: int,
        ts: Timestep,
    ) -> None:
        """soon"""
        beta = self._rng.choice(self.n_thetas)
        q_target = ts.reward
        current_value = ts.state[ts.action] @ self.theta[:, beta]
        if not ts.terminal:
            q_target += self.gamma * self._next_state_value(
                ts.state, ts.action, ts.next_state
            )
        q_update = q_target - current_value
        self.theta[:, beta] += alpha * q_update * ts.state[ts.action]


class LFAAsRQA(MSEAgent):
    """soon"""

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        gamma: float,
        rho: DecayParameter,
        env: Environment,
        features: MakeFeatures,
        eval_config: EvalConfig = DEFAULT_EVAL_CONFIG,
        max_steps: int = 10000,
        seed: int = 1234,
    ) -> None:
        """soon"""
        super().__init__(
            theta_config, alpha, gamma, features, env, eval_config, max_steps, seed
        )
        self.rho: DecayParameter = rho

    def _set_eval_theta(self) -> None:
        """soon"""
        self.eval_theta = np.mean(self.theta, axis=1)

    def _evaluation_select_action(self, state: np.ndarray) -> int:
        """soon"""
        return int(np.argmax(state @ self.eval_theta))

    def _squared_error(self) -> float:
        """soon"""
        theta_star = self.env.solver.theta[:, 0]
        squared_error = np.linalg.norm(np.mean(self.theta, axis=1) - theta_star)
        return float(squared_error**2)

    def _next_state_value(self, next_state: np.ndarray, rho: float) -> np.ndarray:
        """soon"""
        mean_theta = np.mean(self.theta, axis=1)
        state_values = next_state @ mean_theta
        return np.max(state_values - np.sqrt(rho) * np.linalg.norm(next_state, axis=1))

    def _update_step(
        self,
        alpha: float,
        t: int,
        ts: Timestep,
    ) -> None:
        """soon"""
        beta = self._rng.choice(self.n_thetas)
        q_target = ts.reward
        current_value = ts.state[ts.action] @ self.theta[:, beta]
        if not ts.terminal:
            q_target += self.gamma * self._next_state_value(
                ts.next_state, self.rho.decay(t)
            )
        q_update = q_target - current_value
        self.theta[:, beta] += alpha * q_update * ts.state[ts.action]
