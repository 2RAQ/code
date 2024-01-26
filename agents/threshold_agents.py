from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from agents.base_agents import ThresholdAgent

if TYPE_CHECKING:
    from agents.configurations import EvalConfig, ThetaConfig
    from agents.parameter import DecayParameter
    from environments.base_environment import ENV, Environment, Timestep
    from environments.make_features import MakeFeatures


class LFASyVanillaQA(ThresholdAgent):
    """soon"""

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        epsilon: DecayParameter,
        gamma: float,
        env: ENV | Environment,
        features: MakeFeatures,
        eval_config: EvalConfig,
        max_steps: int = 200,
        seed: int = 1234,
    ) -> None:
        """soon"""
        theta_config.n_thetas = 1
        super().__init__(
            theta_config,
            alpha,
            epsilon,
            gamma,
            env,
            features,
            eval_config,
            max_steps,
            seed,
        )

    def _set_eval_theta(self) -> None:
        """soon"""
        self.eval_theta = self.theta[:, 0]

    def _evaluation_select_action(self, state: np.ndarray) -> int:
        """soon"""
        return int(np.argmax(state @ self.eval_theta))

    def _update_step(self, alpha: float, t: int, ts: Timestep) -> float | None:
        """soon"""
        q_target = ts.reward
        current_value = ts.state[ts.action] @ self.theta
        if not ts.terminal:
            q_target += self.gamma * np.max(ts.next_state @ self.theta)
        q_update = q_target - current_value
        self.theta[:, 0] += alpha * q_update * ts.state[ts.action]
        return q_update**2


class LFASyDQA(ThresholdAgent):
    """soon"""

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        epsilon: DecayParameter,
        gamma: float,
        env: ENV | Environment,
        features: MakeFeatures,
        eval_config: EvalConfig,
        max_steps: int = 200,
        seed: int = 1234,
    ) -> None:
        """soon"""
        theta_config.n_thetas = 2
        super().__init__(
            theta_config,
            alpha,
            epsilon,
            gamma,
            env,
            features,
            eval_config,
            max_steps,
            seed,
        )

    def _set_eval_theta(self) -> None:
        """soon"""
        self.eval_theta = self.theta[:, 0]

    def _evaluation_select_action(self, state: np.ndarray) -> int:
        """soon"""
        return int(np.argmax(state @ self.eval_theta))

    def _update_step(self, alpha: float, t: int, ts: Timestep) -> float | None:
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
        return q_update**2


class LFASyDQAA(LFASyDQA):
    """soon"""

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        epsilon: DecayParameter,
        gamma: float,
        env: ENV | Environment,
        features: MakeFeatures,
        eval_config: EvalConfig,
        max_steps: int = 200,
        seed: int = 1234,
    ) -> None:
        theta_config.n_thetas = 2
        super().__init__(
            theta_config,
            alpha,
            epsilon,
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


class LFASyMMQA(ThresholdAgent):
    """soon"""

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        epsilon: DecayParameter,
        gamma: float,
        env: ENV | Environment,
        features: MakeFeatures,
        eval_config: EvalConfig,
        max_steps: int = 200,
        seed: int = 1234,
    ) -> None:
        """soon"""
        super().__init__(
            theta_config,
            alpha,
            epsilon,
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

    def _next_state_value(
        self, state: np.ndarray, action: int, next_state: np.ndarray
    ) -> np.ndarray:
        """soon"""
        min_theta = np.argmin(state[action] @ self.theta)
        return np.max(next_state @ self.theta[:, min_theta])

    def _update_step(self, alpha: float, t: int, ts: Timestep) -> float | None:
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
        return q_update**2


class LFASyRQA(ThresholdAgent):
    """soon"""

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        epsilon: DecayParameter,
        gamma: float,
        rho: DecayParameter,
        env: ENV | Environment,
        features: MakeFeatures,
        eval_config: EvalConfig,
        max_steps: int = 200,
        seed: int = 1234,
    ) -> None:
        """soon"""
        super().__init__(
            theta_config,
            alpha,
            epsilon,
            gamma,
            env,
            features,
            eval_config,
            max_steps,
            seed,
        )
        self.rho: DecayParameter = rho

    def _set_eval_theta(self) -> None:
        """soon"""
        self.eval_theta = np.mean(self.theta, axis=1)

    def _evaluation_select_action(self, state: np.ndarray) -> int:
        """soon"""
        return int(np.argmax(state @ self.eval_theta))

    def _next_state_value(self, next_state: np.ndarray, rho: float) -> np.ndarray:
        """soon"""
        mean_theta = np.mean(self.theta, axis=1)
        state_values = next_state @ mean_theta
        return np.max(state_values - np.sqrt(rho) * np.linalg.norm(next_state, axis=1))

    def _update_step(self, alpha: float, t: int, ts: Timestep) -> float | None:
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
        return q_update**2
