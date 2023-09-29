from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from agents.dqn_base_agent import NNAgent

if TYPE_CHECKING:
    from agents.configurations import EvalConfig
    from agents.parameter import DecayParameter
    from environments.utils import ENV, Timestep


class VanillaDQA(NNAgent):
    """soon"""

    def __init__(
        self,
        n_weights: int,
        alpha: float,
        epsilon: DecayParameter,
        gamma: float,
        env: ENV,
        eval_config: EvalConfig,
        max_steps: int = 1000,
        seed: int = 1234,
    ) -> None:
        n_weights = 1
        super().__init__(
            n_weights,
            alpha,
            epsilon,
            gamma,
            env,
            eval_config,
            max_steps,
            seed,
        )

    def _update_step(self, t: int, ts: Timestep) -> float:
        """soon"""
        q_target = self.models[0](ts.state).numpy()
        q_update = ts.reward
        if not ts.terminal:
            next_values = self.models[0](ts.next_state)
            q_update += self.gamma * np.max(next_values)
        q_target[0, ts.action] = q_update
        return self.update_model(self.models[0], ts.state, q_target)


class DDQAA(NNAgent):
    """soon"""

    def __init__(
        self,
        n_weights: int,
        alpha: float,
        epsilon: DecayParameter,
        gamma: float,
        env: ENV,
        eval_config: EvalConfig,
        max_steps: int = 1000,
        seed: int = 1234,
    ) -> None:
        n_weights = 2
        super().__init__(
            n_weights,
            alpha,
            epsilon,
            gamma,
            env,
            eval_config,
            max_steps,
            seed,
        )

    def _update_step(self, t: int, ts: Timestep) -> float:
        """soon"""
        beta = self._rng.choice(self.n_weights)
        q_target = self.models[beta](ts.state).numpy()
        q_update = ts.reward
        if not ts.terminal:
            next_values = self.models[1 - beta](ts.next_state)
            q_update += self.gamma * np.max(next_values)
        q_target[0, ts.action] = q_update
        return self.update_model(self.models[beta], ts.state, q_target)


class MMDQA(NNAgent):
    """soon"""

    def __init__(
        self,
        n_weights: int,
        alpha: float,
        epsilon: DecayParameter,
        gamma: float,
        env: ENV,
        eval_config: EvalConfig,
        max_steps: int = 1000,
        seed: int = 1234,
    ) -> None:
        super().__init__(
            n_weights,
            alpha,
            epsilon,
            gamma,
            env,
            eval_config,
            max_steps,
            seed,
        )

    def _next_state_value(
        self, state: np.ndarray, action: int, next_state: np.ndarray
    ) -> np.ndarray:
        """soon"""
        values = np.zeros(self.n_weights.shape)
        for i in self.n_weights:
            values[i] = self.models[i](state).numpy()[0, action]
        min_weight = int(np.argmin(values))
        return np.max(self.models[min_weight](next_state).numpy())

    def _update_step(self, t: int, ts: Timestep) -> float:
        """soon"""
        beta = self._rng.choice(self.n_weights)
        q_target = self.models[beta](ts.state).numpy()
        q_update = ts.reward
        if not ts.terminal:
            q_update += self.gamma * self._next_state_value(
                ts.state, ts.action, ts.next_state
            )
        q_target[0, ts.action] = q_update
        return self.update_model(self.models[beta], ts.state, q_target)


class RRADQA(NNAgent):
    """soon"""

    def __init__(
        self,
        n_weights: int,
        alpha: float,
        epsilon: DecayParameter,
        gamma: float,
        rho: DecayParameter,
        env: ENV,
        eval_config: EvalConfig,
        max_steps: int = 1000,
        seed: int = 1234,
    ) -> None:
        super().__init__(
            n_weights,
            alpha,
            epsilon,
            gamma,
            env,
            eval_config,
            max_steps,
            seed,
        )
        self.rho: DecayParameter = rho

    def _next_state_value(self, next_state: np.ndarray, rho: float) -> np.ndarray:
        """soon"""
        self._get_eval_model()
        mean_value_prediction = self.eval_model(next_state).numpy()
        return np.max(
            mean_value_prediction - np.sqrt(rho) * np.linalg.norm(next_state, axis=1)
        )

    def _update_step(self, t: int, ts: Timestep) -> float:
        """soon"""
        beta = self._rng.choice(self.n_weights)
        q_target = self.models[beta](ts.state).numpy()
        q_update = ts.reward
        if not ts.terminal:
            q_update += self.gamma * self._next_state_value(
                ts.next_state, self.rho.decay(t)
            )
        q_target[0, ts.action] = q_update
        return self.update_model(self.models[beta], ts.state, q_target)
