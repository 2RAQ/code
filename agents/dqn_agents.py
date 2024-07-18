from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from agents.dqn_base_agent import NNAgent

if TYPE_CHECKING:
    from gym import Env

    from agents.configurations import EvalConfig
    from agents.parameter import DecayParameter
    from environments.base_environment import Timestep


class VanillaDQA(NNAgent):
    """Watkins' Q-learning - C. Watkins (1989).

    Attributes: Same as the ThresholdAgent base-class.
    """

    def __init__(
        self,
        n_weights: int,
        alpha: float,
        epsilon: DecayParameter,
        gamma: float,
        env: Env,
        eval_config: EvalConfig,
        max_steps: int = 1000,
        seed: int = 1234,
    ) -> None:
        """Initialises the agent and overwrites the passed n_weights with one
        as required for Watkins' Q-learning.
        """
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
        """Watkins' Q-learning update step, returns the loss of the update."""
        q_target = self.models[0](ts.state).numpy()
        q_update = np.array(ts.reward)
        if not ts.terminal:
            next_values = self.models[0](ts.next_state)
            q_update += self.gamma * np.max(next_values)
        q_target[0, ts.action] = q_update
        return self.update_model(self.models[0], ts.state, q_target)


class DDQAA(NNAgent):
    """Double Q-learning - H. Hasselt (2010).

    Attributes: Same as the ThresholdAgent base-class.
    """

    def __init__(
        self,
        n_weights: int,
        alpha: float,
        epsilon: DecayParameter,
        gamma: float,
        env: Env,
        eval_config: EvalConfig,
        max_steps: int = 1000,
        seed: int = 1234,
    ) -> None:
        """Initialises the agent and overwrites the passed n_weights with two
        as required for Double Q-learning.
        """
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
        """Double Q-learning update step, randomly selects one of the two parameter
        sets to update. Also returns the loss of the update.
        """
        beta = self._rng.choice(self.n_weights)
        q_target = self.models[beta](ts.state).numpy()
        q_update = np.array(ts.reward)
        if not ts.terminal:
            next_values = self.models[1 - beta](ts.next_state)
            q_update += self.gamma * np.max(next_values)
        q_target[0, ts.action] = q_update
        return self.update_model(self.models[beta], ts.state, q_target)


class MMDQA(NNAgent):
    """Maxmin Q-learning - Lan et al. (2020).

    Attributes: Same as the ThresholdAgent base-class.
    """

    def __init__(
        self,
        n_weights: int,
        alpha: float,
        epsilon: DecayParameter,
        gamma: float,
        env: Env,
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
        """Returns the maximum Q-value obtained from the set of minimum Q-values
        as opbtained from the estimates of the N parameter sets for each action
        at the given state.
        """
        values = np.zeros(self.n_weights.shape)
        for i in self.n_weights:
            values[i] = self.models[i](state).numpy()[0, action]
        min_weight = int(np.argmin(values))
        return np.max(self.models[min_weight](next_state).numpy())

    def _update_step(self, t: int, ts: Timestep) -> float:
        """Maxmin Q-learning update step where one of the N parameter sets is
        radomly chosen to update at each step. Also returns the loss of the update.
        """
        beta = self._rng.choice(self.n_weights)
        q_target = self.models[beta](ts.state).numpy()
        q_update = np.array(ts.reward)
        if not ts.terminal:
            q_update += self.gamma * self._next_state_value(
                ts.state, ts.action, ts.next_state
            )
        q_target[0, ts.action] = q_update
        return self.update_model(self.models[beta], ts.state, q_target)


class RRADQA(NNAgent):
    """2RA Q-learning.

    Additional Attributes:
        rho: DecayParameter - The decay parameter rho value as used in the 2RA
            Q-learning algorithm.
    """

    def __init__(
        self,
        n_weights: int,
        alpha: float,
        epsilon: DecayParameter,
        gamma: float,
        rho: DecayParameter,
        env: Env,
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
        """Calculates the closed form estimator as described in the paper.

        Returns the action corresponding to the maximum Q-value given that a
        minimizing estimator is chosen from the ambiguity set centered around the
        mean theta.
        """
        self._get_eval_model()
        mean_value_prediction = self.eval_model(next_state).numpy()
        return np.max(
            mean_value_prediction - np.sqrt(rho) * np.linalg.norm(next_state, axis=1)
        )

    def _update_step(self, t: int, ts: Timestep) -> float:
        """2RA Q-learning update step where one of the N sets of parameters is
        radomly chosen to update at each step. Also returns the loss of the update.
        """
        beta = self._rng.choice(self.n_weights)
        q_target = self.models[beta](ts.state).numpy()
        q_update = np.array(ts.reward)
        if not ts.terminal:
            q_update += self.gamma * self._next_state_value(
                ts.next_state, self.rho.decay(t)
            )
        q_target[0, ts.action] = q_update
        return self.update_model(self.models[beta], ts.state, q_target)
