from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from agents.base_agents import ThresholdAgent

if TYPE_CHECKING:
    from agents.configurations import (
        EvalConfig,
        ThetaConfig,
    )
    from agents.parameter import DecayParameter
    from environments.base_environment import (
        Environment,
        Timestep,
    )
    from environments.gym_envs import IntegratedCartPole
    from environments.make_features import MakeFeatures


class LFASyVanillaQA(ThresholdAgent):
    """Watkins' Q-learning - C. Watkins (1989).

    Attributes: Same as the ThresholdAgent base-class.
    """

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        epsilon: DecayParameter,
        gamma: float,
        env: IntegratedCartPole | Environment,
        features: MakeFeatures,
        eval_config: EvalConfig,
        max_steps: int = 200,
        seed: int = 1234,
    ) -> None:
        """Initialises the agent and overwrites any theta configuration with a
        single theta as required for Watkins' Q-learning.
        """
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
        """Returns the single existing theta for evaluation."""
        self.eval_theta = self.theta[:, 0]

    def _evaluation_select_action(self, state: np.ndarray) -> int:
        """Selects greedy action based in highest Q-value."""
        return int(np.argmax(state @ self.eval_theta))

    def _update_step(self, alpha: float, t: int, ts: Timestep) -> float | None:
        """Watkins' Q-learning update step, returns the squared TD-error."""
        q_target = np.array(ts.reward)
        current_value = ts.state[ts.action] @ self.theta
        if not ts.terminal:
            q_target += self.gamma * np.max(ts.next_state @ self.theta)
        q_update = q_target - current_value
        self.theta[:, 0] += alpha * q_update * ts.state[ts.action]
        return q_update**2


class LFASyDQA(ThresholdAgent):
    """Double Q-learning - H. Hasselt (2010).

    Attributes: Same as the ThresholdAgent base-class.
    """

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        epsilon: DecayParameter,
        gamma: float,
        env: IntegratedCartPole | Environment,
        features: MakeFeatures,
        eval_config: EvalConfig,
        max_steps: int = 200,
        seed: int = 1234,
    ) -> None:
        """Initialises the agent and overwrites any theta configuration with two
        thetas as required for Double Q-learning.
        """
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
        """Uses one of the two thetas for evaluation."""
        self.eval_theta = self.theta[:, 0]

    def _evaluation_select_action(self, state: np.ndarray) -> int:
        """Selects greedy action based in highest Q-value estimated by the evaluation
        theta.
        """
        return int(np.argmax(state @ self.eval_theta))

    def _update_step(self, alpha: float, t: int, ts: Timestep) -> float | None:
        """Double Q-learning update step, randomly selects one of the two thetas to
        update. Also returns the squared TD-error.
        """
        beta = self._rng.choice(self.n_thetas)
        q_target = np.array(ts.reward)
        current_value = ts.state[ts.action] @ self.theta[:, beta]
        if not ts.terminal:
            ns = ts.next_state
            max_action = np.argmax(ns @ self.theta[:, 1 - beta])
            q_target += self.gamma * (ns @ self.theta[:, beta])[max_action]
        q_update = q_target - current_value
        self.theta[:, beta] += alpha * q_update * ts.state[ts.action]
        return q_update**2


class LFASyDQAA(LFASyDQA):
    """Double Q-learning - H. Hasselt (2010).

    Difference to the above implementation is that, instead of one of the two
    thetas, the average of the two thetas is used to obtain the evaluation theta.

    Attributes: Same as the ThresholdAgent base-class.
    """

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        epsilon: DecayParameter,
        gamma: float,
        env: IntegratedCartPole | Environment,
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
        """Uses the average of the two thetas for evaluation."""
        self.eval_theta = np.mean(self.theta, axis=1)


class LFASyMMQA(ThresholdAgent):
    """Maxmin Q-learning - Lan et al. (2020).

    Attributes: Same as the ThresholdAgent base-class.
    """

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        epsilon: DecayParameter,
        gamma: float,
        env: IntegratedCartPole | Environment,
        features: MakeFeatures,
        eval_config: EvalConfig,
        max_steps: int = 200,
        seed: int = 1234,
    ) -> None:
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
        """Uses the average of the N thetas for evaluation."""
        self.eval_theta = np.mean(self.theta, axis=1)

    def _evaluation_select_action(self, state: np.ndarray) -> int:
        """Selects greedy action based in highest Q-value estimated by the evaluation
        theta.
        """
        min_values = np.min(state @ self.theta, axis=1)
        return int(np.argmax(min_values))

    def _next_state_value(
        self, state: np.ndarray, next_state: np.ndarray
    ) -> np.ndarray:
        """Returns the maximum Q-value obtained from the set of minimum Q-values
        as opbtained from the estimates of the N thetas for each action at the
        given state.
        """
        min_thetas = np.argmin(state @ self.theta, axis=1)
        next_state_values = np.diagonal(next_state @ self.theta[:, min_thetas])
        return np.max(next_state_values)

    def _update_step(
        self,
        alpha: float,
        t: int,
        ts: Timestep,
    ) -> None:
        """Maxmin Q-learning update step where one of the N thetas is radomly chosen
        to update at each step. Also returns the squared TD-error.
        """
        beta = self._rng.choice(self.n_thetas)
        q_target = np.array(ts.reward)
        current_value = ts.state[ts.action] @ self.theta[:, beta]
        if not ts.terminal:
            q_target += self.gamma * self._next_state_value(ts.state, ts.next_state)
        q_update = q_target - current_value
        self.theta[:, beta] += alpha * q_update * ts.state[ts.action]
        return q_update**2


class LFASyRQA(ThresholdAgent):
    """2RA Q-learning.

    Additional Attributes:
        rho: DecayParameter - The decay parameter rho value as used in the 2RA
            Q-learning algorithm.
    """

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        epsilon: DecayParameter,
        gamma: float,
        rho: DecayParameter,
        env: IntegratedCartPole | Environment,
        features: MakeFeatures,
        eval_config: EvalConfig,
        max_steps: int = 200,
        seed: int = 1234,
    ) -> None:
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
        """Uses the average of the N thetas for evaluation."""
        self.eval_theta = np.mean(self.theta, axis=1)

    def _evaluation_select_action(self, state: np.ndarray) -> int:
        """Selects greedy action based in highest Q-value estimated by the evaluation
        theta.
        """
        return int(np.argmax(state @ self.eval_theta))

    def _next_state_value(self, next_state: np.ndarray, rho: float) -> np.ndarray:
        """Calculates the closed form estimator as described in the paper.

        Returns the action corresponding to the maximum Q-value given that a
        minimizing estimator is chosen from the ambiguity set centered around the
        mean theta.
        """
        mean_theta = np.mean(self.theta, axis=1)
        state_values = next_state @ mean_theta
        return np.max(state_values - np.sqrt(rho) * np.linalg.norm(next_state, axis=1))

    def _update_step(self, alpha: float, t: int, ts: Timestep) -> float | None:
        """2RA Q-learning update step where one of the N thetas is radomly chosen
        to update at each step. Also returns the squared TD-error.
        """
        beta = self._rng.choice(self.n_thetas)
        q_target = np.array(ts.reward)
        current_value = ts.state[ts.action] @ self.theta[:, beta]
        if not ts.terminal:
            q_target += self.gamma * self._next_state_value(
                ts.next_state, self.rho.decay(t)
            )
        q_update = q_target - current_value
        self.theta[:, beta] += alpha * q_update * ts.state[ts.action]
        return q_update**2


class LFASyAverageQA(ThresholdAgent):
    """Averaged Q-learning - Amschel et al. (2017).

    Additional Attributes:
        k: The number of past thetas to average over.
        theta_trace: A deque to store the last k thetas.
    """

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        epsilon: DecayParameter,
        gamma: float,
        k: int,
        env: Environment,
        features: MakeFeatures,
        eval_config: EvalConfig,
        max_steps: int = 10000,
        seed: int = 1234,
    ) -> None:
        """Initialises the agent and overwrites any theta configuration with two
        thetas as required for Averaged Q-learning.
        """
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
        self.k: int = k
        self.theta_trace: deque[np.ndarray] = deque(maxlen=k)
        self.theta_trace.append(self.theta.copy())

    def _set_eval_theta(self) -> None:
        """Uses the average of the past k thetas for evaluation."""
        self.eval_theta = np.mean(self.theta_trace, axis=0)

    def _evaluation_select_action(self, state: np.ndarray) -> int:
        """Selects greedy action based in highest Q-value estimated by the evaluation
        theta.
        """
        return int(np.argmax(state @ self.eval_theta))

    def _update_step(
        self,
        alpha: float,
        t: int,
        ts: Timestep,
    ) -> None:
        """Averaged Q-learning update step, the current values of theta are stored
        in the history and then theta is updated. Also returns the squared TD-error.
        """
        self.theta_trace.append(self.theta.copy())
        q_target = np.array(ts.reward)
        current_value = ts.state[ts.action] @ self.theta
        if not ts.terminal:
            q_target += self.gamma * np.max(
                ts.next_state @ np.mean(self.theta_trace, axis=0)
            )
        q_update = q_target - current_value
        self.theta[:, 0] += alpha * q_update * ts.state[ts.action]
        return q_update**2


class LFAAyREDQA(ThresholdAgent):
    """RED Q-learning - Chen  et al. (2021).

    Additional Attributes:
        g: The number updates to perform in each update step.
        m: The number of theta indices to sample for the update step.
    """

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        epsilon: DecayParameter,
        gamma: float,
        g: int,
        m: int,
        env: Environment,
        features: MakeFeatures,
        eval_config: EvalConfig,
        max_steps: int = 10000,
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
        self.g: int = g
        self.m: int = m

    def _set_eval_theta(self) -> None:
        """Uses the average of the N thetas for evaluation."""
        self.eval_theta = np.mean(self.theta, axis=1)

    def _evaluation_select_action(self, state: np.ndarray) -> int:
        """Selects greedy action based in highest Q-value estimated by the evaluation
        theta.
        """
        return int(np.argmax(state @ self.eval_theta))

    def _next_state_value(
        self, next_state: np.ndarray, betas: np.ndarray
    ) -> np.ndarray:
        """Returns the minimum Q-value from the set obtained of all maximum
        Q-values as estimated by the M randomly sampled thetas.
        """
        value_estimates = next_state @ self.theta[:, betas]
        max_values_indices = np.argmax(value_estimates, axis=0)
        return np.min(value_estimates[max_values_indices, range(self.m)])

    def _update_step(
        self,
        alpha: float,
        t: int,
        ts: Timestep,
    ) -> None:
        """RED Q-learning update step where, g times, M randomly sampled thetas
        out of the set of N thetas are used to generate the update value.
        """
        for _ in range(self.g):
            betas = self._rng.choice(self.n_thetas, self.m, replace=False)
            q_target = np.array(ts.reward)
            if not ts.terminal:
                q_target += self.gamma * self._next_state_value(ts.next_state, betas)
            current_values = ts.state[ts.action] @ self.theta
            q_update = (q_target - current_values)[np.newaxis, :]
            update_mask = ts.state[ts.action, np.newaxis]
            self.theta += alpha * update_mask.T * q_update
        return q_update**2
