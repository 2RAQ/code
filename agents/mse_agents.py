from __future__ import annotations

from collections import (
    defaultdict,
    deque,
)
from typing import TYPE_CHECKING

import numpy as np

from agents.base_agents import MSEAgent
from agents.configurations import DEFAULT_EVAL_CONFIG

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
    from environments.make_features import MakeFeatures


class LFAAsVanillaQA(MSEAgent):
    """Watkins' Q-learning - C. Watkins (1989).

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
        """Initialises the agent and overwrites any theta configuration with a
        single theta as required for Watkins' Q-learning.
        """

        theta_config.n_thetas = 1
        super().__init__(
            theta_config, alpha, gamma, features, env, eval_config, max_steps, seed
        )

    def _set_eval_theta(self) -> None:
        """Returns the single existing theta for evaluation."""
        self.eval_theta = self.theta[:, 0]

    def _evaluation_select_action(self, state: np.ndarray) -> int:
        """Selects greedy action based in highest Q-value."""
        return int(np.argmax(state @ self.eval_theta))

    def _squared_error(self) -> float:
        """Returns the squared error between the agents theta and the optimal theta
        as obtained from the solved environment.
        """
        squared_error = np.linalg.norm(self.theta - self.env.solver.theta)
        return float(squared_error**2)

    def _update_step(
        self,
        alpha: float,
        t: int,
        ts: Timestep,
    ) -> None:
        """Watkins' Q-learning update step"""
        q_target = np.array(ts.reward)
        current_value = ts.state[ts.action] @ self.theta
        if not ts.terminal:
            q_target += self.gamma * np.max(ts.next_state @ self.theta)
        q_update = q_target - current_value
        self.theta[:, 0] += alpha * q_update * ts.state[ts.action]


class LFAAsDQA(MSEAgent):
    """Double Q-learning - H. Hasselt (2010).

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
        """Initialises the agent and overwrites any theta configuration with two
        thetas as required for Double Q-learning.
        """
        theta_config.n_thetas = 2
        super().__init__(
            theta_config, alpha, gamma, features, env, eval_config, max_steps, seed
        )

    def _set_eval_theta(self) -> None:
        """Uses one of the two thetas for evaluation."""
        self.eval_theta = self.theta[:, 0]

    def _evaluation_select_action(self, state: np.ndarray) -> int:
        """Selects greedy action based in highest Q-value estimated by the evaluation
        theta.
        """
        return int(np.argmax(state @ self.eval_theta))

    def _squared_error(self) -> float:
        """Returns the squared error between the agents first theta and the
        optimal theta as obtained from the solved environment.
        """
        theta_star = self.env.solver.theta
        squared_error = np.linalg.norm(self.theta[:, 0] - theta_star[:, 0])
        return float(squared_error**2)

    def _update_step(
        self,
        alpha: float,
        t: int,
        ts: Timestep,
    ) -> None:
        """Double Q-learning update step, randomly selects one of the two thetas to
        update.
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


class LFAAsDQAAL(LFAAsDQA):
    """Double Q-learning - H. Hasselt (2010).

    Difference to the above implementation is that, instead of one of the two
    thetas, the average of the two thetas is used to obtain the evaluation theta.

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
        """Uses the average of the two thetas for evaluation."""
        self.eval_theta = np.mean(self.theta, axis=1)

    def _squared_error(self) -> float:
        """Returns the squared error between the agents mean theta and the
        optimal theta as obtained from the solved environment.
        """
        theta_star = self.env.solver.theta[:, 0]
        squared_error = np.linalg.norm(np.mean(self.theta, axis=1) - theta_star)
        return float(squared_error**2)


class LFAAsMMQA(MSEAgent):
    """Maxmin Q-learning - Lan et al. (2020).

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
        super().__init__(
            theta_config, alpha, gamma, features, env, eval_config, max_steps, seed
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

    def _squared_error(self) -> float:
        """Returns the squared error between the agents mean theta and the
        optimal theta as obtained from the solved environment.
        """
        theta_star = self.env.solver.theta[:, 0]
        squared_error = np.linalg.norm(np.mean(self.theta, axis=1) - theta_star)
        return float(squared_error**2)

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
        to update at each step.
        """
        beta = self._rng.choice(self.n_thetas)
        q_target = np.array(ts.reward)
        current_value = ts.state[ts.action] @ self.theta[:, beta]
        if not ts.terminal:
            q_target += self.gamma * self._next_state_value(ts.state, ts.next_state)
        q_update = q_target - current_value
        self.theta[:, beta] += alpha * q_update * ts.state[ts.action]


class LFAAsRQA(MSEAgent):
    """2RA Q-learning.

    Additional Attributes:
        rho: DecayParameter - The decay parameter rho value as used in the 2RA
            Q-learning algorithm.
    """

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
        super().__init__(
            theta_config, alpha, gamma, features, env, eval_config, max_steps, seed
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

    def _squared_error(self) -> float:
        """Returns the squared error between the agents mean theta and the
        optimal theta as obtained from the solved environment.
        """
        theta_star = self.env.solver.theta[:, 0]
        squared_error = np.linalg.norm(np.mean(self.theta, axis=1) - theta_star)
        return float(squared_error**2)

    def _next_state_value(self, next_state: np.ndarray, rho: float) -> np.ndarray:
        """Calculates the closed form estimator as described in the paper.

        Returns the action corresponding to the maximum Q-value given that a
        minimizing estimator is chosen from the ambiguity set centered around the
        mean theta.
        """
        mean_theta = np.mean(self.theta, axis=1)
        state_values = next_state @ mean_theta
        return np.max(state_values - np.sqrt(rho) * np.linalg.norm(next_state, axis=1))

    def _update_step(
        self,
        alpha: float,
        t: int,
        ts: Timestep,
    ) -> None:
        """2RA Q-learning update step where one of the N thetas is radomly chosen
        to update at each step.
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


class LFAAsAverageQA(MSEAgent):
    """Averaged Q-learning - Amschel et al. (2017).

    Additional Attributes:
        k: The number of past thetas to average over.
        theta_trace: A deque to store the last k thetas.
    """

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        gamma: float,
        k: int,
        env: Environment,
        features: MakeFeatures,
        eval_config: EvalConfig = DEFAULT_EVAL_CONFIG,
        max_steps: int = 10000,
        seed: int = 1234,
    ) -> None:
        """Initialises the agent and overwrites any theta configuration with two
        thetas as required for Averaged Q-learning.
        """
        theta_config.n_thetas = 1
        super().__init__(
            theta_config, alpha, gamma, features, env, eval_config, max_steps, seed
        )
        self.k: int = k
        self.theta_trace: deque[np.ndarray] = deque(maxlen=k)

    def _set_eval_theta(self) -> None:
        """Uses the average of the past k thetas for evaluation."""
        self.eval_theta = np.mean(self.theta_trace, axis=0)

    def _evaluation_select_action(self, state: np.ndarray) -> int:
        """Selects greedy action based in highest Q-value estimated by the evaluation
        theta.
        """
        return int(np.argmax(state @ self.eval_theta))

    def _squared_error(self) -> float:
        """Returns the squared error between the agents current theta and the
        optimal theta as obtained from the solved environment.
        """
        squared_error = np.linalg.norm(self.theta - self.env.solver.theta)
        return float(squared_error**2)

    def _update_step(
        self,
        alpha: float,
        t: int,
        ts: Timestep,
    ) -> None:
        """Averaged Q-learning update step, the current values of theta are stored
        in the history and then theta is updated.
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


class LFAAsREDQA(MSEAgent):
    """RED Q-learning - Chen  et al. (2021).

    Additional Attributes:
        g: The number updates to perform in each update step.
        m: The number of theta indices to sample for the update step.
    """

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        gamma: float,
        g: int,
        m: int,
        env: Environment,
        features: MakeFeatures,
        eval_config: EvalConfig = DEFAULT_EVAL_CONFIG,
        max_steps: int = 10000,
        seed: int = 1234,
    ) -> None:
        super().__init__(
            theta_config, alpha, gamma, features, env, eval_config, max_steps, seed
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

    def _squared_error(self) -> float:
        """Returns the squared error between the agents mean theta and the
        optimal theta as obtained from the solved environment.
        """
        theta_star = self.env.solver.theta[:, 0]
        squared_error = np.linalg.norm(np.mean(self.theta, axis=1) - theta_star)
        return float(squared_error**2)

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


class LFAAsVRQA(MSEAgent):
    """Variance Reduced Q-learning - M. J. Wainwright (2019).

    Additional Attributes:
        d: The number of samples to use in the update step.
        sample_buffer: A dictionary to store the samples for each state-action pair.
    """

    def __init__(
        self,
        theta_config: ThetaConfig,
        alpha: DecayParameter,
        gamma: float,
        d: int,
        env: Environment,
        features: MakeFeatures,
        eval_config: EvalConfig = DEFAULT_EVAL_CONFIG,
        max_steps: int = 10000,
        seed: int = 1234,
    ) -> None:
        super().__init__(
            theta_config, alpha, gamma, features, env, eval_config, max_steps, seed
        )
        self.d: int = d
        self.sample_buffer: None | dict[Timestep] = None

    def _set_eval_theta(self) -> None:
        """Uses the first theta for evaluation"""
        self.eval_theta = self.theta[:, 0]

    def _reset_model(self) -> None:
        """Resets the models theta as well as the sample buffer"""
        self.theta = self._init_theta()
        self.sample_buffer = None

    def _evaluation_select_action(self, state: np.ndarray) -> int:
        """Selects greedy action based in highest Q-value estimated by the evaluation
        theta.
        """
        return int(np.argmax(state @ self.eval_theta))

    def _squared_error(self) -> float:
        """Returns the squared error between the agents theta and the optimal
        theta as obtained from the solved environment.
        """
        squared_error = np.linalg.norm(self.theta - self.env.solver.theta)
        return float(squared_error**2)

    def _generate_sample_buffer(self) -> None:
        """Fills the sample buffer with satte-action pairs and a list their
        realized next states based on the sample trajectory generated to train the
        model.
        """
        self.sample_buffer = defaultdict(list)
        for sample in self.samples:
            self.sample_buffer[(sample.state.tobytes(), sample.action)].append(
                sample.next_state
            )

    def _next_state_value(self, state: np.ndarray, action: int) -> np.ndarray:
        """Generates an update value by sampling d realized next states from the
        sample buffer that contains realized transitions based on current state
        action pair. The current theta estimator is then used to generate Q-values
        for the sampled next states. Then the mean of the maximum Q-values resulting
        from the previous steps is returned and used as the update value.
        """
        state_action_buffer = self.sample_buffer[(state.tobytes(), action)]
        sample_indices = self._rng.integers(0, len(state_action_buffer), self.d)
        sampled_steps = np.array([state_action_buffer[i] for i in sample_indices])
        next_values = np.max(sampled_steps @ self.theta, axis=1)
        return np.mean(next_values)

    def _update_step(
        self,
        alpha: float,
        t: int,
        ts: Timestep,
    ) -> None:
        """Variace Reduced Q-learning update step where"""
        if self.sample_buffer is None:
            self._generate_sample_buffer()
        q_target = np.array(ts.reward)
        current_value = ts.state[ts.action] @ self.theta
        if not ts.terminal:
            q_target += self.gamma * self._next_state_value(ts.state, ts.action)
        q_update = q_target - current_value
        self.theta[:, 0] += alpha * q_update * ts.state[ts.action]
