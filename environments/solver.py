from __future__ import annotations

import copy
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from absl import logging

from agents.parameter import DecayByT

if TYPE_CHECKING:
    from agents.parameter import DecayParameter
    from environments.make_features import MakeFeatures
    from environments.utils import Environment


class Solver:
    """Interface for a solver that finds optimal policies in a given CMC.

    Attributes:
        alpha: Learning rate
        env: The environment which to solve.
        policy: The optimal policy.
        action_values: The action values of the solved environment.
        theta: The optimal theta in case of function approximation.
    """

    alpha: DecayParameter
    env: Environment
    policy: np.ndarray
    action_values: np.ndarray
    theta: np.ndarray

    def __init__(self, alpha: DecayParameter | None, gamma: float) -> None:
        if alpha:
            self.alpha = alpha
        else:
            self.alpha = DecayByT(0.3, 100)
        self.gamma: float = gamma

    def add_env(self, env: Environment) -> None:
        self.env = env
        self.solve_env()

    @abstractmethod
    def solve_env(self) -> None:
        pass


class TabularEnvSolver(Solver):
    """Solves CMC in the tabular case."""

    def __init__(self, alpha: DecayParameter | None = None, gamma: float = 0.9) -> None:
        super().__init__(alpha, gamma)

    def solve_env(self) -> None:
        """soon"""
        logging.debug(
            f"Solving environment with {self.env.n_states} states and "
            f"{self.env.n_actions} actions."
        )
        self.action_values = np.zeros((self.env.n_states, self.env.n_actions))
        for t in range(1000000):
            old = copy.deepcopy(self.action_values)
            for s in range(self.env.n_states):
                for a in range(self.env.n_actions):
                    update = self._get_update(s, a)
                    self.action_values[s, a] += self.alpha.decay(t) * update
            error = np.linalg.norm((self.action_values - old).flatten())
            if error < 1e-10:
                break
        logging.debug(f"Solved environment in {t} iterations.")
        self.policy = np.zeros((self.env.n_states, self.env.n_actions))
        self.policy[
            np.arange(self.env.n_states), np.argmax(self.action_values, axis=1)
        ] = 1

    def _get_update(self, s: int, a: int) -> float:
        """soon"""
        value_expectation = np.sum(
            self.action_values * self.env.transition_probs[:, s][:, np.newaxis], axis=0
        )
        update = (
            self.env.rewards[s, a]
            + self.gamma * np.max(value_expectation)
            - self.action_values[s, a]
        )
        return update


class LFAEnvSolver(Solver):
    """soon"""

    def __init__(
        self,
        theta_size: int,
        features: MakeFeatures,
        alpha: DecayParameter | None = None,
        gamma: float = 0.9,
    ) -> None:
        """soon"""
        super().__init__(alpha, gamma)
        self.theta: np.ndarray = np.zeros((theta_size, 1))
        self.features: MakeFeatures = features

    def solve_env(self) -> None:
        """soon"""
        logging.debug(
            f"Solving environment with {self.env.n_states} states and "
            f"{self.env.n_actions} actions."
        )
        self.theta = np.zeros(self.theta.shape)
        for t in range(1000000):
            theta = self._update_theta(self.alpha.decay(t), copy.deepcopy(self.theta))
            error = np.linalg.norm(theta - self.theta)
            self.theta = theta
            if error < 1e-10:
                break
        logging.debug(f"Solved environment in {t} iterations.")
        self.action_values = np.zeros((self.env.n_states, self.env.n_actions))
        for s, state in enumerate(self.features.features):
            self.action_values[s] = (state @ self.theta)[:, 0]
        self.policy = np.zeros((self.env.n_states, self.env.n_actions))
        self.policy[
            np.arange(self.env.n_states), np.argmax(self.action_values, axis=1)
        ] = 1

    def _update_theta(self, alpha: float, theta: np.ndarray) -> np.ndarray:
        """soon"""
        for s, state in enumerate(self.features.features):
            for a in range(self.env.n_actions):
                update = self._get_update(s, a, state, theta)
                theta[:, 0] += alpha * update * state[a]
        return theta

    def _get_update(
        self, s: int, a: int, state: np.ndarray, theta: np.ndarray
    ) -> float:
        """soon"""
        value_expectation = np.sum(
            (self.features.features @ theta)[:, :, 0]
            * self.env.transition_probs[a, s][:, np.newaxis],
            axis=0,
        )
        value_target = self.env.rewards[s, a] + self.gamma * np.max(value_expectation)
        update = value_target - state[a] @ theta
        return update
