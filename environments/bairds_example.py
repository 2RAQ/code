from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from absl import logging
import numpy as np

from environments.base_environment import Environment

if TYPE_CHECKING:
    from environments.base_environment import Solver


class BairdsExample(Environment):
    """Implementation of Baird's Example"""

    states: np.ndarray

    def __init__(
        self,
        states: int = 6,
        reward_range: tuple[float, float] = (0.0, 0.0),
        seed: int = 123,
        solver: Solver | None = None,
    ) -> None:
        super().__init__(states, 2, seed, solver)
        self.rewards: np.ndarray = self._gen_rewards(reward_range)
        self.transition_probs: np.ndarray = self._gen_transition_probs()
        self.start_state_dist: np.ndarray = self._gen_start_state_distribution()
        self.state: int = 1
        self.states_prob: np.ndarray = self.start_state_dist.copy()
        if solver:
            self.solver = solver
            self.solver.add_env(self)
        logging.debug(
            f"Bairds example with {states} states initialized; Using seed: {seed}"
        )

    def _gen_rewards(self, reward_range: tuple[float, float]) -> np.ndarray:
        """Generates a random reward for each state-action pair"""
        lower, upper = reward_range
        return (upper - lower) * self._rng.random(
            (self.n_states, self.n_actions)
        ) + lower

    def _gen_transition_probs(self) -> np.ndarray:
        """Constructs the transition probability matrix"""
        tp = np.zeros((2, self.n_states, self.n_states))
        tp[1, :, -1] += 1
        tp[0, :, :-1] += 1 / (self.n_states - 1)
        return tp

    def _gen_start_state_distribution(self) -> np.ndarray:
        """Constructs distribution that deterministically starts in state 0"""
        start_state_dist = np.zeros((1, self.n_states))
        start_state_dist[0, 0] = 1
        return start_state_dist

    def copy_parameters(self, other: BairdsExample) -> None:
        """ "Copies parameters from another Bairds example environment"""
        self.states = copy.deepcopy(other.states)
        self.rewards = copy.deepcopy(other.rewards)
        self.transition_probs = copy.deepcopy(other.transition_probs)
        self.start_state_dist = copy.deepcopy(other.start_state_dist)
        self.state = self._rng.choice(self.states, p=self.start_state_dist[0])
        self.states_prob = copy.deepcopy(self.start_state_dist)
        logging.debug("Copied features from other Bairds example environment")

    def reset(self) -> int:
        """Resets the environment and returns the initial state"""
        self.state = 1
        self.states_prob = self.start_state_dist.copy()
        return self.state

    def step(self, action) -> tuple[int, int, bool]:
        """
        Performs an action and returns the next state, reward, and False since there
        is no terminal state
        """
        reward = self.rewards[self.state, action]
        self.state = self._rng.choice(
            self.states, p=self.transition_probs[action, self.state]
        )
        self.states_prob = self.states_prob @ self.transition_probs[action]
        return self.state, reward, False
