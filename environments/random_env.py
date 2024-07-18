from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from absl import logging
import numpy as np

from environments.base_environment import Environment

if TYPE_CHECKING:
    from environments.base_environment import Solver


class RandomEnvironment(Environment):
    """Controlled Markov Chain"""

    states: np.ndarray
    actions: np.ndarray

    def __init__(
        self,
        states: int,
        actions: int,
        q: float,
        p: float,
        dirichlet_alpha: float = 0.1,
        seed: int = 123,
        solver: Solver | None = None,
    ) -> None:
        super().__init__(states, actions, seed, solver)
        self.rewards: np.ndarray = self._gen_rewards(q, p)
        self.dirichlet_alpha: float = dirichlet_alpha
        self.transition_probs: np.ndarray = self._gen_transition_probs()
        self.start_state_dist: np.ndarray = self._gen_start_state_distribution()
        self.state: int = self._rng.choice(self.states, p=self.start_state_dist[0])
        self.states_prob: np.ndarray = self.start_state_dist.copy()
        self.action_seq: list[int] = []
        if solver:
            self.solver = solver
            self.solver.add_env(self)

    def _gen_rewards(self, q: float, p: float) -> np.ndarray:
        """Generate a reward matrix based on linear quadratic rewards"""
        rewards = np.zeros((self.n_states, self.n_actions))
        for s in self.states:
            rewards[s] = -q * s**2 - p * self.actions**2
        return rewards

    def _gen_transition_probs(self) -> np.ndarray:
        """Generates a random the transition probability matrix"""
        alpha_parameter = np.ones(len(self.states)) * self.dirichlet_alpha
        transition_array = self._rng.dirichlet(
            alpha_parameter, (self.n_actions, self.n_states)
        )
        return transition_array

    def _gen_start_state_distribution(self) -> np.ndarray:
        """Generates a random start state distribution"""
        start_state_dist = self._rng.dirichlet(
            np.ones(self.n_states) * self.dirichlet_alpha, 1
        )
        return start_state_dist

    def copy_parameters(self, other: RandomEnvironment) -> None:
        """ "Copies parameters from another random environment"""
        self.states = copy.deepcopy(other.states)
        self.actions = copy.deepcopy(other.actions)
        self.rewards = copy.deepcopy(other.rewards)
        self.transition_probs = copy.deepcopy(other.transition_probs)
        self.start_state_dist = copy.deepcopy(other.start_state_dist)
        self.state = self._rng.choice(self.states, p=self.start_state_dist[0])
        self.states_prob = copy.deepcopy(self.start_state_dist)
        self.action_seq.clear()
        logging.debug("Copied features from other CMC environment")

    def reset(self) -> int:
        """Resets the environment and returns the initial state"""
        self.state = np.random.choice(self.states, p=self.start_state_dist[0])
        self.states_prob = self.start_state_dist.copy()
        self.action_seq.clear()
        return self.state

    def step(self, action: int) -> tuple[int, int, bool]:
        """
        Performs an action and returns the next state, reward, and False since there
        is no terminal state
        """
        reward = self.rewards[self.state, action]
        self.action_seq.append(action)
        self.state = self._rng.choice(
            self.states, p=self.transition_probs[action, self.state]
        )
        self.states_prob = self.states_prob @ self.transition_probs[action]
        return self.state, reward, False
