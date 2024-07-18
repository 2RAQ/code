from __future__ import annotations

from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import gym
import numpy as np

if TYPE_CHECKING:
    from environments.solver import Solver

EPS = np.finfo(np.float32).eps.item()
Env = gym.wrappers.time_limit.TimeLimit
RNG = np.random._generator.Generator


@dataclass
class Timestep:
    """Timestep storate dataclass.

    Allows to store one timestep within a given MDP.

    Attributes:
        state: The state in which the timestep begins.
        action: The action performed in the timestep.
        reward: The reward as realized by performing 'action' in 'state'.
        next_state: The state the environment transitioned to after performing
            'action' in 'state'.
        terminal: Whether or not 'next_state' is a terminal state.
    """

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminal: bool


class Environment:
    """Provides the interface for controlled Markov-Chain environments.


    Attributes:
        rewards: A matrix that holds the state-action -> reward mapping.
        transition_probs: A matrix that holds the state-action -> next_state
            transition probabilities with actions on the 0th dimension, states on
            the 1st dimension and next_states on the 2nd dimension.
        start_state_dist: A distribution for the starting state of the CMC.
        solver: (optional) A solver that solves the CMC and stores the optimal
            policy for further usage.
        seed: The random seed for the environment.
        _rng: The random numbers generator as used within the environment.
        states: A vector containing all states in the environment.
        actions: A vector containing the indices of all actions in the environment.
        models: (optional) A selection of models (a set of start state distributions
            as well as transition probabilities) which can be used to synchronously
            change the transition dynamics of a set of environments.
    """

    rewards: np.ndarray
    transition_probs: np.ndarray
    start_state_dist: np.ndarray
    solver: Solver

    def __init__(
        self, states: int, actions: int, seed: int, solver: Solver | None
    ) -> None:
        """Inits the environment with args."""
        self.seed: int = seed
        self._rng: np.random._generator.Generator = np.random.default_rng(seed)
        self.states: np.ndarray = np.arange(states)
        self.actions: np.ndarray = np.arange(actions)
        self.models: None | deque[tuple[np.ndarray, np.ndarray]] = None

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    @abstractmethod
    def _gen_transition_probs(self) -> np.ndarray:
        """Generates and returns a transition probability array for
        the environment.
        """

    @abstractmethod
    def _gen_start_state_distribution(self) -> np.ndarray:
        """Generates and returns a starting state distribution array for
        the environment.
        """

    @abstractmethod
    def copy_parameters(self, other: Environment) -> None:
        """Copies all environment defining parameters from a passed, other
        environment to self.
        """

    @abstractmethod
    def reset(self) -> int:
        """Resets the dynamic parameters of the environment and returns the new,
        initial state as drawn from the starting state distribution.
        """

    @abstractmethod
    def step(self, action: int) -> tuple[int, int, bool]:
        """Performs the action as passed and returns the next state, the realized
        reward, and whether or not the next state is terminal.
        """

    def set_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def gen_model_sequence(self, n_models: int) -> None:
        """Generates and stores a sequence of 'n_models' transition probabilities
        and starting state distribution tuples.
        """
        self.models = deque(
            (self._gen_transition_probs(), self._gen_start_state_distribution())
            for _ in range(n_models)
        )

    def set_new_transition_dynamics(self) -> None:
        """Interface methods that sets new environment dynamics from the set of
        stored models if available and resets the environment, which enables the
        use of only one method in the agent implementation.
        """
        if self.models:
            self.transition_probs, self.start_state_dist = self.models.pop()
            if self.solver:
                self.solver.solve_env()
        self.reset()
