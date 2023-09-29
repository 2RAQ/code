from __future__ import annotations

from abc import abstractmethod

import numpy as np

from environments.base_environment import ENV


class MakeFeatures:
    """Interface class to allow creation of different features"""

    features: np.ndarray

    @abstractmethod
    def make_features(self, state: int | np.ndarray) -> np.ndarray:
        """Takes a state as input and returns a feature array"""
        pass


class PassThrough(MakeFeatures):
    """Returns the untouched state as passed"""

    def make_features(self, state: int | np.ndarray) -> np.ndarray:
        return np.array(state)


class MakeArray(MakeFeatures):
    """Returns the passed state as numpy array for cases where the datatype
    is required.
    """

    def __init__(self, states: int) -> None:
        self.features: np.ndarray = np.arange(states)[:, np.newaxis]

    def make_features(self, state: int | np.ndarray) -> np.ndarray:
        return self.features[state]


class MakeOneHot(MakeFeatures):
    """Creates a One-Hot representation of the passed state for all actions"""

    def __init__(self, states: int, actions: int) -> None:
        self.features: np.ndarray = self._gen_features(states, actions)

    def _gen_features(self, states: int, actions: int) -> np.ndarray:
        identity = np.identity(states * actions)
        return identity.reshape((states, actions, states * actions))

    def make_features(self, state: int | np.ndarray) -> np.ndarray:
        return self.features[state]


class StackStateAction(MakeFeatures):
    """Creates a stacked state-action representation and returns the passed state
    for all actions.

    The stacked feature vector is a vector of dimension |S|+|A| that consists of
    two stacked One-Hot vectors. One for the state and one for the action.

    E.g. with |S| = 4 and |A| = 3 the state-action pair (2, 1) qould be:
    [0, 0, 1, 0, 0, 1, 0]
    """

    def __init__(self, states: int, actions: int) -> None:
        """soon"""
        self.features: np.ndarray = self._gen_features(states, actions)

    def _gen_features(self, states: int, actions: int) -> np.ndarray:
        """soon"""
        features = np.zeros((states, actions, states + actions))
        for s in range(states):
            for a in range(actions):
                features[s, a, s] = 1
                features[s, a, states + a] = 1
        return features

    def make_features(self, state: int | np.ndarray) -> np.ndarray:
        """soon"""
        return self.features[state]


class BairdsFeatures(MakeFeatures):
    """soon"""

    def __init__(self, states: int) -> None:
        """soon"""
        self.features: np.ndarray = self._gen_features(states)

    def _gen_features(self, states: int) -> np.ndarray:
        """soon"""
        identiy_m = np.identity(2 * states)
        features = np.zeros((states, 2, 2 * states))
        for s in range(states):
            if s == 0:
                features[s, 0] = identiy_m[2 * states - 1]
            else:
                features[s, 0] = identiy_m[s]
            if s == states - 1:
                features[s, 0] = 2 * identiy_m[0] + identiy_m[2 * states - 1]
                features[s, 1] = identiy_m[states - 1]
            else:
                features[s, 1] = identiy_m[0] + 2 * identiy_m[states + s]
        return features

    def make_features(self, state: int | np.ndarray) -> np.ndarray:
        """soon"""
        return self.features[state]


class DiscretizeCartpole(MakeFeatures):
    """Discretize the Cartpole Environment similar to Weng et al. (2020)"""

    def __init__(self, env: ENV, buckets: list[int]) -> None:
        """soon"""
        self.buckets: list[int] = buckets
        self.features: np.ndarray = self._gen_features()
        self.upper_bounds: np.ndarray = np.array(
            [
                env.observation_space.high[0],
                0.5,
                env.observation_space.high[2],
                np.radians(50),
            ]
        )
        self.lower_bounds: np.ndarray = -self.upper_bounds

    def _gen_features(self) -> np.ndarray:
        """soon"""
        n_state_actions = int(np.prod(self.buckets) * 2)
        identity = np.identity(n_state_actions)
        return identity.reshape(self.buckets + [2, n_state_actions])

    def _discretize_state(self, state: np.ndarray) -> list[int]:
        """soon"""
        ratios = [
            (state[i] - self.lower_bounds[i])
            / (self.upper_bounds[i] - self.lower_bounds[i])
            for i in range(len(state))
        ]
        new_state = [
            int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(state))
        ]
        new_state = [
            min(self.buckets[i] - 1, max(0, new_state[i])) for i in range(len(state))
        ]
        return new_state

    def make_features(self, state: int | np.ndarray) -> np.ndarray:
        """soon"""
        assert isinstance(
            state, np.ndarray
        ), f"State has type {type(state)}; expected state of type np.ndarray"
        return self.features[tuple(self._discretize_state(state))]
