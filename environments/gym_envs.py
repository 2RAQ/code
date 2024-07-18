from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import gym
import numpy as np

if TYPE_CHECKING:
    from typing import Optional

    from gym import Env
    from gym.spaces import Space


class IntegratedCartPole:
    """Interface Class to allow the usage of OpenAI gym environments within
    our environment/agent framework.
    """

    def __init__(
        self, env_name: str = "CartPole-v0", render_mode: Optional[str] = None
    ):
        """Initializes the IntegratedCartPole class with an action array as required
        for the agent framework.
        """
        self.env: Env = gym.make(env_name, render_mode=render_mode)
        self.actions: np.ndarray = np.arange(self.env.action_space.n)  # type: ignore[attr-defined]

    @property
    def observation_space(self) -> Space[Any]:
        return self.env.observation_space

    @property
    def action_space(self) -> Space[Any]:
        return self.env.action_space

    def reset(
        self,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        return self.env.reset(seed=seed)  # type: ignore[return-value]

    def step(self, action):
        return self.env.step(action)[:3]

    def set_seed(self, seed: int) -> None:
        self.env.reset(seed=int(seed))
