from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gym.envs.classic_control.cartpole import CartPoleEnv

if TYPE_CHECKING:
    from typing import Optional


class IntegratedCartPole(CartPoleEnv):
    """Interface Class to allow the usage of openAI gym environments within
    our environment/agent framework.
    """

    def __init__(self, render_mode: Optional[str] = None):
        """soon"""
        super().__init__(render_mode)
        self.actions: np.ndarray = np.arange(self.action_space.n)  # type: ignore[attr-defined]

    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> np.ndarray:
        """soon"""
        return super().reset(seed=seed, return_info=return_info, options=options)

    def step(self, action):
        """soon"""
        return super().step(action)[:3]

    def set_seed(self, seed: int) -> None:
        """soon"""
        self.reset(seed=int(seed))
