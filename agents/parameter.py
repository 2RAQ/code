from __future__ import annotations

from abc import abstractmethod

import numpy as np


class DecayParameter:
    """Interface for decaying hyperparameter.

    Holds the initial value as well as the current value.
    Decays the parameter via the decay method.
    """

    def __init__(self, value: float) -> None:
        """Inits the Parameter with its initial value"""
        self.initial_value: float = value
        self.value: float = value

    @abstractmethod
    def decay(self, timestep: int) -> float:
        """Decays the initial value based on the passed timestep.

        Should return the decayed value and set 'value' to the current value.
        """


class ConstantParameter(DecayParameter):
    """Constant parameter class for modular use within the modular system"""

    def decay(self, timestep: int) -> float:
        """Always returns the constant 'value' equal to the initial value"""
        return self.value


class WeightedDecayParameter(DecayParameter):
    """Expands the DecayParameter interface class by a weighted decay.

    Adds an additional class attribute 'weight' that can be used to dampen the
    parameter decay by applying weight to the timesteps.
    """

    def __init__(self, value: float, weight: int) -> None:
        """Inits the WeightedParameterDecay with an initial value and the decay
        weight.
        """
        super().__init__(value)
        self.weight: int = weight
        if weight == 0:
            self.weight = 1


class DecayByT(WeightedDecayParameter):
    """Decays the parameter by the weighted timesteps"""

    def __init__(self, value: float, weight: int) -> None:
        super().__init__(value, weight)

    def decay(self, timestep: int) -> float:
        """Decays the parameter by: weight / (timestep + weight)
        Different values for the weight control the speed at which the linear
        decay affects the parameter.
        """
        self.value = self.initial_value * self.weight / (timestep + self.weight)
        return self.value


class DecayByTSquared(WeightedDecayParameter):
    """Decays the parameter by the weighted squared timestpes"""

    def __init__(self, value: float, weight: int) -> None:
        super().__init__(value, weight)

    def decay(self, timestep: int) -> float:
        """Decays the parameter by: weight / (timestep^2 + weight)
        Different values for the weight control the speed at which the squared
        decay affects the parameter.
        """
        self.value = self.initial_value * self.weight / (timestep**2 + self.weight)
        return self.value


class WengEtAl2020MaxMin(DecayParameter):
    """Epsilon decay for eps-greedy exploration as used by Weng et al. (2020).

    Slightly modified for t>1000.
    """

    def __init__(self) -> None:
        super().__init__(1)

    def decay(self, t: int) -> float:
        if t <= 1000:
            self.value = np.max([0.1, np.min([1.0, 1 - np.log((t + 1) / 200)])])
        else:
            self.value = 0.01
        return self.value


class BasicEpsDecay(DecayParameter):
    """A simple epsilon decay for eps-greedy exploration used in the LunarLander
    experiments.
    """

    def __init__(self) -> None:
        super().__init__(1)

    def decay(self, t: int) -> float:
        if t <= 20:
            self.value = 1
        else:
            self.value = np.max([0.995 ** (t - 20), 0.01])
        return self.value
