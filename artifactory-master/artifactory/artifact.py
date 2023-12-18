from abc import ABC, abstractmethod

import numpy as np


class Artifact(ABC):
    """Base class for artifacts."""

    def __init__(self, min_width: int = 5, max_width: int = 30) -> None:
        self.min_width = min_width
        self.max_width = max_width
        self.generator = np.random.default_rng()

    @abstractmethod
    def generate(self, min_rate: float, max_rate: float) -> np.ndarray:
        pass


class Saw(Artifact):
    """Sawtooth artifact."""

    def generate(self, min_rate=0.05, max_rate=0.25) -> np.ndarray:
        width = self.generator.integers(self.min_width, self.max_width, endpoint=True)
        activation = self.generator.integers(width)
        rate = self.generator.uniform(min_rate, max_rate) * np.sign(
            self.generator.uniform(-1, 1)
        )
        time = np.arange(width)
        ramp = rate * time
        return ramp - (rate * (width - 1) * (time >= activation))
