import logging
import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tsdb
from artifact import Artifact
from torch.utils.data import Dataset, IterableDataset

# disable logging
tsdb.utils.logging.logger.setLevel(logging.ERROR)


class ArtifactDataset(IterableDataset):
    """Artifact dataset."""

    def __init__(
        self,
        data: list[np.ndarray],
        artifact: Artifact,
        width: int,
        padding: str | int = "center",
        weight: Optional[list[float]] = None,
    ) -> None:
        """"""
        # properties
        self.data = data
        self.max_rates = [(s.max() - s.min()) / 10 for s in self.data]
        self.min_rates = [(s.max() - s.min()) / 20 for s in self.data]
        self.artifact = artifact
        self.width = width
        # fixed position
        self._position = width // 2 - artifact.max_width // 2
        self._position_mask = np.zeros(width, dtype=np.float32)
        self._position_mask[self._position] = 1
        # padding
        self.padding = padding
        # random generator and weight for sampling
        self.rng = np.random.default_rng()
        self.weight = weight

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate()

    def generate(self):
        """Generate an artifact.

        Ensures that each window has some activity.

        Returns:
            A dictionary containing the window, the artifact, the mask and the position.

        """
        # pick a sequence
        # i = self.rng.integers(0, len(self.data) - 1)
        i = self.rng.choice(len(self.data), p=self.weight)
        sequence = self.data[i]
        # generate a window
        while True:
            length = self.rng.integers(0, len(sequence) - self.width)
            window = sequence[length : length + self.width]
            if window.sum() > 0.01:
                break
        # generate artifact
        artifact = self.artifact.generate(
            max_rate=self.max_rates[i], min_rate=self.min_rates[i]
        )
        length = len(artifact)
        # generate position
        if self.padding == "center":
            position = self._position
            position_mask = self._position_mask
        else:
            position = self.rng.integers(
                self.padding, self.width - (self.artifact.max_width + self.padding)
            )
            position_mask = np.zeros_like(window, dtype=np.float32)
            position_mask[position : position + length] = 1
        # generate delta
        delta = np.zeros_like(window, dtype=np.float32)
        delta[position : position + length] = artifact
        # mask
        m = np.zeros_like(window, dtype=np.float32)
        m[position : position + length] = 1
        # return
        return {
            "data": window,
            "artifact": delta,
            "mask": m,
            "position": position,
            "position_mask": position_mask,
        }


class CachedArtifactDataset(Dataset):
    """Artifact dataset."""

    def __init__(
        self, data: Optional[list] = None, file: Union[str, Path, None] = None
    ) -> None:
        if data is not None:
            self.data = data
        elif file is not None:
            self.data = pickle.load(open(file, "rb"))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.data[i]

    @classmethod
    def generate(
        cls, dataset: ArtifactDataset, n: int, to: Union[str, Path, None] = None
    ):
        data = [next(dataset) for _ in range(n)]
        if to is not None:
            pickle.dump(data, open(to, "wb"))
        return cls(data=data)


def load_files(files: np.ndarray | str | Path | list[str | Path]) -> np.ndarray:
    """Load data from multiple files."""
    if isinstance(files, Path) or isinstance(files, str):
        files = [files]
    data = list()
    for file in files:
        data.extend(load_file(file))
    return data


def load_file(file: str | Path) -> list[np.ndarray]:
    """Load data from multiple files."""
    with open(file, "rb") as f:
        return pickle.load(f)
