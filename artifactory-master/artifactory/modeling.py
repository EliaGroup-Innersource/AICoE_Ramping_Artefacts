import numpy as np
import torch
from torch.nn import (
    Conv1d,
    Dropout,
    Embedding,
    Linear,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
)

activations = {
    "relu": ReLU(),
    "sigmoid": Sigmoid(),
    "tanh": Tanh(),
}


def _convolutions(
    convolution_features: list[int],
    convolution_width: int | list[int],
    convolution_dilation: int | list[int] = 1,
    convolution_dropout: int = 0,
    activation: str = "sigmoid",
    last: bool = True,
    pad: bool = False,
) -> Sequential:
    """Create a sequence of convolutional layers.

    Args:
        convolution_features: Number of features in each layer.
        convolution_width: Width of each layer. If an integer,
            all layers will have the same width. If a list,
            each layer will have the corresponding width.
        last: If False, the last layer will not have an activation.

    """
    if isinstance(convolution_dilation, int):
        convolution_dilation = [convolution_dilation] * (len(convolution_features) - 1)
    if isinstance(convolution_width, int):
        convolution_width = [convolution_width] * (len(convolution_features) - 1)
    layers = Sequential()
    for i in range(len(convolution_features) - 1):
        layers.append(
            Conv1d(
                in_channels=convolution_features[i],
                out_channels=convolution_features[i + 1],
                kernel_size=convolution_width[i],
                dilation=convolution_dilation[i],
                padding="same" if pad else 0,
            )
        )
        if i < len(convolution_features) - 2 or last:
            layers.append(activations[activation])
        if convolution_dropout > 0:
            layers.append(Dropout(convolution_dropout))
    return layers


def _linear(
    features: list[int], activation: Module = Sigmoid(), last: bool = True
) -> Sequential:
    """Create a sequence of linear layers."""
    layers = Sequential()
    for i in range(len(features) - 1):
        layers.append(Linear(features[i], features[i + 1]))
        if i < len(features) - 2 or last:
            layers.append(activation)
    return layers


def _size(s: int, layers: Sequential) -> int:
    """Compute the size of the output."""
    for layer in layers:
        if isinstance(layer, Conv1d):
            s = s - layer.kernel_size[0] + 1
    return s


class SinusoidalPositionEmbedding(Module):
    """Classic sinusoidal positions."""

    def __init__(self, dimension: int, length: int):
        """

        Args:
            dimension: Embedding dimension.
            length: Sequence length.

        """
        super().__init__()
        self.dimension = dimension
        self.length = length

        # initialise position encoding
        idx = torch.arange(self.length).unsqueeze(1)
        den = torch.exp(torch.arange(0, dimension, 2) * (-np.log(10000.0) / dimension))
        pe = torch.zeros(1, self.length, self.dimension)
        pe[0, :, 0::2] = torch.sin(idx * den)
        pe[0, :, 1::2] = torch.cos(idx * den)
        self.register_buffer("position", pe)

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return x + self.position


class LearnedPositionEmbedding(Module):
    """Learn position embedding."""

    def __init__(self, dimension: int, length: int):
        super().__init__()
        self.dimension = dimension
        self.length = length
        self.embedding = Embedding(length, dimension)
        self.register_buffer("ids", torch.arange(length).unsqueeze(0))

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return x + self.embedding(self.ids)
