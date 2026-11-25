"""Neural channel estimators with matched causal sequence inputs."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_model_sizes(
    input_dim: int,
    output_dim: int,
    hidden_size: int,
    depth: int,
) -> None:
    if min(input_dim, output_dim, hidden_size, depth) <= 0:
        raise ValueError("model dimensions, hidden size, and depth must be positive")


class GRUEstimator(nn.Module):
    """Sequence-to-sequence GRU; the unidirectional form is causal."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        depth: int,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        _validate_model_sizes(input_dim, output_dim, hidden_size, depth)
        self.recurrent = nn.GRU(
            input_dim,
            hidden_size,
            num_layers=depth,
            batch_first=True,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.output = nn.Linear(hidden_size * directions, output_dim)

    def forward(self, values: Tensor) -> Tensor:
        states, _ = self.recurrent(values)
        return self.output(states)


class LSTMEstimator(nn.Module):
    """Sequence-to-sequence LSTM matched to the GRU's input/output contract."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        depth: int,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        _validate_model_sizes(input_dim, output_dim, hidden_size, depth)
        self.recurrent = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=depth,
            batch_first=True,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.output = nn.Linear(hidden_size * directions, output_dim)

    def forward(self, values: Tensor) -> Tensor:
        states, _ = self.recurrent(values)
        return self.output(states)


class WindowedFNNEstimator(nn.Module):
    """Per-slot MLP over the same fixed-length trailing window as the GRU."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        depth: int,
        window_length: int,
    ) -> None:
        super().__init__()
        _validate_model_sizes(input_dim, output_dim, hidden_size, depth)
        if window_length <= 0:
            raise ValueError("window_length must be positive")
        self.window_length = window_length
        layers: list[nn.Module] = []
        layer_input_dim = input_dim * window_length
        for _ in range(depth):
            layers.extend((nn.Linear(layer_input_dim, hidden_size), nn.ReLU()))
            layer_input_dim = hidden_size
        layers.append(nn.Linear(layer_input_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, values: Tensor) -> Tensor:
        left_padding = self.window_length - 1
        padded = F.pad(values, (0, 0, left_padding, 0))
        windows = padded.unfold(1, self.window_length, 1)
        features = windows.transpose(-1, -2).flatten(start_dim=-2)
        return self.network(features)


class CausalCNNEstimator(nn.Module):
    """Temporal CNN with exactly the configured causal receptive field."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        depth: int,
        window_length: int,
    ) -> None:
        super().__init__()
        _validate_model_sizes(input_dim, output_dim, hidden_size, depth)
        if window_length <= 0:
            raise ValueError("window_length must be positive")
        dilations = [2**index for index in range(depth)]
        kernel_sizes = [1] * depth
        remaining_steps = window_length - 1
        for index in reversed(range(depth)):
            kernel_steps = remaining_steps // dilations[index]
            kernel_sizes[index] = kernel_steps + 1
            remaining_steps -= kernel_steps * dilations[index]
        if remaining_steps:
            raise ValueError("could not construct the requested CNN window")

        blocks: list[nn.Module] = []
        layer_input_dim = input_dim
        for dilation, kernel_size in zip(
            dilations, kernel_sizes, strict=True
        ):
            blocks.append(
                nn.Conv1d(
                    layer_input_dim,
                    hidden_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
            )
            layer_input_dim = hidden_size
        self.blocks = nn.ModuleList(blocks)
        self.dilations = dilations
        self.kernel_sizes = kernel_sizes
        self.output = nn.Conv1d(hidden_size, output_dim, kernel_size=1)

    def forward(self, values: Tensor) -> Tensor:
        states = values.transpose(1, 2)
        for convolution, dilation, kernel_size in zip(
            self.blocks,
            self.dilations,
            self.kernel_sizes,
            strict=True,
        ):
            left_padding = (kernel_size - 1) * dilation
            states = F.relu(convolution(F.pad(states, (left_padding, 0))))
        return self.output(states).transpose(1, 2)


class _TanhMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        depth: int,
        output_scale: float,
    ) -> None:
        super().__init__()
        _validate_model_sizes(input_dim, output_dim, hidden_size, depth)
        if output_scale <= 0:
            raise ValueError("output_scale must be positive")
        layers: list[nn.Module] = []
        layer_input_dim = input_dim
        for _ in range(depth):
            layers.extend((nn.Linear(layer_input_dim, hidden_size), nn.ReLU()))
            layer_input_dim = hidden_size
        layers.extend((nn.Linear(layer_input_dim, output_dim), nn.Tanh()))
        self.network = nn.Sequential(*layers)
        self.output_scale = output_scale

    def forward(self, values: Tensor) -> Tensor:
        return self.output_scale * self.network(values)


class SelectiveInputPredictionDNN(nn.Module):
    """Gao et al.'s parallel R-DNN/MP-DNN mapping from high-res inputs."""

    def __init__(
        self,
        n_antennas: int,
        high_resolution_indices: tuple[int, ...] | list[int],
        hidden_size: int,
        depth: int,
        output_scale: float = 3.0,
    ) -> None:
        super().__init__()
        if n_antennas <= 0:
            raise ValueError("n_antennas must be positive")
        indices = tuple(high_resolution_indices)
        if (
            not indices
            or len(set(indices)) != len(indices)
            or min(indices) < 0
            or max(indices) >= n_antennas
            or len(indices) >= n_antennas
        ):
            raise ValueError("high-resolution antenna indices are invalid")
        high_set = set(indices)
        low_indices = tuple(
            index for index in range(n_antennas) if index not in high_set
        )
        self.n_antennas = n_antennas
        self.high_count = len(indices)
        self.low_count = len(low_indices)
        self.register_buffer(
            "high_feature_indices",
            torch.tensor(
                (*indices, *(index + n_antennas for index in indices)),
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "high_real_indices", torch.tensor(indices, dtype=torch.long)
        )
        self.register_buffer(
            "high_imag_indices",
            torch.tensor(
                tuple(index + n_antennas for index in indices),
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "low_real_indices", torch.tensor(low_indices, dtype=torch.long)
        )
        self.register_buffer(
            "low_imag_indices",
            torch.tensor(
                tuple(index + n_antennas for index in low_indices),
                dtype=torch.long,
            ),
        )
        input_dim = 2 * self.high_count
        self.refinement_dnn = _TanhMLP(
            input_dim,
            input_dim,
            hidden_size,
            depth,
            output_scale,
        )
        self.modified_prediction_dnn = _TanhMLP(
            input_dim,
            2 * self.low_count,
            hidden_size,
            depth,
            output_scale,
        )

    def forward(self, values: Tensor) -> Tensor:
        selected = values.index_select(-1, self.high_feature_indices)
        high_estimate = self.refinement_dnn(selected)
        low_estimate = self.modified_prediction_dnn(selected)
        result = values.new_zeros((*values.shape[:-1], 2 * self.n_antennas))
        result[..., self.high_real_indices] = high_estimate[..., : self.high_count]
        result[..., self.high_imag_indices] = high_estimate[..., self.high_count :]
        result[..., self.low_real_indices] = low_estimate[..., : self.low_count]
        result[..., self.low_imag_indices] = low_estimate[..., self.low_count :]
        return result
