from dataclasses import replace

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from chest.config import load_stage2_config
from chest.data import complex_to_real, generate_stage2_dataset
from chest.models import (
    CausalCNNEstimator,
    GRUEstimator,
    LSTMEstimator,
    SelectiveInputPredictionDNN,
    WindowedFNNEstimator,
)
from chest.train import evaluate_nmse, fit_model
from experiments.run_stage2 import run_experiment


@pytest.fixture
def stage2_config():
    return load_stage2_config("configs/stage2.yaml")


def test_stage2_config_loads_explicit_training_protocol(stage2_config) -> None:
    assert stage2_config.n_train_sequences == 2000
    assert stage2_config.window_length == stage2_config.n_slots == 20
    assert stage2_config.high_resolution_indices == (0, 8, 16, 24)
    assert stage2_config.hidden_sizes == (64, 128)
    assert stage2_config.depths == (2, 3)
    assert stage2_config.optimizer == "Adam"
    assert stage2_config.loss == "mean_squared_error"
    assert stage2_config.bidirectional is False
    assert stage2_config.sequence_target_mode == "all_slots"
    assert stage2_config.window_padding == "left_zero"
    assert stage2_config.cnn_causal is True


def test_stage2_config_rejects_bidirectional_temporal_leakage(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(
        Path("configs/stage2.yaml").read_text(encoding="utf-8")
    )
    raw["training"]["bidirectional"] = True
    invalid_config = tmp_path / "stage2-bidirectional.yaml"
    invalid_config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unidirectional"):
        load_stage2_config(invalid_config)


def test_stage2_dataset_is_reproducible_and_uses_mixed_adc(stage2_config) -> None:
    observations, channels = generate_stage2_dataset(stage2_config, 8, 17)
    repeated_observations, repeated_channels = generate_stage2_dataset(
        stage2_config, 8, 17
    )
    np.testing.assert_array_equal(observations, repeated_observations)
    np.testing.assert_array_equal(channels, repeated_channels)
    assert observations.shape == channels.shape == (8, 20, 32)

    low_indices = [
        index
        for index in range(stage2_config.n_antennas)
        if index not in stage2_config.high_resolution_indices
    ]
    assert set(np.unique(observations[..., low_indices].real)).issubset(
        {-1.0, 1.0}
    )
    assert np.unique(observations[..., list(stage2_config.high_resolution_indices)].real).size > 2
    assert complex_to_real(channels).shape == (8, 20, 64)


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: GRUEstimator(64, 64, hidden_size=8, depth=2),
        lambda: LSTMEstimator(64, 64, hidden_size=8, depth=2),
        lambda: WindowedFNNEstimator(64, 64, 8, 2, window_length=20),
        lambda: CausalCNNEstimator(64, 64, 8, 2, window_length=20),
        lambda: SelectiveInputPredictionDNN(
            32,
            [0, 8, 16, 24],
            hidden_size=8,
            depth=2,
        ),
    ],
)
def test_neural_estimators_are_causal_and_shape_matched(model_factory) -> None:
    torch.manual_seed(3)
    model = model_factory().eval()
    inputs = torch.randn(2, 20, 64)
    prediction = model(inputs)
    assert prediction.shape == inputs.shape

    changed_future = inputs.clone()
    changed_future[:, 9:, :] += 50.0
    changed_prediction = model(changed_future)
    torch.testing.assert_close(prediction[:, :9], changed_prediction[:, :9])


def test_gao_sip_uses_only_high_resolution_observation_features() -> None:
    torch.manual_seed(5)
    model = SelectiveInputPredictionDNN(
        32,
        [0, 8, 16, 24],
        hidden_size=8,
        depth=2,
    ).eval()
    inputs = torch.randn(2, 20, 64)
    changed_low_resolution = inputs.clone()
    high_real = {0, 8, 16, 24}
    high_imag = {index + 32 for index in high_real}
    low_features = [
        index
        for index in range(64)
        if index not in high_real and index not in high_imag
    ]
    changed_low_resolution[..., low_features] += 50.0
    torch.testing.assert_close(
        model(inputs),
        model(changed_low_resolution),
    )


@pytest.mark.parametrize("depth", [2, 3])
def test_causal_cnn_receptive_field_matches_configured_window(depth) -> None:
    model = CausalCNNEstimator(64, 64, 8, depth, window_length=20)
    receptive_field = 1 + sum(
        (kernel_size - 1) * dilation
        for kernel_size, dilation in zip(
            model.kernel_sizes,
            model.dilations,
            strict=True,
        )
    )
    assert receptive_field == 20


def test_fit_model_restores_best_validation_checkpoint() -> None:
    torch.manual_seed(7)
    inputs = torch.randn(12, 4, 6)
    targets = 0.25 * inputs
    model, validation_nmse, best_epoch = fit_model(
        nn.Sequential(nn.Linear(6, 6)),
        inputs,
        targets,
        inputs[:4],
        targets[:4],
        epochs=2,
        batch_size=4,
        learning_rate=0.01,
        seed=11,
    )
    assert np.isfinite(validation_nmse)
    assert best_epoch in (1, 2)
    assert evaluate_nmse(model, inputs[:4], targets[:4], batch_size=2) == pytest.approx(
        validation_nmse
    )


def test_stage2_experiment_smoke_runs_all_models(stage2_config) -> None:
    config = replace(
        stage2_config,
        n_train_sequences=4,
        n_validation_sequences=2,
        n_test_sequences=4,
        batch_size=2,
        epochs=1,
        hidden_sizes=(4,),
        depths=(1,),
    )
    results, search, metadata = run_experiment(
        config,
        {"smoke_test": True},
        torch.device("cpu"),
        "smoke-test",
    )
    assert len(search) == 5
    assert search["selected"].sum() == 5
    assert len(results) == 8
    assert results["passes_two_scalar_guard"].dtype == bool
    assert metadata["split_seeds"]["train"] != metadata["split_seeds"]["test"]
