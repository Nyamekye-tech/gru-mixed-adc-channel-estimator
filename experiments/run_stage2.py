"""Run the configured, validation-selected neural comparison."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import torch
import yaml
from torch import nn

from chest.baselines import build_mixed_adc_baselines
from chest.channel import exponential_spatial_covariance, jakes_rho
from chest.config import Stage2Config, load_stage2_config
from chest.data import complex_to_real, generate_stage2_dataset
from chest.metrics import nmse
from chest.models import (
    CausalCNNEstimator,
    GRUEstimator,
    LSTMEstimator,
    SelectiveInputPredictionDNN,
    WindowedFNNEstimator,
)
from chest.train import evaluate_nmse, fit_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = (
    "GRU",
    "LSTM",
    "Windowed-FNN",
    "Causal-CNN",
    "Gao-SIP-DNN",
)
CLASSICAL_NAMES = (
    "Two-scalar",
    "Bussgang-LMMSE",
    "Bussgang-Kalman (true rho; white residual)",
)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError(
            "Stage 2 requires a clean worktree so results identify the exact "
            "committed source revision"
        )
    return result.stdout.strip()


def _split_seed(base_seed: int, split_id: int) -> int:
    sequence = np.random.SeedSequence([base_seed, split_id])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _make_model(
    name: str,
    config: Stage2Config,
    hidden_size: int,
    depth: int,
) -> nn.Module:
    feature_dim = 2 * config.n_antennas
    if name == "GRU":
        return GRUEstimator(
            feature_dim,
            feature_dim,
            hidden_size,
            depth,
            config.bidirectional,
        )
    if name == "LSTM":
        return LSTMEstimator(
            feature_dim,
            feature_dim,
            hidden_size,
            depth,
            config.bidirectional,
        )
    if name == "Windowed-FNN":
        return WindowedFNNEstimator(
            feature_dim,
            feature_dim,
            hidden_size,
            depth,
            config.window_length,
        )
    if name == "Causal-CNN":
        return CausalCNNEstimator(
            feature_dim,
            feature_dim,
            hidden_size,
            depth,
            config.window_length,
        )
    if name == "Gao-SIP-DNN":
        return SelectiveInputPredictionDNN(
            config.n_antennas,
            config.high_resolution_indices,
            hidden_size,
            depth,
            config.sip_output_scale,
        )
    raise ValueError(f"unknown neural model: {name}")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device is cuda, but CUDA is unavailable")
    return torch.device(requested)


def _trial_seed(base_seed: int, model_index: int, hidden_size: int, depth: int) -> int:
    values = np.random.SeedSequence(
        [base_seed, model_index, hidden_size, depth]
    ).generate_state(1, dtype=np.uint32)
    return int(values[0])


def _fit_search(
    config: Stage2Config,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    validation_inputs: torch.Tensor,
    validation_targets: torch.Tensor,
    device: torch.device,
    config_json: str,
    git_commit: str,
) -> tuple[dict[str, nn.Module], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    selected_models: dict[str, nn.Module] = {}
    search_rows: list[dict[str, Any]] = []
    selected_settings: dict[str, dict[str, Any]] = {}

    for model_index, name in enumerate(MODEL_NAMES):
        best_validation_nmse = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        best_hidden_size = 0
        best_depth = 0
        best_epoch = 0
        best_seed = 0

        for hidden_size in config.hidden_sizes:
            for depth in config.depths:
                seed = _trial_seed(config.seed, model_index, hidden_size, depth)
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                model = _make_model(name, config, hidden_size, depth)
                model, validation_nmse, best_epoch_for_trial = fit_model(
                    model,
                    inputs,
                    targets,
                    validation_inputs,
                    validation_targets,
                    config.epochs,
                    config.batch_size,
                    config.learning_rate,
                    seed + 1,
                )
                search_rows.append(
                    {
                        "model": name,
                        "hidden_size": hidden_size,
                        "depth": depth,
                        "trial_seed": seed,
                        "best_epoch": best_epoch_for_trial,
                        "validation_nmse": validation_nmse,
                        "validation_nmse_db": float(
                            10.0 * np.log10(validation_nmse)
                        ),
                        "selected": False,
                        "config_json": config_json,
                        "git_commit": git_commit,
                    }
                )
                if validation_nmse < best_validation_nmse:
                    best_validation_nmse = validation_nmse
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    best_hidden_size = hidden_size
                    best_depth = depth
                    best_epoch = best_epoch_for_trial
                    best_seed = seed
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if best_state is None:
            raise RuntimeError(f"search produced no model for {name}")
        model = _make_model(
            name,
            config,
            best_hidden_size,
            best_depth,
        )
        model.load_state_dict(best_state)
        selected_models[name] = model.to(device)
        selected_settings[name] = {
            "hidden_size": best_hidden_size,
            "depth": best_depth,
            "best_epoch": best_epoch,
            "trial_seed": best_seed,
            "validation_nmse": best_validation_nmse,
        }
        selected_row = next(
            row
            for row in reversed(search_rows)
            if row["model"] == name
            and row["hidden_size"] == best_hidden_size
            and row["depth"] == best_depth
        )
        selected_row["selected"] = True
        gc.collect()

    return selected_models, search_rows, selected_settings


def _run_baselines(
    observations: np.ndarray,
    channels: np.ndarray,
    config: Stage2Config,
) -> dict[str, float]:
    covariance = exponential_spatial_covariance(
        config.n_antennas,
        config.spatial_alpha,
        config.channel_power,
    )
    noise_variance = (
        abs(config.pilot_symbol) ** 2
        * config.channel_power
        * 10.0 ** (-config.snr_db / 10.0)
    )
    rho = jakes_rho(
        config.velocity_kmh,
        config.pilot_spacing_s,
        config.carrier_frequency_hz,
    )
    baselines = build_mixed_adc_baselines(
        covariance,
        noise_variance,
        config.high_resolution_indices,
        config.low_resolution_bits,
        config.clip_sigma,
        config.one_bit_levels,
        config.pilot_symbol,
    )
    estimates = (
        baselines.estimate_two_scalar(observations),
        baselines.estimate_bussgang_lmmse(observations),
        baselines.estimate_bussgang_kalman(observations, rho),
    )
    return {
        name: nmse(estimate, channels)
        for name, estimate in zip(CLASSICAL_NAMES, estimates, strict=True)
    }


def run_experiment(
    config: Stage2Config,
    config_snapshot: dict[str, Any],
    device: torch.device,
    git_commit: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Train/search all neural models, then evaluate once on the held-out set."""
    config_json = json.dumps(
        config_snapshot,
        sort_keys=True,
        separators=(",", ":"),
    )
    training_observations, training_channels = generate_stage2_dataset(
        config,
        config.n_train_sequences,
        _split_seed(config.seed, 0),
    )
    validation_observations, validation_channels = generate_stage2_dataset(
        config,
        config.n_validation_sequences,
        _split_seed(config.seed, 1),
    )
    test_observations, test_channels = generate_stage2_dataset(
        config,
        config.n_test_sequences,
        _split_seed(config.seed, 2),
    )

    train_inputs = torch.from_numpy(complex_to_real(training_observations)).to(
        device
    )
    train_targets = torch.from_numpy(complex_to_real(training_channels)).to(
        device
    )
    validation_inputs = torch.from_numpy(
        complex_to_real(validation_observations)
    ).to(device)
    validation_targets = torch.from_numpy(
        complex_to_real(validation_channels)
    ).to(device)
    test_inputs = torch.from_numpy(complex_to_real(test_observations)).to(
        device
    )
    test_targets = torch.from_numpy(complex_to_real(test_channels)).to(device)

    selected_models, search_rows, selected_settings = _fit_search(
        config,
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
        device,
        config_json,
        git_commit,
    )
    classical_nmse = _run_baselines(
        test_observations,
        test_channels,
        config,
    )
    scalar_nmse = classical_nmse["Two-scalar"]

    result_rows: list[dict[str, Any]] = []
    for name in CLASSICAL_NAMES:
        value = classical_nmse[name]
        result_rows.append(
            {
                "model": name,
                "kind": "classical",
                "hidden_size": None,
                "depth": None,
                "best_epoch": None,
                "validation_nmse": None,
                "trial_seed": None,
                "test_nmse": value,
                "test_nmse_db": float(10.0 * np.log10(value)),
                "two_scalar_nmse": scalar_nmse,
                "passes_two_scalar_guard": value
                <= scalar_nmse + config.comparison_tolerance_nmse,
                "config_json": config_json,
                "git_commit": git_commit,
            }
        )

    for name, model in selected_models.items():
        settings = selected_settings[name]
        value = evaluate_nmse(
            model,
            test_inputs,
            test_targets,
            config.batch_size,
        )
        result_rows.append(
            {
                "model": name,
                "kind": "neural",
                "hidden_size": settings["hidden_size"],
                "depth": settings["depth"],
                "best_epoch": settings["best_epoch"],
                "validation_nmse": settings["validation_nmse"],
                "trial_seed": settings["trial_seed"],
                "test_nmse": value,
                "test_nmse_db": float(10.0 * np.log10(value)),
                "two_scalar_nmse": scalar_nmse,
                "passes_two_scalar_guard": value
                <= scalar_nmse + config.comparison_tolerance_nmse,
                "config_json": config_json,
                "git_commit": git_commit,
            }
        )

    results = pd.DataFrame(result_rows)
    search = pd.DataFrame(search_rows)
    failures = results.loc[
        ~results["passes_two_scalar_guard"],
        ["model", "test_nmse", "two_scalar_nmse"],
    ].to_dict(orient="records")
    metadata = {
        "experiment": config.experiment_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "device": str(device),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "pandas_version": pd.__version__,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor() or "CPU"
            ),
            "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "config": config_snapshot,
        "split_seeds": {
            "train": _split_seed(config.seed, 0),
            "validation": _split_seed(config.seed, 1),
            "test": _split_seed(config.seed, 2),
        },
        "selected_models": selected_settings,
        "candidate_count_per_model": len(config.hidden_sizes)
        * len(config.depths),
        "two_scalar_guard_tolerance_nmse": config.comparison_tolerance_nmse,
        "two_scalar_guard_failures": failures,
    }
    return results, search, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "stage2.yaml",
    )
    args = parser.parse_args()
    config = load_stage2_config(args.config)
    config_snapshot = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config_snapshot, dict):
        raise ValueError("configuration root must be a mapping")

    device = _resolve_device(config.device)
    if device.type == "cpu":
        torch.set_num_threads(
            min(config.cpu_num_threads, torch.get_num_threads())
        )
    commit = _git_commit()
    output_dir = PROJECT_ROOT / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "stage2_neural_results.csv"
    search_path = output_dir / "stage2_neural_search.csv"
    metadata_path = output_dir / "stage2_neural_metadata.json"
    if any(path.exists() for path in (results_path, search_path, metadata_path)):
        raise FileExistsError(
            f"refusing to overwrite existing Stage 2 outputs in {output_dir}"
        )

    results, search, metadata = run_experiment(
        config,
        config_snapshot,
        device,
        commit,
    )
    results.to_csv(results_path, index=False)
    search.to_csv(search_path, index=False)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failures = metadata["two_scalar_guard_failures"]
    if failures:
        raise RuntimeError(
            "Stage 2 neural comparison guard failed. Results were saved for "
            f"diagnosis; do not report the failed models as passing: {failures}"
        )

    print(f"Saved final comparison to {results_path}")
    print(f"Saved validation search to {search_path}")
    print(f"Saved config and source revision to {metadata_path}")


if __name__ == "__main__":
    main()
