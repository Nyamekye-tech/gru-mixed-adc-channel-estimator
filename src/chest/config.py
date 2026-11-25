"""Validated loading for the Stage 1 experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Stage1Config:
    experiment_name: str
    seed: int
    n_sequences_per_point: int
    batch_size: int
    output_dir: Path
    n_antennas: int
    n_slots: int
    channel_power: float
    carrier_frequency_hz: float
    pilot_spacing_s: float
    pilot_symbol: complex
    high_resolution_indices: tuple[int, ...]
    spatial_alphas: tuple[float, ...]
    low_resolution_bits: int
    supported_bit_depths: tuple[int, ...]
    clip_sigma: float
    scaling_model: str
    one_bit_levels: tuple[float, float]
    snr_db: tuple[float, ...]
    snr_sweep_velocity_kmh: float
    velocity_kmh: tuple[float, ...]
    velocity_sweep_snr_db: float
    kalman_observation_model: str
    kalman_residual_model: str
    kalman_uses_true_rho: bool


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _require_number_list(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return tuple(_require_number(item, f"{name} item") for item in value)


def load_stage1_config(path: str | Path) -> Stage1Config:
    """Load YAML and reject missing or invalid Stage 1 settings."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = _require_mapping(yaml.safe_load(file), "configuration")

    experiment = _require_mapping(raw.get("experiment"), "experiment")
    system = _require_mapping(raw.get("system"), "system")
    channel = _require_mapping(raw.get("channel"), "channel")
    adc = _require_mapping(raw.get("adc"), "adc")
    sweeps = _require_mapping(raw.get("sweeps"), "sweeps")
    kalman = _require_mapping(raw.get("kalman"), "kalman")

    experiment_name = experiment.get("name")
    if not isinstance(experiment_name, str) or not experiment_name:
        raise ValueError("experiment.name must be a non-empty string")
    if channel.get("temporal_model") != "gauss_markov_jakes":
        raise ValueError("channel.temporal_model must be gauss_markov_jakes")
    if channel.get("spatial_model") != "exponential_toeplitz_ula":
        raise ValueError(
            "channel.spatial_model must be exponential_toeplitz_ula"
        )

    indices_raw = system.get("high_resolution_indices")
    if not isinstance(indices_raw, list) or any(
        isinstance(index, bool) or not isinstance(index, int)
        for index in indices_raw
    ):
        raise ValueError("system.high_resolution_indices must be a list of integers")

    n_antennas = _require_positive_int(system.get("n_antennas"), "system.n_antennas")
    indices = tuple(indices_raw)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("high-resolution indices must be non-empty and unique")
    if min(indices) < 0 or max(indices) >= n_antennas:
        raise ValueError("high-resolution antenna index is out of range")

    pilot_symbol_raw = system.get("pilot_symbol")
    if not isinstance(pilot_symbol_raw, list) or len(pilot_symbol_raw) != 2:
        raise ValueError("system.pilot_symbol must be [real, imaginary]")
    pilot_symbol = complex(
        _require_number(pilot_symbol_raw[0], "system.pilot_symbol real"),
        _require_number(pilot_symbol_raw[1], "system.pilot_symbol imaginary"),
    )
    if abs(pilot_symbol) == 0:
        raise ValueError("system.pilot_symbol must be non-zero")

    bits = adc.get("low_resolution_bits")
    if isinstance(bits, bool) or bits not in (1, 2, 3):
        raise ValueError("adc.low_resolution_bits must be 1, 2, or 3")
    supported_bits_raw = adc.get("supported_bits")
    if (
        not isinstance(supported_bits_raw, list)
        or any(
            isinstance(value, bool) or value not in (1, 2, 3)
            for value in supported_bits_raw
        )
        or len(supported_bits_raw) != 3
        or set(supported_bits_raw) != {1, 2, 3}
    ):
        raise ValueError("adc.supported_bits must contain 1, 2, and 3")
    if bits not in supported_bits_raw:
        raise ValueError("configured bit depth must be listed in supported_bits")

    spatial_alphas = _require_number_list(
        channel.get("spatial_alphas"), "channel.spatial_alphas"
    )
    if any(not 0.0 <= alpha < 1.0 for alpha in spatial_alphas):
        raise ValueError("channel.spatial_alphas values must be in [0, 1)")

    clip_sigma = _require_number(adc.get("clip_sigma"), "adc.clip_sigma")
    if clip_sigma <= 0:
        raise ValueError("adc.clip_sigma must be positive")
    scaling_model = adc.get("scaling")
    if scaling_model != "known_power_agc":
        raise ValueError("adc.scaling must be known_power_agc")
    one_bit_levels_raw = adc.get("one_bit_output_levels")
    if (
        not isinstance(one_bit_levels_raw, list)
        or len(one_bit_levels_raw) != 2
    ):
        raise ValueError("adc.one_bit_output_levels must contain two values")
    one_bit_levels = tuple(
        _require_number(level, "adc.one_bit_output_levels item")
        for level in one_bit_levels_raw
    )
    if (
        one_bit_levels[0] >= 0
        or one_bit_levels[1] <= 0
        or not isclose(one_bit_levels[0], -one_bit_levels[1])
    ):
        raise ValueError("one-bit output levels must be symmetric about zero")
    carrier_frequency_hz = _require_number(
        system.get("carrier_frequency_hz"), "system.carrier_frequency_hz"
    )
    channel_power = _require_number(
        system.get("channel_power"), "system.channel_power"
    )
    pilot_spacing_s = _require_number(
        system.get("pilot_spacing_s"), "system.pilot_spacing_s"
    )
    if carrier_frequency_hz <= 0 or channel_power <= 0:
        raise ValueError("carrier frequency and channel power must be positive")
    if pilot_spacing_s <= 0:
        raise ValueError("system.pilot_spacing_s must be positive")

    if kalman.get("observation_model") != "bussgang_linearized":
        raise ValueError("kalman.observation_model must be bussgang_linearized")
    if kalman.get("temporal_residual_model") != "white_approximation":
        raise ValueError(
            "kalman.temporal_residual_model must be white_approximation"
        )
    if kalman.get("use_true_jakes_rho") is not True:
        raise ValueError("kalman.use_true_jakes_rho must be true")

    snr_db = _require_number_list(sweeps.get("snr_db"), "sweeps.snr_db")
    velocity_kmh = _require_number_list(
        sweeps.get("velocity_kmh"), "sweeps.velocity_kmh"
    )
    if any(speed < 0 for speed in velocity_kmh):
        raise ValueError("sweep velocities must be non-negative")

    output_dir_raw = experiment.get("output_dir")
    if not isinstance(output_dir_raw, str) or not output_dir_raw:
        raise ValueError("experiment.output_dir must be a non-empty string")

    return Stage1Config(
        experiment_name=experiment_name,
        seed=_require_nonnegative_int(experiment.get("seed"), "experiment.seed"),
        n_sequences_per_point=_require_positive_int(
            experiment.get("n_sequences_per_point"),
            "experiment.n_sequences_per_point",
        ),
        batch_size=_require_positive_int(
            experiment.get("batch_size"), "experiment.batch_size"
        ),
        output_dir=Path(output_dir_raw),
        n_antennas=n_antennas,
        n_slots=_require_positive_int(system.get("n_slots"), "system.n_slots"),
        channel_power=channel_power,
        carrier_frequency_hz=carrier_frequency_hz,
        pilot_spacing_s=pilot_spacing_s,
        pilot_symbol=pilot_symbol,
        high_resolution_indices=indices,
        spatial_alphas=spatial_alphas,
        low_resolution_bits=int(bits),
        supported_bit_depths=tuple(supported_bits_raw),
        clip_sigma=clip_sigma,
        scaling_model=scaling_model,
        one_bit_levels=(one_bit_levels[0], one_bit_levels[1]),
        snr_db=snr_db,
        snr_sweep_velocity_kmh=_require_number(
            sweeps.get("snr_sweep_velocity_kmh"),
            "sweeps.snr_sweep_velocity_kmh",
        ),
        velocity_kmh=velocity_kmh,
        velocity_sweep_snr_db=_require_number(
            sweeps.get("velocity_sweep_snr_db"),
            "sweeps.velocity_sweep_snr_db",
        ),
        kalman_observation_model=str(kalman["observation_model"]),
        kalman_residual_model=str(kalman["temporal_residual_model"]),
        kalman_uses_true_rho=True,
    )
