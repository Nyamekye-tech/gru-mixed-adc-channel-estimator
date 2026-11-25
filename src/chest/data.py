"""Reproducible neural-model datasets from the configured channel model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chest.channel import (
    add_awgn,
    exponential_spatial_covariance,
    generate_channel_sequences,
    jakes_rho,
)
from chest.config import Stage2Config
from chest.quantizers import mixed_adc_quantize

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float32]


def complex_to_real(values: ComplexArray) -> RealArray:
    """Stack real and imaginary antenna features in the project convention."""
    return np.concatenate((values.real, values.imag), axis=-1).astype(
        np.float32,
        copy=False,
    )


def generate_stage2_dataset(
    config: Stage2Config,
    n_sequences: int,
    seed: int,
) -> tuple[ComplexArray, ComplexArray]:
    """Generate quantized observations and matching channels for one split."""
    if n_sequences <= 0:
        raise ValueError("n_sequences must be positive")

    rng = np.random.default_rng(seed)
    covariance = exponential_spatial_covariance(
        config.n_antennas,
        config.spatial_alpha,
        config.channel_power,
    )
    rho = jakes_rho(
        config.velocity_kmh,
        config.pilot_spacing_s,
        config.carrier_frequency_hz,
    )
    snr_linear = 10.0 ** (config.snr_db / 10.0)
    noise_variance = (
        abs(config.pilot_symbol) ** 2 * config.channel_power / snr_linear
    )
    channels = generate_channel_sequences(
        rng,
        n_sequences,
        config.n_slots,
        covariance,
        rho,
    )
    received = add_awgn(
        rng,
        config.pilot_symbol * channels,
        noise_variance,
    )
    component_std = np.sqrt(
        (
            abs(config.pilot_symbol) ** 2 * config.channel_power
            + noise_variance
        )
        / 2.0
    )
    observations = mixed_adc_quantize(
        received,
        config.high_resolution_indices,
        config.low_resolution_bits,
        component_std,
        config.clip_sigma,
        config.one_bit_levels,
    )
    return observations, channels
