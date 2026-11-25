"""Spatially correlated, time-varying channel generation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import j0

ComplexArray = NDArray[np.complex128]


def doppler_frequency_hz(
    velocity_kmh: float,
    carrier_frequency_hz: float = 2.6e9,
    speed_of_light_m_s: float = 3.0e8,
) -> float:
    """Return maximum Doppler frequency for a given user speed."""
    if velocity_kmh < 0:
        raise ValueError("velocity_kmh must be non-negative")
    if carrier_frequency_hz <= 0 or speed_of_light_m_s <= 0:
        raise ValueError("carrier frequency and speed of light must be positive")
    return (velocity_kmh / 3.6) * carrier_frequency_hz / speed_of_light_m_s


def jakes_rho(
    velocity_kmh: float,
    pilot_spacing_s: float = 5e-4,
    carrier_frequency_hz: float = 2.6e9,
) -> float:
    """Return the one-step Jakes correlation used by the Gauss-Markov model."""
    if pilot_spacing_s <= 0:
        raise ValueError("pilot_spacing_s must be positive")
    fd = doppler_frequency_hz(velocity_kmh, carrier_frequency_hz)
    return float(j0(2.0 * np.pi * fd * pilot_spacing_s))


def exponential_spatial_covariance(
    n_antennas: int,
    alpha: float,
    channel_power: float = 1.0,
) -> ComplexArray:
    """Build the ULA exponential covariance R[i,j] = alpha**abs(i-j)."""
    if n_antennas <= 0:
        raise ValueError("n_antennas must be positive")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    if channel_power <= 0:
        raise ValueError("channel_power must be positive")
    indices = np.arange(n_antennas)
    return np.asarray(
        channel_power
        * alpha ** np.abs(indices[:, None] - indices[None, :]),
        dtype=np.complex128,
    )


def _correlate_standard_normals(
    white: ComplexArray,
    covariance: ComplexArray,
) -> ComplexArray:
    cholesky = np.linalg.cholesky(covariance)
    return np.einsum("...j,ij->...i", white, cholesky, optimize=True)


def generate_channel_sequences(
    rng: np.random.Generator,
    n_sequences: int,
    n_slots: int,
    covariance: ComplexArray,
    rho: float,
) -> ComplexArray:
    """Generate stationary CN(0,R) Gauss-Markov channels with shape (B,T,N)."""
    if n_sequences <= 0 or n_slots <= 0:
        raise ValueError("n_sequences and n_slots must be positive")
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if not np.allclose(covariance, covariance.conj().T):
        raise ValueError("covariance must be Hermitian")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be in [-1, 1]")

    n_antennas = covariance.shape[0]
    white = (
        rng.standard_normal((n_sequences, n_slots, n_antennas))
        + 1j * rng.standard_normal((n_sequences, n_slots, n_antennas))
    ) / np.sqrt(2.0)
    innovations = _correlate_standard_normals(white, covariance)

    channels = np.empty_like(innovations)
    channels[:, 0, :] = innovations[:, 0, :]
    innovation_scale = np.sqrt(max(0.0, 1.0 - rho**2))
    for slot in range(1, n_slots):
        channels[:, slot, :] = (
            rho * channels[:, slot - 1, :]
            + innovation_scale * innovations[:, slot, :]
        )
    return channels


def add_awgn(
    rng: np.random.Generator,
    channels: ComplexArray,
    noise_variance: float,
) -> ComplexArray:
    """Add circular complex Gaussian noise with E[|n|^2] = noise_variance."""
    if noise_variance < 0:
        raise ValueError("noise_variance must be non-negative")
    noise = (
        rng.standard_normal(channels.shape)
        + 1j * rng.standard_normal(channels.shape)
    ) * np.sqrt(noise_variance / 2.0)
    return channels + noise


def high_resolution_mask(
    n_antennas: int,
    high_resolution_indices: Sequence[int],
) -> NDArray[np.bool_]:
    """Return a mask for the configured high-resolution antenna indices."""
    indices = np.asarray(high_resolution_indices, dtype=int)
    if indices.ndim != 1:
        raise ValueError("high_resolution_indices must be one-dimensional")
    if np.unique(indices).size != indices.size:
        raise ValueError("high_resolution_indices must not contain duplicates")
    if np.any(indices < 0) or np.any(indices >= n_antennas):
        raise ValueError("high-resolution antenna index is out of range")
    mask = np.zeros(n_antennas, dtype=bool)
    mask[indices] = True
    return mask
