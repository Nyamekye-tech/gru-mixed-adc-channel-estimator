"""Memoryless and temporal estimators for mixed-ADC channel observations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from chest.quantizers import (
    gaussian_quantizer_moments,
    quantized_real_covariance,
)

ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class MixedADCBaselines:
    """Linear-estimation statistics for one channel/SNR/ADC configuration."""

    channel_covariance: ComplexArray
    observation_covariance: ComplexArray
    quantized_covariance: ComplexArray
    channel_quantized_covariance: ComplexArray
    observation_matrix: ComplexArray
    kalman_noise_covariance: ComplexArray
    high_resolution_indices: tuple[int, ...]
    low_resolution_indices: tuple[int, ...]
    scalar_gain_high: complex
    scalar_gain_low: complex

    def estimate_two_scalar(self, observations: ComplexArray) -> ComplexArray:
        """Apply one scalar gain per ADC class to current-slot observations."""
        self._validate_observations(observations)
        estimate = np.empty_like(observations, dtype=np.complex128)
        estimate[..., self.high_resolution_indices] = (
            self.scalar_gain_high * observations[..., self.high_resolution_indices]
        )
        estimate[..., self.low_resolution_indices] = (
            self.scalar_gain_low * observations[..., self.low_resolution_indices]
        )
        return estimate

    def estimate_bussgang_lmmse(self, observations: ComplexArray) -> ComplexArray:
        """Apply the full spatial Bussgang-LMMSE matrix independently per slot."""
        self._validate_observations(observations)
        cross = self.channel_quantized_covariance
        weights = np.linalg.solve(
            self.quantized_covariance.T,
            cross.T,
        ).T
        return np.einsum("...j,ij->...i", observations, weights, optimize=True)

    def estimate_bussgang_kalman(
        self,
        observations: ComplexArray,
        rho: float,
    ) -> ComplexArray:
        """Filter quantized slots with genie rho and white Bussgang residuals.

        The filter uses the stationary spatial prior and a linearized observation
        model. Temporal correlation in quantization residuals is approximated as
        white; this is not the exact nonlinear Bayesian filter.
        """
        self._validate_observations(observations)
        if observations.ndim != 3:
            raise ValueError("Kalman observations must have shape (batch, time, antennas)")
        if not -1.0 <= rho <= 1.0:
            raise ValueError("rho must be in [-1, 1]")

        batch_size, n_slots, n_antennas = observations.shape
        state_mean = np.zeros((batch_size, n_antennas), dtype=np.complex128)
        state_covariance = self.channel_covariance.copy()
        matrix = self.observation_matrix
        identity = np.eye(n_antennas, dtype=np.complex128)
        estimates = np.empty_like(observations, dtype=np.complex128)

        for slot in range(n_slots):
            innovation_covariance = (
                matrix @ state_covariance @ matrix.conj().T
                + self.kalman_noise_covariance
            )
            cross_covariance = state_covariance @ matrix.conj().T
            kalman_gain = np.linalg.solve(
                innovation_covariance.T,
                cross_covariance.T,
            ).T

            innovation = observations[:, slot, :] - state_mean @ matrix.T
            state_mean = state_mean + innovation @ kalman_gain.T
            residual_map = identity - kalman_gain @ matrix
            state_covariance = (
                residual_map
                @ state_covariance
                @ residual_map.conj().T
                + kalman_gain
                @ self.kalman_noise_covariance
                @ kalman_gain.conj().T
            )
            state_covariance = (
                state_covariance + state_covariance.conj().T
            ) / 2.0
            estimates[:, slot, :] = state_mean

            if slot + 1 < n_slots:
                state_mean = rho * state_mean
                state_covariance = (
                    rho**2 * state_covariance
                    + (1.0 - rho**2) * self.channel_covariance
                )

        return estimates

    def theoretical_nmse(
        self,
        estimator: Literal["two_scalar", "bussgang_lmmse"],
    ) -> float:
        """Return the one-slot linear estimator NMSE from its covariance."""
        covariance = self.channel_covariance
        cross = self.channel_quantized_covariance
        if estimator == "bussgang_lmmse":
            error_covariance = covariance - cross @ np.linalg.solve(
                self.quantized_covariance,
                cross.conj().T,
            )
        elif estimator == "two_scalar":
            gains = np.empty(covariance.shape[0], dtype=np.complex128)
            gains[list(self.high_resolution_indices)] = self.scalar_gain_high
            gains[list(self.low_resolution_indices)] = self.scalar_gain_low
            gain_matrix = np.diag(gains)
            error_covariance = (
                covariance
                - cross @ gain_matrix
                - gain_matrix @ cross.conj().T
                + gain_matrix
                @ self.quantized_covariance
                @ gain_matrix
            )
        else:
            raise ValueError(f"unknown estimator: {estimator}")
        return float(np.real(np.trace(error_covariance)) / np.real(np.trace(covariance)))

    def _validate_observations(self, observations: ComplexArray) -> None:
        if observations.shape[-1] != self.channel_covariance.shape[0]:
            raise ValueError("last observation dimension must match antenna count")


def build_mixed_adc_baselines(
    channel_covariance: ComplexArray,
    noise_variance: float,
    high_resolution_indices: tuple[int, ...] | list[int],
    bits: int = 1,
    clip_sigma: float = 3.0,
    one_bit_levels: Sequence[float] = (-1.0, 1.0),
    pilot_symbol: complex = 1.0 + 0.0j,
) -> MixedADCBaselines:
    """Build exact second-order Bussgang statistics for the configured ADCs."""
    covariance = np.asarray(channel_covariance, dtype=np.complex128)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("channel_covariance must be square")
    if not np.allclose(covariance, covariance.conj().T, atol=1e-10):
        raise ValueError("channel_covariance must be Hermitian")
    channel_diagonal = np.real(np.diag(covariance))
    if np.any(channel_diagonal <= 0.0) or not np.allclose(
        channel_diagonal, channel_diagonal[0], rtol=1e-10, atol=1e-12
    ):
        raise ValueError("channel covariance must have equal positive diagonal")
    if noise_variance < 0:
        raise ValueError("noise_variance must be non-negative")
    if not np.isfinite(pilot_symbol) or abs(pilot_symbol) == 0:
        raise ValueError("pilot_symbol must be finite and non-zero")
    if isinstance(bits, bool) or bits not in (1, 2, 3):
        raise ValueError("bits must be 1, 2, or 3")

    n_antennas = covariance.shape[0]
    high_indices = tuple(int(index) for index in high_resolution_indices)
    if len(set(high_indices)) != len(high_indices):
        raise ValueError("high-resolution indices must be unique")
    if any(index < 0 or index >= n_antennas for index in high_indices):
        raise ValueError("high-resolution antenna index is out of range")
    high_set = set(high_indices)
    low_indices = tuple(index for index in range(n_antennas) if index not in high_set)

    observation_covariance = (
        abs(pilot_symbol) ** 2 * covariance
        + noise_variance * np.eye(n_antennas)
    )
    diagonal = np.real(np.diag(observation_covariance))
    if not np.allclose(diagonal, diagonal[0], rtol=1e-10, atol=1e-12):
        raise ValueError("quantized antennas must have equal marginal variance")
    component_std = sqrt(float(diagonal[0]) / 2.0)
    xy_moment, quantized_power = gaussian_quantizer_moments(
        bits,
        component_std,
        clip_sigma,
        one_bit_levels,
    )
    low_beta = xy_moment / component_std**2

    beta = np.ones(n_antennas, dtype=np.float64)
    beta[list(low_indices)] = low_beta
    quantized_covariance = np.empty_like(observation_covariance)
    for row in range(n_antennas):
        for column in range(n_antennas):
            row_is_high = row in high_set
            column_is_high = column in high_set
            if row_is_high and column_is_high:
                value = observation_covariance[row, column]
            elif row_is_high:
                value = observation_covariance[row, column] * beta[column]
            elif column_is_high:
                value = beta[row] * observation_covariance[row, column]
            else:
                correlation = (
                    observation_covariance[row, column] / diagonal[0]
                )
                real_covariance = quantized_real_covariance(
                    float(np.clip(correlation.real, -1.0, 1.0)),
                    bits,
                    component_std,
                    clip_sigma,
                    one_bit_levels=one_bit_levels,
                )
                imag_covariance = quantized_real_covariance(
                    float(np.clip(correlation.imag, -1.0, 1.0)),
                    bits,
                    component_std,
                    clip_sigma,
                    one_bit_levels=one_bit_levels,
                )
                value = 2.0 * (real_covariance + 1j * imag_covariance)
            quantized_covariance[row, column] = value
    quantized_covariance = (
        quantized_covariance + quantized_covariance.conj().T
    ) / 2.0

    eigenvalues = np.linalg.eigvalsh(quantized_covariance)
    if eigenvalues[0] <= 0.0:
        raise ValueError(
            "quantized observation covariance is not positive definite "
            f"(minimum eigenvalue {eigenvalues[0]:.3g})"
        )

    channel_quantized_covariance = (
        np.conj(pilot_symbol) * covariance @ np.diag(beta)
    )
    observation_matrix = pilot_symbol * np.diag(beta.astype(np.complex128))
    kalman_noise_covariance = (
        quantized_covariance
        - observation_matrix @ covariance @ observation_matrix.conj().T
    )
    kalman_noise_covariance = (
        kalman_noise_covariance + kalman_noise_covariance.conj().T
    ) / 2.0
    kalman_eigenvalues = np.linalg.eigvalsh(kalman_noise_covariance)
    if kalman_eigenvalues[0] < -1e-8:
        raise ValueError(
            "Bussgang observation residual covariance is not positive "
            f"semidefinite (minimum eigenvalue {kalman_eigenvalues[0]:.3g})"
        )

    channel_power = float(channel_diagonal[0])
    scalar_gain_high = (
        np.conj(pilot_symbol) * channel_power
        / (abs(pilot_symbol) ** 2 * channel_power + noise_variance)
    )
    scalar_gain_low = (
        np.conj(pilot_symbol)
        * channel_power
        * low_beta
        / (2.0 * quantized_power)
    )
    return MixedADCBaselines(
        channel_covariance=covariance,
        observation_covariance=observation_covariance,
        quantized_covariance=quantized_covariance,
        channel_quantized_covariance=channel_quantized_covariance,
        observation_matrix=observation_matrix,
        kalman_noise_covariance=kalman_noise_covariance,
        high_resolution_indices=high_indices,
        low_resolution_indices=low_indices,
        scalar_gain_high=scalar_gain_high,
        scalar_gain_low=scalar_gain_low,
    )
