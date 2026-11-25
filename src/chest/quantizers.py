"""Mixed-ADC quantizers and Gaussian moments used by Bussgang baselines."""

from __future__ import annotations

from collections.abc import Sequence
from math import exp, pi, sqrt
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr


class QuantizerParameters(NamedTuple):
    levels: NDArray[np.float64]
    thresholds: NDArray[np.float64]


def _validate_one_bit_levels(levels: Sequence[float]) -> tuple[float, float]:
    if len(levels) != 2:
        raise ValueError("one_bit_levels must contain two values")
    negative, positive = float(levels[0]), float(levels[1])
    if negative >= 0.0 or positive <= 0.0 or not np.isclose(negative, -positive):
        raise ValueError("one_bit_levels must be a symmetric pair around zero")
    return negative, positive


def quantizer_parameters(
    bits: int,
    component_std: float,
    clip_sigma: float = 3.0,
    one_bit_levels: Sequence[float] = (-1.0, 1.0),
) -> QuantizerParameters:
    """Return output levels and decision thresholds for one real component."""
    if isinstance(bits, bool) or bits not in (1, 2, 3):
        raise ValueError("bits must be 1, 2, or 3")
    if component_std <= 0 or clip_sigma <= 0:
        raise ValueError("component_std and clip_sigma must be positive")

    if bits == 1:
        return QuantizerParameters(
            levels=np.asarray(_validate_one_bit_levels(one_bit_levels)),
            thresholds=np.asarray([0.0]),
        )

    n_levels = 2**bits
    step = 2.0 * clip_sigma * component_std / n_levels
    levels = -clip_sigma * component_std + step * (
        np.arange(n_levels, dtype=np.float64) + 0.5
    )
    thresholds = -clip_sigma * component_std + step * np.arange(
        1, n_levels, dtype=np.float64
    )
    return QuantizerParameters(levels=levels, thresholds=thresholds)


def quantize_real(
    values: NDArray[np.float64] | float,
    bits: int,
    component_std: float,
    clip_sigma: float = 3.0,
    one_bit_levels: Sequence[float] = (-1.0, 1.0),
) -> NDArray[np.float64]:
    """Quantize one real-valued component; outer bins saturate."""
    params = quantizer_parameters(
        bits, component_std, clip_sigma, one_bit_levels
    )
    values_array = np.asarray(values, dtype=np.float64)
    indices = np.searchsorted(params.thresholds, values_array, side="right")
    return np.asarray(params.levels[indices], dtype=np.float64)


def quantize_complex(
    values: NDArray[np.complex128],
    bits: int,
    component_std: float,
    clip_sigma: float = 3.0,
    one_bit_levels: Sequence[float] = (-1.0, 1.0),
) -> NDArray[np.complex128]:
    """Quantize I and Q independently using the same real quantizer."""
    real = quantize_real(
        values.real, bits, component_std, clip_sigma, one_bit_levels
    )
    imag = quantize_real(
        values.imag, bits, component_std, clip_sigma, one_bit_levels
    )
    return real + 1j * imag


def mixed_adc_quantize(
    values: NDArray[np.complex128],
    high_resolution_indices: Sequence[int],
    bits: int,
    component_std: float,
    clip_sigma: float = 3.0,
    one_bit_levels: Sequence[float] = (-1.0, 1.0),
) -> NDArray[np.complex128]:
    """Pass configured high-resolution antennas and quantize all others."""
    if values.ndim < 1:
        raise ValueError("values must have an antenna axis")
    n_antennas = values.shape[-1]
    indices = np.asarray(high_resolution_indices, dtype=int)
    if indices.ndim != 1:
        raise ValueError("high_resolution_indices must be one-dimensional")
    if np.unique(indices).size != indices.size:
        raise ValueError("high_resolution_indices must not contain duplicates")
    if np.any(indices < 0) or np.any(indices >= n_antennas):
        raise ValueError("high-resolution antenna index is out of range")

    quantized = np.array(values, dtype=np.complex128, copy=True)
    low_resolution_mask = np.ones(n_antennas, dtype=bool)
    low_resolution_mask[indices] = False
    quantized[..., low_resolution_mask] = quantize_complex(
        values[..., low_resolution_mask],
        bits,
        component_std,
        clip_sigma,
        one_bit_levels,
    )
    return quantized


def gaussian_quantizer_moments(
    bits: int,
    component_std: float,
    clip_sigma: float = 3.0,
    one_bit_levels: Sequence[float] = (-1.0, 1.0),
) -> tuple[float, float]:
    """Return E[X Q(X)] and E[Q(X)^2] for X~N(0, component_std^2)."""
    params = quantizer_parameters(
        bits, component_std, clip_sigma, one_bit_levels
    )
    edges = np.concatenate(([-np.inf], params.thresholds, [np.inf]))
    standardized_edges = edges / component_std
    probabilities = np.diff(ndtr(standardized_edges))

    def standard_normal_pdf(value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        return exp(-0.5 * value**2) / sqrt(2.0 * pi)

    boundary_pdf = np.asarray(
        [standard_normal_pdf(float(edge)) for edge in standardized_edges]
    )
    cross_moment = component_std * float(
        np.dot(params.levels, boundary_pdf[:-1] - boundary_pdf[1:])
    )
    quantized_power = float(np.dot(params.levels**2, probabilities))
    return cross_moment, quantized_power


def quantized_real_covariance(
    correlation: float,
    bits: int,
    component_std: float,
    clip_sigma: float = 3.0,
    quadrature_order: int = 64,
    one_bit_levels: Sequence[float] = (-1.0, 1.0),
) -> float:
    """Return E[Q(X)Q(Y)] for correlated, equal-variance Gaussian X and Y."""
    if not -1.0 <= correlation <= 1.0:
        raise ValueError("correlation must be in [-1, 1]")
    params = quantizer_parameters(
        bits, component_std, clip_sigma, one_bit_levels
    )

    if bits == 1:
        amplitude = params.levels[1]
        return float(amplitude**2 * 2.0 / pi * np.arcsin(correlation))

    _, quantized_power = gaussian_quantizer_moments(
        bits, component_std, clip_sigma, one_bit_levels
    )
    if correlation >= 1.0 - 1e-12:
        return quantized_power
    if correlation <= -1.0 + 1e-12:
        return -quantized_power

    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
    standard_samples = np.sqrt(2.0) * nodes
    normal_weights = weights / np.sqrt(pi)
    quantized_samples = quantize_real(
        component_std * standard_samples,
        bits,
        component_std,
        clip_sigma,
        one_bit_levels,
    )

    conditional_std = sqrt(1.0 - correlation**2)
    conditional_mean = np.zeros_like(standard_samples)
    standardized_edges = np.concatenate(
        (
            [-np.inf],
            params.thresholds / component_std,
            [np.inf],
        )
    )
    for index, level in enumerate(params.levels):
        lower = (
            standardized_edges[index] - correlation * standard_samples
        ) / conditional_std
        upper = (
            standardized_edges[index + 1] - correlation * standard_samples
        ) / conditional_std
        conditional_mean += level * (ndtr(upper) - ndtr(lower))

    return float(
        np.dot(normal_weights, quantized_samples * conditional_mean)
    )
