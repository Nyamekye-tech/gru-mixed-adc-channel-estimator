"""Channel-estimation error metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def nmse(estimate: ArrayLike, target: ArrayLike) -> float:
    """Compute the ratio of total squared error to total target power."""
    estimate_array = np.asarray(estimate)
    target_array = np.asarray(target)
    if estimate_array.shape != target_array.shape:
        raise ValueError("estimate and target must have the same shape")
    if target_array.size == 0:
        raise ValueError("estimate and target must not be empty")

    error_power = float(np.sum(np.abs(estimate_array - target_array) ** 2))
    target_power = float(np.sum(np.abs(target_array) ** 2))
    if target_power <= 0:
        raise ValueError("target must have positive total power")
    return error_power / target_power


def nmse_db(estimate: ArrayLike, target: ArrayLike) -> float:
    """Return ratio-of-total-power NMSE in decibels."""
    value = nmse(estimate, target)
    if value == 0:
        return float("-inf")
    return float(10.0 * np.log10(value))
