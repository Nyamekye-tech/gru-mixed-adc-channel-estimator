import numpy as np
import pytest

from chest.baselines import build_mixed_adc_baselines
from chest.channel import (
    add_awgn,
    exponential_spatial_covariance,
    generate_channel_sequences,
    jakes_rho,
)
from chest.config import load_stage1_config
from chest.metrics import nmse
from chest.quantizers import (
    mixed_adc_quantize,
    quantize_real,
    quantizer_parameters,
)


def test_quantizer_levels_and_saturation() -> None:
    sigma = 0.8
    expected_level_counts = {1: 2, 2: 4, 3: 8}

    for bits, expected_count in expected_level_counts.items():
        parameters = quantizer_parameters(bits, sigma, clip_sigma=3.0)
        quantized = quantize_real(
            np.asarray([-100.0, 0.0, 100.0]),
            bits,
            sigma,
            clip_sigma=3.0,
        )
        assert parameters.levels.size == expected_count
        assert np.all(np.diff(parameters.levels) > 0)
        assert quantized[0] == parameters.levels[0]
        assert quantized[-1] == parameters.levels[-1]

    assert quantizer_parameters(1, sigma).levels.tolist() == [-1.0, 1.0]
    assert quantizer_parameters(2, sigma).levels == pytest.approx(
        [-1.8, -0.6, 0.6, 1.8]
    )


def test_one_bit_output_levels_are_configurable() -> None:
    values = np.asarray([-1.0, 1.0])
    assert quantize_real(
        values, 1, component_std=0.5, one_bit_levels=(-2.0, 2.0)
    ).tolist() == [-2.0, 2.0]


def test_mixed_adc_keeps_selected_antennas_unquantized() -> None:
    values = np.asarray([0.3 + 0.7j, -0.4 - 0.2j, 1.1 + 0.1j, -0.9 + 1.5j])
    quantized = mixed_adc_quantize(
        values,
        high_resolution_indices=[0, 2],
        bits=1,
        component_std=1.0,
    )

    np.testing.assert_array_equal(quantized[[0, 2]], values[[0, 2]])
    assert set(quantized[[1, 3]].real).issubset({-1.0, 1.0})
    assert set(quantized[[1, 3]].imag).issubset({-1.0, 1.0})


def test_spatial_covariance_is_unit_diagonal_and_positive_definite() -> None:
    covariance = exponential_spatial_covariance(16, 0.7)
    np.testing.assert_allclose(np.diag(covariance), 1.0)
    np.testing.assert_allclose(covariance, covariance.conj().T)
    assert np.linalg.eigvalsh(covariance)[0] > 0.0
    np.testing.assert_array_equal(
        exponential_spatial_covariance(4, 0.0),
        np.eye(4),
    )


def test_jakes_rho_values_match_configured_physics() -> None:
    assert jakes_rho(0.0) == pytest.approx(1.0)
    assert jakes_rho(3.0) == pytest.approx(0.9998713032810587, abs=1e-12)
    assert jakes_rho(60.0) == pytest.approx(0.949178434829805, abs=1e-12)
    assert jakes_rho(130.0) == pytest.approx(0.772543463230408, abs=1e-12)


def test_generated_channel_has_expected_stationary_spatial_covariance() -> None:
    covariance = exponential_spatial_covariance(3, 0.35)
    channels = generate_channel_sequences(
        np.random.default_rng(12),
        n_sequences=12000,
        n_slots=1,
        covariance=covariance,
        rho=0.8,
    )[:, 0, :]
    empirical_covariance = channels.T @ channels.conj() / channels.shape[0]
    np.testing.assert_allclose(empirical_covariance, covariance, atol=0.035)


def test_nmse_is_ratio_of_total_error_to_total_target_power() -> None:
    target = np.asarray([1.0, 10.0])
    estimate = np.asarray([0.0, 12.0])
    assert nmse(estimate, target) == pytest.approx(5.0 / 101.0)


@pytest.mark.parametrize("alpha", [0.0, 0.7])
def test_bussgang_lmmse_is_not_worse_than_two_scalar(alpha: float) -> None:
    covariance = exponential_spatial_covariance(16, alpha)
    model = build_mixed_adc_baselines(
        covariance,
        noise_variance=0.1,
        high_resolution_indices=[0, 4, 8, 12],
        bits=1,
    )
    assert model.theoretical_nmse("bussgang_lmmse") <= (
        model.theoretical_nmse("two_scalar") + 1e-12
    )


def test_iid_one_bit_two_scalar_baseline_matches_known_30_db_floor() -> None:
    covariance = exponential_spatial_covariance(32, 0.0)
    model = build_mixed_adc_baselines(
        covariance,
        noise_variance=10.0**(-30.0 / 10.0),
        high_resolution_indices=[0, 8, 16, 24],
        bits=1,
    )
    estimate_db = 10.0 * np.log10(model.theoretical_nmse("two_scalar"))
    assert estimate_db == pytest.approx(-4.98, abs=0.03)


def test_bussgang_covariance_is_positive_for_supported_bit_depths() -> None:
    covariance = exponential_spatial_covariance(8, 0.4)
    for bits in (1, 2, 3):
        model = build_mixed_adc_baselines(
            covariance,
            noise_variance=0.1,
            high_resolution_indices=[0, 4],
            bits=bits,
        )
        assert np.linalg.eigvalsh(model.quantized_covariance)[0] > 0.0
        assert np.linalg.eigvalsh(model.kalman_noise_covariance)[0] >= -1e-8


@pytest.mark.parametrize("bits", [1, 2, 3])
def test_bussgang_quantized_covariance_matches_monte_carlo(bits: int) -> None:
    rng = np.random.default_rng(31 + bits)
    covariance = exponential_spatial_covariance(4, 0.4)
    noise_variance = 0.1
    observation_covariance = covariance + noise_variance * np.eye(4)
    model = build_mixed_adc_baselines(
        covariance,
        noise_variance,
        high_resolution_indices=[0, 2],
        bits=bits,
    )
    component_std = np.sqrt((1.0 + noise_variance) / 2.0)
    cholesky = np.linalg.cholesky(observation_covariance)
    white = (
        rng.standard_normal((50000, 4))
        + 1j * rng.standard_normal((50000, 4))
    ) / np.sqrt(2.0)
    received = white @ cholesky.T
    quantized = mixed_adc_quantize(
        received,
        [0, 2],
        bits,
        component_std,
    )
    empirical_covariance = quantized.T @ quantized.conj() / len(quantized)
    np.testing.assert_allclose(
        empirical_covariance,
        model.quantized_covariance,
        atol=0.025,
    )


def test_true_rho_kalman_is_not_worse_than_scalar_by_more_than_noise() -> None:
    rng = np.random.default_rng(23)
    covariance = exponential_spatial_covariance(4, 0.35)
    noise_variance = 0.1
    rho = jakes_rho(3.0)
    channels = generate_channel_sequences(
        rng,
        n_sequences=512,
        n_slots=20,
        covariance=covariance,
        rho=rho,
    )
    observations = add_awgn(rng, channels, noise_variance)
    model = build_mixed_adc_baselines(
        covariance,
        noise_variance=noise_variance,
        high_resolution_indices=[0, 1, 2, 3],
        bits=1,
    )

    scalar_estimate = model.estimate_two_scalar(observations)
    kalman_estimate = model.estimate_bussgang_kalman(observations, rho)
    assert nmse(kalman_estimate, channels) <= (
        nmse(scalar_estimate, channels) + noise_variance
    )


def test_stage1_yaml_loads_explicit_approved_protocol() -> None:
    config = load_stage1_config("configs/stage1.yaml")
    assert config.spatial_alphas == (0.0, 0.7)
    assert config.high_resolution_indices == (0, 8, 16, 24)
    assert config.low_resolution_bits == 1
    assert config.one_bit_levels == (-1.0, 1.0)
