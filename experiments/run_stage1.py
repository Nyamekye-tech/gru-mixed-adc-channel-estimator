"""Run configured classical-baseline sweeps and save raw results."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from chest.baselines import build_mixed_adc_baselines
from chest.channel import (
    add_awgn,
    exponential_spatial_covariance,
    generate_channel_sequences,
    jakes_rho,
)
from chest.config import Stage1Config, load_stage1_config
from chest.quantizers import mixed_adc_quantize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_NAMES = (
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
    return result.stdout.strip()


def _noise_variance(snr_db: float, pilot_power: float) -> float:
    return float(pilot_power * 10.0 ** (-snr_db / 10.0))


def _condition_seed(
    base_seed: int,
    alpha_index: int,
    sweep_index: int,
    point_index: int,
) -> int:
    sequence = np.random.SeedSequence(
        [base_seed, alpha_index, sweep_index, point_index]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _run_condition(
    config: Stage1Config,
    alpha: float,
    alpha_index: int,
    sweep_name: str,
    sweep_index: int,
    point_index: int,
    sweep_value: float,
    git_commit: str,
    config_json: str,
) -> list[dict[str, Any]]:
    if sweep_name == "snr":
        snr_db = sweep_value
        velocity_kmh = config.snr_sweep_velocity_kmh
    elif sweep_name == "velocity":
        snr_db = config.velocity_sweep_snr_db
        velocity_kmh = sweep_value
    else:
        raise ValueError(f"unsupported sweep: {sweep_name}")

    pilot_power = abs(config.pilot_symbol) ** 2
    noise_variance = _noise_variance(
        snr_db,
        pilot_power * config.channel_power,
    )
    rho = jakes_rho(
        velocity_kmh,
        config.pilot_spacing_s,
        config.carrier_frequency_hz,
    )
    covariance = exponential_spatial_covariance(
        config.n_antennas,
        alpha,
        config.channel_power,
    )
    baselines = build_mixed_adc_baselines(
        covariance,
        noise_variance,
        config.high_resolution_indices,
        bits=config.low_resolution_bits,
        clip_sigma=config.clip_sigma,
        one_bit_levels=config.one_bit_levels,
        pilot_symbol=config.pilot_symbol,
    )

    condition_seed = _condition_seed(
        config.seed,
        alpha_index,
        sweep_index,
        point_index,
    )
    rng = np.random.default_rng(condition_seed)
    error_power = {name: 0.0 for name in BASELINE_NAMES}
    target_power = 0.0
    component_std = np.sqrt(
        (pilot_power * config.channel_power + noise_variance) / 2.0
    )

    for start in range(0, config.n_sequences_per_point, config.batch_size):
        batch_size = min(
            config.batch_size,
            config.n_sequences_per_point - start,
        )
        channels = generate_channel_sequences(
            rng,
            batch_size,
            config.n_slots,
            covariance,
            rho,
        )
        received = add_awgn(
            rng,
            config.pilot_symbol * channels,
            noise_variance,
        )
        quantized = mixed_adc_quantize(
            received,
            config.high_resolution_indices,
            config.low_resolution_bits,
            component_std,
            config.clip_sigma,
            config.one_bit_levels,
        )
        estimates = (
            baselines.estimate_two_scalar(quantized),
            baselines.estimate_bussgang_lmmse(quantized),
            baselines.estimate_bussgang_kalman(quantized, rho),
        )
        target_power += float(np.sum(np.abs(channels) ** 2))
        for name, estimate in zip(BASELINE_NAMES, estimates, strict=True):
            error_power[name] += float(
                np.sum(np.abs(estimate - channels) ** 2)
            )

    scalar_nmse = error_power["Two-scalar"] / target_power
    rows = []
    for name in BASELINE_NAMES:
        value = error_power[name] / target_power
        if name == "Bussgang-Kalman (true rho; white residual)":
            allowed_excess = noise_variance
        elif name == "Bussgang-LMMSE":
            allowed_excess = 1e-8
        else:
            allowed_excess = 0.0
        rows.append(
            {
                "experiment": config.experiment_name,
                "sweep": sweep_name,
                "sweep_value": sweep_value,
                "snr_db": snr_db,
                "velocity_kmh": velocity_kmh,
                "rho": rho,
                "spatial_alpha": alpha,
                "bits_per_real_component": config.low_resolution_bits,
                "n_sequences": config.n_sequences_per_point,
                "seed": condition_seed,
                "baseline": name,
                "nmse": value,
                "nmse_db": float(10.0 * np.log10(value)),
                "reference_two_scalar_nmse": scalar_nmse,
                "allowed_excess_nmse": allowed_excess,
                "passes_comparison_guard": value
                <= scalar_nmse + allowed_excess,
                "config_json": config_json,
                "git_commit": git_commit,
            }
        )
    return rows


def run_experiment(
    config: Stage1Config,
    config_snapshot: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run both configured baseline sweeps."""
    commit = _git_commit()
    config_json = json.dumps(config_snapshot, sort_keys=True, separators=(",", ":"))
    rows: list[dict[str, Any]] = []
    for alpha_index, alpha in enumerate(config.spatial_alphas):
        for point_index, snr_db in enumerate(config.snr_db):
            rows.extend(
                _run_condition(
                    config,
                    alpha,
                    alpha_index,
                    "snr",
                    0,
                    point_index,
                    snr_db,
                    commit,
                    config_json,
                )
            )
        for point_index, velocity_kmh in enumerate(config.velocity_kmh):
            rows.extend(
                _run_condition(
                    config,
                    alpha,
                    alpha_index,
                    "velocity",
                    1,
                    point_index,
                    velocity_kmh,
                    commit,
                    config_json,
                )
            )

    frame = pd.DataFrame(rows)
    metadata = {
        "experiment": config.experiment_name,
        "git_commit": commit,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config_snapshot,
        "baseline_gate_failures": frame.loc[
            ~frame["passes_comparison_guard"],
            ["sweep", "sweep_value", "spatial_alpha", "baseline"],
        ].to_dict(orient="records"),
    }
    return frame, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "stage1.yaml",
    )
    args = parser.parse_args()
    config = load_stage1_config(args.config)
    config_snapshot = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config_snapshot, dict):
        raise ValueError("configuration root must be a mapping")
    output_dir = PROJECT_ROOT / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "stage1_baselines.csv"
    metadata_path = output_dir / "stage1_metadata.json"
    if csv_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing Stage 1 results in {output_dir}"
        )

    results, metadata = run_experiment(config, config_snapshot)
    results.to_csv(csv_path, index=False)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    failures = metadata["baseline_gate_failures"]
    if failures:
        raise RuntimeError(
            "Baseline comparison guard failed; results were saved for "
            f"diagnosis: {failures}"
        )

    print(f"Saved {len(results)} baseline rows to {csv_path}")
    print(f"Saved config and source revision to {metadata_path}")


if __name__ == "__main__":
    main()
