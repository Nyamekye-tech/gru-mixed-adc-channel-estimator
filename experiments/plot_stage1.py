"""Plot Stage 1 curves exclusively from the saved result CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ORDER = (
    "Two-scalar",
    "Bussgang-LMMSE",
    "Bussgang-Kalman (true rho; white residual)",
)


def plot_sweep(
    results: pd.DataFrame,
    sweep: str,
    x_column: str,
    x_label: str,
    output_path: Path,
) -> None:
    subset = results.loc[results["sweep"] == sweep]
    if subset.empty:
        raise ValueError(f"saved results do not contain the {sweep!r} sweep")

    figure, axis = plt.subplots(figsize=(8, 5))
    for alpha in sorted(subset["spatial_alpha"].unique()):
        alpha_results = subset.loc[subset["spatial_alpha"] == alpha]
        for baseline in BASELINE_ORDER:
            curve = alpha_results.loc[
                alpha_results["baseline"] == baseline
            ].sort_values(x_column)
            if curve.empty:
                raise ValueError(
                    f"missing {baseline!r} results for spatial alpha {alpha}"
                )
            axis.plot(
                curve[x_column],
                curve["nmse_db"],
                marker="o",
                label=f"{baseline}, spatial alpha={alpha:g}",
            )

    axis.set_xlabel(x_label)
    axis.set_ylabel("NMSE (dB)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "results" / "stage1_baselines.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures",
    )
    args = parser.parse_args()
    results = pd.read_csv(args.input)
    plot_sweep(
        results,
        "snr",
        "snr_db",
        "SNR (dB)",
        args.output_dir / "stage1_nmse_vs_snr.png",
    )
    plot_sweep(
        results,
        "velocity",
        "velocity_kmh",
        "User velocity (km/h)",
        args.output_dir / "stage1_nmse_vs_velocity.png",
    )
    print(f"Saved plots derived from {args.input}")


if __name__ == "__main__":
    main()
