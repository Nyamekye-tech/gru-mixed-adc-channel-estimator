# GRU Mixed-ADC Channel Estimator

Reproducible research code for temporal channel estimation in mixed-ADC
Massive MIMO. The project is being rebuilt from the thesis and its original
notebook; the legacy notebook is preserved separately and is not the source
of the new implementation.

## Current status

Stage 0: project scaffold and v1 review. No new scientific implementation is
included yet.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The Stage 1 protocol is configured in `configs/stage1.yaml`. Heavy datasets,
model checkpoints, and large raw results belong outside Git (for example, on
Google Drive); small result CSV/JSON files may be tracked. PyTorch wheels may
need a platform-specific install; Colab already provides a GPU-compatible
build.

## Stage 1: classical baselines

After installing requirements, run the CPU baseline sweeps and make plots
from their saved CSV:

```powershell
py -m experiments.run_stage1
py -m experiments.plot_stage1
py -m pytest -q
```

The Kalman result is explicitly a Bussgang-linearized filter with the true
Jakes coefficient and a white temporal-residual approximation, not an exact
nonlinear Bayesian filter. Stage 1 uses NumPy/SciPy and should not require
Colab; GPU work becomes useful for neural-model training and full sweeps.

## Layout

- `legacy/` — preserved v1 notebook and local reference copy of the thesis
- `src/chest/` — research package (implementation begins after Stage 0)
- `configs/` — experiment configurations
- `tests/` — focused scientific and reproducibility tests
- `experiments/` — command-line experiment entry points
- `results/` — small tracked results; large/raw output is ignored
- `notebooks/` — thin interactive runners

See [PROJECT_BRIEF.md](PROJECT_BRIEF.md) for scope and staged research rules.
