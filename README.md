# GRU Mixed-ADC Channel Estimator

Reproducible research code for temporal channel estimation in mixed-ADC
Massive MIMO. The project is being rebuilt from the thesis and its original
notebook; the legacy notebook is preserved separately and is not the source
of the new implementation.

## Current status

Stage 1 classical baselines are implemented and tested. Stage 2 neural models
and their bounded validation search are implemented; run the configured
comparison on Colab or another CUDA machine before treating the neural results
as complete.

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

After installing requirements, run the CPU baseline sweep and make plots from
its saved CSV:

```powershell
py -m experiments.run_stage1
py -m experiments.plot_stage1
py -m pytest -q
```

The Kalman result is explicitly a Bussgang-linearized filter with the true
Jakes coefficient and a white temporal-residual approximation, not an exact
nonlinear Bayesian filter. Stage 1 uses NumPy/SciPy and should not require
Colab; GPU work becomes useful for neural-model training and full sweeps.

## Stage 2: neural comparisons

The Stage 2 protocol is explicit in `configs/stage2.yaml`. It trains and
validation-selects GRU, LSTM, causal windowed FNN/CNN, and the published
Gao et al. selective-input prediction DNN. The FNN and CNN use the same
20-slot causal history as the recurrent models. The Gao SIP-DNN is a
single-slot comparator: its parallel refinement and modified-prediction
networks use only high-resolution ADC observations, as in the original paper.
It is not the signal-reconstruction cascade described under the same name in
the thesis.

Run the bounded search and held-out evaluation with:

```powershell
py -m experiments.run_stage2 --config configs/stage2.yaml
```

The configured search has four candidates per neural architecture, the same
train/validation/test splits and training budget, and selects settings using
validation NMSE only. Results include all classical estimators and a strict
two-scalar comparison guard. If a method fails that guard, its CSV/JSON
diagnostic outputs are still saved and the command exits unsuccessfully.
The script requires a clean Git worktree so each result records the exact
source commit. A CUDA GPU is recommended for the 20-candidate search; local
CPU execution is suitable for tests and small checks.

The verified original reference is Shen Gao et al., “Deep Learning Based
Channel Estimation for Massive MIMO With Mixed-Resolution ADCs,” *IEEE
Communications Letters*, vol. 23, no. 11, pp. 1989–1993, 2019,
doi: [10.1109/LCOMM.2019.2936393](https://doi.org/10.1109/LCOMM.2019.2936393),
also available as [arXiv:1908.06245](https://arxiv.org/abs/1908.06245).

## Layout

- `legacy/` — preserved v1 notebook and local reference copy of the thesis
- `src/chest/` — research package (implementation begins after Stage 0)
- `configs/` — experiment configurations
- `tests/` — focused scientific and reproducibility tests
- `experiments/` — command-line experiment entry points
- `results/` — small tracked results; large/raw output is ignored
- `notebooks/` — thin interactive runners

See [PROJECT_BRIEF.md](PROJECT_BRIEF.md) for scope and staged research rules.
