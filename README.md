# Channel Estimator

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

The experiment configuration and runtime dependencies will be pinned and
validated as the new implementation is introduced. Heavy datasets, model
checkpoints, and large raw results belong outside Git (for example, on Google
Drive); small result CSV/JSON files may be tracked.

## Layout

- `legacy/` — preserved v1 notebook and local reference copy of the thesis
- `src/chest/` — research package (implementation begins after Stage 0)
- `configs/` — experiment configurations
- `tests/` — focused scientific and reproducibility tests
- `experiments/` — command-line experiment entry points
- `results/` — small tracked results; large/raw output is ignored
- `notebooks/` — thin interactive runners

See [PROJECT_BRIEF.md](PROJECT_BRIEF.md) for scope and staged research rules.
