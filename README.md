# Bayesian Model of Ad-hoc Role Specialization

This repository hosts our work on a research paper investigating a Bayesian model of ad-hoc role specialization.

## Repository Structure

### `analysis/`
Shared analysis workspace for both collaborators. Contains the `shared/` Python package, experiment folders, and data.

```
analysis/
├── shared/             # Importable Python package (tracked in git)
├── data/               # Experiment data + env configs (gitignored, see below)
├── experiments/        # Date-prefixed analysis folders (tracked in git)
├── legacy/             # Old notebooks before restructuring (gitignored)
├── pyproject.toml      # uv project config
└── .venv/              # Python virtual environment (gitignored)
```

**See [CLAUDE.md](CLAUDE.md) for detailed documentation** of the shared package, how to set up, and how to create experiments.

### `computational_model/`
Contains the computational model: MDP solver, simulation, and value matrix generation. The `analysis/` subfolder here contains legacy analysis scripts (`online_model_sim.py`, `baseline_benchmarks.py`) that are being migrated to `analysis/experiments/`.

### `human_experiment/`
Contains the human experiment built with Empirica. This folder includes the experimental platform for collecting behavioral data from human participants.

### `cogsci_paper/`
Contains the Typst template for the paper submission.

## Quick Start (Analysis)

```bash
cd analysis
uv sync                   # install dependencies
uv pip install -e .       # install shared package in editable mode
uv run jupyter lab        # launch Jupyter
```

Select the `.venv` kernel in Jupyter. Then in any notebook:

```python
from shared.data_loading import load_all_exports, to_dataframe
from shared.constants import ROLE_MAP, ROLE_COLORS
from shared.inference import bayesian_update, utility_based_prior

records = load_all_exports()
df = to_dataframe(records)
```

## Data Setup

Data is gitignored. To set up, place these under `analysis/data/`:

```
analysis/data/
├── exports/                         # Empirica CSV exports
│   ├── bayesian-role-specialization-2026-01-25-16-53-58/
│   ├── bayesian-role-specialization-2026-01-28-10-09-10/
│   ├── bayesian-role-specialization-2026-02-13-10-37-44/
│   ├── bayesian-role-specialization-2026-03-06-09-54-19/
│   └── bayesian-role-specialization-2026-03-18-15-47-09/
├── envs/                            # ~7500 MDP env configs (config.py + values.npy)
└── human_envs_value_matrices/       # 8 role-combo env configs (FFF, FTM, etc.)
```
