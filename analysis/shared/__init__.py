"""Shared analysis utilities for the Bayesian Role Specialization experiment."""

from pathlib import Path

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ANALYSIS_ROOT / "data"
EXPORTS_DIR = DATA_ROOT / "exports"
ENVS_DIR = DATA_ROOT / "envs"
HUMAN_ENVS_DIR = DATA_ROOT / "human_envs_value_matrices"
