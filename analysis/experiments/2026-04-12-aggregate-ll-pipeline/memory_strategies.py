"""Consolidated memory-strategy definitions for Stage 1 and Stage 2.

Strategies control how the joint posterior is carried across stages:

- ``full``: keep the running posterior with no reset.
- ``window`` (param ``w``): recompute from the original prior over the last
  ``w`` stages only.
- ``drift_prior`` (param ``delta``): at each stage boundary,
  ``pi <- (1 - delta) * pi + delta * original_prior``.
- ``drift_uniform`` (param ``delta``): same but toward the flat 1/27 prior.
- ``temper`` (param ``gamma``): at each stage boundary,
  ``pi <- pi ** gamma`` (then renormalise). ``gamma = 1`` is the identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class MemoryStrategy:
    name: str
    kind: str          # full | window | drift_prior | drift_uniform | temper
    param: float | int | None


def build_strategy_grid() -> list[MemoryStrategy]:
    """Expanded stage-1 strategy space."""
    strats: list[MemoryStrategy] = []
    strats.append(MemoryStrategy("full", "full", None))
    for w in (1, 2, 3, 4):
        strats.append(MemoryStrategy(f"window_{w}", "window", int(w)))
    for d in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9):
        strats.append(MemoryStrategy(f"drift_prior_{d:.3f}", "drift_prior", float(d)))
    for d in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        strats.append(MemoryStrategy(f"drift_uniform_{d:.3f}", "drift_uniform", float(d)))
    for g in (0.1, 0.2, 0.3, 0.35, 0.41, 0.5, 0.7, 1.0):
        strats.append(MemoryStrategy(f"temper_{g:.3f}", "temper", float(g)))
    return strats


def apply_boundary(current: np.ndarray, original_prior: np.ndarray,
                   kind: str, param: float | int | None) -> np.ndarray:
    """Apply the boundary operation for a non-windowed memory strategy.

    Returns a normalised (3,3,3) array. Callers should already have applied
    the per-turn Bayesian updates for the stage; this function only performs
    the between-stage memory operation.
    """
    if kind in ("full", None):
        return current
    if kind == "drift_prior":
        out = (1.0 - param) * current + param * original_prior
    elif kind == "drift_uniform":
        uniform = np.ones_like(original_prior) / original_prior.size
        out = (1.0 - param) * current + param * uniform
    elif kind == "temper":
        if param == 1.0:
            return current
        log_cur = np.log(np.clip(current, 1e-300, None))
        log_cur = param * log_cur
        log_cur -= log_cur.max()
        out = np.exp(log_cur)
    else:
        raise ValueError(f"Unknown memory-strategy kind: {kind!r}")
    total = out.sum()
    if total > 0:
        out = out / total
    return out


def strategy_from_params(memory_strategy: str,
                          window: int | None,
                          drift_delta: float | int | None) -> MemoryStrategy:
    """Rebuild a ``MemoryStrategy`` from a stage-1 params.json record.

    Handles both the new keys (``memory_strategy`` = kind-prefixed name) and
    the legacy 04-09 schema (``window`` / ``drift_delta`` fields).
    """
    name = memory_strategy or "full"
    if name.startswith("window_"):
        return MemoryStrategy(name, "window", int(name.split("_", 1)[1]))
    if name.startswith("drift_prior_"):
        return MemoryStrategy(name, "drift_prior", float(name.split("_", 2)[2]))
    if name.startswith("drift_uniform_"):
        return MemoryStrategy(name, "drift_uniform", float(name.split("_", 2)[2]))
    if name.startswith("temper_"):
        return MemoryStrategy(name, "temper", float(name.split("_", 1)[1]))
    if name == "full":
        # Back-compat with 04-09: window=N means "window" strategy, else drift_delta.
        if window is not None:
            return MemoryStrategy(f"window_{int(window)}", "window", int(window))
        if drift_delta and drift_delta > 0:
            return MemoryStrategy(f"drift_prior_{float(drift_delta):.3f}",
                                  "drift_prior", float(drift_delta))
        return MemoryStrategy("full", "full", None)
    raise ValueError(f"Unrecognised memory strategy name: {name!r}")
