"""Bot-round model predictions for the human's role only.

The metric-comparison models in
`2026-05-12-current-export-metric-comparison/models.py` produce a joint
distribution + a *position-averaged* marginal. For bot rounds we want
the marginal at the human's in-game position (the bot positions hold
constants, so averaging would dilute the signal).

Each function returns `human_marginal` — a length-3 array of P(role) for
the human — alongside the trajectory stage so the renderer can directly
plot it on the role-frequency bars.
"""

from __future__ import annotations

import numpy as np

from shared.inference import softmax_role_dist

from pipeline import posterior_marginal


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _stick_dist(role):
    d = np.zeros(3)
    d[role] = 1.0
    return d


# ──────────────────────────────────────────────────────────────────────
# Bayesian Belief
# ──────────────────────────────────────────────────────────────────────

def belief_predict(trajectory, human_pid):
    return [
        {
            "stage": s_idx,
            "human_marginal": posterior_marginal(stage["prior"], human_pid),
        }
        for s_idx, stage in enumerate(trajectory)
    ]


# ──────────────────────────────────────────────────────────────────────
# Bayesian Value (softmax over expected utilities) — needs `values`
# ──────────────────────────────────────────────────────────────────────

def value_predict(trajectory, human_pid, values, tau_softmax):
    if values is None:
        return None
    out = []
    for s_idx, stage in enumerate(trajectory):
        d = softmax_role_dist(
            human_pid, stage["intent"], stage["thp"], stage["ehp"],
            stage["prior"], values, tau_softmax,
        )
        out.append({"stage": s_idx, "human_marginal": d})
    return out


# ──────────────────────────────────────────────────────────────────────
# Bayesian Walk — ε-mixture of stick-with-prev and softmax-over-values
# ──────────────────────────────────────────────────────────────────────

def walk_predict(trajectory, human_pid, values, tau_softmax, epsilon_switch):
    if values is None:
        return None
    out = []
    for s_idx, stage in enumerate(trajectory):
        switch = softmax_role_dist(
            human_pid, stage["intent"], stage["thp"], stage["ehp"],
            stage["prior"], values, tau_softmax,
        )
        prev = stage["prev_roles"]
        if prev is None:
            d = switch
        else:
            stick = _stick_dist(prev[human_pid])
            d = (1.0 - epsilon_switch) * stick + epsilon_switch * switch
        out.append({"stage": s_idx, "human_marginal": d})
    return out


# ──────────────────────────────────────────────────────────────────────
# Driver: run the three models on every record
# ──────────────────────────────────────────────────────────────────────

def run_bot_predictions(records, trajectories,
                          tau_value=13.71598290227467,
                          tau_walk=7.20651148477258,
                          eps_walk=0.5589855617201609):
    """Returns predictions[model_short][record_index] -> list[per-stage dict]
    or None when the model is unavailable (missing `values`)."""
    out = {"B": [], "V": [], "W": []}
    for r, traj in zip(records, trajectories):
        pid = r["human_pid"]
        values = r["env_config"]["values"]
        out["B"].append(belief_predict(traj, pid))
        out["V"].append(value_predict(traj, pid, values, tau_value))
        out["W"].append(walk_predict(traj, pid, values, tau_walk, eps_walk))
    return out
