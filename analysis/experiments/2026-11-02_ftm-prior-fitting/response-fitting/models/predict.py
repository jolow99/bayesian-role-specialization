"""Per-player role marginals (n, 3) for the 13 response models.

Bayesian models from 2026-10-30 fit_helpers.py; baselines from 2026-05-28
paper_figures.py (predict_*), reduced to each player's marginal.
"""

from __future__ import annotations

import numpy as np

from shared.constants import TURNS_PER_STAGE

BAYES = ["Bayesian-BR", "Bayesian-Walk-BR", "Bayesian-Threshold-BR",
         "Bayesian-PS", "Bayesian-Walk-PS", "Bayesian-Threshold-PS"]
BASE = ["Optimal", "Top-5", "Random-to-Optimal", "Copy Others",
        "Contradict Others", "Random Walk", "Random"]
TOP_K = 5


def softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def walk(ob, base, eps_switch):
    """Stay with prob 1 − ε_switch, otherwise re-decide from base."""
    M = base.copy()
    h = ob.has_prev
    M[h] = (1 - eps_switch) * ob.stick[h] + eps_switch * base[h]
    return M


def br(ob, EV, tau, eps_switch=None):
    sm = softmax_rows(EV / tau)
    return sm if eps_switch is None else walk(ob, sm, eps_switch)


def _allowed(ob, X, delta):
    """Current role, plus roles beating it by more than delta."""
    h = ob.has_prev
    cur = np.where(h, X[np.arange(ob.n), np.maximum(ob.prev, 0)], 0.0)
    allow = (X - cur[:, None]) > delta
    allow[h, ob.prev[h]] = True
    allow[~h] = True
    return allow


def thresh_ps(ob, PM, delta):
    M = PM * _allowed(ob, PM, delta)
    return M / M.sum(axis=1, keepdims=True)


def thresh_br(ob, EV, tau, delta):
    return softmax_rows(np.where(_allowed(ob, EV, delta), EV / tau, -np.inf))


def random_walk(ob, eps):
    M = np.full((ob.n, 3), 1 / 3)
    h = ob.has_prev
    M[h] = (1 - eps) * ob.stick[h] + eps / 2 * (1 - ob.stick[h])
    return M


def bayes_marginals(ob, PM, EV, m, p):
    if m == "Bayesian-BR":
        return br(ob, EV, p["tau_softmax"])
    if m == "Bayesian-Walk-BR":
        return br(ob, EV, p["tau_softmax"], p["epsilon_switch"])
    if m == "Bayesian-Threshold-BR":
        return thresh_br(ob, EV, p["tau_softmax"], p["delta"])
    if m == "Bayesian-PS":
        return PM
    if m == "Bayesian-Walk-PS":
        return walk(ob, PM, p["epsilon_switch"])
    if m == "Bayesian-Threshold-PS":
        return thresh_ps(ob, PM, p["delta"])
    raise KeyError(m)


def _joint_marginal(joint, pos):
    return joint.reshape(3, 3, 3).sum(axis=tuple(j for j in range(3) if j != pos))


def fixed_baselines(records, ob):
    """Baselines without free parameters (Top-k at k = 5).

    Returns ({model: (n, 3) marginals}, {model: (n, 27) team joints}); joints only
    for the value baselines, whose joints don't factorise into the marginals.
    """
    out = {m: np.empty((ob.n, 3)) for m in BASE if m != "Random Walk"}
    joints = {m: np.empty((ob.n, 27)) for m in ("Optimal", "Top-5", "Random-to-Optimal")}
    for q, (ri, s, pos) in enumerate(zip(ob.ri, ob.s, ob.pos)):
        rec = records[ri]
        env, lds = rec["env_config"], rec["lds"]
        values = env["values"]
        ti = s * TURNS_PER_STAGE
        v = values[:, lds[ti] if ti < len(lds) else 0,
                   min(int(env["team_max_hp"]), values.shape[2] - 1),
                   min(int(env["enemy_max_hp"]), values.shape[3] - 1)]
        opt = (np.abs(v - v.max()) < 1e-8).astype(float)
        top = np.zeros(27)
        top[sorted(range(27), key=lambda c: -v[c])[:TOP_K]] = 1.0
        opt_m = _joint_marginal(opt / opt.sum(), pos)
        alpha = s / max(len(rec["stage_roles"]) - 1, 1)
        out["Optimal"][q] = opt_m
        out["Top-5"][q] = _joint_marginal(top / TOP_K, pos)
        out["Random-to-Optimal"][q] = (1 - alpha) / 3 + alpha * opt_m
        out["Random"][q] = 1 / 3
        joints["Optimal"][q] = opt / opt.sum()
        joints["Top-5"][q] = top / TOP_K
        joints["Random-to-Optimal"][q] = (1 - alpha) / 27 + alpha * opt / opt.sum()
        if s == 0:
            out["Copy Others"][q] = out["Contradict Others"][q] = 1 / 3
            continue
        others = np.zeros(3)
        prev = rec["stage_roles"][s - 1]
        for j in range(3):
            if j != pos:
                others["FTM".index(prev[j])] = 1.0
        out["Copy Others"][q] = others / others.sum()
        w = np.where(others > 0, 0.1, 1.0)
        out["Contradict Others"][q] = w / w.sum()
    return out, joints
