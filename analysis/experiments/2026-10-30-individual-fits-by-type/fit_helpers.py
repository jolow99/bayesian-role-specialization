"""Per-player scoring, per-group fitting and panel drawing for individual_fits.py.

Every model predicts each player's role independently (Copy / Contradict
Others: their exact per-player marginal), so a group's fit uses only the
chosen roles of that group's own players:

  agg-LL  per (env, stage, stat class) cell, the group's empirical role
          frequencies scored against the model's mean predicted marginal;
          players with identical stats in an environment share a cell.
  mean-LL mean log P(chosen role) per environment, averaged over environments.
  marg-r  Pearson r between mean predicted and empirical role frequencies per
          (env, stage).
Lapse λ = 0.01 on each player's marginal, (1 − λ)·p + λ/3, no floor.
Models with free parameters are fit by maximising per-player agg-LL, with the
same searches as the 10-20 lapse refit.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

EXP = Path(__file__).resolve().parent.parent
for d in ("2026-05-28-paper-figures", "2026-05-25-full-pipeline",
          "2026-10-20-tolerant-threshold-ps", "2026-10-20-tolerant-threshold-br"):
    sys.path.insert(0, str(EXP / d))

import paper_figures as pf  # noqa: E402
import tolerant_threshold_br as tbr  # noqa: E402
import tolerant_threshold_ps as tps  # noqa: E402
from pipeline import posterior_marginal, precompute_trajectories, strategy_from_params  # noqa: E402,F401

TYPES_CSV = EXP / "2026-10-27-inference-soft-classifier" / "participant_types.csv"
THRESHOLD = 0.5
LAM = 0.01
ROLE_IDX = tps.ROLE_IDX
BAYES = ["Bayesian-Value", "Bayesian Walk", "Bayesian Threshold",
         "Bayesian-Belief", "Bayesian Walk-PS", "Bayesian Thresh-PS"]
BASE_FIXED = {"Optimal": pf.predict_optimal, "Random-to-Optimal": pf.predict_random_to_optimal,
              "Copy Others": pf.predict_copy_others, "Contradict Others": pf.predict_contradict,
              "Random": pf.predict_random}
MODEL_ORDER = tps.MODEL_ORDER  # paper Fig. 5 sort / stacking order
GROUP_TITLE = {"Population": "Population", "Bayesian": "Bayesians",
               "Non-Bayesian": "Non-Bayesians"}


def _softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


# ── per-player observations ──────────────────────────────────────────

class Obs:
    """One row per (team-round, stage, position) choice."""

    def __init__(self, records, types):
        rows = []
        cells, mcells, envs = {}, {}, {}
        for ri, rec in enumerate(records):
            stats = rec["env_config"]["player_stats"]
            for s, combo in enumerate(rec["stage_roles"]):
                prev = rec["stage_roles"][s - 1] if s > 0 else None
                for pos in range(3):
                    pid = rec["participant_ids"][pos]
                    assert pid in types, pid
                    cell = (rec["env_id"], s, tuple(int(x) for x in stats[pos]))
                    rows.append((ri, s, pos, ROLE_IDX[combo[pos]],
                                 -1 if prev is None else ROLE_IDX[prev[pos]],
                                 cells.setdefault(cell, len(cells)),
                                 mcells.setdefault((rec["env_id"], s), len(mcells)),
                                 envs.setdefault(rec["env_id"], len(envs)),
                                 types[pid] >= THRESHOLD))
        a = np.array(rows, dtype=object)
        self.ri, self.s, self.pos = (a[:, k].astype(int) for k in range(3))
        self.a, self.prev = a[:, 3].astype(int), a[:, 4].astype(int)
        self.cell, self.mcell, self.env = (a[:, k].astype(int) for k in (5, 6, 7))
        self.nonbayes = a[:, 8].astype(bool)
        self.n = len(rows)
        self.stick = np.zeros((self.n, 3))
        has = self.prev >= 0
        self.stick[np.flatnonzero(has), self.prev[has]] = 1.0
        self.has_prev = has


def metrics(M, ob, mask):
    """Per-player agg-LL, mean-LL, marg-r of predicted marginals M on rows `mask`."""
    M, a = M[mask], ob.a[mask]
    cell = np.unique(ob.cell[mask], return_inverse=True)[1]
    mcell = np.unique(ob.mcell[mask], return_inverse=True)[1]
    env = np.unique(ob.env[mask], return_inverse=True)[1]
    onehot = np.eye(3)[a]

    nc = np.bincount(cell)
    mean_pred = np.zeros((len(nc), 3)); np.add.at(mean_pred, cell, M)
    counts = np.zeros((len(nc), 3)); np.add.at(counts, cell, onehot)
    agg = float(np.sum(counts * np.log((1 - LAM) * mean_pred / nc[:, None] + LAM / 3)) / len(a))

    lp = np.log((1 - LAM) * M[np.arange(len(a)), a] + LAM / 3)
    mean_ll = float(np.mean(np.bincount(env, lp) / np.bincount(env)))

    nm = np.bincount(mcell)[:, None]
    pm = np.zeros((len(nm), 3)); np.add.at(pm, mcell, M)
    hm = np.zeros((len(nm), 3)); np.add.at(hm, mcell, onehot)
    marg_r = float(np.corrcoef((pm / nm).ravel(), (hm / nm).ravel())[0, 1])
    return {"agg_ll": agg, "mean_ll": mean_ll, "marg_r": marg_r}


# ── model marginals (vectorised over observations) ──────────────────

class Variant:
    """Posterior marginals and expected values per observation for one belief setting."""

    def __init__(self, name, records, ob, s1, strat):
        self.name = name
        self.trajs = precompute_trajectories(records, s1["tau_prior"], s1["epsilon"],
                                             memory_strategy=strat)
        self.evs = tbr.precompute_evs(records, self.trajs)
        missing = sum(int(s >= len(self.trajs[ri])) for ri, s in zip(ob.ri, ob.s))
        assert missing == 0, f"{missing} observations past the end of a trajectory"
        self.PM = np.array([posterior_marginal(self.trajs[ri][s]["prior"], p)
                            for ri, s, p in zip(ob.ri, ob.s, ob.pos)])
        self.EV = np.array([self.evs[ri][s][p] for ri, s, p in zip(ob.ri, ob.s, ob.pos)])
        prev_traj = np.array([-1 if self.trajs[ri][s]["prev_roles"] is None
                              else self.trajs[ri][s]["prev_roles"][p]
                              for ri, s, p in zip(ob.ri, ob.s, ob.pos)])
        assert np.array_equal(prev_traj, ob.prev)


def walk(ob, base, eps_switch):
    M = base.copy()
    h = ob.has_prev
    M[h] = (1 - eps_switch) * ob.stick[h] + eps_switch * base[h]
    return M


def br(ob, V, tau, eps_switch=None):
    sm = _softmax_rows(V.EV / tau)
    return sm if eps_switch is None else walk(ob, sm, eps_switch)


def _allowed(ob, X, delta):
    h = ob.has_prev
    cur = np.where(h, X[np.arange(ob.n), np.maximum(ob.prev, 0)], 0.0)
    allow = (X - cur[:, None]) > delta
    allow[h, ob.prev[h]] = True
    allow[~h] = True
    return allow


def thresh_ps(ob, V, delta):
    M = V.PM * _allowed(ob, V.PM, delta)
    return M / M.sum(axis=1, keepdims=True)


def thresh_br(ob, V, tau, delta):
    z = np.where(_allowed(ob, V.EV, delta), V.EV / tau, -np.inf)
    return _softmax_rows(z)


def random_walk(ob, eps):
    M = np.full((ob.n, 3), 1 / 3)
    h = ob.has_prev
    M[h] = (1 - eps) * ob.stick[h] + eps / 2 * (1 - ob.stick[h])
    return M


def baseline_marginals(records, ob, dist_fn):
    fn = pf._wrap_baseline(dist_fn)
    per = [[pf._marginals_from_joint(p["predicted_dist"]) for p in fn(rec)] for rec in records]
    return np.array([per[ri][s][p] for ri, s, p in zip(ob.ri, ob.s, ob.pos)])


def bayes_marginals(ob, V, m, p):
    if m == "Bayesian-Value":
        return br(ob, V, p["tau_softmax"])
    if m == "Bayesian Walk":
        return br(ob, V, p["tau_softmax"], p["epsilon_switch"])
    if m == "Bayesian Walk-PS":
        return walk(ob, V.PM, p["epsilon_switch"])
    if m == "Bayesian-Belief":
        return V.PM
    if m == "Bayesian Thresh-PS":
        return thresh_ps(ob, V, p["delta"])
    if m == "Bayesian Threshold":
        return thresh_br(ob, V, p["tau_softmax"], p["delta"])
    raise KeyError(m)


def base_marginals(ob, base, m, p):
    if m == "Random Walk":
        return random_walk(ob, p["eps"])
    if m == "Top-7":
        return base["topk"][p["k"]]
    return base[m]


# ── fitting (maximise per-player agg-LL) ─────────────────────────────

def _fit_1d(f, grid, lo, hi):
    cache = {}
    def at(x):
        x = round(float(x), 6)
        if x not in cache:
            cache[x] = f(x)
        return cache[x]
    for x in grid:
        at(x)
    xs = sorted(cache)
    i = max(range(len(xs)), key=lambda j: cache[xs[j]])
    a, b = xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]
    if b > a:
        minimize_scalar(lambda x: -at(x), bounds=(max(lo, a), min(hi, b)),
                        method="bounded", options={"xatol": 1e-4})
    return max(cache, key=cache.get)


def fit_models(ob, mask, V):
    """Fitted params of the six Bayesian models on belief setting V."""
    score = lambda M: metrics(M, ob, mask)["agg_ll"]
    P = {"Bayesian-Belief": {}}
    P["Bayesian-Value"] = {"tau_softmax": _fit_1d(lambda t: score(br(ob, V, t)),
                                                  np.geomspace(0.5, 50, 20), 0.01, 50)}
    cache = {}
    def wbr(v):
        k = (round(float(v[0]), 6), round(float(v[1]), 6))
        if k not in cache:
            cache[k] = score(br(ob, V, *k))
        return cache[k]
    for t in np.geomspace(0.5, 50, 15):
        for e in np.linspace(0, 1, 21):
            wbr((t, e))
    x0 = max(cache, key=cache.get)
    minimize(lambda v: -wbr(v), x0=list(x0), method="L-BFGS-B",
             bounds=[(0.01, 50), (0, 1)], options={"eps": 1e-4})
    t, e = max(cache, key=cache.get)
    P["Bayesian Walk"] = {"tau_softmax": t, "epsilon_switch": e}
    P["Bayesian Walk-PS"] = {"epsilon_switch": _fit_1d(lambda e: score(walk(ob, V.PM, e)),
                                                       np.linspace(0, 1, 21), 0, 1)}

    # Threshold-PS: piecewise constant in δ; lower end of the best plateau (as in 10-20)
    cache = {}
    def tps_at(d):
        d = round(float(d), 4)
        if d not in cache:
            cache[d] = score(thresh_ps(ob, V, d))
        return cache[d]
    grid = np.round(np.linspace(-1, 1, 201), 4)
    for d in grid:
        tps_at(d)
    best = max(cache.values())
    tops = [d for d in grid if cache[d] >= best - 1e-9]
    for d in np.round(np.arange(max(-1, min(tops) - 0.01), min(1, max(tops) + 0.01) + 1e-9, 0.0005), 4):
        tps_at(d)
    best = max(cache.values())
    P["Bayesian Thresh-PS"] = {"delta": min(d for d, v in cache.items() if v >= best - 1e-9)}

    # Threshold-BR: coarse τ × δ grid, then τ refined on a fine δ grid near the best (as in 10-20)
    cache = {}
    def tbr_at(t, d):
        k = (round(float(t), 6), round(float(d), 4))
        if k not in cache:
            cache[k] = score(thresh_br(ob, V, *k))
        return cache[k]
    for d in np.arange(tbr.DMIN, tbr.DMAX + 1e-9, 1.0):
        for t in np.geomspace(0.5, 50, 20):
            tbr_at(t, d)
    (t0, d0) = max(cache, key=cache.get)
    rows = [(d0, t0, cache[(t0, d0)])]
    for d in np.round(np.arange(max(tbr.DMIN, d0 - 2), min(tbr.DMAX, d0 + 2) + 1e-9, 0.25), 4):
        r = minimize_scalar(lambda t: -tbr_at(t, d), bounds=(0.01, 50), method="bounded",
                            options={"xatol": 1e-3})
        rows.append((float(d), float(r.x), -float(r.fun)))
    best = max(r[2] for r in rows)
    d, t, _ = sorted(r for r in rows if r[2] >= best - 1e-6)[0]
    P["Bayesian Threshold"] = {"tau_softmax": t, "delta": d}
    return P


def pstr(p):
    out = []
    if "tau_softmax" in p:
        out.append(f"τ_v = {p['tau_softmax']:.2f}")
    if "epsilon_switch" in p:
        out.append(f"ε_s = {1 - p['epsilon_switch']:.2f}")
    if "delta" in p:
        out.append(f"δ = {p['delta']:.3f}")
    if "eps" in p:
        out.append(f"ε = {p['eps']:.3f}")
    if "k" in p:
        out.append(f"k = {p['k']}")
    return ", ".join(out) or "—"


# ── individual fits ──────────────────────────────────────────────────

def panel_lls(ob, records, mask, V, base, params):
    """{model: {participant: LL}} for one panel; params = {model: fitted params}."""
    pids = np.array([records[ri]["participant_ids"][p] for ri, p in zip(ob.ri, ob.pos)])
    lls = {}
    for m, p in params.items():
        M = bayes_marginals(ob, V, m, p) if m in BAYES else base_marginals(ob, base, m, p)
        lp = np.log((1 - LAM) * M[np.arange(ob.n), ob.a] + LAM / 3)
        d = defaultdict(float)
        for pid, x in zip(pids[mask], lp[mask]):
            d[pid] += float(x)
        lls[m] = dict(d)
    return lls


def drop_collapsed(lls):
    """Drop a model whose per-participant LLs match an earlier model's to < 1e-3."""
    keep = []
    for m in MODEL_ORDER:
        dup = next((k for k in keep
                    if all(abs(lls[m][p] - lls[k][p]) < 1e-3 for p in lls[m])), None)
        if dup is None:
            keep.append(m)
        else:
            print(f"    drop {tps.dn(m)} (same as {tps.dn(dup)})")
    return keep


def draw(ax, post, models):
    dom = {p: max(models, key=lambda m: v[m]) for p, v in post.items()}
    idx = {m: i for i, m in enumerate(models)}
    pids = sorted(post, key=lambda p: (idx[dom[p]], -post[p][dom[p]]))
    x = np.arange(len(pids))
    bottoms = np.zeros(len(pids))
    for m in models:
        vals = np.array([post[p][m] for p in pids])
        ax.bar(x, vals, bottom=bottoms, width=1.0, color=tps.PALETTE[m],
               edgecolor="white", linewidth=0.3)
        bottoms += vals
    assert np.allclose(bottoms, 1.0, atol=1e-9)
    ax.set_xlim(-0.5, len(pids) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.spines[["top", "right"]].set_visible(False)
    return dom
