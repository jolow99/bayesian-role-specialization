"""Fit the 13 response models per group (updaters, non-updaters, population).

Beliefs follow each player's type (inference-fitting/participant_types.csv,
P(non-updater) > 0.5): updaters use the mixed posterior at the inference fit's
(tau, eps, beta), non-updaters the fixed FTM prior. Free parameters maximise
per-player agg-LL on the group's rows, with the 2026-10-20 lapse-refit searches.

Outputs: fit-results/<group>/{bayes-models,non-bayes-models}.json, each with
per model: params, agg_ll, mean_ll, marg_r (on the group), combo_r (team-level, on all
teams with the group's params, each player on their own type's belief), participant_ll.

Run from analysis/:
    uv run python experiments/2026-11-02_ftm-prior-fitting/response-fitting/fit.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from models.beliefs import player_beliefs
from models.data import Obs, load_records
from models.predict import BASE, BAYES, br, bayes_marginals, fixed_baselines, random_walk, thresh_br, thresh_ps, walk
from models.scoring import combo_r, metrics, participant_lls

SCRIPT_DIR = Path(__file__).resolve().parent
INFERENCE_DIR = SCRIPT_DIR.parent / "inference-fitting"
OUT_DIR = SCRIPT_DIR / "fit-results"
DMIN, DMAX = -70.0, 70.0  # Threshold-BR δ range


def fit_1d(f, grid, lo, hi):
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


def fit_bayes(ob, mask, PM, EV, objective="agg_ll"):
    """Fitted params of the six Bayesian models (2026-10-30 fit_helpers.fit_models)."""
    score = lambda M: metrics(M, ob, mask)[objective]
    P = {"Bayesian-PS": {}}
    P["Bayesian-BR"] = {"tau_softmax": fit_1d(lambda t: score(br(ob, EV, t)),
                                              np.geomspace(0.5, 50, 20), 0.01, 50)}
    cache = {}
    def wbr(v):
        k = (round(float(v[0]), 6), round(float(v[1]), 6))
        if k not in cache:
            cache[k] = score(br(ob, EV, *k))
        return cache[k]
    for t in np.geomspace(0.5, 50, 15):
        for e in np.linspace(0, 1, 21):
            wbr((t, e))
    minimize(lambda v: -wbr(v), x0=list(max(cache, key=cache.get)), method="L-BFGS-B",
             bounds=[(0.01, 50), (0, 1)], options={"eps": 1e-4})
    t, e = max(cache, key=cache.get)
    P["Bayesian-Walk-BR"] = {"tau_softmax": t, "epsilon_switch": e}
    P["Bayesian-Walk-PS"] = {"epsilon_switch": fit_1d(lambda e: score(walk(ob, PM, e)),
                                                      np.linspace(0, 1, 21), 0, 1)}

    # Threshold-PS: piecewise constant in δ; lower end of the best plateau
    cache = {}
    def tps_at(d):
        d = round(float(d), 4)
        if d not in cache:
            cache[d] = score(thresh_ps(ob, PM, d))
        return cache[d]
    grid = np.round(np.linspace(-1, 1, 201), 4)
    for d in grid:
        tps_at(d)
    best = max(cache.values())
    tops = [d for d in grid if cache[d] >= best - 1e-9]
    for d in np.round(np.arange(max(-1, min(tops) - 0.01), min(1, max(tops) + 0.01) + 1e-9, 0.0005), 4):
        tps_at(d)
    best = max(cache.values())
    P["Bayesian-Threshold-PS"] = {"delta": min(d for d, v in cache.items() if v >= best - 1e-9)}

    # Threshold-BR: coarse τ × δ grid, then τ refined on a fine δ grid near the best
    cache = {}
    def tbr_at(t, d):
        k = (round(float(t), 6), round(float(d), 4))
        if k not in cache:
            cache[k] = score(thresh_br(ob, EV, *k))
        return cache[k]
    for d in np.arange(DMIN, DMAX + 1e-9, 1.0):
        for t in np.geomspace(0.5, 50, 20):
            tbr_at(t, d)
    t0, d0 = max(cache, key=cache.get)
    rows = [(d0, t0, cache[(t0, d0)])]
    for d in np.round(np.arange(max(DMIN, d0 - 2), min(DMAX, d0 + 2) + 1e-9, 0.25), 4):
        r = minimize_scalar(lambda t: -tbr_at(t, d), bounds=(0.01, 50), method="bounded",
                            options={"xatol": 1e-3})
        rows.append((float(d), float(r.x), -float(r.fun)))
    best = max(r[2] for r in rows)
    d, t, _ = sorted(r for r in rows if r[2] >= best - 1e-6)[0]
    P["Bayesian-Threshold-BR"] = {"tau_softmax": t, "delta": d}
    return P


def fit_group(ob, mask, PM, EV, fixed, joints, records):
    """{"bayes-models": {...}, "non-bayes-models": {...}} for one group."""
    P = fit_bayes(ob, mask, PM, EV)
    eps = fit_1d(lambda e: metrics(random_walk(ob, e), ob, mask)["agg_ll"],
                 np.linspace(0, 1, 21), 0, 1)
    out = {"bayes-models": {}, "non-bayes-models": {}}
    for m in BAYES + BASE:
        if m in BAYES:
            p, M = P[m], bayes_marginals(ob, PM, EV, m, P[m])
        elif m == "Random Walk":
            p, M = {"eps": eps}, random_walk(ob, eps)
        else:
            p, M = ({"k": 5} if m == "Top-5" else {}), fixed[m]
        out["bayes-models" if m in BAYES else "non-bayes-models"][m] = {
            "params": p, **metrics(M, ob, mask), "combo_r": combo_r(M, ob, records, joints.get(m)),
            "participant_ll": participant_lls(M, ob, mask)}
    return out


def main():
    params = json.load(open(INFERENCE_DIR / "params.json"))
    with open(INFERENCE_DIR / "participant_types.csv") as f:
        p_nu = {r["participant_id"]: float(r["p_non_updater"]) for r in csv.DictReader(f)}
    records = load_records()
    ob = Obs(records, p_nu)
    PM, EV = player_beliefs(records, ob, params, ob.non_updater)
    fixed, joints = fixed_baselines(records, ob)

    groups = {"updaters": ~ob.non_updater, "non-updaters": ob.non_updater,
              "population": np.ones(ob.n, bool)}
    for g, mask in groups.items():
        res = fit_group(ob, mask, PM, EV, fixed, joints, records)
        (OUT_DIR / g).mkdir(parents=True, exist_ok=True)
        for fam, r in res.items():
            with open(OUT_DIR / g / f"{fam}.json", "w") as f:
                json.dump(r, f, indent=2)
        print(f"\n{g}: {len(np.unique(ob.pid[mask]))} participants, {mask.sum()} choices")
        for m, r in {**res["bayes-models"], **res["non-bayes-models"]}.items():
            print(f"  {m:23s} agg-LL {r['agg_ll']:.4f}  mean-LL {r['mean_ll']:.4f}  "
                  f"marg-r {r['marg_r']:.3f}  combo-r {r['combo_r']:.3f}  {r['params']}")


if __name__ == "__main__":
    main()
