"""How sticky is updaters' Walk-BR under other objectives, and how often do updaters switch?

1. Refit Bayesian-Walk-BR on updaters by maximising agg-LL (reproduces fit-results), mean-LL
   and marg-r, with the fit.py search (15 × 21 grid over τ_v × ε_switch, then L-BFGS-B).
   combo-r is not used as an objective: it needs whole teams, which mix types.
2. Empirical switch rate: fraction of updaters' stage transitions where the role changed,
   overall and per transition, against the switch rates Walk-BR (agg-LL fit) and pure
   Bayesian-BR (updaters' fit in response-fitting/fit-results) predict.
ε_s = 1 − ε_switch is the stick probability; the model switches on at most ε_switch of stages.

Run from analysis/:
    uv run python experiments/2026-11-02_ftm-prior-fitting/checks/walk-br-stickiness/walk_br_stickiness.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

OUT_DIR = Path(__file__).resolve().parent
EXP_DIR = OUT_DIR.parents[1]
sys.path.insert(0, str(EXP_DIR / "response-fitting"))

from models.beliefs import player_beliefs  # noqa: E402
from models.data import Obs, load_records  # noqa: E402
from models.predict import br  # noqa: E402
from models.scoring import metrics  # noqa: E402

OBJECTIVES = ["agg_ll", "mean_ll", "marg_r"]


def fit_walk_br(ob, mask, EV, objective):
    cache = {}
    def score(v):
        k = (round(float(v[0]), 6), round(float(v[1]), 6))
        if k not in cache:
            cache[k] = metrics(br(ob, EV, *k), ob, mask)[objective]
        return cache[k]
    for t in np.geomspace(0.5, 50, 15):
        for e in np.linspace(0, 1, 21):
            score((t, e))
    minimize(lambda v: -score(v), x0=list(max(cache, key=cache.get)), method="L-BFGS-B",
             bounds=[(0.01, 50), (0, 1)], options={"eps": 1e-4})
    return max(cache, key=cache.get)


def main():
    params = json.load(open(EXP_DIR / "inference-fitting" / "params.json"))
    with open(EXP_DIR / "inference-fitting" / "participant_types.csv") as f:
        p_nu = {r["participant_id"]: float(r["p_non_updater"]) for r in csv.DictReader(f)}
    records = load_records()
    ob = Obs(records, p_nu)
    _, EV = player_beliefs(records, ob, params, ob.non_updater)
    mask = ~ob.non_updater

    fits = {}
    for obj in OBJECTIVES:
        tau, eps_switch = fit_walk_br(ob, mask, EV, obj)
        fits[obj] = {"tau_v": tau, "eps_s": 1 - eps_switch,
                     **metrics(br(ob, EV, tau, eps_switch), ob, mask)}

    rows = mask & ob.has_prev
    switched = ob.a != ob.prev
    tau, eps_switch = fits["agg_ll"]["tau_v"], 1 - fits["agg_ll"]["eps_s"]
    M = br(ob, EV, tau, eps_switch)
    model_switch = 1 - M[np.arange(ob.n), np.maximum(ob.prev, 0)]
    tau_br = json.load(open(EXP_DIR / "response-fitting" / "fit-results" / "updaters" /
                            "bayes-models.json"))["Bayesian-BR"]["params"]["tau_softmax"]
    br_switch = 1 - br(ob, EV, tau_br)[np.arange(ob.n), np.maximum(ob.prev, 0)]
    by_stage = {f"{s} -> {s + 1}": {"n": int((rows & (ob.s == s)).sum()),
                                     "human": float(switched[rows & (ob.s == s)].mean()),
                                     "model": float(model_switch[rows & (ob.s == s)].mean()),
                                     "bayesian_br": float(br_switch[rows & (ob.s == s)].mean())}
                for s in range(1, ob.s.max() + 1) if (rows & (ob.s == s)).any()}
    out = {"fits": fits, "switch_rate": {"n": int(rows.sum()), "human": float(switched[rows].mean()),
                                         "model": float(model_switch[rows].mean()),
                                         "bayesian_br": float(br_switch[rows].mean()),
                                         "model_max": eps_switch, "by_stage": by_stage}}

    lines = ["# Updaters' Walk-BR stickiness", "",
             "## Walk-BR refit on updaters by objective", "",
             "| Objective | τ_v | ε_s (stick) | agg-LL | mean-LL | marg-r |", "|---|--:|--:|--:|--:|--:|"]
    lines += [f"| {o} | {f['tau_v']:.2f} | {f['eps_s']:.2f} | {f['agg_ll']:.4f} | {f['mean_ll']:.4f} "
              f"| {f['marg_r']:.3f} |" for o, f in fits.items()]
    sr = out["switch_rate"]
    lines += ["", f"## Switch rate ({sr['n']} updater stage transitions)", "",
              f"Human {sr['human']:.1%}; Walk-BR (agg-LL fit) predicts {sr['model']:.1%}, "
              f"at most {sr['model_max']:.1%}; Bayesian-BR (τ_v = {tau_br:.2f}) predicts "
              f"{sr['bayesian_br']:.1%}.", "",
              "| Transition | n | human | Walk-BR | Bayesian-BR |", "|---|--:|--:|--:|--:|"]
    lines += [f"| {k} | {v['n']} | {v['human']:.1%} | {v['model']:.1%} | {v['bayesian_br']:.1%} |"
              for k, v in by_stage.items()]
    (OUT_DIR / "walk_br_stickiness.md").write_text("\n".join(lines) + "\n")
    with open(OUT_DIR / "walk_br_stickiness.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
