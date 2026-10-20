"""Refit every parametric model's parameters under none / lapse / floor (agg-LL objective).

Same procedure as 2026-05-25 run_comparison.fit_model (coarse grid -> L-BFGS-B) but the
objective is agg-LL scored with the variant's zero-probability rule. Threshold-PS is the PURE
version (delta only). Writes fitted_params.json / fitted_params.md.

Run from analysis/:  uv run python experiments/2026-10-20-lapse-vs-floor/fit_params.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "2026-05-25-full-pipeline"))
import lapse_vs_floor as lf  # noqa: E402
from lapse_vs_floor import pf, pt, with_lapse, agg_ll, mean_ll, run_predictions, FLOOR  # noqa: E402
from pipeline import precompute_trajectories, strategy_from_params  # noqa: E402
import run_comparison as rc  # noqa: E402

VARIANTS = ["none", "lapse", "floor"]
SPECS = {  # name -> (param names, bounds, coarse grids, builder(records, trajs, *params) -> predict_fn)
    "Bayesian-BR": rc.MODELS["bayesian_value"],
    "Bayesian-Walk-BR": rc.MODELS["bayesian_walk"],
    "Bayesian-Walk-PS": rc.MODELS["bayesian_walk_ps"],
    "Bayesian-Threshold-BR": rc.MODELS["bayesian_thresh"],
}
KEY = {"Bayesian-BR": "bayesian_value", "Bayesian-Walk-BR": "bayesian_walk",
       "Bayesian-Walk-PS": "bayesian_walk_ps", "Bayesian-Threshold-BR": "bayesian_thresh"}


def make_objective(records, trajs, build, variant, metric="agg_ll"):
    def f(params):
        fn = build(*params)
        if variant == "lapse":
            fn = with_lapse(fn)
        full = run_predictions(records, fn)
        floor = FLOOR if variant == "floor" else 0.0
        return (agg_ll if metric == "agg_ll" else mean_ll)(full, floor)
    return f


def fit(f, spec):
    grids, bounds = spec["coarse"], spec["bounds"]
    best, bp = -np.inf, None
    if len(grids) == 1:
        cand = [[float(a)] for a in grids[0]]
    else:
        cand = [[float(a), float(b)] for a in grids[0] for b in grids[1]]
    for p in cand:
        v = f(p)
        if np.isfinite(v) and v > best:
            best, bp = v, p
    if bp is None:
        return None, -np.inf
    neg = lambda x: (-f(list(x)) if np.isfinite(f(list(x))) else 1e10)
    o = minimize(neg, bp, method="L-BFGS-B", bounds=bounds, options={"maxiter": 50, "ftol": 1e-6})
    v = f(list(o.x))
    return (list(map(float, o.x)), v) if np.isfinite(v) and v >= best else (bp, best)


def main():
    s1 = json.load(open(pf.STAGE1_PARAMS))
    strat = strategy_from_params(s1["memory_strategy"], s1.get("window"), s1.get("drift_delta", 0.0))
    records = pf.load_records_with_pids(verbose=False)
    trajs = precompute_trajectories(records, s1["tau_prior"], s1["epsilon"], memory_strategy=strat)
    out = {v: {} for v in VARIANTS}

    for v in VARIANTS:
        for name, spec in SPECS.items():
            mk = rc.FACTORY_MAP[KEY[name]]
            build = lambda *p, mk=mk: mk(*p)(records, trajs)
            p, val = fit(make_objective(records, trajs, build, v), spec)
            out[v][name] = {"params": dict(zip(spec["params"], p)) if p else None, "agg_ll": val}
            print(v, name, out[v][name], flush=True)
        # pure Threshold-PS delta (1-D, fine grid) under agg-LL and mean-LL
        for metric in ("agg_ll", "mean_ll"):
            f = make_objective(records, trajs, lambda d: pt.thresh_ps_pure_factory(d)(records, trajs), v, metric)
            prof = {float(d): f([float(d)]) for d in np.round(np.arange(0, 1.0001, 0.01), 2)}
            fin = {d: x for d, x in prof.items() if np.isfinite(x)}
            if fin:
                b = max(fin.values()); plat = [d for d, x in fin.items() if x >= b - 1e-9]
                res = {"delta": min(plat), "plateau": [min(plat), max(plat)], "value": b}
            else:
                res = {"delta": None, "plateau": None, "value": float("-inf")}
            out[v][f"Bayesian-Threshold-PS [{metric}]"] = res
            print(v, "Thresh-PS", metric, res, flush=True)
        # Random Walk eps (switch prob; stay = 1-eps)
        f = make_objective(records, trajs, lambda e: pf._wrap_baseline(pf.make_predict_random_walk(e)), v)
        prof = {float(e): f([float(e)]) for e in np.round(np.arange(0, 1.0001, 0.01), 2)}
        fin = {e: x for e, x in prof.items() if np.isfinite(x)}
        e_best = max(fin, key=fin.get) if fin else None
        out[v]["Random Walk"] = {"params": {"eps_switch": e_best}, "agg_ll": fin[e_best] if fin else float("-inf")}
        print(v, "Random Walk", out[v]["Random Walk"], flush=True)

    json.dump(out, open(HERE / "fitted_params.json", "w"), indent=2, default=float)

    def fp(x):
        if x is None or x.get("params") is None and "delta" not in x:
            return "undefined (agg-LL = −∞ everywhere)"
        if "delta" in x:
            return "undefined (−∞ at every δ)" if x["delta"] is None else \
                f"δ = {x['delta']:.2f}" + (f" (plateau {x['plateau'][0]:.2f}–{x['plateau'][1]:.2f})" if x['plateau'][1] - x['plateau'][0] > 0.02 else "")
        return ", ".join(f"{k} = {val:.3f}" for k, val in x["params"].items())
    rows = ["Bayesian-BR", "Bayesian-Walk-BR", "Bayesian-Walk-PS", "Bayesian-Threshold-BR",
            "Bayesian-Threshold-PS [agg_ll]", "Bayesian-Threshold-PS [mean_ll]", "Random Walk"]
    L = ["# Fitted parameters by variant (agg-LL objective unless noted)", "",
         "Code names: `epsilon_switch` = weight on the switch/sampling distribution (paper's ε = 1 − epsilon_switch); "
         "Random Walk `eps_switch` = total switch prob.", "",
         "| Model | none | lapse (λ=0.05) | floor (1e-12) |", "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r} | " + " | ".join(fp(out[v][r]) for v in VARIANTS) + " |")
    (HERE / "fitted_params.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
