"""Sensitivity of everything to the lapse size lambda (none = 0 for reference).
Run from analysis/:  uv run python experiments/2026-10-20-lapse-vs-floor/lambda_sweep.py"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lapse_vs_floor as lf  # noqa: E402
from lapse_vs_floor import pf, pt, ak, with_lapse, agg_ll, run_predictions, individual_post  # noqa: E402
from pipeline import precompute_trajectories, strategy_from_params  # noqa: E402

LAMBDAS = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05]
s1 = json.load(open(pf.STAGE1_PARAMS))
strat = strategy_from_params(s1["memory_strategy"], s1.get("window"), s1.get("drift_delta", 0.0))
cells = json.load(open(pf.PIPELINE_RESULTS))["cells"]
records = pf.load_records_with_pids(verbose=False)
trajs = precompute_trajectories(records, s1["tau_prior"], s1["epsilon"], memory_strategy=strat)
base, _ = pf.build_predict_fns(records, trajs, cells)
base.pop("Mixture-PS"); base[ak.NAME] = pf._wrap_baseline(ak.predict_always_stick)
order = lf.MODEL_ORDER
out = {}
for lam in LAMBDAS:
    lf.LAMBDA = lam
    prof = {d: agg_ll(run_predictions(records, with_lapse(pt.thresh_ps_pure_factory(float(d))(records, trajs))), 0.0)
            for d in np.round(np.arange(0, 1.0001, 0.02), 2)}
    b = max(prof.values()); plat = [d for d, v in prof.items() if v >= b - 1e-9]
    fns = dict(base); fns["Bayesian Thresh-PS"] = pt.thresh_ps_pure_factory(float(min(plat)))(records, trajs)
    fns = {m: fns[m] for m in order}
    agg = {m: agg_ll(run_predictions(records, with_lapse(fn)), 0.0) for m, fn in fns.items()}
    margs = pf.precompute_player_marginals(records, fns)
    post, ll = individual_post(records, margs, "lapse")
    dom = Counter(max(order, key=lambda m: p[m]) for p in post.values())
    tot = {m: sum(ll[m].values()) for m in order}
    out[lam] = {"delta": [min(plat), max(plat)], "agg_ll": agg, "dominant": dict(dom), "total_ind_ll": tot}
    print(lam, "delta plateau", (min(plat), max(plat)), "| dominant", {pt.dn(m): c for m, c in dom.most_common(6)}, flush=True)
json.dump(out, open(HERE / "lambda_sweep.json", "w"), indent=2, default=float)
L = ["# Lapse-size sweep", "", "| λ | Thresh-PS δ plateau | agg-LL Walk-BR | Random Walk | Thresh-PS | Always Stick | Optimal | dominant: Top-7 / Walk-BR / Optimal / Thresh-PS / Always Stick |", "|--:|---|--:|--:|--:|--:|--:|---|"]
for lam, r in out.items():
    a, d = r["agg_ll"], r["dominant"]
    L.append(f"| {lam} | {r['delta'][0]:.2f}–{r['delta'][1]:.2f} | {a['Bayesian Walk']:.3f} | {a['Random Walk']:.3f} | {a['Bayesian Thresh-PS']:.3f} | {a['Always Stick']:.3f} | {a['Optimal']:.3f} | "
             f"{d.get('Top-7',0)} / {d.get('Bayesian Walk',0)} / {d.get('Optimal',0)} / {d.get('Bayesian Thresh-PS',0)} / {d.get('Always Stick',0)} |")
(HERE / "lambda_sweep.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
