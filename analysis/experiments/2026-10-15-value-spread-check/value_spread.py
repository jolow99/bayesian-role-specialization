"""2026-10-15 — scale of the value softmax temperature tau_v vs. the spread of expected values.

Xuan (2026-10-12 call): tau_v only makes sense relative to the spread of the
expected role values V-hat (paper Eq. for \\tilde V_t(r_i)). If the max-min gap across
roles were always << tau_v the softmax would be ~uniform and the fit uninformative.

For every agent-stage of the 204 clean human team-rounds (same records, Stage-1 params and
trajectories as the 05-25 pipeline) we compute V-hat for the three candidate roles and report
the gap (max - min over roles, in points; V is on a 0-100 scale), and the softmax's
probability on the top role at the fitted tau_v of Bayesian-Walk-BR (12.09) and Bayesian-BR (11.62).

Run from analysis/:  uv run python experiments/2026-10-15-value-spread-check/value_spread.py
"""
import json, sys
from pathlib import Path
import numpy as np

EXP = Path(__file__).resolve().parent.parent
PIPE = EXP / "2026-05-25-full-pipeline"
sys.path.insert(0, str(PIPE))
from pipeline import load_human_team_records, precompute_trajectories, strategy_from_params  # noqa: E402
from models import _expected_values_per_role  # noqa: E402

s1 = json.load(open(PIPE / "stage1_inference/best_inference_params.json"))
strat = strategy_from_params(s1["memory_strategy"], s1.get("window"), s1.get("drift_delta", 0.0))
recs = load_human_team_records(verbose=False)
trajs = precompute_trajectories(recs, s1["tau_prior"], s1["epsilon"], memory_strategy=strat)
cells = json.load(open(PIPE / "results.json"))["cells"]
TAU = {"Bayesian-Walk-BR": cells["bayesian_walk"]["agg_ll"]["fitted_params"]["tau_softmax"],
       "Bayesian-BR": cells["bayesian_value"]["agg_ll"]["fitted_params"]["tau_softmax"]}

EV = []
for r, t in zip(recs, trajs):
    V = r["env_config"]["values"]
    for st in t:
        for i in range(3):
            EV.append(_expected_values_per_role(i, st["intent"], st["thp"], st["ehp"], st["prior"], V))
EV = np.array(EV)                       # (n_agent_stages, 3)
gap = EV.max(1) - EV.min(1)

def top_prob(tau):
    z = EV / tau
    z -= z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    return p.max(1)

out = {"n_agent_stages": int(len(gap)), "n_team_rounds": len(recs),
       "gap": {"mean": float(gap.mean()), "median": float(np.median(gap)),
               "p10": float(np.percentile(gap, 10)), "p25": float(np.percentile(gap, 25)),
               "p75": float(np.percentile(gap, 75)), "p90": float(np.percentile(gap, 90)),
               "max": float(gap.max())},
       "by_model": {}}
for name, tau in TAU.items():
    tp = top_prob(tau)
    out["by_model"][name] = {
        "tau_v": float(tau),
        "frac_gap_gt_tau": float((gap > tau).mean()),
        "frac_gap_gt_half_tau": float((gap > tau / 2).mean()),
        "mean_top_prob": float(tp.mean()), "median_top_prob": float(np.median(tp)),
        "frac_top_prob_gt_0.5": float((tp > 0.5).mean()),
        "frac_top_prob_gt_0.8": float((tp > 0.8).mean()),
        "mean_top_prob_when_gap_gt_tau": float(tp[gap > tau].mean()),
        "mean_top_prob_when_gap_le_tau": float(tp[gap <= tau].mean()),
    }
out["uniform_top_prob"] = 1 / 3
json.dump(out, open(Path(__file__).with_name("results.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
