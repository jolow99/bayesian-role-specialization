"""How flat is the BR softmax at each group's fitted τ_v, relative to the value range?

For each choice in a group (each player on their own type's belief): the spread of
expected values over their three roles (best − worst), and the softmax P ∝ exp(EV / τ_v)
at the group's fitted τ_v: P(worst) / P(best) = exp(−spread / τ_v), P(top role), and
the distance from uniform (TVD; 0 = uniform, 2/3 = deterministic). Medians [IQR].

Run from analysis/:
    uv run python experiments/2026-11-02_ftm-prior-fitting/checks/softmax-flatness/softmax_flatness.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent
EXP_DIR = OUT_DIR.parents[1]
sys.path.insert(0, str(EXP_DIR / "response-fitting"))

from models.beliefs import player_beliefs  # noqa: E402
from models.data import Obs, load_records  # noqa: E402
from models.predict import softmax_rows  # noqa: E402

MODELS = ["Bayesian-BR", "Bayesian-Walk-BR", "Bayesian-Threshold-BR"]


def q(x):
    lo, med, hi = np.percentile(x, [25, 50, 75])
    return {"median": float(med), "q25": float(lo), "q75": float(hi)}


def main():
    params = json.load(open(EXP_DIR / "inference-fitting" / "params.json"))
    with open(EXP_DIR / "inference-fitting" / "participant_types.csv") as f:
        p_nu = {r["participant_id"]: float(r["p_non_updater"]) for r in csv.DictReader(f)}
    records = load_records()
    ob = Obs(records, p_nu)
    _, EV = player_beliefs(records, ob, params, ob.non_updater)

    fmt = lambda s, d=2: f"{s['median']:.{d}f} [{s['q25']:.{d}f}, {s['q75']:.{d}f}]"
    out = {}
    lines = ["# Softmax flatness at the fitted τ_v", "",
             "Per choice: value spread = best − worst expected value over the player's 3 roles; "
             "softmax at the group's fitted τ_v. Median [IQR] over the group's choices. "
             "Uniform: P(top) = 0.33, TVD = 0; deterministic: P(top) = 1, TVD = 0.67.", ""]
    for group, mask in (("updaters", ~ob.non_updater), ("non-updaters", ob.non_updater)):
        fit = json.load(open(EXP_DIR / "response-fitting" / "fit-results" / group / "bayes-models.json"))
        ev = EV[mask]
        spread = ev.max(axis=1) - ev.min(axis=1)
        out[group] = {"value_spread": q(spread)}
        lines += [f"## {group} ({mask.sum()} choices): value spread {fmt(out[group]['value_spread'], 1)}", "",
                  "| Model | τ_v | spread / τ_v | P(worst) / P(best) | P(top role) | TVD from uniform |",
                  "|---|--:|--:|--:|--:|--:|"]
        for m in MODELS:
            tau = fit[m]["params"]["tau_softmax"]
            P = softmax_rows(ev / tau)
            r = {"tau": tau, "spread_over_tau": q(spread / tau), "ratio": q(np.exp(-spread / tau)),
                 "p_top": q(P.max(axis=1)), "tvd_uniform": q(0.5 * np.abs(P - 1 / 3).sum(axis=1))}
            out[group][m] = r
            lines.append(f"| {m} | {tau:.2f} | {fmt(r['spread_over_tau'])} | {fmt(r['ratio'])} "
                         f"| {fmt(r['p_top'])} | {fmt(r['tvd_uniform'])} |")
        lines.append("")

    (OUT_DIR / "softmax_flatness.md").write_text("\n".join(lines))
    with open(OUT_DIR / "softmax_flatness.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
