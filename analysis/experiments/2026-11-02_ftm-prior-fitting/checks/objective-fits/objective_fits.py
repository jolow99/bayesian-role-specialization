"""SI Table S4: the six Bayesian models fit to updaters under each objective.

Same fit as response-fitting/fit.py (fit_bayes), maximising marg-r, agg-LL or mean-LL
on updaters' choices; all four metrics at each fit (combo-r on all teams, non-updaters
on their FTM belief). combo-r is not used as an objective: it needs whole teams.
Writes objective_fits.json and objective_fits.tex (table rows).

Run from analysis/:
    uv run python experiments/2026-11-02_ftm-prior-fitting/checks/objective-fits/objective_fits.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
EXP_DIR = OUT_DIR.parents[1]
sys.path.insert(0, str(EXP_DIR / "response-fitting"))

from fit import fit_bayes  # noqa: E402
from models.beliefs import player_beliefs  # noqa: E402
from models.data import Obs, load_records  # noqa: E402
from models.predict import bayes_marginals  # noqa: E402
from models.scoring import combo_r, metrics  # noqa: E402

OBJECTIVES = {"marg_r": "marg-$r$", "agg_ll": "agg-LL", "mean_ll": "mean-LL"}
ORDER = ["Bayesian-PS", "Bayesian-Walk-PS", "Bayesian-Threshold-PS", "Bayesian-BR",
         "Bayesian-Walk-BR", "Bayesian-Threshold-BR"]


def pstr(p):
    out = []
    if "tau_softmax" in p:
        out.append(f"$\\tau_v{{=}}{p['tau_softmax']:.2f}$")
    if "epsilon_switch" in p:
        out.append(f"$\\varepsilon_s{{=}}{1 - p['epsilon_switch']:.2f}$")
    if "delta" in p:
        out.append(f"$\\delta_{{\\mathrm{{{'PS' if abs(p['delta']) <= 1 and 'tau_softmax' not in p else 'BR'}}}}}"
                   f"{{=}}{p['delta']:.2f}$".replace("=}-", "=}{-}"))
    return ", ".join(out) or "none"


def num(x):
    return f"{x:.3f}".replace("-", "$-$")


def main():
    params = json.load(open(EXP_DIR / "inference-fitting" / "params.json"))
    with open(EXP_DIR / "inference-fitting" / "participant_types.csv") as f:
        p_nu = {r["participant_id"]: float(r["p_non_updater"]) for r in csv.DictReader(f)}
    records = load_records()
    ob = Obs(records, p_nu)
    PM, EV = player_beliefs(records, ob, params, ob.non_updater)
    mask = ~ob.non_updater

    out = {}
    for obj in OBJECTIVES:
        P = fit_bayes(ob, mask, PM, EV, obj)
        out[obj] = {}
        for m, p in P.items():
            M = bayes_marginals(ob, PM, EV, m, p)
            out[obj][m] = {"params": p, "combo_r": combo_r(M, ob, records), **metrics(M, ob, mask)}
    with open(OUT_DIR / "objective_fits.json", "w") as f:
        json.dump(out, f, indent=2)

    rows = []
    for m in ORDER:
        objs = ["agg_ll"] if m == "Bayesian-PS" else list(OBJECTIVES)
        for k, obj in enumerate(objs):
            r = out[obj][m]
            name = m if k == 0 else ""
            fit_on = "(all)" if m == "Bayesian-PS" else OBJECTIVES[obj]
            rows.append(f"{name} & {fit_on} & {pstr(r['params'])} & "
                        + " & ".join(num(r[c]) for c in ("combo_r", "marg_r", "agg_ll", "mean_ll")) + " \\\\")
        rows.append("\\midrule")
    (OUT_DIR / "objective_fits.tex").write_text("\n".join(rows[:-1]) + "\n")
    print("\n".join(rows[:-1]))


if __name__ == "__main__":
    main()
