"""Does Walk-BR still fit the Fig. 4 team (main text) and the 4 SI team cases under the new fits?

Probability Walk-BR gives each player's chosen role, per stage:
  paper  paper fit (τ_v = 11.67, ε_s = 0.462) on the 05-25 Stage-1 beliefs; reproduces the caption
  new-a  updaters' fit, everyone on the mixed posterior (as the figure draws it)
  new-b  updaters' fit, each player on their own type's belief (non-updaters: FTM prior)

Run from analysis/:
    uv run python experiments/2026-11-02_ftm-prior-fitting/checks/fig4-team/team_fit.py
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
from models.predict import bayes_marginals  # noqa: E402

CASES = [("2F8H1E", 2, "Fig. 4"), ("11B0J4", 6, "SI"), ("11B0J4", 8, "SI"), ("K72DV0", 5, "SI"),
         ("RBRN0Z", 8, "SI")]
STAGE1_0525 = {"tau_prior": 4.6385, "epsilon": 0.0624, "beta": 0.5}  # paper's inference fit
PAPER_WALK_BR = {"tau_softmax": 11.67, "epsilon_switch": 1 - 0.462}


def main():
    params = json.load(open(EXP_DIR / "inference-fitting" / "params.json"))
    with open(EXP_DIR / "inference-fitting" / "participant_types.csv") as f:
        p_nu = {r["participant_id"]: float(r["p_non_updater"]) for r in csv.DictReader(f)}
    records = load_records()
    ob = Obs(records, p_nu)
    walk_br = json.load(open(EXP_DIR / "response-fitting" / "fit-results" / "updaters" /
                             "bayes-models.json"))["Bayesian-Walk-BR"]["params"]

    variants = {"paper": ({**params, **STAGE1_0525}, np.zeros(ob.n, bool), PAPER_WALK_BR),
                "new-a": (params, np.zeros(ob.n, bool), walk_br),
                "new-b": (params, ob.non_updater, walk_br)}
    chosen = {}
    for name, (bp, ftm_rows, wp) in variants.items():
        PM, EV = player_beliefs(records, ob, bp, ftm_rows)
        M = bayes_marginals(ob, PM, EV, "Bayesian-Walk-BR", wp)
        chosen[name] = M[np.arange(ob.n), ob.a]

    lines = ["# Walk-BR on the team-case figures", "",
             "Probability on each player's chosen role (chance 0.33).", ""]
    for suffix, rnd, where in CASES:
        ri = next(i for i, r in enumerate(records)
                  if r["game_id"].endswith(suffix) and r["round_number"] == rnd)
        rows = np.flatnonzero(ob.ri == ri)
        lines += team_table(f"…{suffix}, round {rnd} ({where})", rows, ob,
                            {v: chosen[v][rows] for v in variants})
    (OUT_DIR / "team_fit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def team_table(title, rows, ob, probs):
    variants = list(probs)
    lines = [f"## {title}", "",
             "| Stage | Player | Type | Role | " + " | ".join(variants) + " |",
             "|--:|--:|---|:-:|" + "--:|" * len(variants)]
    for k, q in enumerate(rows):
        typ = "non-updater" if ob.non_updater[q] else "updater"
        lines.append(f"| {ob.s[q] + 1} | P{ob.pos[q] + 1} | {typ} | {'FTM'[ob.a[q]]} | "
                     + " | ".join(f"{probs[v][k]:.2f}" for v in variants) + " |")
    lines += ["", "| Stage mean | " + " | ".join(variants) + " |", "|--:|" + "--:|" * len(variants)]
    for s in np.unique(ob.s[rows]):
        m = ob.s[rows] == s
        lines.append(f"| {s + 1} | " + " | ".join(f"{probs[v][m].mean():.2f}" for v in variants) + " |")
    lines += ["| all | " + " | ".join(f"{probs[v].mean():.2f}" for v in variants) + " |", ""]
    return lines


if __name__ == "__main__":
    main()
