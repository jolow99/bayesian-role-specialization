"""Do the Bayesian response models fitted to non-updaters differ meaningfully from Random / Random Walk?

For each Bayesian model (each group's own fit, each player on their own type's belief):
  TVD      mean total variation distance between its predicted role distribution and
           Random's / Random Walk's, over the group's choices (0 = identical, 1 = disjoint)
  ΔLL      per-participant log-likelihood minus Random Walk's, per choice; how many
           participants it fits better, and a Wilcoxon signed-rank p-value
Updaters are shown as the reference.

Run from analysis/:
    uv run python experiments/2026-11-02_ftm-prior-fitting/checks/bayes-vs-rw/bayes_vs_rw.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

OUT_DIR = Path(__file__).resolve().parent
EXP_DIR = OUT_DIR.parents[1]
sys.path.insert(0, str(EXP_DIR / "response-fitting"))

from models.beliefs import player_beliefs  # noqa: E402
from models.data import Obs, load_records  # noqa: E402
from models.predict import BAYES, bayes_marginals, random_walk  # noqa: E402


def tvd(A, B):
    return float(0.5 * np.abs(A - B).sum(axis=1).mean())


def main():
    params = json.load(open(EXP_DIR / "inference-fitting" / "params.json"))
    with open(EXP_DIR / "inference-fitting" / "participant_types.csv") as f:
        p_nu = {r["participant_id"]: float(r["p_non_updater"]) for r in csv.DictReader(f)}
    records = load_records()
    ob = Obs(records, p_nu)
    PM, EV = player_beliefs(records, ob, params, ob.non_updater)
    n_choices = Counter(ob.pid)

    out = {}
    lines = ["# Bayesian response models vs Random / Random Walk", "",
             "Each group's own fit. TVD: mean total variation distance of predicted role "
             "distributions over the group's choices. ΔLL vs RW: per-participant log-likelihood "
             "minus Random Walk's, per choice (mean over participants).", ""]
    for group, mask in (("non-updaters", ob.non_updater), ("updaters", ~ob.non_updater)):
        fit = {}
        for fam in ("bayes-models", "non-bayes-models"):
            fit.update(json.load(open(EXP_DIR / "response-fitting" / "fit-results" / group / f"{fam}.json")))
        rw = random_walk(ob, fit["Random Walk"]["params"]["eps"])[mask]
        rw_ll = fit["Random Walk"]["participant_ll"]
        lines += [f"## {group} ({len(rw_ll)} participants, {mask.sum()} choices)", "",
                  "| Model | TVD vs Random | TVD vs Random Walk | ΔLL vs RW / choice "
                  "| better than RW | Wilcoxon p |", "|---|--:|--:|--:|--:|--:|"]
        out[group] = {}
        for m in BAYES:
            M = bayes_marginals(ob, PM, EV, m, fit[m]["params"])[mask]
            d = np.array([(fit[m]["participant_ll"][p] - rw_ll[p]) / n_choices[p] for p in rw_ll])
            r = {"tvd_random": tvd(M, np.full_like(M, 1 / 3)), "tvd_random_walk": tvd(M, rw),
                 "dll_per_choice": float(d.mean()), "n_better": int((d > 0).sum()),
                 "n": len(d), "wilcoxon_p": float(wilcoxon(d).pvalue)}
            out[group][m] = r
            lines.append(f"| {m} | {r['tvd_random']:.3f} | {r['tvd_random_walk']:.3f} "
                         f"| {r['dll_per_choice']:+.4f} | {r['n_better']}/{r['n']} "
                         f"| {r['wilcoxon_p']:.3g} |")
        lines.append("")

    (OUT_DIR / "bayes_vs_rw.md").write_text("\n".join(lines))
    with open(OUT_DIR / "bayes_vs_rw.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
