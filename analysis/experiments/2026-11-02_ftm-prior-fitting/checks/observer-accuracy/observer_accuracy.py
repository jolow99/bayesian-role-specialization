"""Human inference accuracy against two Bayesian reference observers, by type.

On the same answers, from the unmixed posterior P_N at the inference fit's parameters:
  MAP      reports its most likely role
  sampler  samples its report from P_N (expected accuracy = P_N(true role))
human − sampler gets a 95% CI from a cluster bootstrap over team-rounds.
Non-updater = P(non-updater) > 0.5.

Run from analysis/:
    uv run python experiments/2026-11-02_ftm-prior-fitting/checks/observer-accuracy/observer_accuracy.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent
FITTING_DIR = OUT_DIR.parents[1] / "inference-fitting"
sys.path.insert(0, str(FITTING_DIR))

from pipeline import answer_marginals, load_answers  # noqa: E402

N_BOOT = 10_000


def boot_diff_ci(team, x, y):
    """95% CI of mean(x) − mean(y), cluster bootstrap over teams."""
    rng = np.random.default_rng(0)
    members = [np.flatnonzero(team == t) for t in np.unique(team)]
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([members[t] for t in rng.integers(0, len(members), len(members))])
        d[b] = x[idx].mean() - y[idx].mean()
    return np.percentile(d, [2.5, 97.5]).tolist()


def main():
    params = json.load(open(FITTING_DIR / "params.json"))
    teams, answers = load_answers()
    with open(FITTING_DIR / "participant_types.csv") as f:
        p_nu = {r["participant_id"]: float(r["p_non_updater"]) for r in csv.DictReader(f)}
    is_nu = np.array([p_nu[answers["participant_ids"][i]] > 0.5 for i in answers["participant"]])
    human = (answers["inferred"] == answers["true"]).astype(float)
    P = answer_marginals(teams, answers, params["tau_prior"], params["epsilon"], params["beta"])
    map_correct = (P.argmax(axis=1) == answers["true"]).astype(float)
    sampler = P[np.arange(len(P)), answers["true"]]

    out = {}
    for name, m in [("updater", ~is_nu), ("non_updater", is_nu), ("all", np.ones_like(is_nu))]:
        out[name] = {"n_answers": int(m.sum()), "human_acc": float(human[m].mean()),
                     "map_acc": float(map_correct[m].mean()), "sampler_acc": float(sampler[m].mean()),
                     "human_minus_sampler_ci": boot_diff_ci(answers["team"][m], human[m], sampler[m])}
    with open(OUT_DIR / "observer_accuracy.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
