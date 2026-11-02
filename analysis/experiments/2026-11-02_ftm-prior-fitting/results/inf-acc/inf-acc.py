"""Inference accuracy of updaters vs non-updaters (P(non-updater) > 0.5).

An answer is correct if the reported role equals the target's role in the
previous stage. Chance = 33.3%. Writes inf-acc.json next to this script.

Run from analysis/:
    uv run python experiments/2026-11-02_ftm-prior-fitting/results/inf-acc/inf-acc.py
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

from pipeline import load_answers  # noqa: E402


def main():
    _, answers = load_answers()
    with open(FITTING_DIR / "participant_types.csv") as f:
        p_nu = {r["participant_id"]: float(r["p_non_updater"]) for r in csv.DictReader(f)}
    is_nu = np.array([p_nu[pid] > 0.5 for pid in answers["participant_ids"]])
    correct = answers["inferred"] == answers["true"]
    acc = np.array([correct[ix].mean() for ix in answers["by_participant"]])

    results = {"threshold": 0.5, "chance": 1 / 3}
    for name, mask in [("updater", ~is_nu), ("non_updater", is_nu),
                       ("all", np.ones_like(is_nu))]:
        a = acc[mask]
        ans = correct[mask[answers["participant"]]]
        results[name] = {"n_participants": int(mask.sum()), "n_answers": len(ans),
                         "pooled_acc": float(ans.mean()),
                         "participant_mean_acc": float(a.mean()),
                         "participant_se": float(a.std(ddof=1) / np.sqrt(len(a)))}
    with open(OUT_DIR / "inf-acc.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
