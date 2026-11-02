"""Within-round adaptation in bot rounds: stage 1 vs stage 5 in the same rounds, by type.

Only bot rounds reaching stage 5, so stage 1 and stage 5 compare the same rounds (no
composition change from rounds ending early). Rates of playing the deviate-optimal and
stat-optimal role. Non-updater = P(non-updater) > 0.5.

Run from analysis/:
    uv run python experiments/2026-11-02_ftm-prior-fitting/checks/within-round-adaptation/within_round.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from shared import EXPORTS_DIR
from shared.data_loading import load_all_exports

OUT_DIR = Path(__file__).resolve().parent
TYPES_CSV = OUT_DIR.parents[1] / "inference-fitting" / "participant_types.csv"


def main():
    with open(TYPES_CSV) as f:
        p_nu = {r["participant_id"]: float(r["p_non_updater"]) for r in csv.DictReader(f)}
    prs = load_all_exports(data_dirs=sorted(EXPORTS_DIR.glob("bayesian-role-specialization-*")))
    dropout_games = {pr.game_id for pr in prs if pr.is_dropout}
    rounds = [pr for pr in prs if pr.game_id not in dropout_games and pr.round.round_type == "bot"
              and pr.round.config.get("optimalDeviateRolesId") and len(pr.round.stages) >= 5]
    out = {}
    for name, keep in (("updaters", lambda p: p <= 0.5), ("non-updaters", lambda p: p > 0.5)):
        rs = [pr for pr in rounds if keep(p_nu[pr.participant_id])]
        role = lambda pr, s: int(pr.round.stages[s].role_idx)
        dev = {s: float(np.mean([role(pr, s) == int(pr.round.deviate_roles[0]) for pr in rs])) for s in (0, 4)}
        stat = {s: float(np.mean([role(pr, s) == int(pr.round.optimal_roles[0]) for pr in rs])) for s in (0, 4)}
        out[name] = {"n_rounds": len(rs), "dev_stage1": dev[0], "dev_stage5": dev[4],
                     "stat_stage1": stat[0], "stat_stage5": stat[4]}
    with open(OUT_DIR / "within_round.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
