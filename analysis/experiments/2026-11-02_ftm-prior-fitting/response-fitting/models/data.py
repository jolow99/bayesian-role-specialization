"""Clean human team-rounds with value matrices, and one row per (team-round, stage, position) choice."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from shared import EXPORTS_DIR, HUMAN_ENVS_DIR
from shared.constants import ROLE_CHAR_TO_IDX, ROLE_SHORT
from shared.data_loading import load_all_exports
from shared.env_loading import load_env_config


def load_records():
    prs = load_all_exports(data_dirs=sorted(EXPORTS_DIR.glob("bayesian-role-specialization-*")))
    dropout_games = {pr.game_id for pr in prs if pr.is_dropout}
    teams = defaultdict(list)
    for pr in prs:
        if pr.game_id not in dropout_games and pr.round.round_type == "human":
            teams[(pr.game_id, pr.round.round_number)].append(pr)

    envs, records = {}, []
    for team in teams.values():
        if len(team) != 3:
            continue
        team = sorted(team, key=lambda p: p.player_id)
        rnd = team[0].round
        n_stages = min(len(pr.round.stages) for pr in team)
        if n_stages == 0:
            continue
        optimal = "".join(ROLE_SHORT[int(r)] for r in rnd.config["optimalRoles"])
        env_dir = f"{rnd.stat_profile_id}__{optimal}"
        if env_dir not in envs:
            envs[env_dir] = load_env_config(HUMAN_ENVS_DIR / env_dir)
        records.append({
            "game_id": team[0].game_id,
            "round_number": int(rnd.round_number),
            "env_id": str(rnd.config.get("envId", "")),
            "stat_profile": rnd.stat_profile_id,
            "optimal_roles": optimal,
            "lds": [int(c) for c in rnd.enemy_intent_sequence],
            "stage_roles": ["".join(ROLE_SHORT[pr.round.stages[s].role_idx] for pr in team)
                            for s in range(n_stages)],
            "env_config": envs[env_dir],
            "participant_ids": [pr.participant_id for pr in team],
        })
    print(f"[data] {len(records)} clean human team-rounds")
    return records


class Obs:
    """One row per (team-round, stage, position) choice."""

    def __init__(self, records, p_non_updater):
        rows, cells, mcells, envs = [], {}, {}, {}
        for ri, rec in enumerate(records):
            stats = rec["env_config"]["player_stats"]
            for s, combo in enumerate(rec["stage_roles"]):
                prev = rec["stage_roles"][s - 1] if s > 0 else None
                for pos in range(3):
                    pid = rec["participant_ids"][pos]
                    cell = (rec["env_id"], s, tuple(int(x) for x in stats[pos]))
                    rows.append((ri, s, pos, ROLE_CHAR_TO_IDX[combo[pos]],
                                 -1 if prev is None else ROLE_CHAR_TO_IDX[prev[pos]],
                                 cells.setdefault(cell, len(cells)),
                                 mcells.setdefault((rec["env_id"], s), len(mcells)),
                                 envs.setdefault(rec["env_id"], len(envs)),
                                 p_non_updater[pid] > 0.5))
        a = np.array(rows, dtype=int)
        self.ri, self.s, self.pos, self.a, self.prev, self.cell, self.mcell, self.env = a[:, :8].T
        self.non_updater = a[:, 8].astype(bool)
        self.pid = np.array([records[ri]["participant_ids"][p] for ri, p in zip(self.ri, self.pos)])
        self.n = len(rows)
        self.has_prev = self.prev >= 0
        self.stick = np.zeros((self.n, 3))
        self.stick[np.flatnonzero(self.has_prev), self.prev[self.has_prev]] = 1.0
