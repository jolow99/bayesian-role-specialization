"""Shared scaffolding for the 2026-06-16 human-team case-study figure
(R3_team_case) — the human-round analogue of the 06-07 bot-adaptation
storyboard (R3_adaptation_case).

Joins two views of the SAME clean human team-rounds:
  * PlayerRounds (roles, logged turns/HP/actions, reported inferences,
    stats) via 2026-06-05-paper-figures-v2/common.load_clean_human_teams;
  * value matrices (env_config) via the 05-12 pipeline's
    load_human_team_records — needed for the best-response computation.
Both are keyed by (export_name, game_id, round_number).

Bayesian observer posteriors use the fitted Stage-1 params (tau_prior,
epsilon, memory drift) from the 05-25 full pipeline, cross-checked
byte-identical to common.py's 05-12 fit. posteriors[s] is the belief at
the START of stage s (= inference time for reports made AT stage s,
about stage s-1), exactly as in the bot-round experiment.

Best-response (the "should they switch?" reference): at stage s, player
i knows their own role; their belief over the two TEAMMATES' roles is
the joint posteriors[s] marginalized over player i's own axis. The
best-response role maximizes the eap-weighted expected team VALUE
(section2/06-05 value convention) at the stage-start HP, holding that
teammate belief fixed. A player "stayed against best-response" at stage
s if role[s] == role[s-1] (no switch) yet the best-response role differs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
V2_DIR = SCRIPT_DIR.parent / "2026-06-05-paper-figures-v2"
PIPELINE_DIR = SCRIPT_DIR.parent / "2026-05-12-current-export-metric-comparison"
sys.path.insert(0, str(V2_DIR))

from common import (  # noqa: E402
    compute_posteriors, load_clean_human_teams, load_human_team_records,
    load_stage1, target_marginal,
)
from shared.constants import (  # noqa: E402
    ROLE_CHAR_TO_IDX, ROLE_SHORT, SYMMETRIC_PROFILES, TURNS_PER_STAGE,
)

FULL_PIPELINE_STAGE1 = (SCRIPT_DIR.parent / "2026-05-25-full-pipeline"
                        / "stage1_inference" / "best_inference_params.json")

OUT_DIR = SCRIPT_DIR / "stuff to incorporate"
OUT_DIR.mkdir(exist_ok=True)


def load_stage1_canonical():
    """05-25 Stage-1 params + MemoryStrategy, cross-checked vs common.py."""
    with open(FULL_PIPELINE_STAGE1) as f:
        s1_canon = json.load(f)
    s1, strat = load_stage1()
    for k in ("tau_prior", "epsilon", "memory_strategy"):
        assert s1_canon[k] == s1[k], (
            f"Stage-1 param mismatch on '{k}': 05-25 has {s1_canon[k]!r}, "
            f"common.py (05-12) has {s1[k]!r}.")
    return s1_canon, strat


# ──────────────────────────────────────────────────────────────────────
# Join PlayerRounds with value matrices
# ──────────────────────────────────────────────────────────────────────

def _env_by_key():
    """{(export, game_id, round_number): env_config} from the value pipeline."""
    recs = load_human_team_records(verbose=False)
    return {(r["export_name"], r["game_id"], r["round_number"]): r["env_config"]
            for r in recs}


def human_round_record(key, team_prs, env_config):
    """Normalize one clean human team-round into a flat dict.

    Everything is in IN-GAME POSITION order (player_id), the same order the
    value matrices and inferred_roles keys use.
    """
    export_name, game_id, round_number = key
    rnd = team_prs[0].round
    cfg = rnd.config
    parts = rnd.stat_profile_id.split("_")
    player_stats = np.array([[int(c) for c in p] for p in parts], dtype=float)

    by_pid = {pr.player_id: pr for pr in team_prs}
    n_stages = max(len(pr.round.stages) for pr in team_prs)

    # per-player role per stage (position order)
    roles = {pid: [int(s.role_idx) for s in pr.round.stages]
             for pid, pr in by_pid.items()}
    role_seq = []
    for s in range(n_stages):
        rr = [0, 0, 0]
        for pid, rs in roles.items():
            if s < len(rs):
                rr[pid] = rs[s]
        role_seq.append(rr)

    # logged turns: list per stage of dicts (action, teamHealth, enemyHealth).
    # All three players log the same team/enemy HP; actions differ. Use the
    # per-player action from each player's own stage.turns.
    stage_turns = []
    for s in range(n_stages):
        turns_s = []
        ref_pr = next(pr for pr in team_prs if s < len(pr.round.stages))
        n_turns = len(ref_pr.round.stages[s].turns)
        for ti in range(n_turns):
            thp = float(ref_pr.round.stages[s].turns[ti]["teamHealth"])
            ehp = float(ref_pr.round.stages[s].turns[ti]["enemyHealth"])
            actions = {}
            for pid, pr in by_pid.items():
                if s < len(pr.round.stages) and ti < len(pr.round.stages[s].turns):
                    actions[pid] = pr.round.stages[s].turns[ti]["action"]
            turns_s.append({"team_hp": thp, "enemy_hp": ehp, "actions": actions})
        stage_turns.append(turns_s)

    n_turns_total = sum(len(t) for t in stage_turns)
    turn_intent = [int(c) for c in rnd.enemy_intent_sequence[:n_turns_total]]

    # reports made AT stage s (about stage s-1), keyed by observer pid:
    # {observer_pid: {target_pid: guessed_role}}
    inferred = {}
    for pid, pr in by_pid.items():
        for si, stage in enumerate(pr.round.stages):
            if si == 0 or not stage.inferred_roles:
                continue
            inferred.setdefault(si, {})[pid] = dict(stage.inferred_roles)

    return {
        "export_name": export_name,
        "game_id": game_id,
        "round_number": int(round_number),
        "outcome": rnd.outcome,
        "stat_profile_id": rnd.stat_profile_id,
        "symmetry": SYMMETRIC_PROFILES.get(rnd.stat_profile_id),
        "player_stats": player_stats,
        "roles": roles,                 # {pid: [role per stage]}
        "role_seq": role_seq,           # [ [r0,r1,r2] per stage ]
        "stage_turns": stage_turns,
        "turn_intent": turn_intent,
        "inferred": inferred,
        "team_max_hp": int(cfg.get("maxTeamHealth", 15)),
        "enemy_max_hp": int(cfg.get("maxEnemyHealth", 30)),
        "boss_damage": float(cfg.get("bossDamage", 2)),
        "eis": rnd.enemy_intent_sequence,
        "env_config": env_config,
        "n_stages": n_stages,
    }


def load_human_records(verbose: bool = True):
    """Clean human team-rounds that also have a value matrix attached."""
    teams = load_clean_human_teams(verbose=verbose)
    env_by_key = _env_by_key()
    out = []
    for key, team_prs in teams.items():
        if key not in env_by_key:
            continue
        out.append(human_round_record(key, team_prs, env_by_key[key]))
    if verbose:
        print(f"[common_human] {len(out)} clean human team-rounds with values")
    return out


# ──────────────────────────────────────────────────────────────────────
# Bayesian observer posteriors + best-response
# ──────────────────────────────────────────────────────────────────────

def human_posteriors(rec, s1, strat):
    """posteriors[s] = start-of-stage-s joint belief (3,3,3)."""
    data = {
        "player_stats": rec["player_stats"],
        "boss_damage": rec["boss_damage"],
        "team_max_hp": rec["team_max_hp"],
        "enemy_max_hp": rec["enemy_max_hp"],
        "eis": rec["eis"],
        "role_seq": rec["role_seq"],
        "queries": [],
    }
    return compute_posteriors(data, s1["tau_prior"], s1["epsilon"], strat)


def stage_start_hp(rec):
    """(team_hp, enemy_hp) at the START of each stage, from LOGGED turns."""
    out = [(float(rec["team_max_hp"]), float(rec["enemy_max_hp"]))]
    for s in range(rec["n_stages"]):
        turns = rec["stage_turns"][s]
        if turns:
            out.append((turns[-1]["team_hp"], turns[-1]["enemy_hp"]))
        else:
            out.append(out[-1])
    return out[:rec["n_stages"]]


def value_vector_27(rec, thp_f, ehp_f):
    """eap-weighted expected value of all 27 position-ordered combos."""
    values = rec["env_config"]["values"]
    lds = [int(c) for c in rec["eis"]]
    eap = sum(lds) / len(lds) if lds else 0.5
    thp = int(np.clip(int(thp_f), 0, values.shape[2] - 1))
    ehp = int(np.clip(int(ehp_f), 0, values.shape[3] - 1))
    return (1.0 - eap) * values[:, 0, thp, ehp] + eap * values[:, 1, thp, ehp]


def _combo_index(role_by_pos):
    return role_by_pos[0] * 9 + role_by_pos[1] * 3 + role_by_pos[2]


def best_response(rec, posteriors, s, pid):
    """Best-response role for player `pid` at stage s under the posterior.

    Belief over the two teammates = posteriors[s] marginalized over pid's
    own axis. Returns (br_role, ev_per_role[3]). Uses stage-start HP.
    """
    thp, ehp = stage_start_hp(rec)[s]
    V = value_vector_27(rec, thp, ehp)             # (27,)
    Pjk = posteriors[s].sum(axis=pid)              # (3,3) over the two others
    others = [p for p in range(3) if p != pid]
    ev = np.zeros(3)
    for c in range(3):                             # candidate role for pid
        tot = 0.0
        for rj in range(3):
            for rk in range(3):
                role_by_pos = [0, 0, 0]
                role_by_pos[pid] = c
                role_by_pos[others[0]] = rj
                role_by_pos[others[1]] = rk
                tot += Pjk[rj, rk] * V[_combo_index(role_by_pos)]
        ev[c] = tot
    return int(np.argmax(ev)), ev


def stage_value_rank(rec, s):
    """Rank (1=best of 27) of the actually-played combo at stage s."""
    thp, ehp = stage_start_hp(rec)[s]
    V = value_vector_27(rec, thp, ehp)
    chosen = _combo_index(rec["role_seq"][s])
    order = np.argsort(-V)
    return int(np.where(order == chosen)[0][0]) + 1


def identical_pairs(rec):
    """List of (pid_a, pid_b) position pairs with identical stat profiles."""
    parts = rec["stat_profile_id"].split("_")
    return [(a, b) for a in range(3) for b in range(a + 1, 3)
            if parts[a] == parts[b]]
