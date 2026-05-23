"""Bot-round data + trajectory pipeline mirroring the human-round path.

Mirrors `2026-05-12-current-export-metric-comparison/pipeline.py` but for the
bot-round side: one human + two stubborn bots whose roles are fixed by
``botPlayers[i].strategy.role``. CLAUDE.md → "Bot Round Ground Truth" is the
source of truth for the permutation between logical and in-game positions —
this module is the single place that mapping happens.

Outputs ``records`` shaped like the human pipeline so the same trajectory
engine + model factories can run on them.
"""

from __future__ import annotations

import re as _re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from shared import HUMAN_ENVS_DIR
from shared.constants import (
    ROLE_SHORT, ROLE_CHAR_TO_IDX, TURNS_PER_STAGE,
)
from shared.data_loading import load_all_exports, build_bot_round_layout
from shared.inference import (
    utility_based_prior, bayesian_update, preferred_action, game_step,
)
from shared.parsing import parse_stat_optimal_roles, parse_deviate_roles

# Bring in the human pipeline so we can reuse helpers + tuned params.
HERE = Path(__file__).resolve().parent
METRIC_DIR = HERE.parent / "2026-05-12-current-export-metric-comparison"
sys.path.insert(0, str(METRIC_DIR))

from pipeline import (  # noqa: E402
    EXPORT_DIRS, discover_dropout_games, filter_clean_prs,
    _parse_value_config, apply_boundary,
)


# ──────────────────────────────────────────────────────────────────────
# Value matrix lookup — any matrix with matching stats + HP + boss damage
# ──────────────────────────────────────────────────────────────────────

def _find_bot_value_dir(cfg: dict, stat_profile: str) -> Path | None:
    """Look up a precomputed value matrix that matches the bot round's env.

    Most bot-round (HP, boss) combos are NOT in `human_envs_value_matrices/`
    (which was precomputed for the human-round game parameters), so we
    return None instead of raising — Bayesian-Belief still works without
    `values` and the visualization can show the other models only where a
    matrix exists.
    """
    max_thp = cfg.get("maxTeamHealth")
    max_ehp = cfg.get("maxEnemyHealth")
    boss = cfg.get("bossDamage")

    for candidate in sorted(HUMAN_ENVS_DIR.glob(f"{stat_profile}__*")):
        if not (candidate / "values.npy").exists():
            continue
        vc = _parse_value_config(candidate / "config.py")
        stats_str = "_".join(
            "".join(str(int(v)) for v in row) for row in vc["player_stats"])
        if (vc["team_max_hp"] == max_thp
                and vc["enemy_max_hp"] == max_ehp
                and abs(vc["boss_damage"] - float(boss)) < 1e-6
                and stats_str == stat_profile):
            return candidate
    return None


def _build_env_config(cfg: dict, stat_profile: str,
                       value_dir: Path | None) -> dict:
    """env_config for bot rounds. HP / boss / stats come from the round
    config + stat_profile_id; values are attached only if a matching
    precomputed matrix exists.
    """
    logical_stats = np.array(
        [[int(c) for c in part] for part in stat_profile.split("_")],
        dtype=float,
    )
    env = {
        "values": None,
        "player_stats": logical_stats,
        "boss_damage": float(cfg.get("bossDamage")),
        "team_max_hp": int(cfg.get("maxTeamHealth")),
        "enemy_max_hp": int(cfg.get("maxEnemyHealth")),
    }
    if value_dir is not None:
        env["values"] = np.load(value_dir / "values.npy")
    return env


# ──────────────────────────────────────────────────────────────────────
# Record loading
# ──────────────────────────────────────────────────────────────────────

def load_bot_round_records(verbose: bool = True) -> list[dict]:
    """One record per bot player-round. Roles and stats are permuted to
    in-game position order, so the human's chosen role lands at index
    ``pr.player_id`` (the resolver from CLAUDE.md) and bot roles at the
    sorted-other positions. Records carry an extra ``human_pid`` key so
    downstream code can extract the human's marginal."""
    all_prs = load_all_exports(data_dirs=EXPORT_DIRS)
    dropout_games = discover_dropout_games(all_prs)
    clean = filter_clean_prs(all_prs, dropout_games)

    env_cache: dict[str, dict] = {}
    n_with_values = 0
    records: list[dict] = []
    for pr in clean:
        if pr.round.round_type != "bot":
            continue
        cfg = pr.round.config
        layout = build_bot_round_layout(pr)
        stat_profile = pr.round.stat_profile_id
        dev_id = cfg.get("optimalDeviateRolesId")
        if not dev_id:
            continue

        # Value matrices are scarce for bot-round HP/boss configs; build
        # the env from cfg + stat_profile and only attach `values` when
        # available.
        cache_key = (stat_profile, cfg.get("maxTeamHealth"),
                     cfg.get("maxEnemyHealth"), cfg.get("bossDamage"))
        if cache_key not in env_cache:
            vd = _find_bot_value_dir(cfg, stat_profile)
            env_cache[cache_key] = _build_env_config(cfg, stat_profile, vd)
        env = dict(env_cache[cache_key])
        if env["values"] is not None:
            n_with_values += 1
        # Replace stats with the in-game permuted version (so position 0/1/2
        # in `values[combo,...]` lines up with the in-game ordering).
        env["player_stats"] = layout.player_stats.astype(float)

        # Stage-by-stage 3-letter combo in in-game position order:
        # human's chosen role at pr.player_id; bot roles constant at the others.
        stage_roles = []
        for stage in pr.round.stages:
            roles = [None, None, None]
            roles[layout.pid] = stage.role_idx
            for pos, role in layout.bot_role_map.items():
                roles[pos] = role
            stage_roles.append("".join(ROLE_SHORT[r] for r in roles))
        if not stage_roles:
            continue

        lds = [int(c) for c in pr.round.enemy_intent_sequence]
        stat_opt_logical = parse_stat_optimal_roles(dev_id)
        dev_opt_logical = parse_deviate_roles(dev_id)

        records.append({
            "export_name": pr.export_name,
            "game_id": pr.game_id,
            "round_id": f"{pr.game_id}_r{pr.round.round_number}",
            "round_type": "bot",
            "round_number": int(pr.round.round_number),
            "treatment_id": f"{stat_profile}__{dev_id}",
            "stat_profile": stat_profile,
            "deviate_roles_id": dev_id,
            "human_pid": int(layout.pid),
            "bot_positions": list(layout.others),
            "bot_role_map": dict(layout.bot_role_map),   # in-game pos -> role idx
            "human_stat_optimal": int(stat_opt_logical[0]),
            "human_deviate_optimal": int(dev_opt_logical[0]),
            "lds": lds,
            "stage_roles": stage_roles,                  # in-game position order
            "human_role_seq": [int(s.role_idx) for s in pr.round.stages],
            "env_config": env,
        })

    if verbose:
        n_treatments = len(set(r["treatment_id"] for r in records))
        print(f"[bot_pipeline] {len(records)} bot player-rounds across "
              f"{n_treatments} treatments (clean games only). "
              f"{n_with_values}/{len(records)} have a matching value matrix "
              f"(Belief works without values; Value/Walk need them).")
    return records


def group_by_treatment(records):
    """Bucket bot records by (stat_profile, deviate_roles_id)."""
    by = defaultdict(list)
    for r in records:
        by[r["treatment_id"]].append(r)
    return by


# ──────────────────────────────────────────────────────────────────────
# Per-stage trajectory: replay actual game using observed roles
# ──────────────────────────────────────────────────────────────────────

def precompute_bot_trajectory(record, tau_prior, epsilon, strategy_kind, strategy_param):
    """Same shape as `pipeline._precompute_human` but uses the bot-permuted
    player_stats and runs the existing trajectory engine.

    The 3 players' "roles" used to compute actions each turn are the in-game
    role combo recorded in stage_roles[stage] — i.e., human's chosen role +
    the two constant bot roles, all already in in-game position order.
    """
    env = record["env_config"]
    player_stats = env["player_stats"]    # already permuted to in-game order
    boss_damage = env["boss_damage"]
    team_max_hp = env["team_max_hp"]
    enemy_max_hp = env["enemy_max_hp"]

    original_prior = utility_based_prior(player_stats, tau=tau_prior)
    stage_roles_list = record["stage_roles"]
    lds = record["lds"]

    role_letter_to_roles = lambda combo: [ROLE_CHAR_TO_IDX[c] for c in combo]

    stages = []
    current = original_prior.copy()
    team_hp = float(team_max_hp)
    enemy_hp = float(enemy_max_hp)
    turn_idx = 0
    prev_roles = None

    for combo in stage_roles_list:
        if turn_idx >= len(lds) or team_hp <= 0 or enemy_hp <= 0:
            break

        intent = int(lds[turn_idx])
        thp = int(min(max(0, team_hp), team_max_hp))
        ehp = int(min(max(0, enemy_hp), enemy_max_hp))

        stages.append({
            "prior": current.copy(),
            "intent": intent,
            "thp": thp,
            "ehp": ehp,
            "human_combo": combo,
            "prev_roles": prev_roles,
        })

        roles = role_letter_to_roles(combo)
        prev_roles = list(roles)
        for _ in range(TURNS_PER_STAGE):
            if turn_idx >= len(lds) or team_hp <= 0 or enemy_hp <= 0:
                break
            intent_t = int(lds[turn_idx])
            actions = [preferred_action(roles[i], intent_t, team_hp, team_max_hp)
                       for i in range(3)]
            current = bayesian_update(current, actions, intent_t,
                                       team_hp, team_max_hp, epsilon)
            team_hp, enemy_hp = game_step(intent_t, team_hp, enemy_hp,
                                           actions, player_stats,
                                           boss_damage, team_max_hp)
            turn_idx += 1

        current = apply_boundary(current, original_prior,
                                  strategy_kind, strategy_param)

    return stages


def precompute_bot_trajectories(records, tau_prior, epsilon, strategy):
    return [
        precompute_bot_trajectory(r, tau_prior, epsilon,
                                   strategy.kind, strategy.param)
        for r in records
    ]


# ──────────────────────────────────────────────────────────────────────
# HP replay (parallel to the human-round one)
# ──────────────────────────────────────────────────────────────────────

def replay_bot_hp_timeline(record):
    env = record["env_config"]
    player_stats = env["player_stats"]
    boss_damage = env["boss_damage"]
    team_max_hp = env["team_max_hp"]
    enemy_max_hp = env["enemy_max_hp"]
    lds = record["lds"]
    stage_roles_list = record["stage_roles"]

    team_hp = [float(team_max_hp)]
    enemy_hp = [float(enemy_max_hp)]
    intents = []
    turn_idx = 0
    thp = float(team_max_hp)
    ehp = float(enemy_max_hp)

    for combo in stage_roles_list:
        roles = [ROLE_CHAR_TO_IDX[c] for c in combo]
        for _ in range(TURNS_PER_STAGE):
            if turn_idx >= len(lds) or thp <= 0 or ehp <= 0:
                break
            intent = int(lds[turn_idx])
            actions = [preferred_action(roles[i], intent, thp, team_max_hp)
                       for i in range(3)]
            thp, ehp = game_step(intent, thp, ehp, actions,
                                  player_stats, boss_damage, team_max_hp)
            intents.append(intent)
            team_hp.append(thp)
            enemy_hp.append(ehp)
            turn_idx += 1
    return team_hp, enemy_hp, intents
