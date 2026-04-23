"""Self-contained pipeline utilities for the 04-23 metric comparison.

Everything Stage 1 / Stage 2 needs lives in this file: memory-strategy
dispatch, the bot-round-safe trajectory precompute, pooled + disaggregated
evaluation, and checkpoint / JSON helpers. Nothing is imported from any
other ``experiments/`` folder.

Scope for this experiment:
    * Only the ``bayesian-role-specialization-2026-04-23-09-12-55`` export.
    * Human-only fit and eval (bot rounds are not loaded).
    * Strict "clean teams" filter: any game with a dropout is excluded
      entirely; remaining human team-rounds must have all three players
      present.
"""

from __future__ import annotations

import json
import os
import re as _re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared import DATA_ROOT, EXPORTS_DIR, BIG_PILOT_VM_DIR
from shared.constants import (
    ROLE_SHORT, ROLE_CHAR_TO_IDX, GAME_ROLE_TO_IDX, TURNS_PER_STAGE,
)
from shared.data_loading import load_all_exports
from shared.parsing import canonical_combo
from shared.inference import (
    utility_based_prior, bayesian_update, preferred_action, game_step,
    combo_marginal,
)
from shared.evaluation import (
    run_predictions, compute_pearson, compute_log_likelihood,
)


VALUE_MATRICES_DIR = BIG_PILOT_VM_DIR
LEGACY_VALUE_MATRICES_DIR = DATA_ROOT / "human_envs_value_matrices"
ENVS_DIR = DATA_ROOT / "envs"


class MissingValueMatrixError(RuntimeError):
    """Raised when a round config has no matching precomputed value matrix.

    Silently dropping such records produces metric comparisons over a biased
    subset (see project memory `value_matrix_silent_drop`). We raise so the
    caller has to either generate the missing matrix or change scope
    explicitly.
    """


def _parse_value_config(config_path: Path) -> dict:
    """Read PLAYER_STATS / HP / BOSS_DAMAGE from a value-matrix config.py."""
    text = config_path.read_text()
    team_max_hp = int(_re.search(r"TEAM_MAX_HP\s*=\s*(\d+)", text).group(1))
    enemy_max_hp = int(_re.search(r"ENEMY_MAX_HP\s*=\s*(\d+)", text).group(1))
    boss_damage = float(_re.search(r"BOSS_DAMAGE\s*=\s*([\d.]+)", text).group(1))
    ps_match = _re.search(
        r"PLAYER_STATS\s*=\s*(?:jnp\.array|np\.array)?\(?\s*(\[\[.+?\]\])\s*\)?",
        text, _re.DOTALL)
    rows = _re.findall(r"\[([^\[\]]+)\]", ps_match.group(1))
    player_stats = np.array([[float(x) for x in row.split(",")] for row in rows])
    return {
        "team_max_hp": team_max_hp,
        "enemy_max_hp": enemy_max_hp,
        "boss_damage": boss_damage,
        "player_stats": player_stats,
    }


# ──────────────────────────────────────────────────────────────────────
# Data scope: 04-23 export, clean games only
# ──────────────────────────────────────────────────────────────────────

EXPORT_NAME = "bayesian-role-specialization-2026-04-23-09-12-55"
EXPORT_DIR = EXPORTS_DIR / EXPORT_NAME


def discover_dropout_games(records) -> set[str]:
    """Return the set of game IDs that had any dropout player-round."""
    return {pr.game_id for pr in records if pr.is_dropout}


def filter_clean_prs(records, dropout_game_ids: set[str]):
    """Keep only player-rounds whose game had zero dropouts anywhere."""
    return [pr for pr in records if pr.game_id not in dropout_game_ids]


# ──────────────────────────────────────────────────────────────────────
# Memory-strategy dispatch
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MemoryStrategy:
    name: str
    kind: str          # full | window | drift_prior | drift_uniform | temper
    param: float | int | None


def build_strategy_grid() -> list[MemoryStrategy]:
    """Expanded stage-1 strategy space."""
    strats: list[MemoryStrategy] = []
    strats.append(MemoryStrategy("full", "full", None))
    for w in (1, 2, 3, 4):
        strats.append(MemoryStrategy(f"window_{w}", "window", int(w)))
    for d in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9):
        strats.append(MemoryStrategy(f"drift_prior_{d:.3f}", "drift_prior", float(d)))
    for d in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        strats.append(MemoryStrategy(f"drift_uniform_{d:.3f}", "drift_uniform", float(d)))
    for g in (0.1, 0.2, 0.3, 0.35, 0.41, 0.5, 0.7, 1.0):
        strats.append(MemoryStrategy(f"temper_{g:.3f}", "temper", float(g)))
    return strats


def apply_boundary(current: np.ndarray, original_prior: np.ndarray,
                   kind: str, param) -> np.ndarray:
    if kind in ("full", None):
        return current
    if kind == "drift_prior":
        out = (1.0 - param) * current + param * original_prior
    elif kind == "drift_uniform":
        uniform = np.ones_like(original_prior) / original_prior.size
        out = (1.0 - param) * current + param * uniform
    elif kind == "temper":
        if param == 1.0:
            return current
        log_cur = np.log(np.clip(current, 1e-300, None))
        log_cur = param * log_cur
        log_cur -= log_cur.max()
        out = np.exp(log_cur)
    else:
        raise ValueError(f"Unknown memory-strategy kind: {kind!r}")
    total = out.sum()
    if total > 0:
        out = out / total
    return out


def strategy_from_params(memory_strategy: str,
                          window: int | None,
                          drift_delta: float | int | None) -> MemoryStrategy:
    name = memory_strategy or "full"
    if name.startswith("window_"):
        return MemoryStrategy(name, "window", int(name.split("_", 1)[1]))
    if name.startswith("drift_prior_"):
        return MemoryStrategy(name, "drift_prior", float(name.split("_", 2)[2]))
    if name.startswith("drift_uniform_"):
        return MemoryStrategy(name, "drift_uniform", float(name.split("_", 2)[2]))
    if name.startswith("temper_"):
        return MemoryStrategy(name, "temper", float(name.split("_", 1)[1]))
    if name == "full":
        if window is not None:
            return MemoryStrategy(f"window_{int(window)}", "window", int(window))
        if drift_delta and drift_delta > 0:
            return MemoryStrategy(f"drift_prior_{float(drift_delta):.3f}",
                                  "drift_prior", float(drift_delta))
        return MemoryStrategy("full", "full", None)
    raise ValueError(f"Unrecognised memory strategy name: {name!r}")


# ──────────────────────────────────────────────────────────────────────
# Record loading (human-only, clean games)
# ──────────────────────────────────────────────────────────────────────

def _find_value_dir(cfg: dict) -> Path:
    """Locate a precomputed value matrix for a round config.

    Looks in ``human-envs-big-pilot-matrices/<stat_profile>__<role_combo>``.
    Cross-checks the matrix's config.py against the round config (HP, boss
    damage, stats) so we never pair a matrix with an incompatible env.

    Raises MissingValueMatrixError with diagnostic detail if nothing matches.
    """
    optimal_roles = cfg.get("optimalRoles") or []
    role_combo = "".join(ROLE_SHORT[int(r)] for r in optimal_roles)
    stat_profile = cfg.get("statProfileId", "")
    max_thp = cfg.get("maxTeamHealth")
    max_ehp = cfg.get("maxEnemyHealth")
    boss = cfg.get("bossDamage")

    candidate = VALUE_MATRICES_DIR / f"{stat_profile}__{role_combo}"
    if (candidate / "values.npy").exists():
        vc = _parse_value_config(candidate / "config.py")
        stats_str = "_".join(
            "".join(str(int(v)) for v in row) for row in vc["player_stats"])
        mismatches = []
        if vc["team_max_hp"] != max_thp:
            mismatches.append(f"team_max_hp {vc['team_max_hp']}!={max_thp}")
        if vc["enemy_max_hp"] != max_ehp:
            mismatches.append(f"enemy_max_hp {vc['enemy_max_hp']}!={max_ehp}")
        if abs(vc["boss_damage"] - float(boss)) >= 1e-6:
            mismatches.append(f"boss_damage {vc['boss_damage']}!={boss}")
        if stats_str != stat_profile:
            mismatches.append(f"stats {stats_str}!={stat_profile}")
        if mismatches:
            raise MissingValueMatrixError(
                f"Value matrix at {candidate} mismatches round config "
                f"({stat_profile}__{role_combo}, thp={max_thp}, ehp={max_ehp}, "
                f"boss={boss}): {', '.join(mismatches)}. "
                "Regenerate the matrix or fix the env config.")
        return candidate

    raise MissingValueMatrixError(
        f"No value matrix for stat_profile={stat_profile} role_combo={role_combo} "
        f"(thp={max_thp}, ehp={max_ehp}, boss={boss}). "
        f"Expected at {candidate}. Generate it before running this pipeline.")


def _load_env_config(value_dir: Path) -> dict:
    """Load {values, player_stats, boss_damage, *_max_hp} from a value dir."""
    values = np.load(value_dir / "values.npy")
    vc = _parse_value_config(value_dir / "config.py")
    return {
        "values": values,
        "player_stats": vc["player_stats"],
        "boss_damage": vc["boss_damage"],
        "team_max_hp": vc["team_max_hp"],
        "enemy_max_hp": vc["enemy_max_hp"],
    }


def load_human_team_records(verbose: bool = True):
    """Load clean human team-round records with value matrices attached.

    Scope: 04-23 export only. Games with any dropout player-round are
    dropped entirely. Within the remaining games, keep team-rounds that
    have three clean players. Every record must have a matching precomputed
    value matrix — if any are missing, this raises MissingValueMatrixError
    listing every gap, rather than silently dropping records (which biases
    the comparison toward whichever subset happens to have matrices).
    """
    all_prs = load_all_exports(data_dirs=[EXPORT_DIR])
    dropout_games = discover_dropout_games(all_prs)
    clean = filter_clean_prs(all_prs, dropout_games)

    human_teams = defaultdict(list)
    for pr in clean:
        if pr.round.round_type == "human":
            human_teams[(pr.game_id, pr.round.round_number)].append(pr)
    complete = {k: sorted(v, key=lambda p: p.player_id)
                for k, v in human_teams.items() if len(v) == 3}

    env_cache: dict[str, dict] = {}
    records = []
    missing: list[str] = []

    for (game_id, round_number), team_prs in complete.items():
        rnd = team_prs[0].round
        cfg = rnd.config
        env_id = str(cfg.get("envId", ""))

        try:
            value_dir = _find_value_dir(cfg)
        except MissingValueMatrixError as e:
            missing.append(f"  game={game_id} r{round_number} env={env_id}: {e}")
            continue
        if env_id not in env_cache:
            env_cache[env_id] = _load_env_config(value_dir)

        # Stage-by-stage human-team role combo (in-game position order).
        n_stages = max(len(pr.round.stages) for pr in team_prs)
        stage_roles: list[str] = []
        ok = True
        for s in range(n_stages):
            combo_chars = ["F", "F", "F"]
            for pr in team_prs:
                if s >= len(pr.round.stages):
                    ok = False
                    break
                combo_chars[pr.player_id] = ROLE_SHORT[pr.round.stages[s].role_idx]
            if not ok:
                break
            stage_roles.append("".join(combo_chars))
        if not stage_roles:
            continue

        lds = [int(c) for c in rnd.enemy_intent_sequence]
        stat_profile = rnd.stat_profile_id
        role_combo = "".join(ROLE_SHORT[int(r)] for r in cfg.get("optimalRoles") or [])

        records.append({
            "game_id": game_id,
            "round_id": f"{game_id}_r{round_number}",
            "round_type": "human",
            "round_number": int(round_number),
            "env_id": env_id,
            "stat_profile": stat_profile,
            "optimal_roles": role_combo,
            "lds": lds,
            "stage_roles": stage_roles,
            "env_config": env_cache[env_id],
        })

    if missing:
        raise MissingValueMatrixError(
            f"{len(missing)} of {len(complete)} clean team-rounds have no "
            f"matching value matrix. Generate the missing matrices in "
            f"{VALUE_MATRICES_DIR} (naming: <stat_profile>__<role_combo>) "
            f"before running this pipeline. Misses:\n" + "\n".join(missing))

    if verbose:
        n_games = len({r["game_id"] for r in records})
        print(f"[pipeline] 04-23 export: {len(all_prs)} player-rounds total, "
              f"{len(dropout_games)} games excluded as dropout")
        print(f"[pipeline] Stage 2: {len(records)} team-rounds across "
              f"{n_games} games (from {len(complete)} complete clean teams)")
    return records


def load_human_inference_records(verbose: bool = True):
    """Stage-1 preparation: clean human teams with 3 players.

    Returns a list of per-team dicts with the fields evaluate() needs. No bot
    rounds are loaded — everything in this experiment is human-only.
    """
    all_prs = load_all_exports(data_dirs=[EXPORT_DIR])
    dropout_games = discover_dropout_games(all_prs)
    clean = filter_clean_prs(all_prs, dropout_games)

    human_teams = defaultdict(list)
    for pr in clean:
        if pr.round.round_type == "human":
            human_teams[(pr.game_id, pr.round.round_number)].append(pr)
    human_teams = {k: sorted(v, key=lambda p: p.player_id)
                   for k, v in human_teams.items() if len(v) == 3}

    prepared = [_prepare_human_team(t) for t in human_teams.values()]
    if verbose:
        n_queries = sum(len(d["queries"]) for d in prepared)
        print(f"[pipeline] Stage 1: {len(prepared)} clean human teams, "
              f"{n_queries} inference queries")
    return prepared


def _prepare_human_team(team_prs):
    rnd = team_prs[0].round
    config = rnd.config
    parts = rnd.stat_profile_id.split("_")
    player_stats = np.array([[int(c) for c in p] for p in parts], dtype=float)
    boss_damage = config.get("bossDamage", 2)
    team_max_hp = config.get("maxTeamHealth", 15)
    enemy_max_hp = config.get("maxEnemyHealth", 30)
    eis = rnd.enemy_intent_sequence

    player_roles = {pr.player_id: [s.role_idx for s in pr.round.stages]
                    for pr in team_prs}
    n_stages = max(len(v) for v in player_roles.values())
    role_seq = []
    for s in range(n_stages):
        roles = [0, 0, 0]
        for pid, rs in player_roles.items():
            if s < len(rs):
                roles[pid] = rs[s]
        role_seq.append(roles)

    queries = []
    for pr in team_prs:
        for si, stage in enumerate(pr.round.stages):
            if si == 0 or not stage.inferred_roles:
                continue
            for target_pos, inferred_role in stage.inferred_roles.items():
                if (target_pos not in player_roles
                        or si - 1 >= len(player_roles[target_pos])):
                    continue
                true_role = player_roles[target_pos][si - 1]
                queries.append((si, target_pos, inferred_role, true_role))

    return {
        "player_stats": player_stats,
        "boss_damage": boss_damage,
        "team_max_hp": team_max_hp,
        "enemy_max_hp": enemy_max_hp,
        "eis": eis,
        "role_seq": role_seq,
        "queries": queries,
    }


# ──────────────────────────────────────────────────────────────────────
# Stage 1: posterior + inference likelihood
# ──────────────────────────────────────────────────────────────────────

def compute_posteriors(data, tau_prior, epsilon, strategy: MemoryStrategy):
    """posteriors[s] is the belief at the START of stage s (inference time)."""
    player_stats = data["player_stats"]
    boss_damage = data["boss_damage"]
    team_max_hp = data["team_max_hp"]
    enemy_max_hp = data["enemy_max_hp"]
    eis = data["eis"]
    role_seq = data["role_seq"]
    n_stages = len(role_seq)

    original_prior = utility_based_prior(player_stats, tau=tau_prior)
    posteriors = [original_prior.copy()]

    if strategy.kind == "window":
        w = int(strategy.param)
        team_hp_seq = [float(team_max_hp)]
        hp = float(team_max_hp)
        ehp = float(enemy_max_hp)
        for s in range(n_stages):
            roles = role_seq[s]
            for off in range(TURNS_PER_STAGE):
                ti = s * TURNS_PER_STAGE + off
                if ti >= len(eis) or hp <= 0:
                    break
                intent = int(eis[ti])
                actions = [preferred_action(roles[i], intent, hp, team_max_hp)
                           for i in range(3)]
                hp, ehp = game_step(intent, hp, ehp, actions,
                                    player_stats, boss_damage, team_max_hp)
            team_hp_seq.append(hp)

        for s in range(1, n_stages + 1):
            start = max(0, s - w)
            post = original_prior.copy()
            hp = team_hp_seq[start]
            for ws in range(start, s):
                roles = role_seq[ws]
                for off in range(TURNS_PER_STAGE):
                    ti = ws * TURNS_PER_STAGE + off
                    if ti >= len(eis) or hp <= 0:
                        break
                    intent = int(eis[ti])
                    actions = [preferred_action(roles[i], intent, hp, team_max_hp)
                               for i in range(3)]
                    post = bayesian_update(post, actions, intent,
                                           hp, team_max_hp, epsilon)
                    hp, _ = game_step(intent, hp, float(enemy_max_hp),
                                      actions, player_stats,
                                      boss_damage, team_max_hp)
            posteriors.append(post)
        return posteriors

    current = original_prior.copy()
    hp = float(team_max_hp)
    ehp = float(enemy_max_hp)
    for s in range(n_stages):
        roles = role_seq[s]
        for off in range(TURNS_PER_STAGE):
            ti = s * TURNS_PER_STAGE + off
            if ti >= len(eis) or hp <= 0:
                break
            intent = int(eis[ti])
            actions = [preferred_action(roles[i], intent, hp, team_max_hp)
                       for i in range(3)]
            current = bayesian_update(current, actions, intent,
                                      hp, team_max_hp, epsilon)
            hp, ehp = game_step(intent, hp, ehp, actions,
                                player_stats, boss_damage, team_max_hp)
        current = apply_boundary(current, original_prior,
                                  strategy.kind, strategy.param)
        posteriors.append(current.copy())
    return posteriors


def stage1_evaluate(prepared, tau_prior, epsilon, strategy: MemoryStrategy):
    correct = 0
    total = 0
    ll = []
    floor_hits = 0
    floor_threshold = np.log(1e-10)
    for data in prepared:
        posteriors = compute_posteriors(data, tau_prior, epsilon, strategy)
        for si, target_pos, inferred_role, true_role in data["queries"]:
            if si >= len(posteriors):
                continue
            post = posteriors[si]
            marg = np.sum(post, axis=tuple(j for j in range(3) if j != target_pos))
            t = marg.sum()
            if t > 0:
                marg = marg / t
            pred = int(np.argmax(marg))
            log_lik = np.log(max(marg[inferred_role], 1e-20))
            total += 1
            if pred == true_role:
                correct += 1
            ll.append(log_lik)
            if log_lik <= floor_threshold:
                floor_hits += 1
    return {
        "inference_ll": float(np.mean(ll)) if ll else float("nan"),
        "accuracy": correct / total if total > 0 else 0.0,
        "n": total,
        "floor_hits": floor_hits,
    }


# ──────────────────────────────────────────────────────────────────────
# Stage 2: trajectory precompute (human-only path)
# ──────────────────────────────────────────────────────────────────────

def precompute_trajectories(records, tau_prior, epsilon,
                             memory_strategy: MemoryStrategy):
    """Precompute per-stage posteriors + game state for each human record."""
    kind = memory_strategy.kind
    param = memory_strategy.param
    trajectories = []
    for record in records:
        if record["round_type"] != "human":
            raise ValueError(
                "This pipeline is human-only — got a non-human record.")
        trajectories.append(_precompute_human(record, tau_prior, epsilon,
                                              kind, param))
    return trajectories


def _precompute_human(record, tau_prior, epsilon, kind, param):
    env = record["env_config"]
    player_stats = env["player_stats"]
    boss_damage = env["boss_damage"]
    team_max_hp = env["team_max_hp"]
    enemy_max_hp = env["enemy_max_hp"]

    original_prior = utility_based_prior(player_stats, tau=tau_prior)
    stage_roles_list = record["stage_roles"]
    lds = record["lds"]

    role_letter_to_roles = lambda combo: [ROLE_CHAR_TO_IDX[c] for c in combo]

    if kind == "window":
        return _precompute_window(
            stage_roles_list, lds, player_stats, boss_damage,
            team_max_hp, enemy_max_hp, original_prior, epsilon,
            int(param), role_letter_to_roles)

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

        current = apply_boundary(current, original_prior, kind, param)

    return stages


def _precompute_window(stage_roles_list, lds, player_stats, boss_damage,
                       team_max_hp, enemy_max_hp, original_prior, epsilon,
                       window, role_letter_to_roles):
    n_stages = len(stage_roles_list)

    team_hp_seq = [float(team_max_hp)]
    enemy_hp_seq = [float(enemy_max_hp)]
    hp = float(team_max_hp)
    ehp = float(enemy_max_hp)
    turn_idx = 0
    for s in range(n_stages):
        roles = role_letter_to_roles(stage_roles_list[s])
        for _ in range(TURNS_PER_STAGE):
            if turn_idx >= len(lds) or hp <= 0 or ehp <= 0:
                break
            intent_t = int(lds[turn_idx])
            actions = [preferred_action(roles[i], intent_t, hp, team_max_hp)
                       for i in range(3)]
            hp, ehp = game_step(intent_t, hp, ehp, actions,
                                player_stats, boss_damage, team_max_hp)
            turn_idx += 1
        team_hp_seq.append(hp)
        enemy_hp_seq.append(ehp)

    stages = []
    prev_roles = None
    turn_idx = 0
    for s in range(n_stages):
        if turn_idx >= len(lds) or team_hp_seq[s] <= 0 or enemy_hp_seq[s] <= 0:
            break

        start = max(0, s - window)
        post = original_prior.copy()
        replay_hp = team_hp_seq[start]
        replay_enemy = enemy_hp_seq[start]
        replay_turn = start * TURNS_PER_STAGE
        for ws in range(start, s):
            ws_roles = role_letter_to_roles(stage_roles_list[ws])
            for _ in range(TURNS_PER_STAGE):
                if replay_turn >= len(lds) or replay_hp <= 0 or replay_enemy <= 0:
                    break
                intent_t = int(lds[replay_turn])
                actions = [preferred_action(ws_roles[i], intent_t, replay_hp, team_max_hp)
                           for i in range(3)]
                post = bayesian_update(post, actions, intent_t,
                                       replay_hp, team_max_hp, epsilon)
                replay_hp, replay_enemy = game_step(
                    intent_t, replay_hp, replay_enemy, actions,
                    player_stats, boss_damage, team_max_hp)
                replay_turn += 1

        intent = int(lds[turn_idx]) if turn_idx < len(lds) else 0
        thp = int(min(max(0, team_hp_seq[s]), team_max_hp))
        ehp = int(min(max(0, enemy_hp_seq[s]), enemy_max_hp))

        stages.append({
            "prior": post,
            "intent": intent,
            "thp": thp,
            "ehp": ehp,
            "human_combo": stage_roles_list[s],
            "prev_roles": prev_roles,
        })
        prev_roles = list(role_letter_to_roles(stage_roles_list[s]))
        turn_idx += TURNS_PER_STAGE

    return stages


# ──────────────────────────────────────────────────────────────────────
# Posterior / joint helpers reused by the models
# ──────────────────────────────────────────────────────────────────────

def posterior_marginal(prior, agent_i):
    marg = np.sum(prior, axis=tuple(j for j in range(3) if j != agent_i))
    total = marg.sum()
    return marg / total if total > 0 else np.ones(3) / 3.0


def build_joint_dist(per_agent):
    predicted_dist = {}
    for r0 in range(3):
        for r1 in range(3):
            for r2 in range(3):
                combo = ROLE_SHORT[r0] + ROLE_SHORT[r1] + ROLE_SHORT[r2]
                predicted_dist[combo] = float(
                    per_agent[0][r0] * per_agent[1][r1] * per_agent[2][r2])
    return predicted_dist


# ──────────────────────────────────────────────────────────────────────
# Stage 2 metrics
# ──────────────────────────────────────────────────────────────────────

def filter_switch_stages(all_results):
    """Re-aggregate only the stages where the human's combo changed."""
    filtered = {}
    for env_id, data in all_results.items():
        canon_combos = data["canonical_combos"]
        stat_profile = data["stat_profile"]

        new_team_preds = []
        for team_preds in data["team_predictions"]:
            kept = []
            for s, pred in enumerate(team_preds):
                if s > 0 and pred["human_combo"] != team_preds[s - 1]["human_combo"]:
                    kept.append(pred)
            new_team_preds.append(kept)

        stage_predicted = defaultdict(lambda: defaultdict(float))
        stage_human = defaultdict(lambda: defaultdict(int))
        stage_model_marg = defaultdict(lambda: np.zeros(3))
        stage_human_marg = defaultdict(lambda: np.zeros(3))
        stage_counts = defaultdict(int)
        max_stages = 0

        for team_preds in new_team_preds:
            for s, pred in enumerate(team_preds):
                stage_counts[s] += 1
                max_stages = max(max_stages, s + 1)
                for combo, prob in pred["predicted_dist"].items():
                    stage_predicted[s][canonical_combo(combo, stat_profile)] += prob
                stage_human[s][canonical_combo(pred["human_combo"], stat_profile)] += 1
                stage_model_marg[s] += pred["model_marginal"]
                stage_human_marg[s] += combo_marginal(pred["human_combo"])

        if max_stages == 0:
            continue

        predicted_avg, model_marg_avg, human_marg_avg = {}, {}, {}
        for s in range(max_stages):
            n = stage_counts[s]
            if n > 0:
                predicted_avg[s] = {cc: stage_predicted[s].get(cc, 0.0) / n
                                    for cc in canon_combos}
                model_marg_avg[s] = stage_model_marg[s] / n
                human_marg_avg[s] = stage_human_marg[s] / n

        filtered[env_id] = dict(data)
        filtered[env_id].update({
            "max_stages": max_stages,
            "stage_predicted": predicted_avg,
            "stage_human": dict(stage_human),
            "stage_counts": dict(stage_counts),
            "team_predictions": new_team_preds,
            "stage_model_marginal": model_marg_avg,
            "stage_human_marginal": human_marg_avg,
        })
    return filtered


def compute_aggregate_cross_entropy(all_results):
    """Aggregate cross-entropy comparable to mean_ll."""
    FLOOR = 1e-12
    total_weight = 0.0
    total_ce = 0.0
    n_floor_hits = 0
    per_env = {}
    for env_id, data in all_results.items():
        canon_combos = data["canonical_combos"]
        env_ce = 0.0
        env_weight = 0.0
        for s in range(data["max_stages"]):
            predicted = data["stage_predicted"].get(s)
            human_counts = data["stage_human"].get(s, {})
            n = data["stage_counts"].get(s, 0)
            if predicted is None or n == 0:
                continue
            for cc in canon_combos:
                p_h = human_counts.get(cc, 0) / n
                if p_h <= 0:
                    continue
                p_m = predicted.get(cc, 0.0)
                if p_m < FLOOR:
                    n_floor_hits += 1
                env_ce += n * p_h * np.log(max(p_m, FLOOR))
            env_weight += n
        if env_weight > 0:
            per_env[env_id] = float(env_ce / env_weight)
            total_ce += env_ce
            total_weight += env_weight
    return {
        "global": float(total_ce / total_weight) if total_weight > 0 else float("nan"),
        "per_env": per_env,
        "n_floor_hits": int(n_floor_hits),
    }


def _metric_block(all_results):
    if not all_results:
        return {
            "combo_r": float("nan"), "marg_r": float("nan"),
            "mean_ll": float("nan"), "agg_ll": float("nan"),
            "n_floor_hits": 0,
        }
    corrs = compute_pearson(all_results)
    ll = compute_log_likelihood(all_results)
    g = corrs.get("__global__", {})
    mean_ll = (float(np.mean([v["mean_ll"] for v in ll.values()]))
               if ll else float("nan"))
    ce = compute_aggregate_cross_entropy(all_results)
    return {
        "combo_r": g.get("combo", {}).get("r", float("nan")),
        "marg_r": g.get("marginal", {}).get("r", float("nan")),
        "mean_ll": mean_ll,
        "agg_ll": ce["global"],
        "n_floor_hits": ce["n_floor_hits"],
    }


def eval_subset(records, predict_fn):
    if not records:
        return {
            "n_records": 0,
            **{k: float("nan") for k in
               ("combo_r", "marg_r", "mean_ll", "agg_ll")},
            "n_floor_hits": 0,
            **{k: float("nan") for k in
               ("switch_combo_r", "switch_marg_r", "switch_mean_ll", "switch_agg_ll")},
            "switch_n_floor_hits": 0,
        }
    full = run_predictions(records, predict_fn)
    agg = _metric_block(full)
    sw = _metric_block(filter_switch_stages(full))
    out = {"n_records": len(records)}
    out.update(agg)
    out["switch_combo_r"] = sw["combo_r"]
    out["switch_marg_r"] = sw["marg_r"]
    out["switch_mean_ll"] = sw["mean_ll"]
    out["switch_agg_ll"] = sw["agg_ll"]
    out["switch_n_floor_hits"] = sw["n_floor_hits"]
    return out


def compute_pooled_metric(records, trajectories, predict_fn_factory,
                          metric="agg_ll"):
    predict_fn = predict_fn_factory(records, trajectories)
    if not records:
        return float("nan")
    full = run_predictions(records, predict_fn)
    if metric == "agg_ll":
        return compute_aggregate_cross_entropy(full)["global"]
    if metric == "mean_ll":
        ll = compute_log_likelihood(full)
        return float(np.mean([v["mean_ll"] for v in ll.values()])) if ll else float("nan")
    if metric == "combo_r":
        g = compute_pearson(full).get("__global__", {})
        return g.get("combo", {}).get("r", float("nan"))
    raise ValueError(f"unknown metric: {metric!r}")


# ──────────────────────────────────────────────────────────────────────
# Checkpoint & JSON helpers
# ──────────────────────────────────────────────────────────────────────

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_checkpoint(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def save_checkpoint(path, results):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(results, f, indent=2, default=_json_default)
        os.rename(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def get_completed_keys(results, key_fields):
    return {tuple(r[k] for k in key_fields) for r in results}


def pick_best(results, metric="combo_r"):
    valid = [r for r in results if not np.isnan(r.get(metric, float("nan")))]
    if not valid:
        return results[0] if results else None
    return max(valid, key=lambda r: r[metric])
