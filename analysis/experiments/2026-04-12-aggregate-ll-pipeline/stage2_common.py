"""Shared Stage-2 utilities for 2026-04-12-aggregate-ll-pipeline.

Differs from the 04-09 copy in three ways:
    1. Pooled fit on human + bot rounds, with the bot-round branch of
       ``precompute_trajectories`` built on ``build_bot_round_layout`` so the
       position ordering is correct (CLAUDE.md "Bot Round Ground Truth").
    2. Supports the expanded memory-strategy space: full / window /
       drift_prior / drift_uniform / temper.
    3. Adds the aggregate cross-entropy metric and a disaggregated
       (human/bot/pooled) evaluation path.
"""

from __future__ import annotations

import json
import re as _re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from shared import DATA_ROOT, EXPORTS_DIR
from shared.constants import (
    F, T, M, ROLE_SHORT, ROLE_CHAR_TO_IDX, GAME_ROLE_TO_IDX,
    TURNS_PER_STAGE,
)
from shared.data_loading import build_bot_round_layout
from shared.parsing import canonical_combo, get_canonical_combos
from shared.inference import (
    utility_based_prior, bayesian_update, preferred_action, game_step,
    combo_marginal,
)
from shared.evaluation import run_predictions, compute_pearson, compute_log_likelihood

# Monkey-patch oms for data loading (mirrors the 04-09 setup).
OMS_DIR = Path(DATA_ROOT).parent.parent / "computational_model" / "analysis"
sys.path.insert(0, str(OMS_DIR))
import online_model_sim as oms  # noqa: E402

oms.VALUE_MATRICES_DIR = DATA_ROOT / "human_envs_value_matrices"
oms.ENVS_DIR = DATA_ROOT / "envs"


def _load_config_no_jax(config_path):
    text = Path(config_path).read_text()
    team_max_hp = int(_re.search(r"TEAM_MAX_HP\s*=\s*(\d+)", text).group(1))
    enemy_max_hp = int(_re.search(r"ENEMY_MAX_HP\s*=\s*(\d+)", text).group(1))
    boss_damage = float(_re.search(r"BOSS_DAMAGE\s*=\s*([\d.]+)", text).group(1))
    ps_match = _re.search(
        r"PLAYER_STATS\s*=\s*(?:jnp\.array|np\.array)?\(?\s*(\[\[.+?\]\])\s*\)?",
        text, _re.DOTALL)
    rows = _re.findall(r"\[([^\[\]]+)\]", ps_match.group(1))
    player_stats = np.array([[float(x) for x in row.split(",")] for row in rows])

    class C:
        pass

    cfg = C()
    cfg.TEAM_MAX_HP = team_max_hp
    cfg.ENEMY_MAX_HP = enemy_max_hp
    cfg.BOSS_DAMAGE = boss_damage
    cfg.PLAYER_STATS = player_stats
    return cfg


oms.load_config_module = _load_config_no_jax


# All three Empirica exports with full gameSummary data.
DATA_DIRS = [
    str(EXPORTS_DIR / "bayesian-role-specialization-2026-02-13-10-37-44"),
    str(EXPORTS_DIR / "bayesian-role-specialization-2026-03-06-09-54-19"),
    str(EXPORTS_DIR / "bayesian-role-specialization-2026-03-18-15-47-09"),
]


# ──────────────────────────────────────────────────────────────────────
# Stage 1 params + pooled record loading
# ──────────────────────────────────────────────────────────────────────

def load_stage1_params(experiment_dir):
    """Load best inference params from the experiment's stage1 fit."""
    path = Path(experiment_dir) / "stage1_inference" / "best_inference_params.json"
    with open(path) as f:
        return json.load(f)


def load_records(include_bot_rounds: bool = True):
    """Load pooled (human + bot) team-round records via the patched oms."""
    records = oms.load_team_rounds(
        data_dirs=DATA_DIRS, include_bot_rounds=include_bot_rounds)
    n_envs = len(set(r["env_id"] for r in records))
    n_games = len(set(r["game_id"] for r in records))
    n_human = sum(1 for r in records if r["round_type"] == "human")
    n_bot = sum(1 for r in records if r["round_type"] == "bot")
    print(
        f"Loaded {len(records)} team-rounds "
        f"({n_human} human, {n_bot} bot) "
        f"across {n_envs} envs / {n_games} games"
    )
    return records


# ──────────────────────────────────────────────────────────────────────
# Memory-strategy dispatch
# ──────────────────────────────────────────────────────────────────────

def _apply_boundary(current: np.ndarray, original_prior: np.ndarray,
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
        out /= total
    return out


# ──────────────────────────────────────────────────────────────────────
# Trajectory precompute (dispatches on round_type and memory strategy)
# ──────────────────────────────────────────────────────────────────────

def precompute_trajectories(records, tau_prior, epsilon, memory_strategy):
    """Precompute per-stage posteriors + game state for each record.

    ``memory_strategy`` is a ``memory_strategies.MemoryStrategy`` dataclass
    (or any object with ``.kind``/``.param`` attributes).

    Returns a list of trajectories parallel to ``records``. Each trajectory
    is a list of stage dicts with ``prior``, ``intent``, ``thp``, ``ehp``,
    ``human_combo``, ``prev_roles``.
    """
    kind = memory_strategy.kind
    param = memory_strategy.param

    trajectories = []
    for record in records:
        if record["round_type"] == "bot":
            stages = _precompute_bot(record, tau_prior, epsilon, kind, param)
        else:
            stages = _precompute_human(record, tau_prior, epsilon, kind, param)
        trajectories.append(stages)
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

    return _precompute_common(
        stage_roles_list, lds, player_stats, boss_damage,
        team_max_hp, enemy_max_hp, original_prior, epsilon, kind, param,
        role_letter_to_roles=lambda combo: [ROLE_CHAR_TO_IDX[c] for c in combo],
    )


def _precompute_bot(record, tau_prior, epsilon, kind, param):
    """Bot-round trajectory precompute in **logical order** (human first).

    Stage 2 must use logical order for bot rounds because:
        (a) ``values.npy`` is indexed by joint role combos ``(r0, r1, r2)``
            where role ``r0`` goes with the PLAYER_STATS row 0 from the env's
            ``config.py`` — which is logical-order (human first for bot
            envs), not in-game-position order.
        (b) ``canonical_combo`` with an asymmetric bot-round stat profile
            like ``114_222_222`` (sym=``last_two``) treats the FIRST letter
            as the anchor and canonicalises only the other two. The anchor
            letter must be the human's role, which is logical index 0.

    Using ``build_bot_round_layout`` would be correct for inference ground
    truth (Stage 1) where ``stage.inferred_roles`` keys are in-game
    positions, but wrong for Stage 2 value-matrix consumers.
    """
    pr = record["player_round"]
    rnd = pr.round
    env = record["env_config"]
    boss_damage = env["boss_damage"]
    team_max_hp = env["team_max_hp"]
    enemy_max_hp = env["enemy_max_hp"]

    parts = rnd.stat_profile_id.split("_")
    player_stats = np.array([[int(c) for c in p] for p in parts], dtype=float)
    original_prior = utility_based_prior(player_stats, tau=tau_prior)

    bot_players = rnd.config.get("botPlayers") or []
    if len(bot_players) != 2:
        raise ValueError(
            f"bot round has {len(bot_players)} botPlayers entries, expected 2"
        )
    bot_roles = [int(bp["strategy"]["role"]) for bp in bot_players]

    full_combos = []
    for stage in rnd.stages:
        roles = [int(stage.role_idx), bot_roles[0], bot_roles[1]]
        full_combos.append("".join(ROLE_SHORT[r] for r in roles))

    return _precompute_common(
        full_combos, record["lds"], player_stats, boss_damage,
        team_max_hp, enemy_max_hp, original_prior, epsilon, kind, param,
        role_letter_to_roles=lambda combo: [ROLE_CHAR_TO_IDX[c] for c in combo],
    )


def _precompute_common(stage_roles_list, lds, player_stats, boss_damage,
                       team_max_hp, enemy_max_hp, original_prior, epsilon,
                       kind, param, role_letter_to_roles):
    """Shared posterior+HP trajectory under an arbitrary memory strategy.

    ``stage_roles_list`` holds either a full 3-char combo (human rounds) or a
    single-character human role (bot rounds, with the other two positions
    implicit). ``role_letter_to_roles`` maps the stage entry to the full
    length-3 position-ordered role list used for action simulation.
    """
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

        current = _apply_boundary(current, original_prior, kind, param)

    return stages


def _precompute_window(stage_roles_list, lds, player_stats, boss_damage,
                       team_max_hp, enemy_max_hp, original_prior, epsilon,
                       window, role_letter_to_roles):
    """Windowed memory strategy: replay the last ``window`` stages from prior."""
    n_stages = len(stage_roles_list)

    # First pass: HP trajectory at stage boundaries.
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

    # Second pass: windowed posterior per stage.
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
# Metrics: aggregate cross-entropy, switch-stage filtering, disaggregation
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
    """Aggregate cross-entropy: sum over (env, stage, combo) of p_h * log p_m.

    At each (env, stage), ``p_h`` is the empirical canonical-combo
    distribution over teams and ``p_m`` is the mean predicted canonical-combo
    distribution over teams. Stages are weighted by their team count, so the
    result is a per-sample average comparable to ``mean_ll``.

    Returns ``{"global": float, "per_env": dict, "n_floor_hits": int}``.
    """
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
    """Compute the full {combo_r, marg_r, mean_ll, agg_ll, ...} block on one
    result set (either full or switch-filtered)."""
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
    """Evaluate ``predict_fn`` on one record subset.

    Returns a dict with ``n_records`` plus the aggregate metric block and a
    ``switch_*`` metric block. Empty subsets return NaNs.
    """
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


def _filter_records_and_trajectories(records, trajectories, predicate):
    filt_records = []
    filt_trajs = []
    for rec, traj in zip(records, trajectories):
        if predicate(rec):
            filt_records.append(rec)
            filt_trajs.append(traj)
    return filt_records, filt_trajs


def compute_disaggregated_metrics(records, trajectories, predict_fn_factory):
    """Compute metric blocks for human-only, bot-only, and pooled subsets.

    ``predict_fn_factory`` is a callable ``(records_subset, traj_subset) ->
    predict_fn``; the factory is called once per subset so the predict_fn
    closes over the right trajectory indexing.
    """
    subsets = {}
    for name, predicate in (
        ("human", lambda r: r["round_type"] == "human"),
        ("bot", lambda r: r["round_type"] == "bot"),
        ("pooled", lambda r: True),
    ):
        recs, trajs = _filter_records_and_trajectories(records, trajectories, predicate)
        predict_fn = predict_fn_factory(recs, trajs)
        subsets[name] = eval_subset(recs, predict_fn)
    return subsets


def compute_pooled_metric(records, trajectories, predict_fn_factory, metric="agg_ll"):
    """Fast-path: evaluate ``metric`` on the pooled subset only, for fitting."""
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
# Posterior and distribution helpers (copied from 04-09)
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
