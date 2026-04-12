"""Stage 1: tune (tau_prior, epsilon, memory_strategy) on inference data.

Uses the bot-corrected layout helper from shared.data_loading and the
expanded memory-strategy space (full / window / drift_prior / drift_uniform
/ temper).

Objective: per-query mean log-likelihood of human inferred roles under the
marginal of the Bayesian posterior at inference time. Compared to the 04-09
pipeline this version:

- Fixes the three bot-round bugs documented in CLAUDE.md (uses pr.player_id,
  reads botPlayers[i].strategy.role as int, permutes player_stats from
  logical to position order).
- Adds temper_gamma and drift_uniform_delta branches.
- Aborts on startup if the bot-round query count is not ~578, which is the
  CLAUDE.md ground truth under the fix.
- Emits clip_floor_diagnostics.json with per-query LL histograms, floor-hit
  counts, and "rerun under fix" numbers for the 04-09 and 04-11 winners.
- Reports human-only / bot-only / combined LLs separately.
"""

from __future__ import annotations

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from shared_utils import load_checkpoint, save_checkpoint, get_completed_keys, pick_best
from memory_strategies import MemoryStrategy, build_strategy_grid, apply_boundary

from shared.constants import TURNS_PER_STAGE
from shared.data_loading import load_all_exports, build_bot_round_layout
from shared.inference import (
    utility_based_prior, bayesian_update, preferred_action, game_step,
)

CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
COARSE_PATH = CHECKPOINT_DIR / "coarse_results.json"
REFINED_PATH = CHECKPOINT_DIR / "refined_results.json"
POLISHED_PATH = CHECKPOINT_DIR / "polished_results.json"
OUTPUT_PATH = SCRIPT_DIR / "best_inference_params.json"
DIAG_PATH = SCRIPT_DIR / "clip_floor_diagnostics.json"


BOT_QUERY_EXPECTED = 578  # CLAUDE.md ground truth after the three-bug fix


# ──────────────────────────────────────────────────────────────────────
# Data preparation (per-record precompute of stats, eis, queries)
# ──────────────────────────────────────────────────────────────────────

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
        "origin": "human",
        "player_stats": player_stats,
        "boss_damage": boss_damage,
        "team_max_hp": team_max_hp,
        "enemy_max_hp": enemy_max_hp,
        "eis": eis,
        "role_seq": role_seq,
        "queries": queries,
    }


def _prepare_bot_record(pr):
    """Bot-round preparation using the shared layout helper.

    CLAUDE.md: never use config.humanRole; never look up botPlayers.strategy.role
    through GAME_ROLE_TO_IDX; permute player_stats to in-game-position order.
    """
    layout = build_bot_round_layout(pr)
    rnd = pr.round
    config = rnd.config

    player_stats = layout.player_stats.astype(float)
    boss_damage = config.get("bossDamage", 2)
    team_max_hp = config.get("maxTeamHealth", 15)
    enemy_max_hp = config.get("maxEnemyHealth", 30)
    eis = rnd.enemy_intent_sequence
    n_stages = len(rnd.stages)

    # Full role sequences keyed by in-game position.
    all_roles = {layout.pid: [s.role_idx for s in rnd.stages]}
    for pos, ridx in layout.bot_role_map.items():
        all_roles[pos] = [ridx] * n_stages

    role_seq = []
    for s in range(n_stages):
        roles = [0, 0, 0]
        for pos, rs in all_roles.items():
            roles[pos] = rs[s]
        role_seq.append(roles)

    queries = []
    for si, stage in enumerate(rnd.stages):
        if si == 0 or not stage.inferred_roles:
            continue
        for target_pos, inferred_role in stage.inferred_roles.items():
            if (target_pos not in all_roles
                    or si - 1 >= len(all_roles[target_pos])):
                continue
            true_role = all_roles[target_pos][si - 1]
            queries.append((si, target_pos, inferred_role, true_role))

    return {
        "origin": "bot",
        "player_stats": player_stats,
        "boss_damage": boss_damage,
        "team_max_hp": team_max_hp,
        "enemy_max_hp": enemy_max_hp,
        "eis": eis,
        "role_seq": role_seq,
        "queries": queries,
    }


def load_prepared():
    records = load_all_exports()
    human_teams = defaultdict(list)
    bot_records = []
    for pr in records:
        if pr.is_dropout:
            continue
        if pr.round.round_type == "human":
            human_teams[(pr.game_id, pr.round.round_number)].append(pr)
        elif pr.round.round_type == "bot":
            bot_records.append(pr)
    human_teams = {k: sorted(v, key=lambda p: p.player_id)
                   for k, v in human_teams.items() if len(v) == 3}
    print(f"Loaded {len(human_teams)} human teams, "
          f"{len(bot_records)} bot player-rounds")

    prepared_human = [_prepare_human_team(t) for t in human_teams.values()]
    prepared_bot = [_prepare_bot_record(pr) for pr in bot_records]
    return prepared_human + prepared_bot


# ──────────────────────────────────────────────────────────────────────
# Posterior computation (one path for all strategies)
# ──────────────────────────────────────────────────────────────────────

def compute_posteriors(data, tau_prior, epsilon, strategy: MemoryStrategy):
    """Return posteriors indexed by stage: posteriors[s] is the belief state
    at the START of stage s (i.e. at inference time for stage s)."""
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
        # HP at stage boundaries
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


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate(prepared, tau_prior, epsilon, strategy: MemoryStrategy,
             collect_histogram: bool = False):
    correct_h = correct_b = 0
    total_h = total_b = 0
    ll_h, ll_b = [], []
    floor_hits_h = floor_hits_b = 0
    floor_threshold = np.log(1e-10)
    per_query_ll = [] if collect_histogram else None

    for data in prepared:
        posteriors = compute_posteriors(data, tau_prior, epsilon, strategy)
        is_bot = data["origin"] == "bot"
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

            if is_bot:
                total_b += 1
                if pred == true_role:
                    correct_b += 1
                ll_b.append(log_lik)
                if log_lik <= floor_threshold:
                    floor_hits_b += 1
            else:
                total_h += 1
                if pred == true_role:
                    correct_h += 1
                ll_h.append(log_lik)
                if log_lik <= floor_threshold:
                    floor_hits_h += 1
            if collect_histogram:
                per_query_ll.append(log_lik)

    combined = ll_h + ll_b
    out = {
        "inference_ll": float(np.mean(combined)) if combined else float("nan"),
        "human_inference_ll": float(np.mean(ll_h)) if ll_h else float("nan"),
        "bot_inference_ll": float(np.mean(ll_b)) if ll_b else float("nan"),
        "accuracy": (correct_h + correct_b) / (total_h + total_b)
            if (total_h + total_b) > 0 else 0.0,
        "human_accuracy": correct_h / total_h if total_h > 0 else 0.0,
        "bot_accuracy": correct_b / total_b if total_b > 0 else 0.0,
        "n": total_h + total_b,
        "n_human": total_h,
        "n_bot": total_b,
        "floor_hits": floor_hits_h + floor_hits_b,
        "floor_hits_human": floor_hits_h,
        "floor_hits_bot": floor_hits_b,
    }
    if collect_histogram:
        out["per_query_ll"] = per_query_ll
    return out


# ──────────────────────────────────────────────────────────────────────
# Search phases
# ──────────────────────────────────────────────────────────────────────

def run_coarse(prepared):
    strategies = build_strategy_grid()
    tau_vals = np.linspace(0.5, 15.0, 8)
    eps_vals = np.linspace(0.05, 0.9, 8)

    results = load_checkpoint(str(COARSE_PATH))
    done = get_completed_keys(results, ["strategy_name", "tau_prior", "epsilon"])
    total = len(strategies) * len(tau_vals) * len(eps_vals)
    print(f"Coarse: {len(strategies)} strategies x {len(tau_vals)} tau "
          f"x {len(eps_vals)} eps = {total} points (done {len(done)})")

    count = len(done)
    for si, strat in enumerate(strategies):
        added = False
        for tp in tau_vals:
            for eps in eps_vals:
                key = (strat.name, float(tp), float(eps))
                if key in done:
                    continue
                r = evaluate(prepared, float(tp), float(eps), strat)
                r.update({
                    "strategy_name": strat.name,
                    "kind": strat.kind,
                    "param": strat.param,
                    "tau_prior": float(tp),
                    "epsilon": float(eps),
                })
                results.append(r)
                added = True
                count += 1
        if added:
            save_checkpoint(str(COARSE_PATH), results)
            rs = [r for r in results if r["strategy_name"] == strat.name]
            best = max(rs, key=lambda x: x["inference_ll"])
            print(f"  [{si+1}/{len(strategies)}] {strat.name:<22} "
                  f"LL={best['inference_ll']:.4f} "
                  f"(H={best['human_inference_ll']:.3f}, "
                  f"B={best['bot_inference_ll']:.3f}) "
                  f"tp={best['tau_prior']:.2f} eps={best['epsilon']:.2f} "
                  f"[{count}/{total}]", flush=True)
    return results


def run_refined(prepared, coarse_results):
    best = pick_best(coarse_results, "inference_ll")
    print(f"\nCoarse best: strategy={best['strategy_name']} "
          f"tau={best['tau_prior']:.3f} eps={best['epsilon']:.3f} "
          f"LL={best['inference_ll']:.4f}")

    strat = MemoryStrategy(best["strategy_name"], best["kind"], best["param"])
    tau_step = (15.0 - 0.5) / 7
    eps_step = (0.9 - 0.05) / 7
    tau_vals = np.linspace(max(0.05, best["tau_prior"] - tau_step),
                            min(30.0, best["tau_prior"] + tau_step), 11)
    eps_vals = np.linspace(max(0.001, best["epsilon"] - eps_step),
                            min(0.999, best["epsilon"] + eps_step), 11)

    results = load_checkpoint(str(REFINED_PATH))
    done = get_completed_keys(results, ["strategy_name", "tau_prior", "epsilon"])
    total = len(tau_vals) * len(eps_vals)
    print(f"Refined: {len(tau_vals)} x {len(eps_vals)} = {total} points (done {len(done)})")

    count = len(done)
    for tp in tau_vals:
        added = False
        for eps in eps_vals:
            key = (strat.name, float(tp), float(eps))
            if key in done:
                continue
            r = evaluate(prepared, float(tp), float(eps), strat)
            r.update({
                "strategy_name": strat.name,
                "kind": strat.kind,
                "param": strat.param,
                "tau_prior": float(tp),
                "epsilon": float(eps),
            })
            results.append(r)
            added = True
            count += 1
        if added:
            save_checkpoint(str(REFINED_PATH), results)
            print(f"  [{count}/{total}] ...", flush=True)
    return results


def run_polish(prepared, all_results):
    best = pick_best(all_results, "inference_ll")
    strat = MemoryStrategy(best["strategy_name"], best["kind"], best["param"])
    print(f"\nPolish from: strategy={strat.name} "
          f"tau={best['tau_prior']:.4f} eps={best['epsilon']:.4f}")

    def obj(params):
        return -evaluate(prepared, params[0], params[1], strat)["inference_ll"]

    opt = minimize(obj, [best["tau_prior"], best["epsilon"]],
                   method="L-BFGS-B", bounds=[(0.05, 30.0), (0.001, 0.999)],
                   options={"maxiter": 50, "ftol": 1e-6})
    r = evaluate(prepared, float(opt.x[0]), float(opt.x[1]), strat)
    r.update({
        "strategy_name": strat.name, "kind": strat.kind, "param": strat.param,
        "tau_prior": float(opt.x[0]), "epsilon": float(opt.x[1]),
    })
    save_checkpoint(str(POLISHED_PATH), [r])
    return r


# ──────────────────────────────────────────────────────────────────────
# Diagnostics: clip-floor, "rerun under fix" numbers for 04-09 and 04-11
# ──────────────────────────────────────────────────────────────────────

def write_diagnostics(prepared, best_result):
    strat = MemoryStrategy(best_result["strategy_name"],
                           best_result["kind"], best_result["param"])
    winner_eval = evaluate(prepared, best_result["tau_prior"],
                           best_result["epsilon"], strat,
                           collect_histogram=True)
    per_q = winner_eval.pop("per_query_ll")
    bin_edges = [0.0, -0.5, -1.0, -2.0, -5.0, -10.0, -46.0, -float("inf")]
    hist = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        n = sum(1 for x in per_q if hi < x <= lo)
        hist.append({"hi": lo, "lo": hi, "count": n})

    diagnostics = {
        "winner": {
            "strategy_name": best_result["strategy_name"],
            "tau_prior": best_result["tau_prior"],
            "epsilon": best_result["epsilon"],
            **winner_eval,
            "per_query_ll_histogram": hist,
        },
    }

    # Rerun 04-09 and 04-11 winners under the bot-fix code (for comparison).
    rerun_specs = [
        {
            "label": "04-09 winner (window_1, tau=6.80, eps=0.56)",
            "strategy": MemoryStrategy("window_1", "window", 1),
            "tau_prior": 6.80,
            "epsilon": 0.56,
        },
        {
            "label": "04-11 winner (temper_0.410, tau=1.84, eps=0.42)",
            "strategy": MemoryStrategy("temper_0.410", "temper", 0.41),
            "tau_prior": 1.84,
            "epsilon": 0.42,
        },
    ]
    rerun_results = []
    for spec in rerun_specs:
        r = evaluate(prepared, spec["tau_prior"], spec["epsilon"], spec["strategy"])
        rerun_results.append({
            "label": spec["label"],
            "strategy_name": spec["strategy"].name,
            "tau_prior": spec["tau_prior"],
            "epsilon": spec["epsilon"],
            **r,
        })
    diagnostics["prior_winners_under_fix"] = rerun_results

    with open(DIAG_PATH, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\nDiagnostics written to {DIAG_PATH}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("Stage 1: inference parameter tuning (2026-04-12)")
    print("=" * 66)

    prepared = load_prepared()
    n_queries = sum(len(d["queries"]) for d in prepared)
    n_bot_queries = sum(len(d["queries"]) for d in prepared if d["origin"] == "bot")
    n_human_queries = n_queries - n_bot_queries
    print(f"Prepared {len(prepared)} records, {n_queries} inference queries "
          f"({n_human_queries} human, {n_bot_queries} bot)")

    # BLOCKING assertion: CLAUDE.md expects ~578 bot-round queries under fix.
    # Reject if we're in the "~210" (humanRole bug) or "~0" (GAME_ROLE_TO_IDX
    # bug) regimes.
    assert n_bot_queries == BOT_QUERY_EXPECTED, (
        f"Bot-round query count is {n_bot_queries}, expected "
        f"{BOT_QUERY_EXPECTED} (CLAUDE.md ground truth under the three-bug "
        "fix). Abort before any tuning."
    )
    print(f"  [OK] bot-round query count matches CLAUDE.md ({BOT_QUERY_EXPECTED})")

    # Top-5 strategies report helper
    def top5(results):
        by_strat = defaultdict(list)
        for r in results:
            by_strat[r["strategy_name"]].append(r)
        best_per = [max(rs, key=lambda x: x["inference_ll"]) for rs in by_strat.values()]
        best_per.sort(key=lambda r: -r["inference_ll"])
        return best_per[:5]

    print("\n--- Phase 1: Coarse grid ---")
    coarse_results = run_coarse(prepared)
    print("\n--- Phase 2: Refined grid ---")
    refined_results = run_refined(prepared, coarse_results)

    all_results = coarse_results + refined_results
    print("\n--- Phase 3: L-BFGS-B polish ---")
    polished = run_polish(prepared, all_results)
    all_results.append(polished)

    best = pick_best(all_results, "inference_ll")
    print("\nTop-5 strategies:")
    for r in top5(all_results):
        print(f"  {r['strategy_name']:<24} LL={r['inference_ll']:.4f} "
              f"(H={r['human_inference_ll']:.3f}, B={r['bot_inference_ll']:.3f}) "
              f"tau={r['tau_prior']:.3f} eps={r['epsilon']:.3f}")

    output = {
        "tau_prior": best["tau_prior"],
        "epsilon": best["epsilon"],
        "memory_strategy": best["strategy_name"],
        "kind": best["kind"],
        "param": best["param"],
        # 04-09-compatible fields (for backwards compat in downstream code):
        "window": best["param"] if best["kind"] == "window" else None,
        "drift_delta": best["param"] if best["kind"] == "drift_prior" else 0.0,
        "accuracy": best["accuracy"],
        "inference_ll": best["inference_ll"],
        "human_inference_ll": best["human_inference_ll"],
        "bot_inference_ll": best["bot_inference_ll"],
        "n": best["n"],
        "n_human": best["n_human"],
        "n_bot": best["n_bot"],
        "floor_hits": best["floor_hits"],
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nBest: {best['strategy_name']} tau={best['tau_prior']:.4f} "
          f"eps={best['epsilon']:.4f} LL={best['inference_ll']:.4f}")
    print(f"  human LL={best['human_inference_ll']:.4f} "
          f"bot LL={best['bot_inference_ll']:.4f}")
    print(f"Saved to {OUTPUT_PATH}")

    write_diagnostics(prepared, best)


if __name__ == "__main__":
    main()
