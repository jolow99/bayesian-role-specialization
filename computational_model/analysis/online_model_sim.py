"""
Online (teacher-forced) model-based analysis.

For each human team's round, feeds the model the actual human role choices,
lets it update beliefs, and predicts what it would play at each stage.
Compares predicted distributions with empirical human data via Pearson r.
"""

import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# === Paths ===
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "bayesian-role-specialization-2026-03-06-09-54-19"
VALUE_MATRICES_DIR = PROJECT_DIR / "human_envs_value_matrices"
HUMAN_DIST_FILE = SCRIPT_DIR / "stage_distributions.json"
OUTPUT_DIR = SCRIPT_DIR / "figures" / "model_comparison_online"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Constants ===
F, T, M = 0, 1, 2
ATTACK, DEFEND, HEAL = 0, 1, 2
ROLE_NAMES = {0: "F", 1: "T", 2: "M"}
ROLE_CHAR_TO_IDX = {"F": 0, "T": 1, "M": 2}
GAME_ROLE_TO_IDX = {"FIGHTER": 0, "TANK": 1, "MEDIC": 2}
EPSILON = 1e-10
MAX_STAGES = 5
TURNS_PER_STAGE = 2
TAU = 1.0

DROPOUT_GAME_ID = "01KK14SSY8E64SK69715NN1TMW"

ALL_ROLE_COMBOS = [
    ROLE_NAMES[r0] + ROLE_NAMES[r1] + ROLE_NAMES[r2]
    for r0 in range(3) for r1 in range(3) for r2 in range(3)
]

# === Canonical combo handling ===
SYMMETRIC_PROFILES = {
    "222_222_222": "all",
    "411_222_222": "last_two",
    "114_222_222": "last_two",
    "141_222_222": "last_two",
}


def canonical_combo(combo, stat_profile):
    sym = SYMMETRIC_PROFILES.get(stat_profile)
    if sym == "all":
        return "".join(sorted(combo))
    elif sym == "last_two":
        return combo[0] + "".join(sorted(combo[1:]))
    return combo


def get_canonical_combos(stat_profile):
    seen = set()
    canonical = []
    for c in ALL_ROLE_COMBOS:
        cc = canonical_combo(c, stat_profile)
        if cc not in seen:
            seen.add(cc)
            canonical.append(cc)
    return canonical


# === Utility-based prior ===

# Role-to-stat mapping: Fighter uses STR (col 0), Tank uses DEF (col 1), Medic uses SUP (col 2)
ROLE_STAT_COL = {0: 0, 1: 1, 2: 2}


def utility_based_prior(player_stats, tau=1.0):
    """
    P(r0, r1, r2) ∝ exp(1/τ * Σ_i u_i(r_i))
    where u_i(r) = player i's stat for role r.
    Factorizes into independent softmax per player.
    """
    prior = np.zeros((3, 3, 3))
    for r0 in range(3):
        for r1 in range(3):
            for r2 in range(3):
                utility = (
                    float(player_stats[0, ROLE_STAT_COL[r0]])
                    + float(player_stats[1, ROLE_STAT_COL[r1]])
                    + float(player_stats[2, ROLE_STAT_COL[r2]])
                )
                prior[r0, r1, r2] = utility / tau

    # Numerical stability: subtract max before exp
    prior -= prior.max()
    prior = np.exp(prior)
    prior /= prior.sum()
    return prior


# === Policy functions ===


def action_prob(role, action, intent, team_hp, team_max_hp):
    if role == F:
        preferred = ATTACK
    elif role == T:
        preferred = DEFEND if intent == 1 else ATTACK
    else:
        preferred = HEAL if team_hp < team_max_hp else ATTACK

    if action == preferred:
        return 1.0 - EPSILON
    else:
        return EPSILON / 2.0


def get_action(role, intent, team_hp, team_max_hp):
    if role == F:
        return ATTACK
    elif role == T:
        return DEFEND if intent == 1 else ATTACK
    else:
        return HEAL if team_hp < team_max_hp else ATTACK


# === Bayesian inference ===


def bayesian_update(prior, actions, intent, team_hp, team_max_hp):
    posterior = np.copy(prior)
    for r0 in range(3):
        for r1 in range(3):
            for r2 in range(3):
                likelihood = (
                    action_prob(r0, actions[0], intent, team_hp, team_max_hp)
                    * action_prob(r1, actions[1], intent, team_hp, team_max_hp)
                    * action_prob(r2, actions[2], intent, team_hp, team_max_hp)
                )
                posterior[r0, r1, r2] *= likelihood

    total = posterior.sum()
    if total > 0:
        posterior /= total
    else:
        posterior = np.ones((3, 3, 3)) / 27.0
    return posterior


# === Softmax role selection ===


def softmax_role_dist(agent_i, intent, team_hp, enemy_hp, prior, values, tau=1.0):
    other_agents = [a for a in range(3) if a != agent_i]

    other_probs = np.sum(prior, axis=agent_i)
    total = other_probs.sum()
    if total > 0:
        other_probs /= total
    else:
        other_probs = np.ones((3, 3)) / 9.0

    expected_values = np.zeros(3)

    for r_i in range(3):
        ev = 0.0
        for r_j in range(3):
            for r_k in range(3):
                roles = [0, 0, 0]
                roles[agent_i] = r_i
                roles[other_agents[0]] = r_j
                roles[other_agents[1]] = r_k

                flat_idx = roles[0] * 9 + roles[1] * 3 + roles[2]
                val = float(values[flat_idx, intent, team_hp, enemy_hp])
                ev += other_probs[r_j, r_k] * val

        expected_values[r_i] = ev

    ev_scaled = expected_values / tau
    ev_scaled -= ev_scaled.max()
    exp_ev = np.exp(ev_scaled)
    return exp_ev / exp_ev.sum()


def model_predicted_combo_dist(intent, team_hp, enemy_hp, prior, values, tau=1.0):
    """
    Compute the joint distribution over all 27 role combos predicted by the model.
    Assumes agents independently sample from their softmax distributions.
    Returns dict: combo_str -> probability.
    """
    per_agent = []
    for i in range(3):
        dist = softmax_role_dist(i, intent, team_hp, enemy_hp, prior, values, tau)
        per_agent.append(dist)

    combo_probs = {}
    for r0 in range(3):
        for r1 in range(3):
            for r2 in range(3):
                combo = ROLE_NAMES[r0] + ROLE_NAMES[r1] + ROLE_NAMES[r2]
                prob = per_agent[0][r0] * per_agent[1][r1] * per_agent[2][r2]
                combo_probs[combo] = float(prob)

    return combo_probs


# === Game step ===


def game_step(intent, team_hp, enemy_hp, actions, player_stats, boss_damage, team_max_hp):
    total_attack = sum(
        float(player_stats[i, 0]) for i in range(3) if actions[i] == ATTACK
    )
    defenders = [
        float(player_stats[i, 1]) for i in range(3) if actions[i] == DEFEND
    ]
    max_defense = max(defenders) if defenders else 0.0
    total_heal = sum(
        float(player_stats[i, 2]) for i in range(3) if actions[i] == HEAL
    )

    new_enemy_hp = max(0.0, enemy_hp - total_attack)
    damage = max(0.0, boss_damage - max_defense) if intent == 1 else 0.0
    new_team_hp = team_hp - damage + total_heal
    new_team_hp = max(0.0, min(float(team_max_hp), new_team_hp))

    return new_team_hp, new_enemy_hp


# === Load config modules ===


def load_config_module(config_path):
    spec = importlib.util.spec_from_file_location(
        f"config_{hash(str(config_path))}", config_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# === Load per-team round data ===


def load_team_rounds():
    """
    Load per-team, per-round data: the actual human role choices at each stage,
    plus the env config and LDS.
    """
    with open(DATA_DIR / "game.csv") as f:
        games = {r["id"]: r for r in csv.DictReader(f)}

    with open(DATA_DIR / "round.csv") as f:
        rounds = list(csv.DictReader(f))

    # Cache env configs
    env_cache = {}
    records = []

    for r in rounds:
        game = games.get(r["gameID"])
        if not game or game.get("status") != "ended":
            continue
        if r["gameID"] == DROPOUT_GAME_ID:
            continue

        rnum = r["roundNumber"]
        cfg_key = f"round{rnum}Config"
        if not game.get(cfg_key):
            continue

        cfg = json.loads(game[cfg_key])
        if cfg.get("botPlayers"):
            continue

        env_id = cfg["envId"]
        optimal_roles = cfg["optimalRoles"]
        role_combo = "".join(ROLE_NAMES[ri] for ri in optimal_roles)
        stat_profile = cfg.get("statProfileId", "")
        lds_str = cfg["enemyIntentSequence"]
        lds = [int(c) for c in lds_str]

        # Load env config + values (cached)
        if env_id not in env_cache:
            val_dir = VALUE_MATRICES_DIR / role_combo
            values = np.load(val_dir / "values.npy")
            config_mod = load_config_module(val_dir / "config.py")
            env_cache[env_id] = {
                "values": values,
                "player_stats": np.array(config_mod.PLAYER_STATS, dtype=float),
                "boss_damage": float(config_mod.BOSS_DAMAGE),
                "team_max_hp": int(config_mod.TEAM_MAX_HP),
                "enemy_max_hp": int(config_mod.ENEMY_MAX_HP),
            }

        # Extract human role combos per stage
        stage_roles = []
        for s in range(1, MAX_STAGES + 1):
            turns_data = r.get(f"stage{s}Turns")
            if not turns_data:
                break
            turns = json.loads(turns_data)
            if not turns:
                break
            roles_str = turns[0]["roles"]
            combo_str = "".join(ROLE_NAMES[GAME_ROLE_TO_IDX[rs]] for rs in roles_str)
            stage_roles.append(combo_str)

        if not stage_roles:
            continue

        records.append({
            "game_id": r["gameID"],
            "round_id": r["id"],
            "env_id": env_id,
            "stat_profile": stat_profile,
            "optimal_roles": "".join(ROLE_NAMES[ri] for ri in optimal_roles),
            "lds": lds,
            "stage_roles": stage_roles,
            "outcome": r.get("outcome", ""),
            "env_config": env_cache[env_id],
        })

    return records


# === Teacher-forced prediction ===


def teacher_forced_predictions(record, tau=TAU):
    """
    For one team's round, predict the model's role combo distribution at each stage,
    then feed it the actual human choices.

    Returns list of dicts, one per stage:
        {"predicted_dist": {combo: prob}, "human_combo": str}
    """
    env = record["env_config"]
    values = env["values"]
    player_stats = env["player_stats"]
    boss_damage = env["boss_damage"]
    team_max_hp = env["team_max_hp"]
    enemy_max_hp = env["enemy_max_hp"]
    lds = record["lds"]
    human_stages = record["stage_roles"]

    team_hp = float(team_max_hp)
    enemy_hp = float(enemy_max_hp)
    prior = utility_based_prior(player_stats, tau=tau)

    results = []
    turn_idx = 0

    for stage_idx, human_combo in enumerate(human_stages):
        if turn_idx >= len(lds):
            break
        if team_hp <= 0 or enemy_hp <= 0:
            break

        intent = lds[turn_idx]
        thp = int(min(max(0, team_hp), team_max_hp))
        ehp = int(min(max(0, enemy_hp), enemy_max_hp))

        # PREDICT: model's distribution over combos at this state
        predicted_dist = model_predicted_combo_dist(
            intent, thp, ehp, prior, values, tau
        )

        results.append({
            "predicted_dist": predicted_dist,
            "human_combo": human_combo,
        })

        # OBSERVE + EXECUTE: use human roles to advance the game
        human_roles = [ROLE_CHAR_TO_IDX[c] for c in human_combo]

        for t in range(TURNS_PER_STAGE):
            if turn_idx >= len(lds):
                break
            if team_hp <= 0 or enemy_hp <= 0:
                break

            intent = lds[turn_idx]
            actions = [get_action(human_roles[i], intent, team_hp, team_max_hp) for i in range(3)]
            prior = bayesian_update(prior, actions, intent, team_hp, team_max_hp)
            team_hp, enemy_hp = game_step(
                intent, team_hp, enemy_hp, actions, player_stats, boss_damage, team_max_hp
            )
            turn_idx += 1

    return results


# === Aggregate and compare ===


def run_all_predictions(records, tau=TAU):
    """
    Run teacher-forced predictions for all teams, aggregate per env.
    Returns dict: env_id -> {stage_idx -> {canonical_combo -> total_prob}}
    """
    # Group records by env
    by_env = defaultdict(list)
    for rec in records:
        by_env[rec["env_id"]].append(rec)

    all_results = {}

    for env_id, env_records in by_env.items():
        stat_profile = env_records[0]["stat_profile"]
        optimal = env_records[0]["optimal_roles"]
        canonical_optimal = canonical_combo(optimal, stat_profile)
        canon_combos = get_canonical_combos(stat_profile)
        n_teams = len(env_records)

        # Accumulate predicted distributions across teams
        # For each stage, average the predicted distributions
        stage_predicted = defaultdict(lambda: defaultdict(float))
        stage_human = defaultdict(lambda: defaultdict(int))
        stage_counts = defaultdict(int)
        max_stages_seen = 0

        # Per-team predictions for logging
        team_predictions = []

        for rec in env_records:
            preds = teacher_forced_predictions(rec, tau=tau)
            team_predictions.append(preds)

            for stage_idx, pred in enumerate(preds):
                stage_counts[stage_idx] += 1
                max_stages_seen = max(max_stages_seen, stage_idx + 1)

                # Accumulate predicted probabilities
                for combo, prob in pred["predicted_dist"].items():
                    cc = canonical_combo(combo, stat_profile)
                    stage_predicted[stage_idx][cc] += prob

                # Count actual human choices
                hcc = canonical_combo(pred["human_combo"], stat_profile)
                stage_human[stage_idx][hcc] += 1

        # Average predicted distributions
        stage_predicted_avg = {}
        for stage_idx in range(max_stages_seen):
            n = stage_counts[stage_idx]
            if n > 0:
                stage_predicted_avg[stage_idx] = {
                    cc: stage_predicted[stage_idx].get(cc, 0.0) / n
                    for cc in canon_combos
                }

        all_results[env_id] = {
            "stat_profile": stat_profile,
            "optimal": optimal,
            "canonical_optimal": canonical_optimal,
            "canonical_combos": canon_combos,
            "n_teams": n_teams,
            "max_stages": max_stages_seen,
            "stage_predicted": stage_predicted_avg,
            "stage_human": dict(stage_human),
            "stage_counts": dict(stage_counts),
            "team_predictions": team_predictions,
        }

    return all_results


def compute_pearson(all_results):
    """Pearson correlation between model predicted and human canonical distributions."""
    correlations = {}

    for env_id, data in all_results.items():
        canon_combos = data["canonical_combos"]

        model_vec = []
        human_vec = []

        for stage_idx in range(data["max_stages"]):
            predicted = data["stage_predicted"].get(stage_idx)
            human_counts = data["stage_human"].get(stage_idx, {})
            n_teams = data["stage_counts"].get(stage_idx, 0)

            if predicted is None or n_teams == 0:
                continue

            for cc in canon_combos:
                model_vec.append(predicted.get(cc, 0.0))
                human_vec.append(human_counts.get(cc, 0) / n_teams)

        if len(model_vec) >= 2:
            r, p = pearsonr(model_vec, human_vec)
            correlations[env_id] = {"r": float(r), "p": float(p), "n_points": len(model_vec)}
        else:
            correlations[env_id] = {"r": float("nan"), "p": float("nan"), "n_points": len(model_vec)}

    return correlations


def compute_log_likelihood(all_results):
    """Average log-likelihood of human choices under the model's predictions."""
    ll_by_env = {}

    for env_id, data in all_results.items():
        log_liks = []
        for team_preds in data["team_predictions"]:
            for pred in team_preds:
                human_combo = pred["human_combo"]
                prob = pred["predicted_dist"].get(human_combo, 1e-20)
                log_liks.append(np.log(max(prob, 1e-20)))

        if log_liks:
            ll_by_env[env_id] = {
                "mean_ll": float(np.mean(log_liks)),
                "n_predictions": len(log_liks),
            }

    return ll_by_env


# === Plotting ===


def plot_comparison(all_results, correlations):
    for env_id, data in all_results.items():
        stat_profile = data["stat_profile"]
        canon_combos = data["canonical_combos"]
        optimal_canon = data["canonical_optimal"]

        max_stages = data["max_stages"]
        stages = list(range(1, max_stages + 1))

        # Model predicted (averaged across teams)
        model_probs = {cc: [] for cc in canon_combos}
        for stage_idx in range(max_stages):
            predicted = data["stage_predicted"].get(stage_idx, {})
            for cc in canon_combos:
                model_probs[cc].append(predicted.get(cc, 0.0))

        # Human empirical
        human_probs = {cc: [] for cc in canon_combos}
        for stage_idx in range(max_stages):
            human_counts = data["stage_human"].get(stage_idx, {})
            n = data["stage_counts"].get(stage_idx, 0)
            for cc in canon_combos:
                human_probs[cc].append(human_counts.get(cc, 0) / n if n > 0 else 0)

        # Identify played combos
        played_combos = [
            cc for cc in canon_combos
            if cc == optimal_canon
            or any(p > 0 for p in human_probs[cc])
            or any(p > 0.02 for p in model_probs[cc])
        ]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        for cc in played_combos:
            is_opt = cc == optimal_canon
            color = "red" if is_opt else None
            lw = 2.5 if is_opt else 1.2
            ms = 8 if is_opt else 4
            label = f"{cc} (optimal)" if is_opt else cc

            kwargs = dict(linewidth=lw, markersize=ms, label=label)
            if color:
                kwargs["color"] = color

            ax1.plot(stages, human_probs[cc], "o-", **kwargs)
            ax2.plot(stages, model_probs[cc], "o-", **kwargs)

        corr = correlations.get(env_id, {})
        r_val = corr.get("r", float("nan"))
        r_str = f"r={r_val:.3f}" if not np.isnan(r_val) else "r=N/A"

        for ax, title in [
            (ax1, f"Human ({data['n_teams']} teams)"),
            (ax2, f"Model teacher-forced (τ={TAU})"),
        ]:
            ax.set_xlabel("Stage")
            ax.set_ylabel("P(canonical role combo)")
            ax.set_title(title)
            ax.set_xticks(stages)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"{env_id} | Profile: {stat_profile} | Optimal: {optimal_canon} | {r_str}",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        fname = f"online_{env_id}.png"
        fig.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# === Main ===


def main():
    print("Loading per-team round data...")
    records = load_team_rounds()
    human_only = [r for r in records if not r.get("has_bots")]
    print(f"Loaded {len(records)} team-rounds across {len(set(r['env_id'] for r in records))} envs")

    print(f"\nRunning teacher-forced predictions (τ={TAU})...")
    all_results = run_all_predictions(records, tau=TAU)

    # Print summary
    for env_id in sorted(all_results.keys()):
        data = all_results[env_id]
        print(f"\n  {env_id} (optimal: {data['optimal']}, {data['n_teams']} teams):")
        for stage_idx in range(data["max_stages"]):
            predicted = data["stage_predicted"].get(stage_idx, {})
            top = sorted(predicted.items(), key=lambda x: -x[1])[:5]
            top_str = ", ".join(f"{c}={p:.2f}" for c, p in top)
            print(f"    Stage {stage_idx+1}: {top_str}")

    print("\nComputing Pearson correlations...")
    correlations = compute_pearson(all_results)

    print("\n" + "=" * 70)
    print("PEARSON CORRELATION: Teacher-Forced Model vs Human")
    print("=" * 70)

    all_r = []
    for env_id in sorted(correlations.keys()):
        c = correlations[env_id]
        r_str = f"{c['r']:.3f}" if not np.isnan(c["r"]) else "N/A"
        p_str = f"{c['p']:.4f}" if not np.isnan(c["p"]) else "N/A"
        print(f"  {env_id}: r={r_str}, p={p_str} (n={c['n_points']} points)")
        if not np.isnan(c["r"]):
            all_r.append(c["r"])

    if all_r:
        print(f"\n  Mean r:   {np.mean(all_r):.3f}")
        print(f"  Median r: {np.median(all_r):.3f}")

    print("\nComputing log-likelihood of human choices...")
    ll = compute_log_likelihood(all_results)
    for env_id in sorted(ll.keys()):
        info = ll[env_id]
        print(f"  {env_id}: mean LL={info['mean_ll']:.3f} ({info['n_predictions']} predictions)")

    print("\nGenerating comparison plots...")
    plot_comparison(all_results, correlations)

    # Save results
    output = {
        "params": {"tau": TAU},
        "correlations": correlations,
        "log_likelihood": ll,
    }
    out_path = SCRIPT_DIR / "online_model_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
