"""Generate PNAS-paper extra figures:

1. Individual heterogeneity: per-participant best-fitting model
   (stacked-bar posterior over models + count bar chart).
2. Qualitative aggregate-by-environment: stage-by-stage human marginal
   role distribution vs Bayesian-Walk and Bayesian-Belief model
   predictions, faceted by env_id, for a small set of representative
   environments.

Run:
    cd analysis
    .venv/bin/python experiments/2026-05-12-pnas-figures/make_figures.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = SCRIPT_DIR.parent.parent
REPO_ROOT = ANALYSIS_ROOT.parent
sys.path.insert(0, str(ANALYSIS_ROOT))
sys.path.insert(
    0,
    str(ANALYSIS_ROOT / "experiments" / "2026-05-12-current-export-metric-comparison"),
)

from shared.constants import ROLE_SHORT, ROLE_CHAR_TO_IDX, TURNS_PER_STAGE
from shared.inference import (
    utility_based_prior, bayesian_update,
    preferred_action, game_step, softmax_role_dist,
)

from pipeline import (  # type: ignore
    load_human_team_records, posterior_marginal, build_joint_dist,
)


# Use the fitted Stage-1 + Stage-2 params from the 05-12 metric comparison
RESULTS = json.loads(
    (ANALYSIS_ROOT / "experiments" / "2026-05-12-current-export-metric-comparison" / "results.json").read_text()
)
STAGE1 = RESULTS["stage1_params"]
TAU_PRIOR = STAGE1["tau_prior"]
EPS_INFER = STAGE1["epsilon"]
# memory_strategy = drift_prior_0.500
DRIFT = 0.5
TAU_SOFTMAX_WALK = RESULTS["cells"]["bayesian_walk"]["combo_r"]["fitted_params"]["tau_softmax"]
EPS_SWITCH_WALK = RESULTS["cells"]["bayesian_walk"]["combo_r"]["fitted_params"]["epsilon_switch"]

FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def precompute_trajectory(record):
    """Replicate pipeline.precompute_trajectory using drift_prior 0.5."""
    env = record["env_config"]
    values = env["values"]
    player_stats = env["player_stats"]
    boss_damage = env["boss_damage"]
    team_max_hp, enemy_max_hp = env["team_max_hp"], env["enemy_max_hp"]

    base_prior = utility_based_prior(player_stats, tau=TAU_PRIOR)
    team_hp, enemy_hp = float(team_max_hp), float(enemy_max_hp)
    prior = base_prior.copy()

    out = []
    turn_idx = 0
    prev_roles = None
    for human_combo in record["stage_roles"]:
        if turn_idx >= len(record["lds"]) or team_hp <= 0 or enemy_hp <= 0:
            break
        intent = record["lds"][turn_idx]
        thp = int(min(max(0, team_hp), team_max_hp))
        ehp = int(min(max(0, enemy_hp), enemy_max_hp))
        out.append({
            "prior": prior.copy(),
            "intent": intent, "thp": thp, "ehp": ehp,
            "human_combo": human_combo,
            "prev_roles": list(prev_roles) if prev_roles is not None else None,
        })
        human_roles = [ROLE_CHAR_TO_IDX[c] for c in human_combo]
        prev_roles = human_roles
        for _ in range(TURNS_PER_STAGE):
            if turn_idx >= len(record["lds"]) or team_hp <= 0 or enemy_hp <= 0:
                break
            intent = record["lds"][turn_idx]
            actions = [preferred_action(human_roles[i], intent, team_hp, team_max_hp)
                       for i in range(3)]
            prior = bayesian_update(prior, actions, intent, team_hp, team_max_hp, EPS_INFER)
            team_hp, enemy_hp = game_step(intent, team_hp, enemy_hp, actions,
                                           player_stats, boss_damage, team_max_hp)
            turn_idx += 1
        # apply drift-to-prior memory strategy
        prior = DRIFT * base_prior + (1.0 - DRIFT) * prior
    return out


def predict_walk(trajectory, values):
    out = []
    for stage in trajectory:
        prior = stage["prior"]
        intent, thp, ehp = stage["intent"], stage["thp"], stage["ehp"]
        prev_roles = stage["prev_roles"]
        switch = [softmax_role_dist(i, intent, thp, ehp, prior, values, TAU_SOFTMAX_WALK)
                  for i in range(3)]
        per_agent = []
        for i in range(3):
            if prev_roles is None:
                per_agent.append(switch[i])
            else:
                stick = np.zeros(3)
                stick[prev_roles[i]] = 1.0
                per_agent.append((1.0 - EPS_SWITCH_WALK) * stick + EPS_SWITCH_WALK * switch[i])
        out.append(np.mean(per_agent, axis=0))
    return out


def predict_belief(trajectory):
    out = []
    for stage in trajectory:
        per_agent = [posterior_marginal(stage["prior"], i) for i in range(3)]
        out.append(np.mean(per_agent, axis=0))
    return out


def main():
    records = load_human_team_records()
    print(f"Loaded {len(records)} clean human team-records")

    # Group records by env_id
    by_env = defaultdict(list)
    for rec in records:
        by_env[rec["env_id"]].append(rec)
    print("Env counts:")
    for env_id, recs in sorted(by_env.items(), key=lambda kv: -len(kv[1])):
        print(f"  {env_id}: {len(recs)}")

    # Pick top-5 envs by sample size
    top_envs = sorted(by_env.items(), key=lambda kv: -len(kv[1]))[:5]

    fig, axes = plt.subplots(3, len(top_envs), figsize=(3.0 * len(top_envs), 7),
                              sharex=True, sharey=True)
    role_names = ["Fighter", "Tank", "Medic"]
    role_colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for col, (env_id, recs) in enumerate(top_envs):
        # Aggregate over teams: per-stage role marginal across all 3 players
        human_marg = np.zeros((5, 3))
        walk_marg = np.zeros((5, 3))
        belief_marg = np.zeros((5, 3))
        counts = np.zeros(5)

        for rec in recs:
            traj = precompute_trajectory(rec)
            walk_pred = predict_walk(traj, rec["env_config"]["values"])
            belief_pred = predict_belief(traj)
            for s, combo in enumerate(rec["stage_roles"][:5]):
                # human marginal across 3 players in this team-stage
                hm = np.zeros(3)
                for c in combo:
                    hm[ROLE_CHAR_TO_IDX[c]] += 1
                hm /= 3.0
                human_marg[s] += hm
                if s < len(walk_pred):
                    walk_marg[s] += walk_pred[s]
                    belief_marg[s] += belief_pred[s]
                counts[s] += 1

        for s in range(5):
            if counts[s] > 0:
                human_marg[s] /= counts[s]
                walk_marg[s] /= counts[s]
                belief_marg[s] /= counts[s]

        # Three rows: Human, Bayesian-Walk, Bayesian-Belief
        for row, (data, title) in enumerate([
            (human_marg, "Human"),
            (walk_marg, "Bayesian-Walk"),
            (belief_marg, "Bayesian-Belief"),
        ]):
            ax = axes[row, col]
            x = np.arange(1, 6)
            bottoms = np.zeros(5)
            for r_idx, (rname, rcol) in enumerate(zip(role_names, role_colors)):
                ax.bar(x, data[:, r_idx], bottom=bottoms, color=rcol,
                       label=rname if (row == 0 and col == 0) else None,
                       width=0.85, edgecolor="white", linewidth=0.4)
                bottoms += data[:, r_idx]
            ax.set_ylim(0, 1)
            ax.set_xticks(x)
            if row == 0:
                # Parse env id: STR_DEF_SUP__ROLES
                if "__" in env_id:
                    profile, combo = env_id.split("__")
                else:
                    parts = env_id.split("_")
                    if len(parts) >= 4:
                        profile = "_".join(parts[:3])
                        combo = parts[3]
                    else:
                        profile, combo = env_id, ""
                ax.set_title(f"{combo}\n{profile}\n(n={len(recs)})", fontsize=8)
            if col == 0:
                ax.set_ylabel(title, fontsize=9)
            if row == 2:
                ax.set_xlabel("Stage", fontsize=8)
            ax.grid(True, alpha=0.2, axis="y")

    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(0, 1.55), ncol=3, fontsize=8,
                       frameon=False)
    plt.suptitle("Aggregate role-choice distribution by environment: human vs models",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    out_path = FIG_DIR / "qualitative_by_env.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
