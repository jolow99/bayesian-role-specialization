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
DEFAULT_DATA_DIR = PROJECT_DIR / "bayesian-role-specialization-2026-03-06-09-54-19"
DEFAULT_DATA_DIRS = [
    PROJECT_DIR / "bayesian-role-specialization-2026-03-06-09-54-19",
    PROJECT_DIR / "bayesian-role-specialization-2026-03-18-15-47-09",
]
VALUE_MATRICES_DIR = PROJECT_DIR / "human_envs_value_matrices"
ENVS_DIR = PROJECT_DIR / "envs"
OUTPUT_DIR = SCRIPT_DIR / "figures" / "model_comparison_online"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Role combo → numeric env ID from envs/
ROLE_COMBO_TO_ENV_NUM = {
    "FFF": 82, "FFM": 5189, "FMM": 2712, "FTF": 157,
    "FTM": 139, "MFF": 957, "MMM": 4915, "TFF": 855,
}

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
TAU_PRIOR = 1.0
TAU_SOFTMAX = 1.0
DROPOUT_GAME_IDS = {
    "01KK14SSY8E64SK69715NN1TMW",  # old dataset dropout
    "01KKZZ4T8F90RB51JW9GHR3B9Q",  # 2 players dropped after R1S1
    "01KKZZ4VRMJT8DA43K6G75XABM",  # 1 player dropped at R4S3
    "01KKZZ54V188BNYZ0WCNNHNC13",  # game never started (batch terminated)
}
ROLE_STAT_COL = {0: 0, 1: 1, 2: 2}

ALL_ROLE_COMBOS = [
    ROLE_NAMES[r0] + ROLE_NAMES[r1] + ROLE_NAMES[r2]
    for r0 in range(3) for r1 in range(3) for r2 in range(3)
]

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


# === Core model ===


def utility_based_prior(player_stats, tau=1.0):
    """P(r0,r1,r2) ∝ exp(Σ_i stat_i(r_i) / τ)"""
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
    prior -= prior.max()
    prior = np.exp(prior)
    prior /= prior.sum()
    return prior


def _preferred_action(role, intent, team_hp, team_max_hp):
    if role == F:
        return ATTACK
    elif role == T:
        return DEFEND if intent == 1 else ATTACK
    else:
        return HEAL if team_hp < team_max_hp else ATTACK


def uniform_prior():
    """Flat 1/27 prior over all role combinations."""
    return np.ones((3, 3, 3)) / 27.0


def action_prob(role, action, intent, team_hp, team_max_hp, epsilon=EPSILON):
    preferred = _preferred_action(role, intent, team_hp, team_max_hp)
    return (1.0 - epsilon) if action == preferred else (epsilon / 2.0)


def bayesian_update(prior, actions, intent, team_hp, team_max_hp, epsilon=EPSILON):
    posterior = np.copy(prior)
    for r0 in range(3):
        for r1 in range(3):
            for r2 in range(3):
                likelihood = (
                    action_prob(r0, actions[0], intent, team_hp, team_max_hp, epsilon)
                    * action_prob(r1, actions[1], intent, team_hp, team_max_hp, epsilon)
                    * action_prob(r2, actions[2], intent, team_hp, team_max_hp, epsilon)
                )
                posterior[r0, r1, r2] *= likelihood
    total = posterior.sum()
    if total > 0:
        posterior /= total
    else:
        posterior = np.ones((3, 3, 3)) / 27.0
    return posterior


def softmax_role_dist(agent_i, intent, team_hp, enemy_hp, prior, values, tau=1.0):
    other_agents = [a for a in range(3) if a != agent_i]
    other_probs = np.sum(prior, axis=agent_i)
    total = other_probs.sum()
    other_probs = other_probs / total if total > 0 else np.ones((3, 3)) / 9.0

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
                ev += other_probs[r_j, r_k] * float(values[flat_idx, intent, team_hp, enemy_hp])
        expected_values[r_i] = ev

    ev_scaled = expected_values / tau
    ev_scaled -= ev_scaled.max()
    exp_ev = np.exp(ev_scaled)
    return exp_ev / exp_ev.sum()


def game_step(intent, team_hp, enemy_hp, actions, player_stats, boss_damage, team_max_hp):
    total_attack = sum(float(player_stats[i, 0]) for i in range(3) if actions[i] == ATTACK)
    defenders = [float(player_stats[i, 1]) for i in range(3) if actions[i] == DEFEND]
    max_defense = max(defenders) if defenders else 0.0
    total_heal = sum(float(player_stats[i, 2]) for i in range(3) if actions[i] == HEAL)

    new_enemy_hp = max(0.0, enemy_hp - total_attack)
    damage = max(0.0, boss_damage - max_defense) if intent == 1 else 0.0
    new_team_hp = max(0.0, min(float(team_max_hp), team_hp - damage + total_heal))
    return new_team_hp, new_enemy_hp


# === Data loading ===


def load_config_module(config_path):
    spec = importlib.util.spec_from_file_location(f"config_{hash(str(config_path))}", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_team_rounds(data_dir=None, data_dirs=None, include_bot_rounds=False):
    """Load team-round records (one per team × round).

    Args:
        data_dir / data_dirs: Export directories to load.
        include_bot_rounds: If True, also emit one record per bot round (per
            human player). Bot-round records carry ``round_type="bot"`` and a
            ``player_round`` field containing the PlayerRound object parsed
            from gameSummary, which downstream code can feed into
            ``shared.data_loading.build_bot_round_layout`` to resolve the
            in-game position mapping correctly (see CLAUDE.md → "Bot Round
            Ground Truth"). Human-round records are tagged
            ``round_type="human"``.

    Returns:
        list[dict]: each record has ``game_id``, ``round_id``, ``round_type``,
        ``env_id``, ``stat_profile``, ``optimal_roles``, ``lds``,
        ``stage_roles``, ``env_config``; bot-round records additionally have
        ``player_round`` (a shared.data_loading.PlayerRound) and
        ``round_number``.
    """
    if data_dirs:
        dirs = [Path(d) for d in data_dirs]
    else:
        dirs = [Path(data_dir) if data_dir else DEFAULT_DATA_DIR]

    games = {}
    rounds = []
    for d in dirs:
        with open(d / "game.csv") as f:
            games.update({r["id"]: r for r in csv.DictReader(f)})
    for d in dirs:
        with open(d / "round.csv") as f:
            rounds.extend(csv.DictReader(f))

    env_cache = {}
    records = []

    def _load_env(env_id, role_combo):
        if env_id in env_cache:
            return env_cache[env_id]
        val_dir = VALUE_MATRICES_DIR / role_combo
        if not (val_dir / "values.npy").exists():
            val_dir = ENVS_DIR / str(env_id)
        values = np.load(val_dir / "values.npy")
        config_mod = load_config_module(val_dir / "config.py")
        env_cache[env_id] = {
            "values": values,
            "player_stats": np.array(config_mod.PLAYER_STATS, dtype=float),
            "boss_damage": float(config_mod.BOSS_DAMAGE),
            "team_max_hp": int(config_mod.TEAM_MAX_HP),
            "enemy_max_hp": int(config_mod.ENEMY_MAX_HP),
        }
        return env_cache[env_id]

    for r in rounds:
        game = games.get(r["gameID"])
        if not game or game.get("status") != "ended":
            continue
        if r["gameID"] in DROPOUT_GAME_IDS:
            continue

        rnum = r["roundNumber"]
        cfg_key = f"round{rnum}Config"
        if not game.get(cfg_key):
            continue

        cfg = json.loads(game[cfg_key])
        if cfg.get("botPlayers"):
            # Bot rounds are handled in the second pass via shared.data_loading
            # so that the human's in-game position comes from gameSummary
            # (pr.player_id) rather than the unreliable config.humanRole.
            continue

        env_id = cfg["envId"]
        optimal_roles = cfg["optimalRoles"]
        role_combo = "".join(ROLE_NAMES[ri] for ri in optimal_roles)
        stat_profile = cfg.get("statProfileId", "")
        lds = [int(c) for c in cfg["enemyIntentSequence"]]

        _load_env(env_id, role_combo)

        stage_roles = []
        for s in range(1, MAX_STAGES + 1):
            turns_data = r.get(f"stage{s}Turns")
            if not turns_data:
                break
            turns = json.loads(turns_data)
            if not turns:
                break
            roles = turns[0].get("roles", [])
            if len(roles) != 3:
                break
            combo_str = "".join(ROLE_NAMES[GAME_ROLE_TO_IDX.get(rs, 0)] for rs in roles)
            stage_roles.append(combo_str)

        if not stage_roles:
            continue

        records.append({
            "game_id": r["gameID"],
            "round_id": r["id"],
            "round_type": "human",
            "round_number": int(rnum),
            "env_id": env_id,
            "stat_profile": stat_profile,
            "optimal_roles": role_combo,
            "lds": lds,
            "stage_roles": stage_roles,
            "env_config": env_cache[env_id],
        })

    if include_bot_rounds:
        # Import locally to avoid a hard dependency on the shared package when
        # callers don't need bot rounds.
        import importlib.util as _ilu
        from pathlib import Path as _Path
        _sdl_path = _Path(__file__).resolve().parents[2] / "analysis" / "shared" / "data_loading.py"
        _spec = _ilu.spec_from_file_location("_sdl", str(_sdl_path))
        # Defer to the package-level import so the shared module is reusable.
        import sys as _sys
        _analysis_root = _Path(__file__).resolve().parents[2] / "analysis"
        if str(_analysis_root) not in _sys.path:
            _sys.path.insert(0, str(_analysis_root))
        from shared.data_loading import load_export

        for d in dirs:
            try:
                prs = load_export(d, include_bot_rounds=True, include_dropout_games=True)
            except ValueError as e:
                # v1/v1.5 exports — skip.
                print(f"  load_team_rounds: skipping {d.name}: {e}")
                continue

            for pr in prs:
                if pr.round.round_type != "bot":
                    continue
                if pr.is_dropout:
                    continue
                if pr.game_id in DROPOUT_GAME_IDS:
                    continue

                cfg = pr.round.config
                env_id = str(cfg.get("envId", ""))
                if not env_id:
                    continue
                optimal_roles = cfg.get("optimalRoles") or []
                role_combo = "".join(ROLE_NAMES[int(ri)] for ri in optimal_roles)
                stat_profile = cfg.get("statProfileId", "")
                lds = [int(c) for c in pr.round.enemy_intent_sequence]

                try:
                    _load_env(env_id, role_combo)
                except FileNotFoundError:
                    print(f"  load_team_rounds: no values.npy for bot env {env_id}; skipping")
                    continue

                # Bot-round stage_roles: human's chosen role letter at each stage.
                # Downstream code uses record['player_round'] +
                # build_bot_round_layout to place roles at in-game positions.
                stage_roles = [
                    ROLE_NAMES[int(s.role_idx)] if s.role_idx in (0, 1, 2) else "F"
                    for s in pr.round.stages
                ]
                if not stage_roles:
                    continue

                records.append({
                    "game_id": pr.game_id,
                    "round_id": f"{pr.game_id}_r{pr.round.round_number}_p{pr.player_id}",
                    "round_type": "bot",
                    "round_number": int(pr.round.round_number),
                    "env_id": env_id,
                    "stat_profile": stat_profile,
                    "optimal_roles": role_combo,
                    "lds": lds,
                    "stage_roles": stage_roles,
                    "env_config": env_cache[env_id],
                    "player_round": pr,
                })

        # Sanity assertion per the plan.
        bot_records = [r for r in records if r.get("round_type") == "bot"]
        assert bot_records, \
            "include_bot_rounds=True but no bot records loaded"
        for br in bot_records:
            assert br["player_round"].round.config.get("botPlayers"), \
                f"bot record {br['round_id']} has empty botPlayers"

    return records


# === Teacher-forced prediction ===


def combo_marginal(combo):
    """Role frequencies from a combo string. 'FTM' -> [1/3, 1/3, 1/3]."""
    counts = np.zeros(3)
    for c in combo:
        counts[ROLE_CHAR_TO_IDX[c]] += 1
    return counts / 3.0


def teacher_forced_predictions(record, tau_prior=TAU_PRIOR, tau_softmax=TAU_SOFTMAX, prior_type="utility", epsilon=EPSILON):
    env = record["env_config"]
    values, player_stats = env["values"], env["player_stats"]
    boss_damage = env["boss_damage"]
    team_max_hp, enemy_max_hp = env["team_max_hp"], env["enemy_max_hp"]

    team_hp, enemy_hp = float(team_max_hp), float(enemy_max_hp)
    if prior_type == "uniform":
        prior = uniform_prior()
    else:
        prior = utility_based_prior(player_stats, tau=tau_prior)
    results = []
    turn_idx = 0

    for human_combo in record["stage_roles"]:
        if turn_idx >= len(record["lds"]) or team_hp <= 0 or enemy_hp <= 0:
            break

        intent = record["lds"][turn_idx]
        thp = int(min(max(0, team_hp), team_max_hp))
        ehp = int(min(max(0, enemy_hp), enemy_max_hp))

        # Per-agent softmax → joint combo dist + marginals
        per_agent = [softmax_role_dist(i, intent, thp, ehp, prior, values, tau_softmax) for i in range(3)]

        predicted_dist = {}
        for r0 in range(3):
            for r1 in range(3):
                for r2 in range(3):
                    combo = ROLE_NAMES[r0] + ROLE_NAMES[r1] + ROLE_NAMES[r2]
                    predicted_dist[combo] = float(per_agent[0][r0] * per_agent[1][r1] * per_agent[2][r2])

        results.append({
            "predicted_dist": predicted_dist,
            "human_combo": human_combo,
            "model_marginal": np.mean(per_agent, axis=0),
        })

        # Advance game with human roles
        human_roles = [ROLE_CHAR_TO_IDX[c] for c in human_combo]
        for _ in range(TURNS_PER_STAGE):
            if turn_idx >= len(record["lds"]) or team_hp <= 0 or enemy_hp <= 0:
                break
            intent = record["lds"][turn_idx]
            actions = [_preferred_action(human_roles[i], intent, team_hp, team_max_hp) for i in range(3)]
            prior = bayesian_update(prior, actions, intent, team_hp, team_max_hp, epsilon)
            team_hp, enemy_hp = game_step(intent, team_hp, enemy_hp, actions, player_stats, boss_damage, team_max_hp)
            turn_idx += 1

    return results


# === Aggregation ===


def run_all_predictions(records, tau_prior=TAU_PRIOR, tau_softmax=TAU_SOFTMAX, prior_type="utility", epsilon=EPSILON):
    by_env = defaultdict(list)
    for rec in records:
        by_env[rec["env_id"]].append(rec)

    all_results = {}

    for env_id, env_records in by_env.items():
        stat_profile = env_records[0]["stat_profile"]
        optimal = env_records[0]["optimal_roles"]
        canon_combos = get_canonical_combos(stat_profile)

        stage_predicted = defaultdict(lambda: defaultdict(float))
        stage_human = defaultdict(lambda: defaultdict(int))
        stage_model_marg = defaultdict(lambda: np.zeros(3))
        stage_human_marg = defaultdict(lambda: np.zeros(3))
        stage_counts = defaultdict(int)
        team_predictions = []
        max_stages = 0

        for rec in env_records:
            preds = teacher_forced_predictions(rec, tau_prior=tau_prior, tau_softmax=tau_softmax, prior_type=prior_type, epsilon=epsilon)
            team_predictions.append(preds)

            for s, pred in enumerate(preds):
                stage_counts[s] += 1
                max_stages = max(max_stages, s + 1)

                for combo, prob in pred["predicted_dist"].items():
                    stage_predicted[s][canonical_combo(combo, stat_profile)] += prob
                stage_human[s][canonical_combo(pred["human_combo"], stat_profile)] += 1

                stage_model_marg[s] += pred["model_marginal"]
                stage_human_marg[s] += combo_marginal(pred["human_combo"])

        # Average across teams
        predicted_avg, model_marg_avg, human_marg_avg = {}, {}, {}
        for s in range(max_stages):
            n = stage_counts[s]
            if n > 0:
                predicted_avg[s] = {cc: stage_predicted[s].get(cc, 0.0) / n for cc in canon_combos}
                model_marg_avg[s] = stage_model_marg[s] / n
                human_marg_avg[s] = stage_human_marg[s] / n

        all_results[env_id] = {
            "stat_profile": stat_profile,
            "optimal": optimal,
            "canonical_optimal": canonical_combo(optimal, stat_profile),
            "canonical_combos": canon_combos,
            "n_teams": len(env_records),
            "max_stages": max_stages,
            "stage_predicted": predicted_avg,
            "stage_human": dict(stage_human),
            "stage_counts": dict(stage_counts),
            "team_predictions": team_predictions,
            "stage_model_marginal": model_marg_avg,
            "stage_human_marginal": human_marg_avg,
        }

    return all_results


# === Evaluation ===


def compute_pearson(all_results):
    correlations = {}
    global_combo_m, global_combo_h = [], []
    global_marg_m, global_marg_h = [], []

    for env_id, data in all_results.items():
        combo_m, combo_h, marg_m, marg_h = [], [], [], []

        for s in range(data["max_stages"]):
            predicted = data["stage_predicted"].get(s)
            human_counts = data["stage_human"].get(s, {})
            n = data["stage_counts"].get(s, 0)
            if predicted is None or n == 0:
                continue

            for cc in data["canonical_combos"]:
                combo_m.append(predicted.get(cc, 0.0))
                combo_h.append(human_counts.get(cc, 0) / n)

            mm = data["stage_model_marginal"].get(s)
            hm = data["stage_human_marginal"].get(s)
            if mm is not None and hm is not None:
                marg_m.extend(mm.tolist())
                marg_h.extend(hm.tolist())

        env_corr = {}
        if len(combo_m) >= 2:
            r, p = pearsonr(combo_m, combo_h)
            env_corr["combo"] = {"r": float(r), "p": float(p), "n": len(combo_m)}
        if len(marg_m) >= 2:
            r, p = pearsonr(marg_m, marg_h)
            env_corr["marginal"] = {"r": float(r), "p": float(p), "n": len(marg_m)}

        correlations[env_id] = env_corr
        global_combo_m.extend(combo_m)
        global_combo_h.extend(combo_h)
        global_marg_m.extend(marg_m)
        global_marg_h.extend(marg_h)

    # Global correlations
    global_corr = {}
    if len(global_combo_m) >= 2:
        r, p = pearsonr(global_combo_m, global_combo_h)
        global_corr["combo"] = {"r": float(r), "p": float(p), "n": len(global_combo_m)}
    if len(global_marg_m) >= 2:
        r, p = pearsonr(global_marg_m, global_marg_h)
        global_corr["marginal"] = {"r": float(r), "p": float(p), "n": len(global_marg_m)}
    correlations["__global__"] = global_corr

    return correlations


def compute_log_likelihood(all_results):
    ll_by_env = {}
    for env_id, data in all_results.items():
        log_liks = []
        for team_preds in data["team_predictions"]:
            for pred in team_preds:
                prob = pred["predicted_dist"].get(pred["human_combo"], 1e-20)
                log_liks.append(np.log(max(prob, 1e-20)))
        if log_liks:
            ll_by_env[env_id] = {"mean_ll": float(np.mean(log_liks)), "n": len(log_liks)}
    return ll_by_env


# === Plotting ===

ROLE_COLORS = {"F": "#e74c3c", "T": "#3498db", "M": "#2ecc71"}


def plot_comparison(all_results, correlations, tau_prior=TAU_PRIOR, tau_softmax=TAU_SOFTMAX, epsilon=EPSILON):
    for env_id, data in all_results.items():
        canon_combos = data["canonical_combos"]
        optimal_canon = data["canonical_optimal"]
        max_stages = data["max_stages"]
        stages = list(range(1, max_stages + 1))

        # Build per-combo time series
        model_probs = {cc: [] for cc in canon_combos}
        human_probs = {cc: [] for cc in canon_combos}
        for s in range(max_stages):
            predicted = data["stage_predicted"].get(s, {})
            human_counts = data["stage_human"].get(s, {})
            n = data["stage_counts"].get(s, 0)
            for cc in canon_combos:
                model_probs[cc].append(predicted.get(cc, 0.0))
                human_probs[cc].append(human_counts.get(cc, 0) / n if n > 0 else 0)

        played = [
            cc for cc in canon_combos
            if cc == optimal_canon
            or any(p > 0 for p in human_probs[cc])
            or any(p > 0.02 for p in model_probs[cc])
        ]

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        # Combo distributions
        for cc in played:
            is_opt = cc == optimal_canon
            kw = dict(linewidth=2.5 if is_opt else 1.2, markersize=8 if is_opt else 4,
                       label=f"{cc} (optimal)" if is_opt else cc)
            if is_opt:
                kw["color"] = "red"
            axes[0].plot(stages, human_probs[cc], "o-", **kw)
            axes[1].plot(stages, model_probs[cc], "o-", **kw)

        # Marginals
        for role_idx, role_name in ROLE_NAMES.items():
            h_vals = [data["stage_human_marginal"].get(s, np.zeros(3))[role_idx] for s in range(max_stages)]
            m_vals = [data["stage_model_marginal"].get(s, np.zeros(3))[role_idx] for s in range(max_stages)]
            color = ROLE_COLORS[role_name]
            axes[2].plot(stages, h_vals, "o-", color=color, linewidth=2, label=f"{role_name} human")
            axes[2].plot(stages, m_vals, "s--", color=color, linewidth=2, alpha=0.7, label=f"{role_name} model")

        env_corr = correlations.get(env_id, {})
        combo_r = env_corr.get("combo", {}).get("r", float("nan"))
        marg_r = env_corr.get("marginal", {}).get("r", float("nan"))

        for ax, title in [(axes[0], f"Human ({data['n_teams']} teams)"),
                          (axes[1], f"Model (τ_p={tau_prior}, τ_s={tau_softmax}, ε={epsilon:.4g})")]:
            ax.set_xlabel("Stage")
            ax.set_ylabel("P(combo)")
            ax.set_title(title)
            ax.set_xticks(stages)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        axes[2].set_xlabel("Stage")
        axes[2].set_ylabel("P(role)")
        axes[2].set_title(f"Marginals (r={marg_r:.3f})" if not np.isnan(marg_r) else "Marginals")
        axes[2].set_xticks(stages)
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].legend(fontsize=7)
        axes[2].grid(True, alpha=0.3)

        r_str = f"combo r={combo_r:.3f}" if not np.isnan(combo_r) else "combo r=N/A"
        fig.suptitle(
            f"{env_id} | {data['stat_profile']} | Optimal: {optimal_canon} | {r_str}",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        role_combo = data["optimal"]
        env_num = ROLE_COMBO_TO_ENV_NUM.get(role_combo, "")
        folder_name = f"{env_num}_{env_id}" if env_num else env_id
        env_dir = OUTPUT_DIR / folder_name
        env_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(env_dir / f"online_{env_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Bar chart: per-stage grouped bars (human vs model)
        n_stages = max_stages
        fig_bar, axes_bar = plt.subplots(1, n_stages, figsize=(5 * n_stages, 5), sharey=True)
        if n_stages == 1:
            axes_bar = [axes_bar]

        bar_width = 0.35
        for s in range(n_stages):
            ax = axes_bar[s]
            h_vals = [human_probs[cc][s] for cc in played]
            m_vals = [model_probs[cc][s] for cc in played]
            x = np.arange(len(played))

            bars_h = ax.bar(x - bar_width / 2, h_vals, bar_width, label="Human", color="#3498db", alpha=0.8)
            bars_m = ax.bar(x + bar_width / 2, m_vals, bar_width, label="Model", color="#e74c3c", alpha=0.8)

            # Highlight optimal combo
            for i, cc in enumerate(played):
                if cc == optimal_canon:
                    ax.bar(x[i] - bar_width / 2, h_vals[i], bar_width, color="#3498db", edgecolor="gold", linewidth=2.5)
                    ax.bar(x[i] + bar_width / 2, m_vals[i], bar_width, color="#e74c3c", edgecolor="gold", linewidth=2.5)

            ax.set_xlabel("Role combo")
            ax.set_title(f"Stage {s + 1}")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{cc}*" if cc == optimal_canon else cc for cc in played],
                               rotation=45, ha="right", fontsize=7)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis="y")
            if s == 0:
                ax.set_ylabel("P(combo)")
                ax.legend(fontsize=8)

        r_str = f"combo r={combo_r:.3f}" if not np.isnan(combo_r) else "combo r=N/A"
        fig_bar.suptitle(
            f"{env_id} | {data['stat_profile']} | Optimal: {optimal_canon}* | {r_str} | τ_p={tau_prior}, τ_s={tau_softmax}, ε={epsilon:.4g}",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        fig_bar.savefig(env_dir / f"online_{env_id}_bars.png", dpi=150, bbox_inches="tight")
        plt.close(fig_bar)


# === Main ===


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-prior", type=float, default=TAU_PRIOR, help="Temperature for utility-based prior")
    parser.add_argument("--tau-softmax", type=float, default=TAU_SOFTMAX, help="Temperature for softmax role selection")
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--data-dir", type=str, nargs="+", default=None, help="Path(s) to data directory (can specify multiple)")
    args = parser.parse_args()
    tau_prior, tau_softmax, epsilon = args.tau_prior, args.tau_softmax, args.epsilon

    records = load_team_rounds(data_dirs=args.data_dir)
    n_envs = len(set(r["env_id"] for r in records))
    print(f"Loaded {len(records)} team-rounds across {n_envs} envs")

    all_results = run_all_predictions(records, tau_prior=tau_prior, tau_softmax=tau_softmax, epsilon=epsilon)
    correlations = compute_pearson(all_results)
    ll = compute_log_likelihood(all_results)

    print(f"\n{'='*60}")
    print(f"RESULTS (τ_prior={tau_prior}, τ_softmax={tau_softmax}, ε={epsilon:.4g})")
    print(f"{'='*60}")

    for env_id in sorted(k for k in correlations if k != "__global__"):
        c = correlations[env_id]
        combo = c.get("combo", {})
        marg = c.get("marginal", {})
        ll_info = ll.get(env_id, {})
        print(f"  {env_id}:")
        if combo:
            print(f"    combo  r={combo['r']:.3f}  p={combo['p']:.4f}  (n={combo['n']})")
        if marg:
            print(f"    marg   r={marg['r']:.3f}  p={marg['p']:.4f}  (n={marg['n']})")
        if ll_info:
            print(f"    LL={ll_info['mean_ll']:.3f}  ({ll_info['n']} predictions)")

    g = correlations.get("__global__", {})
    print(f"\n  Global:")
    if g.get("combo"):
        print(f"    combo  r={g['combo']['r']:.3f}  p={g['combo']['p']:.4f}")
    if g.get("marginal"):
        print(f"    marg   r={g['marginal']['r']:.3f}  p={g['marginal']['p']:.4f}")

    output = {"params": {"tau_prior": tau_prior, "tau_softmax": tau_softmax, "epsilon": epsilon}, "correlations": correlations, "log_likelihood": ll}
    out_path = SCRIPT_DIR / "online_model_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("\nTo generate plots, run finetune_tau.py which plots with best-fit params.")


if __name__ == "__main__":
    main()
