"""Stage 1: Tune (tau_prior, epsilon, memory_strategy) on human role inference data.

Shared across ALL Bayesian models since inference code is identical.
Optimizes inference_ll (mean log P(human_inferred_role | model_marginal)).
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.optimize import minimize

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from shared_utils import load_checkpoint, save_checkpoint, get_completed_keys, pick_best

# Shared package
from shared import EXPORTS_DIR, DATA_ROOT
from shared.constants import (
    F, T, M, ROLE_SHORT, ROLE_CHAR_TO_IDX, GAME_ROLE_TO_IDX,
    TURNS_PER_STAGE, DROPOUT_GAME_IDS,
)
from shared.data_loading import load_all_exports
from shared.inference import (
    utility_based_prior, bayesian_update, preferred_action, game_step,
)

# Paths
CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints'
CHECKPOINT_DIR.mkdir(exist_ok=True)
COARSE_PATH = CHECKPOINT_DIR / 'coarse_results.json'
REFINED_PATH = CHECKPOINT_DIR / 'refined_results.json'
POLISHED_PATH = CHECKPOINT_DIR / 'polished_results.json'
OUTPUT_PATH = SCRIPT_DIR / 'best_inference_params.json'


# ── Memory strategies ──────────────────────────────────────────────

MEMORY_STRATEGIES = []
# Full history
MEMORY_STRATEGIES.append(('full', None, 0.0))
# Windowed: 1..4
for w in [1, 2, 3, 4]:
    MEMORY_STRATEGIES.append((f'window_{w}', w, 0.0))
# Drift: 10 values
for d in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    MEMORY_STRATEGIES.append((f'drift_{d}', None, d))

STRATEGY_KEY_MAP = {s[0]: (s[1], s[2]) for s in MEMORY_STRATEGIES}


# ── Data loading ───────────────────────────────────────────────────

def load_data():
    """Load all exports, group into human teams and bot records."""
    records = load_all_exports()

    # Group human rounds by (game_id, round_number) into teams
    human_teams = defaultdict(list)
    bot_records = []

    for pr in records:
        if pr.is_dropout:
            continue
        if pr.round.round_type == 'human':
            key = (pr.game_id, pr.round.round_number)
            human_teams[key].append(pr)
        elif pr.round.round_type == 'bot':
            bot_records.append(pr)

    # Filter to complete teams of 3
    human_teams = {k: sorted(v, key=lambda p: p.player_id)
                   for k, v in human_teams.items() if len(v) == 3}

    print(f"Loaded {len(human_teams)} human teams, {len(bot_records)} bot records")
    return human_teams, bot_records


# ── Posterior computation ──────────────────────────────────────────

def get_windowed_posteriors_team(team_prs, tau_prior, epsilon, window=None, drift_delta=0.0):
    """Compute posteriors with different memory strategies for a human team.

    Returns list of posteriors: posteriors[s] = belief state at inference time for stage s.
    posteriors[0] = prior (before any observations).
    """
    rnd = team_prs[0].round
    config = rnd.config
    stat_profile_id = rnd.stat_profile_id
    parts = stat_profile_id.split('_')
    player_stats = np.array([[int(c) for c in part] for part in parts])

    boss_damage = config.get('bossDamage', 2)
    team_max_hp = config.get('maxTeamHealth', 15)
    enemy_max_hp = config.get('maxEnemyHealth', 30)
    eis = rnd.enemy_intent_sequence

    original_prior = utility_based_prior(player_stats, tau=tau_prior)

    player_roles = {}
    for pr in team_prs:
        player_roles[pr.player_id] = [s.role_idx for s in pr.round.stages]

    n_stages = max(len(roles) for roles in player_roles.values())

    # Track HP at start of each stage (needed for windowed strategy)
    team_hp_per_stage = [float(team_max_hp)]
    running_hp = float(team_max_hp)
    for s in range(n_stages):
        roles = [0, 0, 0]
        for pid, role_list in player_roles.items():
            if s < len(role_list):
                roles[pid] = role_list[s]
        hp = running_hp
        for turn_offset in range(TURNS_PER_STAGE):
            turn_idx = s * TURNS_PER_STAGE + turn_offset
            if turn_idx >= len(eis) or hp <= 0:
                break
            intent = int(eis[turn_idx])
            actions = [preferred_action(roles[i], intent, hp, team_max_hp) for i in range(3)]
            hp, _ = game_step(intent, hp, float(enemy_max_hp),
                              actions, player_stats, boss_damage, team_max_hp)
        running_hp = hp
        team_hp_per_stage.append(hp)

    # Build posteriors under chosen strategy
    posteriors = [original_prior.copy()]  # before stage 0

    if window is not None:
        for s in range(1, n_stages + 1):
            start = max(0, s - window)
            post = original_prior.copy()
            hp = team_hp_per_stage[start]
            for ws in range(start, s):
                roles = [0, 0, 0]
                for pid, role_list in player_roles.items():
                    if ws < len(role_list):
                        roles[pid] = role_list[ws]
                for turn_offset in range(TURNS_PER_STAGE):
                    turn_idx = ws * TURNS_PER_STAGE + turn_offset
                    if turn_idx >= len(eis) or hp <= 0:
                        break
                    intent = int(eis[turn_idx])
                    actions = [preferred_action(roles[i], intent, hp, team_max_hp) for i in range(3)]
                    post = bayesian_update(post, actions, intent, hp, team_max_hp, epsilon)
                    hp, _ = game_step(intent, hp, float(enemy_max_hp),
                                      actions, player_stats, boss_damage, team_max_hp)
            posteriors.append(post)
    elif drift_delta > 0:
        current = original_prior.copy()
        hp = float(team_max_hp)
        for s in range(n_stages):
            roles = [0, 0, 0]
            for pid, role_list in player_roles.items():
                if s < len(role_list):
                    roles[pid] = role_list[s]
            for turn_offset in range(TURNS_PER_STAGE):
                turn_idx = s * TURNS_PER_STAGE + turn_offset
                if turn_idx >= len(eis) or hp <= 0:
                    break
                intent = int(eis[turn_idx])
                actions = [preferred_action(roles[i], intent, hp, team_max_hp) for i in range(3)]
                current = bayesian_update(current, actions, intent, hp, team_max_hp, epsilon)
                hp, _ = game_step(intent, hp, float(enemy_max_hp),
                                  actions, player_stats, boss_damage, team_max_hp)
            current = (1 - drift_delta) * current + drift_delta * original_prior
            total = current.sum()
            if total > 0:
                current /= total
            posteriors.append(current.copy())
    else:
        # Full history
        current = original_prior.copy()
        hp = float(team_max_hp)
        for s in range(n_stages):
            roles = [0, 0, 0]
            for pid, role_list in player_roles.items():
                if s < len(role_list):
                    roles[pid] = role_list[s]
            for turn_offset in range(TURNS_PER_STAGE):
                turn_idx = s * TURNS_PER_STAGE + turn_offset
                if turn_idx >= len(eis) or hp <= 0:
                    break
                intent = int(eis[turn_idx])
                actions = [preferred_action(roles[i], intent, hp, team_max_hp) for i in range(3)]
                current = bayesian_update(current, actions, intent, hp, team_max_hp, epsilon)
                hp, _ = game_step(intent, hp, float(enemy_max_hp),
                                  actions, player_stats, boss_damage, team_max_hp)
            posteriors.append(current.copy())

    return posteriors


def get_windowed_posteriors_bot(pr, tau_prior, epsilon, window=None, drift_delta=0.0):
    """Compute posteriors for a bot round (single human player).

    Same logic as team version but roles come from a single PlayerRound.
    """
    rnd = pr.round
    config = rnd.config
    stat_profile_id = rnd.stat_profile_id
    parts = stat_profile_id.split('_')
    player_stats = np.array([[int(c) for c in part] for part in parts])

    boss_damage = config.get('bossDamage', 2)
    team_max_hp = config.get('maxTeamHealth', 15)
    enemy_max_hp = config.get('maxEnemyHealth', 30)
    eis = rnd.enemy_intent_sequence

    original_prior = utility_based_prior(player_stats, tau=tau_prior)

    # In bot rounds, we have all 3 players' roles from the human's perspective
    # The stages have the human's role, and bot roles come from config
    n_stages = len(rnd.stages)
    human_pos = config.get('humanRole', 0)
    bot_players = config.get('botPlayers', [])

    # Build full role sequences: bot roles are fixed per their strategy
    bot_role_map = {}
    for bp in bot_players:
        bot_pos = bp.get('position', bp.get('playerIndex'))
        bot_role_str = bp.get('strategy', {}).get('role', '')
        if bot_role_str in GAME_ROLE_TO_IDX:
            bot_role_map[bot_pos] = GAME_ROLE_TO_IDX[bot_role_str]

    def get_all_roles(stage_idx):
        roles = [0, 0, 0]
        if stage_idx < n_stages:
            roles[human_pos] = rnd.stages[stage_idx].role_idx
        for pos, ridx in bot_role_map.items():
            roles[pos] = ridx
        return roles

    # Track HP per stage
    team_hp_per_stage = [float(team_max_hp)]
    running_hp = float(team_max_hp)
    for s in range(n_stages):
        roles = get_all_roles(s)
        hp = running_hp
        for turn_offset in range(TURNS_PER_STAGE):
            turn_idx = s * TURNS_PER_STAGE + turn_offset
            if turn_idx >= len(eis) or hp <= 0:
                break
            intent = int(eis[turn_idx])
            actions = [preferred_action(roles[i], intent, hp, team_max_hp) for i in range(3)]
            hp, _ = game_step(intent, hp, float(enemy_max_hp),
                              actions, player_stats, boss_damage, team_max_hp)
        running_hp = hp
        team_hp_per_stage.append(hp)

    # Build posteriors (same 3-branch logic)
    posteriors = [original_prior.copy()]

    if window is not None:
        for s in range(1, n_stages + 1):
            start = max(0, s - window)
            post = original_prior.copy()
            hp = team_hp_per_stage[start]
            for ws in range(start, s):
                roles = get_all_roles(ws)
                for turn_offset in range(TURNS_PER_STAGE):
                    turn_idx = ws * TURNS_PER_STAGE + turn_offset
                    if turn_idx >= len(eis) or hp <= 0:
                        break
                    intent = int(eis[turn_idx])
                    actions = [preferred_action(roles[i], intent, hp, team_max_hp) for i in range(3)]
                    post = bayesian_update(post, actions, intent, hp, team_max_hp, epsilon)
                    hp, _ = game_step(intent, hp, float(enemy_max_hp),
                                      actions, player_stats, boss_damage, team_max_hp)
            posteriors.append(post)
    elif drift_delta > 0:
        current = original_prior.copy()
        hp = float(team_max_hp)
        for s in range(n_stages):
            roles = get_all_roles(s)
            for turn_offset in range(TURNS_PER_STAGE):
                turn_idx = s * TURNS_PER_STAGE + turn_offset
                if turn_idx >= len(eis) or hp <= 0:
                    break
                intent = int(eis[turn_idx])
                actions = [preferred_action(roles[i], intent, hp, team_max_hp) for i in range(3)]
                current = bayesian_update(current, actions, intent, hp, team_max_hp, epsilon)
                hp, _ = game_step(intent, hp, float(enemy_max_hp),
                                  actions, player_stats, boss_damage, team_max_hp)
            current = (1 - drift_delta) * current + drift_delta * original_prior
            total = current.sum()
            if total > 0:
                current /= total
            posteriors.append(current.copy())
    else:
        current = original_prior.copy()
        hp = float(team_max_hp)
        for s in range(n_stages):
            roles = get_all_roles(s)
            for turn_offset in range(TURNS_PER_STAGE):
                turn_idx = s * TURNS_PER_STAGE + turn_offset
                if turn_idx >= len(eis) or hp <= 0:
                    break
                intent = int(eis[turn_idx])
                actions = [preferred_action(roles[i], intent, hp, team_max_hp) for i in range(3)]
                current = bayesian_update(current, actions, intent, hp, team_max_hp, epsilon)
                hp, _ = game_step(intent, hp, float(enemy_max_hp),
                                  actions, player_stats, boss_damage, team_max_hp)
            posteriors.append(current.copy())

    return posteriors


# ── Inference evaluation ───────────────────────────────────────────

def evaluate_inference(tau_prior, epsilon, window, drift_delta, human_teams, bot_records):
    """Evaluate inference accuracy and log-likelihood for given params.

    Returns dict with accuracy, inference_ll, n.
    """
    correct, total = 0, 0
    log_liks = []

    # Human rounds
    for (gid, rnum), team_prs in human_teams.items():
        posteriors = get_windowed_posteriors_team(
            team_prs, tau_prior, epsilon, window=window, drift_delta=drift_delta)

        player_roles = {}
        for pr in team_prs:
            player_roles[pr.player_id] = [s.role_idx for s in pr.round.stages]

        for pr in team_prs:
            for si, stage in enumerate(pr.round.stages):
                if si == 0 or not stage.inferred_roles:
                    continue
                if si >= len(posteriors):
                    continue
                post = posteriors[si]
                for target_pos, human_inferred_role in stage.inferred_roles.items():
                    if target_pos not in player_roles or si - 1 >= len(player_roles[target_pos]):
                        continue
                    # Marginalize posterior for target position
                    marg = np.sum(post, axis=tuple(j for j in range(3) if j != target_pos))
                    t = marg.sum()
                    if t > 0:
                        marg /= t

                    total += 1
                    if int(np.argmax(marg)) == player_roles[target_pos][si - 1]:
                        correct += 1
                    log_liks.append(np.log(max(marg[human_inferred_role], 1e-20)))

    # Bot rounds
    for pr in bot_records:
        posteriors = get_windowed_posteriors_bot(
            pr, tau_prior, epsilon, window=window, drift_delta=drift_delta)

        rnd = pr.round
        config = rnd.config
        human_pos = config.get('humanRole', 0)
        bot_players = config.get('botPlayers', [])

        # Build role map for all positions
        all_roles = {}
        all_roles[human_pos] = [s.role_idx for s in rnd.stages]
        for bp in bot_players:
            bot_pos = bp.get('position', bp.get('playerIndex'))
            bot_role_str = bp.get('strategy', {}).get('role', '')
            if bot_role_str in GAME_ROLE_TO_IDX:
                bot_role_idx = GAME_ROLE_TO_IDX[bot_role_str]
                all_roles[bot_pos] = [bot_role_idx] * len(rnd.stages)

        for si, stage in enumerate(rnd.stages):
            if si == 0 or not stage.inferred_roles:
                continue
            if si >= len(posteriors):
                continue
            post = posteriors[si]
            for target_pos, human_inferred_role in stage.inferred_roles.items():
                if target_pos not in all_roles or si - 1 >= len(all_roles[target_pos]):
                    continue
                marg = np.sum(post, axis=tuple(j for j in range(3) if j != target_pos))
                t = marg.sum()
                if t > 0:
                    marg /= t

                total += 1
                if int(np.argmax(marg)) == all_roles[target_pos][si - 1]:
                    correct += 1
                log_liks.append(np.log(max(marg[human_inferred_role], 1e-20)))

    return {
        'accuracy': correct / total if total > 0 else 0.0,
        'inference_ll': float(np.mean(log_liks)) if log_liks else float('nan'),
        'n': total,
    }


# ── Grid search phases ─────────────────────────────────────────────

def run_coarse_search(human_teams, bot_records):
    """Phase 1: Coarse grid over tau_prior x epsilon x memory_strategy."""
    results = load_checkpoint(str(COARSE_PATH))
    completed = get_completed_keys(results, ['tau_prior', 'epsilon', 'strategy_name'])

    tau_prior_vals = np.linspace(0.1, 20.0, 20)
    eps_vals = np.linspace(0.001, 1.0, 21)

    total = len(tau_prior_vals) * len(eps_vals) * len(MEMORY_STRATEGIES)
    print(f"Coarse grid: {len(tau_prior_vals)} x {len(eps_vals)} x {len(MEMORY_STRATEGIES)} = {total} points")
    print(f"  Already completed: {len(completed)}")

    count = len(completed)
    for ti, tp in enumerate(tau_prior_vals):
        batch_added = False
        for eps in eps_vals:
            for strategy_name, window, drift_delta in MEMORY_STRATEGIES:
                key = (float(tp), float(eps), strategy_name)
                if key in completed:
                    continue

                res = evaluate_inference(tp, eps, window, drift_delta, human_teams, bot_records)
                res.update({
                    'tau_prior': float(tp),
                    'epsilon': float(eps),
                    'strategy_name': strategy_name,
                    'window': window,
                    'drift_delta': drift_delta,
                })
                results.append(res)
                batch_added = True
                count += 1

        if batch_added:
            save_checkpoint(str(COARSE_PATH), results)
            print(f"  [{count}/{total}] saved (tau_prior={tp:.2f})", flush=True)

    if not any(True for _ in []):  # ensure final save
        save_checkpoint(str(COARSE_PATH), results)
    print(f"Coarse search complete: {len(results)} results")
    return results


def run_refined_search(human_teams, bot_records, coarse_results):
    """Phase 2: Refined grid around best coarse result."""
    best = pick_best(coarse_results, 'inference_ll')
    print(f"\nCoarse best: tau_prior={best['tau_prior']:.4f} eps={best['epsilon']:.6f} "
          f"strategy={best['strategy_name']}")
    print(f"  inference_ll={best['inference_ll']:.4f} accuracy={best['accuracy']:.3f}")

    # Fixed strategy from coarse
    strategy_name = best['strategy_name']
    window = best['window']
    drift_delta = best['drift_delta']

    # Refined grid: ±1 step around best
    tau_step = (20.0 - 0.1) / 19  # coarse step
    eps_step = (1.0 - 0.001) / 20

    tau_lo = max(0.01, best['tau_prior'] - tau_step)
    tau_hi = min(50.0, best['tau_prior'] + tau_step)
    eps_lo = max(0.001, best['epsilon'] - eps_step)
    eps_hi = min(0.999, best['epsilon'] + eps_step)

    tau_vals = np.linspace(tau_lo, tau_hi, 11)
    eps_vals = np.linspace(eps_lo, eps_hi, 11)

    results = load_checkpoint(str(REFINED_PATH))
    completed = get_completed_keys(results, ['tau_prior', 'epsilon'])

    total = len(tau_vals) * len(eps_vals)
    print(f"\nRefined grid: {len(tau_vals)} x {len(eps_vals)} = {total} points")
    print(f"  tau_prior: [{tau_lo:.4f}, {tau_hi:.4f}]")
    print(f"  epsilon: [{eps_lo:.6f}, {eps_hi:.6f}]")
    print(f"  strategy: {strategy_name} (fixed)")
    print(f"  Already completed: {len(completed)}")

    count = len(completed)
    for tp in tau_vals:
        batch_added = False
        for eps in eps_vals:
            key = (float(tp), float(eps))
            if key in completed:
                continue

            res = evaluate_inference(tp, eps, window, drift_delta, human_teams, bot_records)
            res.update({
                'tau_prior': float(tp),
                'epsilon': float(eps),
                'strategy_name': strategy_name,
                'window': window,
                'drift_delta': drift_delta,
            })
            results.append(res)
            batch_added = True
            count += 1

        if batch_added:
            save_checkpoint(str(REFINED_PATH), results)
            print(f"  [{count}/{total}] ...", flush=True)

    save_checkpoint(str(REFINED_PATH), results)
    print(f"Refined search complete: {len(results)} results")
    return results


def run_scipy_polish(human_teams, bot_records, all_results):
    """Phase 3: L-BFGS-B polishing on (tau_prior, epsilon) with fixed strategy."""
    best = pick_best(all_results, 'inference_ll')
    strategy_name = best['strategy_name']
    window = best['window']
    drift_delta = best['drift_delta']

    print(f"\nScipy polish starting from: tau_prior={best['tau_prior']:.4f} "
          f"eps={best['epsilon']:.6f} strategy={strategy_name}")

    def objective(params):
        tp, eps = params
        res = evaluate_inference(tp, eps, window, drift_delta, human_teams, bot_records)
        return -res['inference_ll']

    x0 = [best['tau_prior'], best['epsilon']]
    bounds = [(0.01, 50.0), (0.001, 0.999)]

    opt = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 50, 'ftol': 1e-6})

    opt_tp, opt_eps = opt.x
    print(f"  Optimal: tau_prior={opt_tp:.4f} eps={opt_eps:.6f}")
    print(f"  objective={-opt.fun:.4f}")

    opt_res = evaluate_inference(opt_tp, opt_eps, window, drift_delta, human_teams, bot_records)
    opt_res.update({
        'tau_prior': float(opt_tp),
        'epsilon': float(opt_eps),
        'strategy_name': strategy_name,
        'window': window,
        'drift_delta': drift_delta,
    })

    polished = [opt_res]
    save_checkpoint(str(POLISHED_PATH), polished)
    return opt_res


# ── Main ───────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Stage 1: Inference Parameter Tuning")
    print("=" * 60)

    human_teams, bot_records = load_data()

    # Phase 1: Coarse
    print("\n--- Phase 1: Coarse Grid Search ---")
    coarse_results = run_coarse_search(human_teams, bot_records)

    # Phase 2: Refined
    print("\n--- Phase 2: Refined Grid Search ---")
    refined_results = run_refined_search(human_teams, bot_records, coarse_results)

    # Combine all results for best selection
    all_results = coarse_results + refined_results

    # Phase 3: Scipy polish
    print("\n--- Phase 3: Scipy L-BFGS-B Polish ---")
    polished = run_scipy_polish(human_teams, bot_records, all_results)

    # Pick overall best
    all_results.append(polished)
    best = pick_best(all_results, 'inference_ll')

    output = {
        'tau_prior': best['tau_prior'],
        'epsilon': best['epsilon'],
        'memory_strategy': best['strategy_name'],
        'window': best['window'],
        'drift_delta': best['drift_delta'],
        'accuracy': best['accuracy'],
        'inference_ll': best['inference_ll'],
        'n': best['n'],
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Stage 1 Results:")
    print(f"  tau_prior = {output['tau_prior']:.4f}")
    print(f"  epsilon = {output['epsilon']:.6f}")
    print(f"  memory_strategy = {output['memory_strategy']}")
    print(f"  window = {output['window']}")
    print(f"  drift_delta = {output['drift_delta']}")
    print(f"  accuracy = {output['accuracy']:.3f}")
    print(f"  inference_ll = {output['inference_ll']:.4f}")
    print(f"  n = {output['n']}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
