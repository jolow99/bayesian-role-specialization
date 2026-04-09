"""Shared code for Stage 2 model tuning scripts.

Handles:
- Loading stage 1 params
- Data loading with OMS monkey-patching
- Precomputing trajectories with memory strategy
- Switch-stage filtering
- Metric computation helpers
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

from shared import EXPORTS_DIR, DATA_ROOT
from shared.constants import (
    F, T, M, ROLE_SHORT, ROLE_CHAR_TO_IDX, GAME_ROLE_TO_IDX,
    ALL_ROLE_COMBOS, TURNS_PER_STAGE,
)
from shared.parsing import canonical_combo
from shared.inference import (
    utility_based_prior, bayesian_update, preferred_action, game_step,
    softmax_role_dist, combo_marginal,
)
from shared.evaluation import (
    run_predictions, compute_pearson, compute_log_likelihood,
)

# Monkey-patch OMS for data loading
OMS_DIR = Path(DATA_ROOT).parent.parent / 'computational_model' / 'analysis'
sys.path.insert(0, str(OMS_DIR))
import online_model_sim as oms
import re as _re

oms.VALUE_MATRICES_DIR = DATA_ROOT / 'human_envs_value_matrices'
oms.ENVS_DIR = DATA_ROOT / 'envs'


def _load_config_no_jax(config_path):
    text = Path(config_path).read_text()
    team_max_hp = int(_re.search(r'TEAM_MAX_HP\s*=\s*(\d+)', text).group(1))
    enemy_max_hp = int(_re.search(r'ENEMY_MAX_HP\s*=\s*(\d+)', text).group(1))
    boss_damage = float(_re.search(r'BOSS_DAMAGE\s*=\s*([\d.]+)', text).group(1))
    ps_match = _re.search(
        r'PLAYER_STATS\s*=\s*(?:jnp\.array|np\.array)?\(?\s*(\[\[.+?\]\])\s*\)?',
        text, _re.DOTALL)
    rows = _re.findall(r'\[([^\[\]]+)\]', ps_match.group(1))
    player_stats = np.array([[float(x) for x in row.split(',')] for row in rows])

    class Config:
        pass

    cfg = Config()
    cfg.TEAM_MAX_HP = team_max_hp
    cfg.ENEMY_MAX_HP = enemy_max_hp
    cfg.BOSS_DAMAGE = boss_damage
    cfg.PLAYER_STATS = player_stats
    return cfg


oms.load_config_module = _load_config_no_jax

DATA_DIRS = [
    str(EXPORTS_DIR / 'bayesian-role-specialization-2026-03-06-09-54-19'),
    str(EXPORTS_DIR / 'bayesian-role-specialization-2026-03-18-15-47-09'),
]


def load_stage1_params(experiment_dir):
    """Load best inference params from stage 1."""
    path = Path(experiment_dir) / 'stage1_inference' / 'best_inference_params.json'
    with open(path) as f:
        return json.load(f)


def load_records():
    """Load team-round records via OMS."""
    records = oms.load_team_rounds(data_dirs=DATA_DIRS)
    n_envs = len(set(r['env_id'] for r in records))
    print(f'Loaded {len(records)} team-rounds across {n_envs} environments')
    return records


def precompute_trajectories(records, tau_prior, epsilon, window=None, drift_delta=0.0):
    """Precompute posterior + game state per stage, using the memory strategy.

    Returns list of trajectories (one per record), each a list of stage dicts:
    {prior, intent, thp, ehp, human_combo, prev_roles}
    """
    trajectories = []
    for record in records:
        env = record['env_config']
        player_stats = env['player_stats']
        boss_damage = env['boss_damage']
        team_max_hp, enemy_max_hp = env['team_max_hp'], env['enemy_max_hp']

        original_prior = utility_based_prior(player_stats, tau=tau_prior)
        stages = []

        if window is not None:
            # Windowed: need per-stage info to reconstruct
            _precompute_windowed(record, original_prior, epsilon, window,
                                 player_stats, boss_damage, team_max_hp, enemy_max_hp, stages)
        elif drift_delta > 0:
            _precompute_drift(record, original_prior, epsilon, drift_delta,
                              player_stats, boss_damage, team_max_hp, enemy_max_hp, stages)
        else:
            _precompute_full(record, original_prior, epsilon,
                             player_stats, boss_damage, team_max_hp, enemy_max_hp, stages)

        trajectories.append(stages)
    return trajectories


def _precompute_full(record, original_prior, epsilon, player_stats, boss_damage,
                     team_max_hp, enemy_max_hp, stages):
    """Full history posterior computation."""
    team_hp = float(team_max_hp)
    enemy_hp = float(enemy_max_hp)
    prior = original_prior.copy()
    turn_idx = 0
    prev_roles = None

    for human_combo in record['stage_roles']:
        if turn_idx >= len(record['lds']) or team_hp <= 0 or enemy_hp <= 0:
            break

        intent = record['lds'][turn_idx]
        thp = int(min(max(0, team_hp), team_max_hp))
        ehp = int(min(max(0, enemy_hp), enemy_max_hp))

        stages.append({
            'prior': prior.copy(),
            'intent': intent,
            'thp': thp,
            'ehp': ehp,
            'human_combo': human_combo,
            'prev_roles': prev_roles,
        })

        human_roles = [ROLE_CHAR_TO_IDX[c] for c in human_combo]
        prev_roles = list(human_roles)
        for _ in range(TURNS_PER_STAGE):
            if turn_idx >= len(record['lds']) or team_hp <= 0 or enemy_hp <= 0:
                break
            intent_t = record['lds'][turn_idx]
            actions = [preferred_action(human_roles[i], intent_t, team_hp, team_max_hp)
                       for i in range(3)]
            prior = bayesian_update(prior, actions, intent_t, team_hp, team_max_hp, epsilon)
            team_hp, enemy_hp = game_step(intent_t, team_hp, enemy_hp, actions,
                                          player_stats, boss_damage, team_max_hp)
            turn_idx += 1


def _precompute_drift(record, original_prior, epsilon, drift_delta, player_stats,
                      boss_damage, team_max_hp, enemy_max_hp, stages):
    """Drift memory strategy: mix posterior with prior at stage boundaries."""
    team_hp = float(team_max_hp)
    enemy_hp = float(enemy_max_hp)
    current = original_prior.copy()
    turn_idx = 0
    prev_roles = None

    for human_combo in record['stage_roles']:
        if turn_idx >= len(record['lds']) or team_hp <= 0 or enemy_hp <= 0:
            break

        intent = record['lds'][turn_idx]
        thp = int(min(max(0, team_hp), team_max_hp))
        ehp = int(min(max(0, enemy_hp), enemy_max_hp))

        stages.append({
            'prior': current.copy(),
            'intent': intent,
            'thp': thp,
            'ehp': ehp,
            'human_combo': human_combo,
            'prev_roles': prev_roles,
        })

        human_roles = [ROLE_CHAR_TO_IDX[c] for c in human_combo]
        prev_roles = list(human_roles)
        for _ in range(TURNS_PER_STAGE):
            if turn_idx >= len(record['lds']) or team_hp <= 0 or enemy_hp <= 0:
                break
            intent_t = record['lds'][turn_idx]
            actions = [preferred_action(human_roles[i], intent_t, team_hp, team_max_hp)
                       for i in range(3)]
            current = bayesian_update(current, actions, intent_t, team_hp, team_max_hp, epsilon)
            team_hp, enemy_hp = game_step(intent_t, team_hp, enemy_hp, actions,
                                          player_stats, boss_damage, team_max_hp)
            turn_idx += 1

        # Apply drift at stage boundary
        current = (1 - drift_delta) * current + drift_delta * original_prior
        total = current.sum()
        if total > 0:
            current /= total


def _precompute_windowed(record, original_prior, epsilon, window, player_stats,
                         boss_damage, team_max_hp, enemy_max_hp, stages):
    """Windowed memory strategy: only use last `window` stages of turns."""
    lds = record['lds']
    stage_roles_list = record['stage_roles']
    n_stages = len(stage_roles_list)

    # First pass: compute HP trajectory
    team_hp_seq = [float(team_max_hp)]
    enemy_hp_seq = [float(enemy_max_hp)]
    hp = float(team_max_hp)
    ehp = float(enemy_max_hp)
    turn_idx = 0
    for s in range(n_stages):
        human_roles = [ROLE_CHAR_TO_IDX[c] for c in stage_roles_list[s]]
        for _ in range(TURNS_PER_STAGE):
            if turn_idx >= len(lds) or hp <= 0 or ehp <= 0:
                break
            intent_t = lds[turn_idx]
            actions = [preferred_action(human_roles[i], intent_t, hp, team_max_hp)
                       for i in range(3)]
            hp, ehp = game_step(intent_t, hp, ehp, actions,
                                player_stats, boss_damage, team_max_hp)
            turn_idx += 1
        team_hp_seq.append(hp)
        enemy_hp_seq.append(ehp)

    # Second pass: compute windowed posteriors per stage
    prev_roles = None
    turn_idx = 0
    for s in range(n_stages):
        if turn_idx >= len(lds) or team_hp_seq[s] <= 0 or enemy_hp_seq[s] <= 0:
            break

        # Windowed posterior: replay from max(0, s - window) to s
        start = max(0, s - window)
        post = original_prior.copy()
        replay_turn = start * TURNS_PER_STAGE
        replay_hp = team_hp_seq[start]
        for ws in range(start, s):
            ws_roles = [ROLE_CHAR_TO_IDX[c] for c in stage_roles_list[ws]]
            for _ in range(TURNS_PER_STAGE):
                if replay_turn >= len(lds) or replay_hp <= 0:
                    break
                intent_t = lds[replay_turn]
                actions = [preferred_action(ws_roles[i], intent_t, replay_hp, team_max_hp)
                           for i in range(3)]
                post = bayesian_update(post, actions, intent_t, replay_hp, team_max_hp, epsilon)
                replay_hp, _ = game_step(intent_t, replay_hp, float(enemy_max_hp),
                                         actions, player_stats, boss_damage, team_max_hp)
                replay_turn += 1

        intent = lds[turn_idx] if turn_idx < len(lds) else 0
        thp = int(min(max(0, team_hp_seq[s]), team_max_hp))
        ehp = int(min(max(0, enemy_hp_seq[s]), enemy_max_hp))

        stages.append({
            'prior': post,
            'intent': intent,
            'thp': thp,
            'ehp': ehp,
            'human_combo': stage_roles_list[s],
            'prev_roles': prev_roles,
        })

        human_roles = [ROLE_CHAR_TO_IDX[c] for c in stage_roles_list[s]]
        prev_roles = list(human_roles)
        for _ in range(TURNS_PER_STAGE):
            if turn_idx >= len(lds):
                break
            turn_idx += 1


# ── Switch-stage filtering ─────────────────────────────────────────

def filter_switch_stages(all_results):
    """Keep only predictions where the human combo changed from the previous stage."""
    filtered = {}
    for env_id, data in all_results.items():
        new_team_preds = []
        for team_preds in data['team_predictions']:
            filtered_preds = []
            for s, pred in enumerate(team_preds):
                if s > 0 and pred['human_combo'] != team_preds[s - 1]['human_combo']:
                    filtered_preds.append(pred)
            new_team_preds.append(filtered_preds)

        canon_combos = data['canonical_combos']
        stat_profile = data['stat_profile']
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
                for combo, prob in pred['predicted_dist'].items():
                    stage_predicted[s][canonical_combo(combo, stat_profile)] += prob
                stage_human[s][canonical_combo(pred['human_combo'], stat_profile)] += 1
                stage_model_marg[s] += pred['model_marginal']
                stage_human_marg[s] += combo_marginal(pred['human_combo'])

        if max_stages == 0:
            continue

        predicted_avg, model_marg_avg, human_marg_avg = {}, {}, {}
        for s in range(max_stages):
            n = stage_counts[s]
            if n > 0:
                predicted_avg[s] = {cc: stage_predicted[s].get(cc, 0.0) / n for cc in canon_combos}
                model_marg_avg[s] = stage_model_marg[s] / n
                human_marg_avg[s] = stage_human_marg[s] / n

        filtered[env_id] = dict(data)
        filtered[env_id].update({
            'max_stages': max_stages,
            'stage_predicted': predicted_avg,
            'stage_human': dict(stage_human),
            'stage_counts': dict(stage_counts),
            'team_predictions': new_team_preds,
            'stage_model_marginal': model_marg_avg,
            'stage_human_marginal': human_marg_avg,
        })

    return filtered


# ── Metric computation ─────────────────────────────────────────────

def compute_all_metrics(records, predict_fn):
    """Compute aggregate + switch-stage metrics."""
    full_results = run_predictions(records, predict_fn)

    # Aggregate metrics
    corrs = compute_pearson(full_results)
    ll = compute_log_likelihood(full_results)
    g = corrs.get('__global__', {})
    agg = {
        'combo_r': g.get('combo', {}).get('r', float('nan')),
        'marg_r': g.get('marginal', {}).get('r', float('nan')),
        'mean_ll': float(np.mean([v['mean_ll'] for v in ll.values()])) if ll else float('nan'),
    }

    # Switch-stage metrics
    sw_results = filter_switch_stages(full_results)
    if sw_results:
        sw_corrs = compute_pearson(sw_results)
        sw_ll = compute_log_likelihood(sw_results)
        sw_g = sw_corrs.get('__global__', {})
        switch = {
            'switch_combo_r': sw_g.get('combo', {}).get('r', float('nan')),
            'switch_marg_r': sw_g.get('marginal', {}).get('r', float('nan')),
            'switch_mean_ll': float(np.mean([v['mean_ll'] for v in sw_ll.values()])) if sw_ll else float('nan'),
        }
    else:
        switch = {
            'switch_combo_r': float('nan'),
            'switch_marg_r': float('nan'),
            'switch_mean_ll': float('nan'),
        }

    return {**agg, **switch}


def posterior_marginal(prior, agent_i):
    """Marginalize the (3,3,3) joint posterior to get P(role) for agent_i."""
    marg = np.sum(prior, axis=tuple(j for j in range(3) if j != agent_i))
    total = marg.sum()
    return marg / total if total > 0 else np.ones(3) / 3.0


def build_joint_dist(per_agent):
    """Build joint combo distribution from per-agent distributions."""
    predicted_dist = {}
    for r0 in range(3):
        for r1 in range(3):
            for r2 in range(3):
                combo = ROLE_SHORT[r0] + ROLE_SHORT[r1] + ROLE_SHORT[r2]
                predicted_dist[combo] = float(
                    per_agent[0][r0] * per_agent[1][r1] * per_agent[2][r2])
    return predicted_dist
