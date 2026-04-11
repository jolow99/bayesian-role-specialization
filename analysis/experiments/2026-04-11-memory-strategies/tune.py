"""Memory strategy sweep: dense drift, exponential tempering, uniform drift.

Extends the stage1 memory-strategy space with:
  - drift_prior_δ on a denser grid (21 points, 0..1)
  - drift_uniform_δ (11 points, 0..1): mix toward uniform instead of π₀
  - temper_γ (19 points, 0.05..0.95): geometric decay via π ← π^γ at stage boundaries
Plus baselines: full, window_1..4.

For each strategy, re-optimizes (tau_prior, epsilon) over a 6x6 grid so every
strategy is scored at its own best hyperparameters. Metric: mean inference
log-likelihood (same as stage 1).
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent

from shared import EXPORTS_DIR
from shared.constants import (
    ROLE_SHORT, GAME_ROLE_TO_IDX, TURNS_PER_STAGE,
)
from shared.data_loading import load_all_exports
from shared.inference import (
    utility_based_prior, bayesian_update, preferred_action, game_step,
)

RESULTS_PATH = SCRIPT_DIR / 'results.json'
SUMMARY_PATH = SCRIPT_DIR / 'summary.json'


# ── Strategies ─────────────────────────────────────────────────────

def build_strategies():
    strats = [{'name': 'full', 'kind': 'full', 'param': None}]
    for w in [1, 2, 3, 4]:
        strats.append({'name': f'window_{w}', 'kind': 'window', 'param': w})
    for d in np.linspace(0.0, 1.0, 21)[1:]:  # skip 0 (== full)
        strats.append({'name': f'drift_prior_{d:.3f}', 'kind': 'drift_prior',
                       'param': float(d)})
    for d in np.linspace(0.0, 1.0, 11)[1:]:
        strats.append({'name': f'drift_uniform_{d:.3f}', 'kind': 'drift_uniform',
                       'param': float(d)})
    for g in np.linspace(0.05, 0.95, 19):  # skip 1.0 (== full)
        strats.append({'name': f'temper_{g:.3f}', 'kind': 'temper',
                       'param': float(g)})
    return strats


# ── Posterior computation ─────────────────────────────────────────

def compute_posteriors(role_seq, eis, player_stats, boss_damage,
                       team_max_hp, enemy_max_hp,
                       tau_prior, epsilon, strategy):
    """Return posteriors[s] = belief state at the start of stage s (inference time)."""
    n_stages = len(role_seq)
    original_prior = utility_based_prior(player_stats, tau=tau_prior)
    uniform = np.ones_like(original_prior) / original_prior.size
    kind = strategy['kind']
    param = strategy['param']

    posteriors = [original_prior.copy()]

    if kind == 'window':
        w = param
        # Need HP at start of each stage to replay a window
        team_hp_per_stage = [float(team_max_hp)]
        running_hp = float(team_max_hp)
        for s in range(n_stages):
            roles = role_seq[s]
            hp = running_hp
            for turn_offset in range(TURNS_PER_STAGE):
                turn_idx = s * TURNS_PER_STAGE + turn_offset
                if turn_idx >= len(eis) or hp <= 0:
                    break
                intent = int(eis[turn_idx])
                actions = [preferred_action(roles[i], intent, hp, team_max_hp)
                           for i in range(3)]
                hp, _ = game_step(intent, hp, float(enemy_max_hp),
                                  actions, player_stats, boss_damage, team_max_hp)
            running_hp = hp
            team_hp_per_stage.append(hp)

        for s in range(1, n_stages + 1):
            start = max(0, s - w)
            post = original_prior.copy()
            hp = team_hp_per_stage[start]
            for ws in range(start, s):
                roles = role_seq[ws]
                for turn_offset in range(TURNS_PER_STAGE):
                    turn_idx = ws * TURNS_PER_STAGE + turn_offset
                    if turn_idx >= len(eis) or hp <= 0:
                        break
                    intent = int(eis[turn_idx])
                    actions = [preferred_action(roles[i], intent, hp, team_max_hp)
                               for i in range(3)]
                    post = bayesian_update(post, actions, intent, hp,
                                           team_max_hp, epsilon)
                    hp, _ = game_step(intent, hp, float(enemy_max_hp),
                                      actions, player_stats, boss_damage,
                                      team_max_hp)
            posteriors.append(post)
        return posteriors

    # Running posterior with optional boundary op
    current = original_prior.copy()
    hp = float(team_max_hp)
    ehp = float(enemy_max_hp)
    for s in range(n_stages):
        roles = role_seq[s]
        for turn_offset in range(TURNS_PER_STAGE):
            turn_idx = s * TURNS_PER_STAGE + turn_offset
            if turn_idx >= len(eis) or hp <= 0:
                break
            intent = int(eis[turn_idx])
            actions = [preferred_action(roles[i], intent, hp, team_max_hp)
                       for i in range(3)]
            current = bayesian_update(current, actions, intent, hp,
                                      team_max_hp, epsilon)
            hp, ehp = game_step(intent, hp, ehp, actions,
                                player_stats, boss_damage, team_max_hp)

        if kind == 'drift_prior':
            current = (1 - param) * current + param * original_prior
        elif kind == 'drift_uniform':
            current = (1 - param) * current + param * uniform
        elif kind == 'temper':
            log_cur = np.log(np.clip(current, 1e-300, None))
            log_cur = param * log_cur
            log_cur -= log_cur.max()
            current = np.exp(log_cur)
        # kind == 'full' → no boundary op

        total = current.sum()
        if total > 0:
            current = current / total
        posteriors.append(current.copy())

    return posteriors


# ── Data prep ──────────────────────────────────────────────────────

def _stats_from_id(stat_profile_id):
    parts = stat_profile_id.split('_')
    return np.array([[int(c) for c in part] for part in parts])


def prepare_team_data(human_teams):
    prepared = []
    for _, team_prs in human_teams.items():
        rnd = team_prs[0].round
        config = rnd.config
        player_stats = _stats_from_id(rnd.stat_profile_id)
        boss_damage = config.get('bossDamage', 2)
        team_max_hp = config.get('maxTeamHealth', 15)
        enemy_max_hp = config.get('maxEnemyHealth', 30)
        eis = rnd.enemy_intent_sequence

        player_roles = {}
        for pr in team_prs:
            player_roles[pr.player_id] = [s.role_idx for s in pr.round.stages]

        n_stages = max(len(rs) for rs in player_roles.values())
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
                    if (target_pos not in player_roles or
                            si - 1 >= len(player_roles[target_pos])):
                        continue
                    true_role = player_roles[target_pos][si - 1]
                    queries.append((si, target_pos, inferred_role, true_role))

        prepared.append({
            'player_stats': player_stats,
            'boss_damage': boss_damage,
            'team_max_hp': team_max_hp,
            'enemy_max_hp': enemy_max_hp,
            'eis': eis,
            'role_seq': role_seq,
            'queries': queries,
        })
    return prepared


def prepare_bot_data(bot_records):
    prepared = []
    for pr in bot_records:
        rnd = pr.round
        config = rnd.config
        player_stats = _stats_from_id(rnd.stat_profile_id)
        boss_damage = config.get('bossDamage', 2)
        team_max_hp = config.get('maxTeamHealth', 15)
        enemy_max_hp = config.get('maxEnemyHealth', 30)
        eis = rnd.enemy_intent_sequence
        n_stages = len(rnd.stages)

        human_pos = config.get('humanRole', 0)
        bot_players = config.get('botPlayers', [])
        bot_role_map = {}
        for bp in bot_players:
            bot_pos = bp.get('position', bp.get('playerIndex'))
            bot_role_str = bp.get('strategy', {}).get('role', '')
            if bot_role_str in GAME_ROLE_TO_IDX:
                bot_role_map[bot_pos] = GAME_ROLE_TO_IDX[bot_role_str]

        role_seq = []
        for s in range(n_stages):
            roles = [0, 0, 0]
            roles[human_pos] = rnd.stages[s].role_idx
            for pos, ridx in bot_role_map.items():
                roles[pos] = ridx
            role_seq.append(roles)

        all_roles = {human_pos: [s.role_idx for s in rnd.stages]}
        for pos, ridx in bot_role_map.items():
            all_roles[pos] = [ridx] * n_stages

        queries = []
        for si, stage in enumerate(rnd.stages):
            if si == 0 or not stage.inferred_roles:
                continue
            for target_pos, inferred_role in stage.inferred_roles.items():
                if (target_pos not in all_roles or
                        si - 1 >= len(all_roles[target_pos])):
                    continue
                true_role = all_roles[target_pos][si - 1]
                queries.append((si, target_pos, inferred_role, true_role))

        prepared.append({
            'player_stats': player_stats,
            'boss_damage': boss_damage,
            'team_max_hp': team_max_hp,
            'enemy_max_hp': enemy_max_hp,
            'eis': eis,
            'role_seq': role_seq,
            'queries': queries,
        })
    return prepared


def load_prepared():
    records = load_all_exports()
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
    human_teams = {k: sorted(v, key=lambda p: p.player_id)
                   for k, v in human_teams.items() if len(v) == 3}
    print(f"Loaded {len(human_teams)} human teams, {len(bot_records)} bot records",
          flush=True)
    return prepare_team_data(human_teams) + prepare_bot_data(bot_records)


# ── Evaluation ─────────────────────────────────────────────────────

def evaluate(prepared, tau_prior, epsilon, strategy):
    correct, total = 0, 0
    log_liks = []
    for data in prepared:
        posteriors = compute_posteriors(
            data['role_seq'], data['eis'], data['player_stats'],
            data['boss_damage'], data['team_max_hp'], data['enemy_max_hp'],
            tau_prior, epsilon, strategy)
        for si, target_pos, inferred_role, true_role in data['queries']:
            if si >= len(posteriors):
                continue
            post = posteriors[si]
            marg = np.sum(post, axis=tuple(j for j in range(3) if j != target_pos))
            t = marg.sum()
            if t > 0:
                marg = marg / t
            total += 1
            if int(np.argmax(marg)) == true_role:
                correct += 1
            log_liks.append(np.log(max(marg[inferred_role], 1e-20)))
    return {
        'accuracy': correct / total if total > 0 else 0.0,
        'inference_ll': float(np.mean(log_liks)) if log_liks else float('nan'),
        'n': total,
    }


# ── Main sweep ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Memory strategy sweep")
    print("=" * 60)

    prepared = load_prepared()
    print(f"Prepared {len(prepared)} records, "
          f"{sum(len(d['queries']) for d in prepared)} inference queries", flush=True)

    strategies = build_strategies()
    print(f"{len(strategies)} strategies to evaluate", flush=True)

    tau_vals = np.linspace(0.5, 15.0, 6)
    eps_vals = np.linspace(0.1, 0.9, 6)

    results = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    done = {(r['strategy'], r['tau_prior'], r['epsilon']) for r in results}

    total_evals = len(strategies) * len(tau_vals) * len(eps_vals)
    count = len(done)
    print(f"Grid per strategy: {len(tau_vals)} tau_prior x {len(eps_vals)} epsilon "
          f"= {len(tau_vals)*len(eps_vals)} points")
    print(f"Total: {total_evals} (already done: {count})\n", flush=True)

    for si, strat in enumerate(strategies):
        strat_added = False
        for tp in tau_vals:
            for eps in eps_vals:
                key = (strat['name'], float(tp), float(eps))
                if key in done:
                    continue
                res = evaluate(prepared, float(tp), float(eps), strat)
                results.append({
                    'strategy': strat['name'],
                    'kind': strat['kind'],
                    'param': strat['param'],
                    'tau_prior': float(tp),
                    'epsilon': float(eps),
                    **res,
                })
                count += 1
                strat_added = True
        if strat_added:
            with open(RESULTS_PATH, 'w') as f:
                json.dump(results, f, indent=2)
        rs = [r for r in results if r['strategy'] == strat['name']]
        if rs:
            best = max(rs, key=lambda r: r['inference_ll'])
            print(f"  [{si+1}/{len(strategies)}] {strat['name']:<24} "
                  f"LL={best['inference_ll']:.4f} "
                  f"acc={best['accuracy']:.3f} "
                  f"tp={best['tau_prior']:.2f} eps={best['epsilon']:.2f}",
                  flush=True)

    # Summary: best per strategy
    by_strat = defaultdict(list)
    for r in results:
        by_strat[r['strategy']].append(r)
    summary = [max(rs, key=lambda r: r['inference_ll']) for rs in by_strat.values()]
    summary.sort(key=lambda r: -r['inference_ll'])

    with open(SUMMARY_PATH, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}\nTop 15 strategies by inference LL:\n")
    print(f"{'strategy':<26}{'LL':>10}{'acc':>8}{'tp':>8}{'eps':>8}")
    for r in summary[:15]:
        print(f"{r['strategy']:<26}{r['inference_ll']:>10.4f}"
              f"{r['accuracy']:>8.3f}{r['tau_prior']:>8.2f}{r['epsilon']:>8.2f}")


if __name__ == '__main__':
    main()
