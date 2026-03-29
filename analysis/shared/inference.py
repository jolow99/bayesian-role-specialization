"""Core game inference engine: priors, Bayesian updates, action model, game mechanics.

These functions are used by both bayesian-value and bayesian-belief models,
and by any future model that reasons about role choices in the game.

Ported from computational_model/analysis/online_model_sim.py.
"""

from __future__ import annotations

import numpy as np

from .constants import F, T, M, BLOCK, ROLE_CHAR_TO_IDX, ROLE_SHORT

# Action indices (DEFEND is an alias for BLOCK, used in model code)
ATTACK, DEFEND, HEAL = 0, 1, 2


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

def utility_based_prior(player_stats, tau=1.0):
    """P(r0,r1,r2) proportional to exp(sum_i stat_i(r_i) / tau).

    Args:
        player_stats: (3, 3) array — player_stats[player, stat_col].
            stat_col 0 = STR (Fighter), 1 = DEF (Tank), 2 = SUP (Medic).
        tau: Temperature. Higher = flatter prior.

    Returns:
        (3, 3, 3) array — joint prior over role combos.
    """
    # ROLE_STAT_COL: role index -> which stat column matters
    ROLE_STAT_COL = {0: 0, 1: 1, 2: 2}

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


def uniform_prior():
    """Flat 1/27 prior over all role combinations."""
    return np.ones((3, 3, 3)) / 27.0


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

def preferred_action(role, intent, team_hp, team_max_hp):
    """Deterministic action a player would take given their role and game state.

    Fighter always attacks. Tank blocks when enemy attacks, else attacks.
    Medic heals when team HP < max, else attacks.
    """
    if role == F:
        return ATTACK
    elif role == T:
        return DEFEND if intent == 1 else ATTACK
    else:
        return HEAL if team_hp < team_max_hp else ATTACK


def action_prob(role, action, intent, team_hp, team_max_hp, epsilon=1e-10):
    """Probability of observing an action given a role.

    With probability (1 - epsilon) the player takes the preferred action,
    otherwise uniform over the other 2 actions.
    """
    pref = preferred_action(role, intent, team_hp, team_max_hp)
    return (1.0 - epsilon) if action == pref else (epsilon / 2.0)


# ---------------------------------------------------------------------------
# Bayesian update
# ---------------------------------------------------------------------------

def bayesian_update(prior, actions, intent, team_hp, team_max_hp, epsilon=1e-10):
    """Update joint posterior given observed actions for all 3 players.

    Args:
        prior: (3, 3, 3) array — current joint distribution over role combos.
        actions: Length-3 list of action indices for each player.
        intent: 0 or 1 — whether enemy attacks this turn.
        team_hp: Current team HP.
        team_max_hp: Maximum team HP.
        epsilon: Action noise parameter.

    Returns:
        (3, 3, 3) array — updated posterior.
    """
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


# ---------------------------------------------------------------------------
# Game mechanics
# ---------------------------------------------------------------------------

def game_step(intent, team_hp, enemy_hp, actions, player_stats, boss_damage, team_max_hp):
    """Advance game by one turn: compute new HP values.

    Args:
        intent: 0 or 1 — whether enemy attacks.
        team_hp: Current team HP (float).
        enemy_hp: Current enemy HP (float).
        actions: Length-3 list of action indices.
        player_stats: (3, 3) array — player_stats[player, stat_col].
        boss_damage: Base boss damage.
        team_max_hp: Maximum team HP.

    Returns:
        (new_team_hp, new_enemy_hp) tuple.
    """
    total_attack = sum(float(player_stats[i, 0]) for i in range(3) if actions[i] == ATTACK)
    defenders = [float(player_stats[i, 1]) for i in range(3) if actions[i] == DEFEND]
    max_defense = max(defenders) if defenders else 0.0
    total_heal = sum(float(player_stats[i, 2]) for i in range(3) if actions[i] == HEAL)

    new_enemy_hp = max(0.0, enemy_hp - total_attack)
    damage = max(0.0, boss_damage - max_defense) if intent == 1 else 0.0
    new_team_hp = max(0.0, min(float(team_max_hp), team_hp - damage + total_heal))
    return new_team_hp, new_enemy_hp


# ---------------------------------------------------------------------------
# Softmax role distribution (used by bayesian-value model)
# ---------------------------------------------------------------------------

def softmax_role_dist(agent_i, intent, team_hp, enemy_hp, prior, values, tau=1.0):
    """Compute softmax distribution over roles for one agent.

    Marginalizes over other agents' roles using the posterior,
    computes expected value per role, then applies softmax.

    Args:
        agent_i: Agent index (0, 1, or 2).
        intent: 0 or 1 — enemy attack intent.
        team_hp: Current team HP (int, clamped to value table range).
        enemy_hp: Current enemy HP (int, clamped to value table range).
        prior: (3, 3, 3) joint posterior.
        values: (27, 2, H, W) value table.
        tau: Softmax temperature.

    Returns:
        Length-3 array — probability of each role for this agent.
    """
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


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def combo_marginal(combo):
    """Role frequencies from a combo string. 'FTM' -> [1/3, 1/3, 1/3]."""
    counts = np.zeros(3)
    for c in combo:
        counts[ROLE_CHAR_TO_IDX[c]] += 1
    return counts / 3.0
