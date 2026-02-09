"""
mechanics.py: Game dynamics and helper functions for simulation.
"""

import jax
import jax.numpy as jnp
from config import (
    ACTIONS, ROLES, ENEMY_ATTACK_INTENT, ENEMY_NO_ATTACK_INTENT,
    ATTACK, DEFEND, HEAL, ATTACK_STATS, DEFEND_STATS, HEAL_STATS,
    BOSS_DAMAGE, TEAM_MAX_HP, ENEMY_ATTACK_PROB,
    get_intent_and_hps_from_state
)

from pi import choose_policy

def softmax_dist_over_roles(i, s, role_prior, values_array):
    """
    Calculates the softmax distribution over roles for agent i,
    based on the expected value of those roles given beliefs about others.
    
    values_array: The pre-computed V table of shape (27, 2, H, W).
    """
    other_agents = [a for a in range(3) if a != i]
    intent, team_hp, enemy_hp = get_intent_and_hps_from_state(s)

    # Marginalize prior to get probabilities of other agents' roles
    other_probs = jnp.sum(role_prior, axis=i)
    other_probs = other_probs / jnp.sum(other_probs)

    expected_values = jnp.zeros(3)

    for r_i in ROLES:
        ev = 0.0
        # Iterate over all possible roles for the other two agents
        for r_j in ROLES:
            for r_k in ROLES:
                # Construct the role index for the V-table
                # Logic assumes standard product ordering (000, 001, ... 222)
                # index = p0*9 + p1*3 + p2
                curr_roles = [0] * 3
                curr_roles[i] = r_i
                curr_roles[other_agents[0]] = r_j
                curr_roles[other_agents[1]] = r_k
                
                # Convert role combo to flat index 0-26
                flat_role_idx = curr_roles[0] * 9 + curr_roles[1] * 3 + curr_roles[2]
                
                weight = other_probs[r_j, r_k]
                val = values_array[flat_role_idx, intent, team_hp, enemy_hp]
                ev += weight * val

        expected_values = expected_values.at[r_i].set(ev)

    return jax.nn.softmax(expected_values)

def choose_action(role, intent, team_hp, key):
    """Selects an action based on the role's policy."""
    # Compute action probabilities
    probs = jnp.array([
        choose_policy(role, ATTACK, intent, team_hp),
        choose_policy(role, DEFEND, intent, team_hp),
        choose_policy(role, HEAL, intent, team_hp),
    ])

    # Normalize to ensure sum is 1.0
    probs = probs / jnp.sum(probs)
    action = jax.random.choice(key, len(ACTIONS), p=probs)
    return action

def game_step(intent, team_hp, enemy_hp, action_profile, key):
    """
    Advances the game state based on actions and mechanics.
    Replicates logic from MarkovGame.ipynb.
    """
    total_attack = jnp.sum(jnp.where((action_profile == ATTACK), ATTACK_STATS, 0))
    max_defense  = jnp.max(jnp.where((action_profile == DEFEND), DEFEND_STATS, 0))
    total_heal   = jnp.sum(jnp.where((action_profile == HEAL), HEAL_STATS, 0))

    new_enemy_hp = jnp.maximum(0, enemy_hp - total_attack)

    damage_incoming = jnp.where(
        intent == ENEMY_ATTACK_INTENT, 
        jnp.maximum(0, BOSS_DAMAGE - max_defense), 
        0
    )

    new_team_hp = team_hp - damage_incoming + total_heal
    new_team_hp = jnp.maximum(0, jnp.minimum(TEAM_MAX_HP, new_team_hp))

    intent_key, _ = jax.random.split(key)
    new_intent = jnp.where(
        jax.random.uniform(intent_key) < ENEMY_ATTACK_PROB, 
        ENEMY_ATTACK_INTENT, 
        ENEMY_NO_ATTACK_INTENT
    )

    return new_intent, new_team_hp, new_enemy_hp