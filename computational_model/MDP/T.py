"""
Environment dynamics including the transition function.
"""

import jax
import jax.numpy as jnp
from pi import action_profile_prob
from config import (
    ATTACK, DEFEND, HEAL, ENEMY_ATTACK_INTENT, 
    ENEMY_ATTACK_PROB, ENEMY_NO_ATTACK_INTENT,
    TEAM_MAX_HP, BOSS_DAMAGE, ACTION_PROFILES,
    ATTACK_STATS, DEFEND_STATS, HEAL_STATS, get_intent_and_hps_from_state
)

@jax.jit
def T(s, a, s_, r):
    """
    Transition function: T(s, a, s_) = P(s_ | s, a, r)
    """
    # Get action probability
    prob_act    = action_profile_prob(s, a, r)
    a_decoded   = ACTION_PROFILES[a]  # decode joint action

    # Extract current state information
    i, team_hp, enemy_hp = get_intent_and_hps_from_state(s)

    # Calculate action effects
    total_attack    = jnp.sum(jnp.where((a_decoded == ATTACK), ATTACK_STATS, 0))
    max_defense     = jnp.max(jnp.where((a_decoded == DEFEND), DEFEND_STATS, 0))
    total_heal      = jnp.sum(jnp.where((a_decoded == HEAL), HEAL_STATS, 0))

    # Update enemy HP
    new_enemy_hp = jnp.maximum(0, enemy_hp - total_attack)

    # Calculate damage to team (only if enemy attacks)
    damage_incoming = jnp.where(
        i == ENEMY_ATTACK_INTENT, 
        jnp.maximum(0, BOSS_DAMAGE - max_defense), 0)

    # Update team HP
    new_team_hp = team_hp - damage_incoming + total_heal
    new_team_hp = jnp.maximum(0, jnp.minimum(TEAM_MAX_HP, new_team_hp))

    # Extract next state information
    s_intent, s_team_hp, s_enemy_hp = get_intent_and_hps_from_state(s_)

    # Calculate transition probability
    hp_match = (new_team_hp == s_team_hp) & (new_enemy_hp == s_enemy_hp)
    
    intent_prob = (
        (1 - ENEMY_ATTACK_PROB) * (s_intent == ENEMY_NO_ATTACK_INTENT) +
        ENEMY_ATTACK_PROB * (s_intent == ENEMY_ATTACK_INTENT)
    )
    
    return hp_match * intent_prob * prob_act