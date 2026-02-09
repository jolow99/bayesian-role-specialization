import jax
import jax.numpy as jnp
from config import (
    ATTACK, DEFEND, EPSILON, ACTIONS, 
    ENEMY_ATTACK_INTENT, TEAM_MAX_HP, HEAL, 
    ATTACKER, DEFENDER, ACTION_PROFILES, ROLE_COMBOS, get_intent_and_hps_from_state,
)

# ============================================================================
# ROLE POLICIES
# ============================================================================

@jax.jit
def fighter_policy(action):
    """Attacker policy: Always prefer to attack."""
    return jnp.where(
        action == ATTACK,
        1.0 - EPSILON,
        EPSILON / (len(ACTIONS) - 1),
    )


@jax.jit
def defender_policy(action, intent):
    """Defender policy: Defend when enemy attacks, otherwise attack."""
    preferred_action = jnp.where(
        intent == ENEMY_ATTACK_INTENT,
        DEFEND,
        ATTACK,
    )

    return jnp.where(
        action == preferred_action,
        1.0 - EPSILON,
        EPSILON / (len(ACTIONS) - 1),
    )


@jax.jit
def healer_policy(action, team_hp):
    """Healer policy: Heal when team HP is less than full, otherwise attack."""
    preferred_action = jnp.where(
        team_hp < TEAM_MAX_HP,
        HEAL,
        ATTACK,
    )

    return jnp.where(
        action == preferred_action,
        1.0 - EPSILON,
        EPSILON / (len(ACTIONS) - 1),
    )

# ============================================================================
# ACTION PROFILE PROBABILITY
# ============================================================================

@jax.jit
def choose_policy(role, action, intent, team_hp):
    """Select the appropriate policy based on role."""
    return jnp.where(
        role == ATTACKER,
        fighter_policy(action),
        jnp.where(
            role == DEFENDER,
            defender_policy(action, intent),
            healer_policy(action, team_hp),
        ),
    )


@jax.jit
def action_profile_prob(s, a, r):
    """
    Compute the probability of a joint action profile given state and roles.
    """
    intent, team_hp, _ = get_intent_and_hps_from_state(s)
    a_decoded = ACTION_PROFILES[a]  # decode joint action
    r_decoded = ROLE_COMBOS[r]      # decode role profile

    prob = jnp.ones(3)
    prob = prob.at[0].set(choose_policy(r_decoded[0], a_decoded[0], intent, team_hp))
    prob = prob.at[1].set(choose_policy(r_decoded[1], a_decoded[1], intent, team_hp))
    prob = prob.at[2].set(choose_policy(r_decoded[2], a_decoded[2], intent, team_hp))
    
    return jnp.prod(prob)