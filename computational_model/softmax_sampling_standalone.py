"""
Standalone file for verifying the softmax role-sampling logic.

Context:
    We have a 3-player cooperative game (boss fight) where each agent picks one
    of 3 roles: Attacker (0), Defender (1), Healer (2). The game state is a flat
    index encoding three quantities:
        - intent:   whether the boss is attacking this turn (0 = no, 1 = yes)
        - team_hp:  current team HP
        - enemy_hp: current enemy/boss HP

    A value table V[role_combo, intent, team_hp, enemy_hp] has been pre-computed
    via dynamic programming (value iteration over the MDP). It gives the expected
    future reward for a given role assignment in a given state.

    Each agent maintains a joint belief (prior) over all three agents' roles as a
    (3, 3, 3) tensor: role_prior[r0, r1, r2] = P(agent0=r0, agent1=r1, agent2=r2).

    This function answers: "Given my beliefs about what roles the other two agents
    might be playing, what is the expected value of ME choosing each role?"
    It then converts those expected values into a probability distribution via softmax.
"""

import jax
import jax.numpy as jnp

# === Constants (from config.py) ===
ATTACKER, DEFENDER, HEALER = 0, 1, 2
ROLES = [ATTACKER, DEFENDER, HEALER]

# State dimensions (these depend on the environment config)
TEAM_MAX_HP = 15
ENEMY_MAX_HP = 30
NUM_TEAM_HP_STATES = TEAM_MAX_HP + 1    # 16 (0..15)
NUM_ENEMY_HP_STATES = ENEMY_MAX_HP + 1  # 31 (0..30)


@jax.jit
def get_intent_and_hps_from_state(s):
    """
    Decode a flat state index back into its three components.
    The state space is flattened as:
        s = intent * (NUM_TEAM_HP_STATES * NUM_ENEMY_HP_STATES) + team_hp * NUM_ENEMY_HP_STATES + enemy_hp
    """
    i = s // (NUM_TEAM_HP_STATES * NUM_ENEMY_HP_STATES)
    rem = s % (NUM_TEAM_HP_STATES * NUM_ENEMY_HP_STATES)
    return i, rem // NUM_ENEMY_HP_STATES, rem % NUM_ENEMY_HP_STATES


def softmax_dist_over_roles(i, s, role_prior, values_array):
    """
    Calculates the softmax distribution over roles for agent i,
    based on the expected value of those roles given beliefs about others.

    Args:
        i: Agent index (0, 1, or 2).
        s: Flat state index encoding (intent, team_hp, enemy_hp).
        role_prior: A (3, 3, 3) array representing the joint belief distribution
                    over all three agents' roles. role_prior[r0, r1, r2] = P(roles = (r0, r1, r2)).
        values_array: Pre-computed V table of shape (27, 2, H, W).
                      Index 0 is the flat role combo index (r0*9 + r1*3 + r2),
                      index 1 is intent (0 or 1), index 2 is team_hp, index 3 is enemy_hp.

    Returns:
        A (3,) array: softmax probabilities over roles [attacker, defender, healer] for agent i.
    """
    other_agents = [a for a in range(3) if a != i]
    intent, team_hp, enemy_hp = get_intent_and_hps_from_state(s)

    # Marginalize out agent i's axis from the joint prior to get a (3, 3) distribution
    # over the other two agents' roles. E.g. if i=0, sum over axis 0 to get P(r1, r2).
    other_probs = jnp.sum(role_prior, axis=i)
    other_probs = other_probs / jnp.sum(other_probs)

    # For each candidate role r_i that agent i could take, compute the expected value
    # by averaging V over all possible role assignments of the other two agents,
    # weighted by the belief probabilities.
    #
    # EV(r_i) = sum_{r_j, r_k} P(others = r_j, r_k) * V[(r_i, r_j, r_k), state]
    expected_values = jnp.zeros(3)

    for r_i in ROLES:
        ev = 0.0
        for r_j in ROLES:
            for r_k in ROLES:
                # Build the full 3-agent role assignment, placing r_i at position i
                # and r_j, r_k at the other two positions (in agent-index order)
                curr_roles = [0] * 3
                curr_roles[i] = r_i
                curr_roles[other_agents[0]] = r_j
                curr_roles[other_agents[1]] = r_k

                # Map the 3-tuple of roles to a flat index 0..26
                # using base-3 encoding: index = r0*9 + r1*3 + r2
                flat_role_idx = curr_roles[0] * 9 + curr_roles[1] * 3 + curr_roles[2]

                weight = other_probs[r_j, r_k]
                val = values_array[flat_role_idx, intent, team_hp, enemy_hp]
                ev += weight * val

        expected_values = expected_values.at[r_i].set(ev)

    # Convert expected values to a probability distribution via softmax.
    # Higher expected value roles get higher probability, but all roles
    # retain some chance of being selected (soft rather than argmax).
    return jax.nn.softmax(expected_values)
