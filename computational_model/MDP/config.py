"""
Configuration file containing all environment parameters and constants.
"""

import jax
import jax.numpy as jnp
from itertools import product

ATTACKER    = 0
DEFENDER    = 1
HEALER      = 2

ATTACK      = 0
DEFEND      = 1
HEAL        = 2

ENEMY_NO_ATTACK_INTENT  = 0
ENEMY_ATTACK_INTENT     = 1

ROLES   = [ATTACKER, DEFENDER, HEALER]
ACTIONS = [ATTACK, DEFEND, HEAL]

ROLE_COMBOS     = jnp.array(list(product(ROLES, repeat=3)))
ROLE_INDICES    = jnp.array(list(range(27)))

TEAM_MAX_HP     = 8
ENEMY_MAX_HP    = 16

PLAYER_STATS = jnp.array([
    [4, 1, 1],
    [1, 4, 1],
    [1, 1, 4],
])

ATTACK_STATS    = PLAYER_STATS[:, ATTACK]
DEFEND_STATS    = PLAYER_STATS[:, DEFEND]
HEAL_STATS      = PLAYER_STATS[:, HEAL]

BOSS_DAMAGE         = 4
ENEMY_ATTACK_PROB   = 1.0

EPSILON = 1e-10
HORIZON = 10

H = TEAM_MAX_HP + 1
W = ENEMY_MAX_HP + 1

S = jnp.arange(2 * H * W)

ACTION_PROFILES = jnp.array(list(product(ACTIONS, repeat=3)))
A               = jnp.arange(len(ACTION_PROFILES))

@jax.jit
def get_intent_and_hps_from_state(s):
    i = s // (H * W) # to which 'square' (see above) does s belong to?
    rem = s % (H * W)

    team_hp = rem // W
    enemy_hp = rem % W
    return i, team_hp, enemy_hp