import jax
import jax.numpy as jnp
from config import HORIZON, get_intent_and_hps_from_state

@jax.jit
def R(s, t):
    """
    Reward function: Returns reward for reaching state s at time t.
    Returns 100 * (t/HORIZON) if team alive and enemy dead 
    (recursing back in time because that's what memo does)
    """
    _, team_hp, enemy_hp = get_intent_and_hps_from_state(s)
    return ((team_hp != 0) & (enemy_hp == 0)) * 100 * (t / HORIZON)


@jax.jit
def is_terminal(s):
    """Check if state is terminal (either team or enemy has 0 HP)."""
    _, team_hp, enemy_hp = get_intent_and_hps_from_state(s)
    return jnp.logical_or(team_hp == 0, enemy_hp == 0)