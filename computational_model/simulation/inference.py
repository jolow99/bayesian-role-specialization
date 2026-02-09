# type: ignore
"""
inference.py: Cognitive model for role inference using the memo library.
"""

import jax
from memo import memo
from config import ROLES, ACTIONS, get_intent_and_hps_from_state
from pi import choose_policy

# Helper function to extract element from 3D array (needed for memo)
@jax.jit
def get_element(array, i0, i1, i2):
    return array[i0, i1, i2]

def role_policy(role, action, state):
    """Wrapper to align role policy with inference needs."""
    intent, team_hp, _ = get_intent_and_hps_from_state(state)
    return choose_policy(role, action, intent, team_hp)

@memo
def role_inference[r0: ROLES, r1: ROLES, r2: ROLES](role_prior: ..., obs_a0, obs_a1, obs_a2, s):
    """
    Infers the probability of roles given observed actions and priors.
    Derived from MarkovGame.ipynb.
    """
    # Push array axis variables into observer's frame
    observer: knows(r0, r1, r2) 
    
    observer: thinks[
        # Assign roles to each player weighted by the prior
        team: assigned(r0 in ROLES, r1 in ROLES, r2 in ROLES, 
                       wpp=get_element(role_prior, r0, r1, r2)),
        
        # Choose actions according to roles and current state
        team: chooses(a0 in ACTIONS, wpp=role_policy(r0, a0, s)),
        team: chooses(a1 in ACTIONS, wpp=role_policy(r1, a1, s)),
        team: chooses(a2 in ACTIONS, wpp=role_policy(r2, a2, s))
    ]
    
    # Condition on observations
    observer: observes_that[team.a0 == obs_a0]
    observer: observes_that[team.a1 == obs_a1]
    observer: observes_that[team.a2 == obs_a2]
    
    return observer[Pr[r0 == team.r0 and r1 == team.r1 and r2 == team.r2]]