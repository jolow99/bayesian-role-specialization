"""Updater belief chain (stat prior P_0) with both response-model inputs recorded:

  mixed    (1 − β)·P_N + β·P_0   (paper; what the next stage's update starts from)
  unmixed  P_N, the posterior after the previous stage's updates, before the β-mix.

Same recursion as 2026-11-02 models.beliefs.team_states (Bayesian updates at fixed τ_p, ε;
β-mix toward P_0 between stages); the recursion always carries the mixed belief forward,
so "unmixed" only changes what the response model sees. At stage 1 both equal P_0.
"""

from __future__ import annotations

import numpy as np

from models.beliefs import expected_values, own_marginal
from shared.constants import ROLE_CHAR_TO_IDX, TURNS_PER_STAGE
from shared.inference import bayesian_update, game_step, preferred_action, utility_based_prior


def team_states(rec, tau, eps, beta):
    env, lds = rec["env_config"], rec["lds"]
    stats, max_hp, max_ehp = env["player_stats"], env["team_max_hp"], env["enemy_max_hp"]
    prior = utility_based_prior(stats, tau=tau)
    current, unmixed, hp, ehp, t = prior.copy(), prior.copy(), float(max_hp), float(max_ehp), 0
    states = []
    for combo in rec["stage_roles"]:
        if t >= len(lds) or hp <= 0 or ehp <= 0:
            break
        states.append({"mixed": current.copy(), "unmixed": unmixed.copy(), "prior": prior,
                       "intent": lds[t],
                       "thp": int(min(max(0, hp), max_hp)), "ehp": int(min(max(0, ehp), max_ehp))})
        roles = [ROLE_CHAR_TO_IDX[c] for c in combo]
        for _ in range(TURNS_PER_STAGE):
            if t >= len(lds) or hp <= 0 or ehp <= 0:
                break
            actions = [preferred_action(r, lds[t], hp, max_hp) for r in roles]
            current = bayesian_update(current, actions, lds[t], hp, max_hp, eps)
            hp, ehp = game_step(lds[t], hp, ehp, actions, stats, env["boss_damage"], max_hp)
            t += 1
        unmixed = current / current.sum()
        current = (1.0 - beta) * current + beta * prior
        current = current / current.sum()
    return states


def all_states(records, params, beta=None):
    b = params["beta"] if beta is None else beta
    return [team_states(rec, params["tau_prior"], params["epsilon"], b) for rec in records]


def stat_beliefs(records, ob, kind, states):
    """(PM, EV), each (n, 3): every row on the updater chain's `kind` belief (mixed | unmixed)."""
    PM, EV = np.empty((ob.n, 3)), np.empty((ob.n, 3))
    for q, (ri, s, pos) in enumerate(zip(ob.ri, ob.s, ob.pos)):
        st = states[ri][s]
        v = records[ri]["env_config"]["values"][:, st["intent"], st["thp"], st["ehp"]]
        PM[q], EV[q] = own_marginal(st[kind], pos), expected_values(st[kind], pos, v)
    return PM, EV
