"""Per-choice beliefs: updaters use the mixed posterior, non-updaters the fixed FTM prior.

For each row: own-role marginal (PS-type models) and expected value of each own
role under the belief about the two teammates (BR-type models).
"""

from __future__ import annotations

import numpy as np

from shared.constants import ROLE_CHAR_TO_IDX, TURNS_PER_STAGE
from shared.inference import bayesian_update, game_step, preferred_action, utility_based_prior


def team_states(rec, tau, eps, beta):
    """Per stage: mixed belief (1 − β)·P_N + β·P_0 and game state at the stage start.

    Drift-prior branch of 2026-05-25 pipeline._precompute_human.
    """
    env, lds = rec["env_config"], rec["lds"]
    stats, max_hp, max_ehp = env["player_stats"], env["team_max_hp"], env["enemy_max_hp"]
    prior = utility_based_prior(stats, tau=tau)
    current, hp, ehp, t = prior.copy(), float(max_hp), float(max_ehp), 0
    states = []
    for combo in rec["stage_roles"]:
        if t >= len(lds) or hp <= 0 or ehp <= 0:
            break
        states.append({"belief": current.copy(), "intent": lds[t],
                       "thp": int(min(max(0, hp), max_hp)), "ehp": int(min(max(0, ehp), max_ehp))})
        roles = [ROLE_CHAR_TO_IDX[c] for c in combo]
        for _ in range(TURNS_PER_STAGE):
            if t >= len(lds) or hp <= 0 or ehp <= 0:
                break
            actions = [preferred_action(r, lds[t], hp, max_hp) for r in roles]
            current = bayesian_update(current, actions, lds[t], hp, max_hp, eps)
            hp, ehp = game_step(lds[t], hp, ehp, actions, stats, env["boss_damage"], max_hp)
            t += 1
        current = (1.0 - beta) * current + beta * prior
        current = current / current.sum()
    return states


def own_marginal(belief, i):
    m = belief.sum(axis=tuple(j for j in range(3) if j != i))
    return m / m.sum()


def expected_values(belief, i, v):
    """EV of each role for player i; v = (27,) values indexed r0·9 + r1·3 + r2."""
    others = belief.sum(axis=i)
    others = others / others.sum()
    return (np.moveaxis(v.reshape(3, 3, 3), i, 0) * others).sum(axis=(1, 2))


def player_beliefs(records, ob, params, ftm_rows):
    """(PM, EV), each (n, 3). Rows in ftm_rows use the FTM prior p⊗p⊗p, the rest the mixed posterior."""
    p = np.array([params["p_F"], params["p_T"], params["p_M"]])
    ftm = np.einsum("i,j,k->ijk", p, p, p)
    states = [team_states(rec, params["tau_prior"], params["epsilon"], params["beta"])
              for rec in records]
    assert all(s < len(states[ri]) for ri, s in zip(ob.ri, ob.s)), "stage past end of game"
    PM, EV = np.empty((ob.n, 3)), np.empty((ob.n, 3))
    for q, (ri, s, pos) in enumerate(zip(ob.ri, ob.s, ob.pos)):
        st = states[ri][s]
        b = ftm if ftm_rows[q] else st["belief"]
        v = records[ri]["env_config"]["values"][:, st["intent"], st["thp"], st["ehp"]]
        PM[q], EV[q] = own_marginal(b, pos), expected_values(b, pos, v)
    return PM, EV
