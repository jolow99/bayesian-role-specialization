"""Data prep and Bayesian inference likelihood for the two-type classifier.

Trimmed from 2026-10-27-inference-soft-classifier/pipeline.py: clean human
teams only, drift_prior memory (beta), inferences scored against the unmixed
posterior P_N with a lapse (1 − λ)·p + λ/3.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from shared import EXPORTS_DIR
from shared.constants import TURNS_PER_STAGE
from shared.data_loading import load_all_exports
from shared.inference import utility_based_prior, bayesian_update, preferred_action, game_step

EXPORT_DIRS = sorted(EXPORTS_DIR.glob("bayesian-role-specialization-*"))
INFERENCE_LAPSE = 0.01


def _clean_human_teams():
    """(game_id, round_number) -> 3 player-rounds sorted by position, dropout games excluded."""
    prs = load_all_exports(data_dirs=EXPORT_DIRS)
    dropout_games = {pr.game_id for pr in prs if pr.is_dropout}
    teams = defaultdict(list)
    for pr in prs:
        if pr.game_id not in dropout_games and pr.round.round_type == "human":
            teams[(pr.game_id, pr.round.round_number)].append(pr)
    return [sorted(v, key=lambda p: p.player_id) for v in teams.values() if len(v) == 3]


def _prepare_team(team_prs):
    rnd = team_prs[0].round
    config = rnd.config
    player_roles = {pr.player_id: [s.role_idx for s in pr.round.stages] for pr in team_prs}
    n_stages = max(len(v) for v in player_roles.values())
    role_seq = [[rs[s] if s < len(rs) else 0 for _, rs in sorted(player_roles.items())]
                for s in range(n_stages)]

    queries = []  # (participant_id, stage si, target position, reported role, true role at si - 1)
    for pr in team_prs:
        for si, stage in enumerate(pr.round.stages):
            if si == 0 or not stage.inferred_roles:
                continue
            for target, role in stage.inferred_roles.items():
                if target in player_roles and si - 1 < len(player_roles[target]):
                    queries.append((pr.participant_id, si, target, role,
                                    player_roles[target][si - 1]))

    return {
        "player_stats": np.array([[int(c) for c in p] for p in rnd.stat_profile_id.split("_")],
                                 dtype=float),
        "boss_damage": config.get("bossDamage", 2),
        "team_max_hp": config.get("maxTeamHealth", 15),
        "enemy_max_hp": config.get("maxEnemyHealth", 30),
        "eis": rnd.enemy_intent_sequence,
        "role_seq": role_seq,
        "queries": queries,
    }


def load_answers():
    """Returns (teams, answers); answers holds one entry per inference query."""
    teams = [_prepare_team(t) for t in _clean_human_teams()]
    pid_index: dict[str, int] = {}
    rows = [(pid_index.setdefault(pid, len(pid_index)), t, si, target, role, true)
            for t, team in enumerate(teams) for pid, si, target, role, true in team["queries"]]
    cols = np.array(rows, dtype=int)
    answers = {
        "participant_ids": list(pid_index),
        "participant": cols[:, 0],
        "team": cols[:, 1],
        "stage": cols[:, 2],
        "target": cols[:, 3],
        "inferred": cols[:, 4],
        "true": cols[:, 5],
        "by_participant": [np.flatnonzero(cols[:, 0] == p) for p in range(len(pid_index))],
        "by_team": [np.flatnonzero(cols[:, 1] == t) for t in range(len(teams))],
    }
    print(f"[pipeline] {len(teams)} clean human teams, {len(pid_index)} participants, "
          f"{len(cols)} answers")
    return teams, answers


def compute_posteriors(team, tau, eps, beta):
    """posteriors[s] = unmixed belief at the start of stage s; drift_prior beta carried over."""
    prior = utility_based_prior(team["player_stats"], tau=tau)
    posteriors = [prior.copy()]
    current = prior.copy()
    hp, ehp = float(team["team_max_hp"]), float(team["enemy_max_hp"])
    eis = team["eis"]
    for s, roles in enumerate(team["role_seq"]):
        for off in range(TURNS_PER_STAGE):
            ti = s * TURNS_PER_STAGE + off
            if ti >= len(eis) or hp <= 0:
                break
            intent = int(eis[ti])
            actions = [preferred_action(r, intent, hp, team["team_max_hp"]) for r in roles]
            current = bayesian_update(current, actions, intent, hp, team["team_max_hp"], eps)
            hp, ehp = game_step(intent, hp, ehp, actions, team["player_stats"],
                                team["boss_damage"], team["team_max_hp"])
        posteriors.append(current.copy())
        current = (1.0 - beta) * current + beta * prior
        current = current / current.sum()
    return posteriors


def _target_marginal(post, target):
    marg = np.sum(post, axis=tuple(j for j in range(3) if j != target))
    return marg / marg.sum()


def answer_marginals(teams, answers, tau, eps, beta):
    """(Q, 3) target's role marginal per answer under the unmixed P_N."""
    marg = np.empty((len(answers["participant"]), 3))
    for t, idx in enumerate(answers["by_team"]):
        if len(idx) == 0:
            continue
        posteriors = compute_posteriors(teams[t], tau, eps, beta)
        for q in idx:
            marg[q] = _target_marginal(posteriors[answers["stage"][q]], answers["target"][q])
    return marg


def answer_probs(teams, answers, tau, eps, beta):
    """P_N(reported role) per answer."""
    marg = answer_marginals(teams, answers, tau, eps, beta)
    return marg[np.arange(len(marg)), answers["inferred"]]


def participant_logprob(p, answers):
    """Σ_q log[(1 − λ)·p_q + λ/3] per participant."""
    log_p = np.log((1.0 - INFERENCE_LAPSE) * p + INFERENCE_LAPSE / 3.0)
    return np.array([log_p[ix].sum() for ix in answers["by_participant"]])
