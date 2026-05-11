"""Self-contained model factories for the 04-23 metric comparison.

All seven models produce a per-stage ``{predicted_dist, human_combo,
model_marginal}`` record given a precomputed trajectory and the stage-2
tunable params. Nothing here imports from any other ``experiments/`` folder.

Factories return a two-level closure: ``make_factory(params)`` →
``factory(records, trajs)`` → ``predict_fn(record)``. The outer layer fixes
the model's params; the middle layer captures the subset of records under
evaluation (so the predict_fn can resolve a record to its precomputed
trajectory via ``id(record)``).
"""

from __future__ import annotations

import numpy as np

from shared.inference import softmax_role_dist

from pipeline import posterior_marginal, build_joint_dist


# ──────────────────────────────────────────────────────────────────────
# Bayesian Belief (no params)
# ──────────────────────────────────────────────────────────────────────

def _belief_predict(trajectory):
    out = []
    for stage in trajectory:
        prior = stage["prior"]
        per_agent = [posterior_marginal(prior, i) for i in range(3)]
        out.append({
            "predicted_dist": build_joint_dist(per_agent),
            "human_combo": stage["human_combo"],
            "model_marginal": np.mean(per_agent, axis=0),
        })
    return out


def belief_factory():
    def factory(records_subset, traj_subset):
        idx_map = {id(r): i for i, r in enumerate(records_subset)}
        def predict_fn(record):
            return _belief_predict(traj_subset[idx_map[id(record)]])
        return predict_fn
    return factory


# ──────────────────────────────────────────────────────────────────────
# Bayesian Value — softmax over expected utilities
# ──────────────────────────────────────────────────────────────────────

def _value_predict(trajectory, values, tau_softmax):
    out = []
    for stage in trajectory:
        prior = stage["prior"]
        intent, thp, ehp = stage["intent"], stage["thp"], stage["ehp"]
        per_agent = [softmax_role_dist(i, intent, thp, ehp, prior, values, tau_softmax)
                     for i in range(3)]
        out.append({
            "predicted_dist": build_joint_dist(per_agent),
            "human_combo": stage["human_combo"],
            "model_marginal": np.mean(per_agent, axis=0),
        })
    return out


def value_factory(tau_softmax):
    def factory(records_subset, traj_subset):
        idx_map = {id(r): i for i, r in enumerate(records_subset)}
        def predict_fn(record):
            i = idx_map[id(record)]
            return _value_predict(traj_subset[i],
                                   record["env_config"]["values"], tau_softmax)
        return predict_fn
    return factory


# ──────────────────────────────────────────────────────────────────────
# Bayesian Walk — ε-mixture of stick-with-prev and softmax over values
# ──────────────────────────────────────────────────────────────────────

def _walk_predict(trajectory, values, tau_softmax, epsilon_switch):
    out = []
    for stage in trajectory:
        prior = stage["prior"]
        intent, thp, ehp = stage["intent"], stage["thp"], stage["ehp"]
        prev_roles = stage["prev_roles"]
        switch = [softmax_role_dist(i, intent, thp, ehp, prior, values, tau_softmax)
                  for i in range(3)]
        per_agent = []
        for i in range(3):
            if prev_roles is None:
                per_agent.append(switch[i])
            else:
                stick = np.zeros(3)
                stick[prev_roles[i]] = 1.0
                per_agent.append((1.0 - epsilon_switch) * stick
                                 + epsilon_switch * switch[i])
        out.append({
            "predicted_dist": build_joint_dist(per_agent),
            "human_combo": stage["human_combo"],
            "model_marginal": np.mean(per_agent, axis=0),
        })
    return out


def walk_factory(tau_softmax, epsilon_switch):
    def factory(records_subset, traj_subset):
        idx_map = {id(r): i for i, r in enumerate(records_subset)}
        def predict_fn(record):
            i = idx_map[id(record)]
            return _walk_predict(traj_subset[i],
                                  record["env_config"]["values"],
                                  tau_softmax, epsilon_switch)
        return predict_fn
    return factory


# ──────────────────────────────────────────────────────────────────────
# Bayesian Threshold — only switch if another role's EV beats current by δ
# ──────────────────────────────────────────────────────────────────────

def _expected_values_per_role(agent_i, intent, thp, ehp, prior, values):
    other = [a for a in range(3) if a != agent_i]
    other_probs = np.sum(prior, axis=agent_i)
    total = other_probs.sum()
    other_probs = other_probs / total if total > 0 else np.ones((3, 3)) / 9.0
    ev = np.zeros(3)
    for r_i in range(3):
        for r_j in range(3):
            for r_k in range(3):
                roles = [0, 0, 0]
                roles[agent_i] = r_i
                roles[other[0]] = r_j
                roles[other[1]] = r_k
                flat_idx = roles[0] * 9 + roles[1] * 3 + roles[2]
                ev[r_i] += other_probs[r_j, r_k] * float(
                    values[flat_idx, intent, thp, ehp])
    return ev


def _threshold_role_dist(agent_i, intent, thp, ehp, prior, values,
                          current_role, delta, tau):
    ev = _expected_values_per_role(agent_i, intent, thp, ehp, prior, values)
    current_val = ev[current_role]
    candidates = [r for r in range(3) if r != current_role and (ev[r] - current_val) > delta]
    if not candidates:
        d = np.zeros(3); d[current_role] = 1.0; return d
    cv = np.array([ev[r] for r in candidates]) / tau
    cv -= cv.max()
    p = np.exp(cv); p /= p.sum()
    d = np.zeros(3)
    for i, r in enumerate(candidates):
        d[r] = p[i]
    return d


def _thresh_predict(trajectory, values, tau_softmax, delta):
    out = []
    for stage in trajectory:
        prior = stage["prior"]
        intent, thp, ehp = stage["intent"], stage["thp"], stage["ehp"]
        prev_roles = stage["prev_roles"]
        per_agent = []
        for i in range(3):
            if prev_roles is None:
                per_agent.append(softmax_role_dist(
                    i, intent, thp, ehp, prior, values, tau_softmax))
            else:
                per_agent.append(_threshold_role_dist(
                    i, intent, thp, ehp, prior, values,
                    current_role=prev_roles[i], delta=delta, tau=tau_softmax))
        out.append({
            "predicted_dist": build_joint_dist(per_agent),
            "human_combo": stage["human_combo"],
            "model_marginal": np.mean(per_agent, axis=0),
        })
    return out


def thresh_factory(tau_softmax, delta):
    def factory(records_subset, traj_subset):
        idx_map = {id(r): i for i, r in enumerate(records_subset)}
        def predict_fn(record):
            i = idx_map[id(record)]
            return _thresh_predict(traj_subset[i],
                                    record["env_config"]["values"],
                                    tau_softmax, delta)
        return predict_fn
    return factory


# ──────────────────────────────────────────────────────────────────────
# Walk-PS — posterior-sampling version of walk (no value matrix)
# ──────────────────────────────────────────────────────────────────────

def _walk_ps_predict(trajectory, epsilon_switch):
    out = []
    for stage in trajectory:
        prior = stage["prior"]
        prev_roles = stage["prev_roles"]
        switch = [posterior_marginal(prior, i) for i in range(3)]
        per_agent = []
        for i in range(3):
            if prev_roles is None:
                per_agent.append(switch[i])
            else:
                stick = np.zeros(3)
                stick[prev_roles[i]] = 1.0
                per_agent.append((1.0 - epsilon_switch) * stick
                                 + epsilon_switch * switch[i])
        out.append({
            "predicted_dist": build_joint_dist(per_agent),
            "human_combo": stage["human_combo"],
            "model_marginal": np.mean(per_agent, axis=0),
        })
    return out


def walk_ps_factory(epsilon_switch):
    def factory(records_subset, traj_subset):
        idx_map = {id(r): i for i, r in enumerate(records_subset)}
        def predict_fn(record):
            i = idx_map[id(record)]
            return _walk_ps_predict(traj_subset[i], epsilon_switch)
        return predict_fn
    return factory


# ──────────────────────────────────────────────────────────────────────
# Threshold-PS — PS version of threshold
# ──────────────────────────────────────────────────────────────────────

def _thresh_ps_switch_dist(prior, agent_i, prev_role, delta):
    marg = posterior_marginal(prior, agent_i)
    if prev_role is None:
        return marg
    cur = marg[prev_role]
    candidates = [r for r in range(3) if r != prev_role and (marg[r] - cur) > delta]
    if not candidates:
        d = np.zeros(3); d[prev_role] = 1.0; return d
    cp = np.array([marg[r] for r in candidates])
    cp = cp / cp.sum()
    d = np.zeros(3)
    for i, r in enumerate(candidates):
        d[r] = cp[i]
    return d


def _thresh_ps_predict(trajectory, epsilon_switch, delta):
    out = []
    for stage in trajectory:
        prior = stage["prior"]
        prev_roles = stage["prev_roles"]
        per_agent = []
        for i in range(3):
            prev_r = prev_roles[i] if prev_roles is not None else None
            marg = posterior_marginal(prior, i)
            if prev_r is None:
                per_agent.append(marg)
                continue
            switch = _thresh_ps_switch_dist(prior, i, prev_r, delta)
            stick = np.zeros(3); stick[prev_r] = 1.0
            per_agent.append((1.0 - epsilon_switch) * stick + epsilon_switch * switch)
        out.append({
            "predicted_dist": build_joint_dist(per_agent),
            "human_combo": stage["human_combo"],
            "model_marginal": np.mean(per_agent, axis=0),
        })
    return out


def thresh_ps_factory(epsilon_switch, delta):
    def factory(records_subset, traj_subset):
        idx_map = {id(r): i for i, r in enumerate(records_subset)}
        def predict_fn(record):
            i = idx_map[id(record)]
            return _thresh_ps_predict(traj_subset[i], epsilon_switch, delta)
        return predict_fn
    return factory


# ──────────────────────────────────────────────────────────────────────
# Mixture-PS — convex combination of walk_ps and thresh_ps (PS-style)
# ──────────────────────────────────────────────────────────────────────

def _walk_ps_dist(prior, agent_i, prev_role, epsilon_switch):
    marg = posterior_marginal(prior, agent_i)
    if prev_role is None:
        return marg
    stick = np.zeros(3); stick[prev_role] = 1.0
    return (1.0 - epsilon_switch) * stick + epsilon_switch * marg


def _thresh_ps_full(prior, agent_i, prev_role, epsilon_switch, delta):
    if prev_role is None:
        return posterior_marginal(prior, agent_i)
    stick = np.zeros(3); stick[prev_role] = 1.0
    switch = _thresh_ps_switch_dist(prior, agent_i, prev_role, delta)
    return (1.0 - epsilon_switch) * stick + epsilon_switch * switch


def _mixture_predict(trajectory, walk_eps, thresh_eps, thresh_delta, w):
    out = []
    for stage in trajectory:
        prior = stage["prior"]
        prev_roles = stage["prev_roles"]
        per_agent = []
        for i in range(3):
            pr = prev_roles[i] if prev_roles is not None else None
            d_walk = _walk_ps_dist(prior, i, pr, walk_eps)
            d_thresh = _thresh_ps_full(prior, i, pr, thresh_eps, thresh_delta)
            per_agent.append(w * d_walk + (1.0 - w) * d_thresh)
        out.append({
            "predicted_dist": build_joint_dist(per_agent),
            "human_combo": stage["human_combo"],
            "model_marginal": np.mean(per_agent, axis=0),
        })
    return out


def mixture_ps_factory(walk_eps, thresh_eps, thresh_delta, w):
    def factory(records_subset, traj_subset):
        idx_map = {id(r): i for i, r in enumerate(records_subset)}
        def predict_fn(record):
            i = idx_map[id(record)]
            return _mixture_predict(traj_subset[i],
                                     walk_eps, thresh_eps, thresh_delta, w)
        return predict_fn
    return factory
