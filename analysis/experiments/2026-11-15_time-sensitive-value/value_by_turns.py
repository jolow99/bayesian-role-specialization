"""Time-indexed value tables V(t) for the 15 human environments.

The shipped `values.npy` is V(HORIZON) only (shape (27, 2, H, W)): memo's recursion computes V(t) for
every t = 0..HORIZON but only the top slice is saved. This re-implements the same DP in numpy and
keeps every slice, so a choice at the start of a stage with `t` turns left can use V(t).

Mirrors computational_model/MDP/{V,R,T,pi}.py exactly:
  V_t[r, s] = R(s, t) + 0                                 if t == 0 or s terminal
            = R(s, t) + E_{a ~ pi_r(s), s' ~ T}[V_{t-1}[r, s']]   otherwise
  R(s, t)   = 100 * t / HORIZON  if team alive and enemy dead, else 0
  s = (intent, team_hp, enemy_hp); intent is redrawn iid each turn (P(attack) = ENEMY_ATTACK_PROB).

Verification (asserted): the t = HORIZON slice reproduces each env's shipped values.npy.

Output: value_by_turns.npz (one (HORIZON+1, 27, 2, H, W) float32 array per env dir).
"""

from __future__ import annotations

import re
from itertools import product
from pathlib import Path

import numpy as np

from shared import HUMAN_ENVS_DIR
from shared.env_loading import load_env_config

OUT = Path(__file__).resolve().parent / "value_by_turns.npz"
HORIZON = 10
EPS = 1e-10  # config.EPSILON
ATTACK, DEFEND, HEAL = 0, 1, 2


def role_action_prob(role, a, intent, hp, max_hp):
    """pi(a | role, intent, team_hp) as an array over hp. (preferred action w.p. 1-EPS)"""
    if role == 0:
        pref = np.full_like(hp, ATTACK)
    elif role == 1:
        pref = np.full_like(hp, DEFEND if intent == 1 else ATTACK)
    else:
        pref = np.where(hp < max_hp, HEAL, ATTACK)
    return np.where(a == pref, 1.0 - EPS, EPS / 2)


def compute(env_dir: Path):
    cfg = load_env_config(env_dir)
    text = (env_dir / "config.py").read_text()
    p_att = float(re.search(r"ENEMY_ATTACK_PROB\s*=\s*([\d.]+)", text).group(1))
    horizon = int(re.search(r"HORIZON\s*=\s*(\d+)", text.replace("EPSILON, HORIZON = 1e-10, ", "HORIZON = ")).group(1))
    assert horizon == HORIZON, horizon
    stats, dmg = cfg["player_stats"], cfg["boss_damage"]
    mh, me = cfg["team_max_hp"], cfg["enemy_max_hp"]
    H, W = mh + 1, me + 1
    atk, dfn, heal = stats[:, 0], stats[:, 1], stats[:, 2]
    hp = np.arange(H)[:, None] * np.ones((1, W))   # team hp grid (H, W)
    ehp = np.ones((H, 1)) * np.arange(W)[None, :]  # enemy hp grid (H, W)
    win = ((hp > 0) & (ehp == 0)).astype(float)
    terminal = (hp == 0) | (ehp == 0)
    combos = list(product(range(3), repeat=3))
    profiles = list(product(range(3), repeat=3))

    V = np.zeros((HORIZON + 1, 27, 2, H, W))
    for t in range(HORIZON + 1):
        V[t] = (100.0 * t / HORIZON) * win  # R(s, t), broadcast over r and intent
        if t == 0:
            continue
        # intent-averaged next-step value, W_prev[r, hp, ehp]
        Wprev = (1 - p_att) * V[t - 1][:, 0] + p_att * V[t - 1][:, 1]
        for ri, roles in enumerate(combos):
            for i in (0, 1):
                acc = np.zeros((H, W))
                for a in profiles:
                    pa = np.ones((H, W))
                    for k in range(3):
                        pa = pa * role_action_prob(roles[k], a[k], i, hp, mh)
                    tot_atk = sum(atk[k] for k in range(3) if a[k] == ATTACK)
                    tot_heal = sum(heal[k] for k in range(3) if a[k] == HEAL)
                    max_def = max([dfn[k] for k in range(3) if a[k] == DEFEND] or [0])
                    nehp = np.maximum(0, ehp - tot_atk).astype(int)
                    inc = max(0.0, dmg - max_def) if i == 1 else 0.0
                    nhp = np.clip(hp - inc + tot_heal, 0, mh).astype(int)
                    acc += pa * Wprev[ri][nhp, nehp]
                V[t, ri, i] += np.where(terminal, 0.0, acc)
    return V.astype(np.float32), cfg


def main():
    out = {}
    for env_dir in sorted(p for p in HUMAN_ENVS_DIR.iterdir() if (p / "values.npy").exists()):
        V, cfg = compute(env_dir)
        shipped = cfg["values"]
        err = float(np.abs(V[HORIZON] - shipped).max())
        assert err < 1e-3 * max(1.0, float(np.abs(shipped).max())), f"{env_dir.name}: max abs err {err}"
        print(f"{env_dir.name}: V(10) matches values.npy (max abs err {err:.2e}); "
              f"max V(10) = {V[HORIZON].max():.1f}, max V(2) = {V[2].max():.1f}")
        out[env_dir.name] = V
    np.savez_compressed(OUT, **out)
    print(f"saved {len(out)} envs -> {OUT}")


if __name__ == "__main__":
    main()
