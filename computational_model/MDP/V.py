# type: ignore
"""
Value function computation using memo library.
"""

from memo import memo
from functools import cache
from config import (
    ROLE_INDICES, S, A,
    HORIZON, H, W
)
from pi import action_profile_prob
from R import R, is_terminal
from T import T
import subprocess
import jax.numpy as jnp

@cache
@memo
def V[r: ROLE_INDICES, s: S](t):
    observer: knows(r, s)
    observer: chooses(a in A, wpp=action_profile_prob(s, a, r))
    observer: draws(s_ in S, wpp=T(s, a, s_, r))
    
    return E[
        R(s, t) + (
            0.0 if t == 0 else              # recursion depth reached
            0.0 if is_terminal(s) else      # terminal state
            V[r, observer.s_](t - 1)        # continue recursion
        )
    ]


V(0)  # pre-compile
values = V(HORIZON).reshape((27, 2, H, W))
jnp.save("values.npy", values)
subprocess.run(["python", "print.py"])
