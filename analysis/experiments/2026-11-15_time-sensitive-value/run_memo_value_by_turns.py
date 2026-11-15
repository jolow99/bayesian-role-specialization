import time, sys, numpy as np
from memo import memo
from functools import cache
from config import ROLE_INDICES, S, A, HORIZON, H, W
from pi import action_profile_prob
from R import R, is_terminal
from T import T
import jax.numpy as jnp

@cache
@memo
def V[r: ROLE_INDICES, s: S](t):
    observer: knows(r, s)
    observer: chooses(a in A, wpp=action_profile_prob(s, a, r))
    observer: draws(s_ in S, wpp=T(s, a, s_, r))
    return E[
        R(s, t) + (
            0.0 if t == 0 else
            0.0 if is_terminal(s) else
            V[r, observer.s_](t - 1)
        )
    ]

out = []
t0 = time.time()
for t in range(HORIZON + 1):
    out.append(np.array(V(t)).reshape((27, 2, H, W)))
    print(f"t={t} done at {time.time()-t0:.1f}s", flush=True)
np.save("memo_V_by_t.npy", np.stack(out))
