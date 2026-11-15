# 2026-11-15 time-sensitive value function

Robustness checks from the 2026-11-10 meeting (Xuan): does a value function that accounts for the turns
remaining change the response-model fits, and how does that combine with mixed vs unmixed beliefs and with
non-updaters scored on updater beliefs? Result: Bayesian-Walk-BR wins in every variant and the fit barely
moves (updaters: agg-LL within 0.009 across value function x belief). Reported as SI Table S9.

The paper's value tables are `V(HORIZON)` only (`values.npy`, no time axis). A stage-s choice here uses
`V(10 - 2 s)` (2 turns per stage).

| File | What it does |
|---|---|
| `value_by_turns.py` | Numpy DP over all 27 role combos and states, every t = 0..10, all 15 envs -> `value_by_turns.npz`. Asserts its t=10 slice matches each shipped `values.npy`. |
| `run_memo_value_by_turns.py` | The same tables from the original memo `V.py` (run in each env dir next to `R.py`, `T.py`, `pi.py`, `config.py`) -> `value_by_turns_memo.npz`. Needs `memo-lang==1.2.7`, `jax==0.5.0`, Python 3.12 (memo 1.2.10 uses 3.13 syntax). CPU only: 5 s (242 states) to ~570 s (992 states) per env for all 11 slices; the Metal backend was not faster. |
| `fit_time_sensitive.py` | Updaters / non-updaters on updater beliefs, refit with time-aware V. |
| `fit_master.py [numpy\|memo]` | Master table: value (paper / time-aware / time-aware with params frozen at the paper fit) x belief (mixed / unmixed) x group. Asserts that paper-V variants reproduce the stored 2026-11-12 tables. Outputs `master_<source>/{master.md, master.csv, all_fits.json}`. |
| `render_master.py` | Renders `master_summary.png` and `master_all_models.png` via headless Chrome. |

memo and numpy tables agree at every t for all 15 envs (max abs diff 1.9e-5, float32), and the two master tables
agree to <1e-6. Note the paper's eps_s is the STICK probability; `epsilon_switch` in the fit outputs is 1 - eps_s.

Run from `analysis/` (`uv run python experiments/2026-11-15_time-sensitive-value/<script>`): `value_by_turns.py`,
then `fit_master.py numpy`, then `render_master.py` (plain python3).
