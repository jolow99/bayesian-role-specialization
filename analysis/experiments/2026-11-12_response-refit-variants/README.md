# 2026-11-12 response refit variants

Robustness checks from the 2026-11-10 meeting (Xuan). These use the fixed inference fit and the
updater/non-updater split from `../2026-11-02_ftm-prior-fitting/`. Only the six Bayesian response
models are covered.

| Script | Question | Outputs |
|---|---|---|
| `make_tables.py` | Updaters: mixed vs unmixed posterior. Non-updaters: paper (FTM prior) vs updater beliefs (mixed, unmixed), refit on non-updaters | `tables/` |
| `nonupd_as_updaters.py` | Non-updaters scored with updater beliefs and updater parameters (nothing refit), plus individual fits | `nonupd-as-updaters/` |

`beliefs.py` is the updater belief chain from the stat prior, with both the mixed and the unmixed posterior.

Each script checks itself and writes `verify.log`; all checks pass. combo-r is team-level over all
teams, so it is not specific to one group.

Run from `analysis/`, `make_tables.py` first:

    uv run python experiments/2026-11-12_response-refit-variants/make_tables.py
    uv run python experiments/2026-11-12_response-refit-variants/nonupd_as_updaters.py
