# 2026-10-15 — pure Threshold-PS refit

Follow-up to the 2026-10-12 call with Xuan: Bayesian-Threshold-PS was a stick/threshold
*mixture* ((1-ε_s)·stick + ε_s·gated switch), not a pure threshold rule. ε_s is removed;
δ_PS is the only parameter. Everything else (records, Stage-1 params, agg_ll objective,
other models' fits, 5% lapse in the individual fit) is unchanged — the script asserts that
the 11 untouched models reproduce 05-28's numbers to 1e-9. Mixture-PS is dropped (13 models).

Run (≈50 s): `uv run python experiments/2026-10-15-pure-threshold-ps/pure_threshold_ps.py` from `analysis/`.

| File | Contents |
|---|---|
| `summary.md` | δ fits per objective, 13-model aggregate table, dominant-model counts old vs new, δ profile |
| `tables_snippets.tex` | drop-in rows for main-text aggregate table and SI Table S3 |
| `stuff to incorporate/R4_individual_fitting.{pdf,png}` | regenerated Fig. R4 |
| `results.json` | all numbers incl. per-participant log-likelihoods |

Unaffected (don't depend on Threshold-PS): R1 calibration, R2 top-k, R3 team case, R5 adaptation.
