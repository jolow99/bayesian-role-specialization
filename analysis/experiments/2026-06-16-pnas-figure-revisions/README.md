# 2026-06-16 — PNAS figure revisions (R1 / R2 / R3)

Generate / revise the role-self-organization PNAS figures. Final
artifacts go in `stuff to incorporate/` with R-prefixes:
`R1_*` = inference, `R2_*` = model fits, `R3_*` = adaptation. All figures
are vector PDF (+ 300 dpi PNG), colorblind-safe, no in-figure titles
(captions live in the paper), styled to match the 06-07 PNAS figures.

Paper claim these support: **players select roles in a rational way — a
mixture of rational (value-maximizing) choice and sticky-choice
(inertia).**

## Artifacts (`stuff to incorporate/`)

| File | Item | What |
|------|------|------|
| `R3_team_case.pdf` | 1 (NEW, top priority) | human-team case study: rational + sticky + relent |
| `R1_calibration.pdf` | 2 (revise) | calibration; headline now binned-decile r = 0.99 |
| `R2_individual_fitting.pdf` | 3 (revise) | per-participant model posteriors; baselines hatched |
| `R1_accuracy_numbers.md` | 4 (compute) | inference-accuracy prose numbers (replaces a dropped figure) |

`summary.md` collects all the numbers and the case-selection rationale.

## Scripts

| Script | Produces |
|--------|----------|
| `common_human.py` | join clean human team-rounds (PlayerRounds) with value matrices; posteriors; per-player best-response; value-rank |
| `svg_icons.py` | true-vector Twemoji role icons (SVG path → matplotlib `PathPatch`) |
| `case_search.py` | rank candidate rounds for the case study (run to reproduce the shortlist) |
| `team_case.py` | render `R3_team_case` |
| `calibration_revision.py` | render `R1_calibration` |
| `individual_fitting_revision.py` | render `R2_individual_fitting` |
| `accuracy_numbers.py` | compute `R1_accuracy_numbers.md` |

Run from `analysis/`:

```bash
uv run python experiments/2026-06-16-pnas-figure-revisions/team_case.py
uv run python experiments/2026-06-16-pnas-figure-revisions/calibration_revision.py
uv run python experiments/2026-06-16-pnas-figure-revisions/individual_fitting_revision.py
uv run python experiments/2026-06-16-pnas-figure-revisions/accuracy_numbers.py
```

## R3 case selection (committed decision)

Searched all 204 clean human team-rounds for: starts mis-coordinated;
≥1 player stays against best-response; team eventually deviates to a
better combo and wins; bonus = symmetry breaking. 143 rounds qualified;
re-ranked for the **sticks-then-relents** signature (the visual mark of
a rational+sticky *mixture*, not pure stickiness).

Chosen: **`01KRBT30X9RGSB5Q5P3PX48RYB` r4** (`222_222_222`,
TMT→FMT→FFF, rank 26→12→1, WIN). Picked over the specialized-ending
alternative `01KQ6YDA…` r2 (MFT) because the claim is about the *choice
process*: this round shows P1 best-responding immediately while P2/P3
lag one stage (stickiness) then relent (rational switch) — exactly what
Bayesian Walk-PS (the winning fit in `R2_individual_fitting`) predicts.
Its only cost is a homogeneous (all-attack) optimum, acceptable for a
process claim. To re-pin, run `case_search.py` and edit `CASE_GAME_ID` /
`CASE_ROUND` in `team_case.py`.

## Dependencies

Adds `svgpath2mpl` (in `pyproject.toml`). Twemoji role SVGs are committed
under `assets/` (`role_fighter.svg` 🤺, `role_tank.svg` 💂,
`role_medic.svg` 👩‍⚕️ — the game-UI role icons).

## Notes / provenance

- Stage-1 inference params come from the 05-25 full pipeline (the
  paper's source of truth), asserted byte-identical to the 05-12 fit
  loaded by `2026-06-05-paper-figures-v2/common.py`.
- Tasks 2–4 are revisions of `2026-06-07-epistemic-rationality` and
  `2026-06-07-instrumental-rationality`; all their canonical-param and
  `dominant_counts` assertions are preserved and pass.
- R2 collapses only **Copy Others** (negligible: max per-participant
  posterior 0.0004, dominant for 0) into a grey "other baselines"
  segment; the normalizer is left intact so bars remain honest
  P(model | participant).
