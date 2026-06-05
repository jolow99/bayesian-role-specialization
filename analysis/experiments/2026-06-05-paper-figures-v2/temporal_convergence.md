# Task 2 — temporal convergence (value-rank of played combo)

721 team-stage observations, 204 clean human team-rounds. Rank 1 = value-optimal of 27; chance = 14.

| Stage | N | Mean rank [95% CI] | Top-1 | Top-5 |
|------:|--:|--------------------|------:|------:|
| 1 | 204 | 9.31 [8.24, 10.41] | 15.7% | 46.6% |
| 2 | 200 | 9.15 [8.10, 10.21] | 12.5% | 48.0% |
| 3 | 184 | 10.37 [9.28, 11.51] | 9.2% | 37.0% |
| 4 | 107 | 10.32 [9.00, 11.70] | 9.3% | 34.6% |
| 5 | 26 | 10.04 [7.08, 13.19] | 0.0% | 38.5% |

## Fixed cohort (rounds lasting ≥4 stages, n=107)

The all-rounds series is survivorship-confounded: teams that lock onto a top combo kill the boss sooner and leave the sample, so later stages over-represent struggling teams.

| Stage | N | Mean rank [95% CI] | Top-1 | Top-5 |
|------:|--:|--------------------|------:|------:|
| 1 | 107 | 10.64 [9.06, 12.29] | 16.8% | 41.1% |
| 2 | 107 | 9.28 [7.82, 10.79] | 13.1% | 45.8% |
| 3 | 107 | 10.52 [9.09, 12.02] | 6.5% | 32.7% |
| 4 | 107 | 10.32 [8.96, 11.69] | 9.3% | 34.6% |

## Reading

Teams sit **well above chance from stage 1** (mean rank ≈ 9.3 vs 14; top-5 ≈ 47% vs 18.5%) but the per-stage trend is flat-to-slightly-worse, in both the all-rounds series and the fixed cohort. There is no evidence of a not-top-K → top-K shift across stages on this metric: good coordination is mostly present from the start (stat-driven priors), and rounds that lock onto top combos end early by winning. Note normalized optimality (05-28 topk summary) *does* rise across stages (0.60 → 0.78), so 'value of what teams play, relative to what's attainable' improves even though the rank of the chosen combo does not.
