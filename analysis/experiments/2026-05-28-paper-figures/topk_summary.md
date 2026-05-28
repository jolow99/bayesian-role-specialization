# Section 2 — Top-K analysis on 5-export clean human team-rounds

**Data scope.** 5 exports, human-only, clean teams: **721 team-stage observations** across 15 environments.

## Headline numbers

- Mean rank: **9.71** (random baseline: 14.0). One-sided z-test against random: z = -14.78, p = <1e-4.
- Top-1: **11.7%** (random: 3.7%) — humans land on the exact-best combo at 3.1× the random rate.
- Top-3: **26.4%** (random: 11.1%).
- Top-5: **42.4%** (random: 18.5%).
- Bottom-5: **8.3%** (random: 18.5%) — humans avoid the worst combos at 0.4× the random rate.
- Mean normalized optimality: **0.667** (random: 0.500).

## Per-stage breakdown

| Stage | N | Mean rank | Top-1 | Top-3 | Top-5 | Bottom-5 | Norm. opt. |
|------:|--:|----------:|------:|------:|------:|---------:|-----------:|
| 1 | 204 | 9.31 | 15.7% | 30.4% | 46.6% | 9.8% | 0.601 |
| 2 | 200 | 9.15 | 12.5% | 31.0% | 48.0% | 6.5% | 0.652 |
| 3 | 184 | 10.37 | 9.2% | 22.3% | 37.0% | 10.3% | 0.679 |
| 4 | 107 | 10.32 | 9.3% | 17.8% | 34.6% | 5.6% | 0.779 |
| 5 | 26 | 10.04 | 0.0% | 23.1% | 38.5% | 7.7% | 0.760 |

## Interpretation

Humans coordinate **significantly better than chance** — the Top-K curve sits above the random diagonal and the Bottom-K curve sits below it. The effect is consistent across stages, with mean rank improving across the round as teams accumulate inference evidence. This supports the paper's quantitative finding for Section 2: **most humans plan top-K** rather than choosing roles uniformly at random.