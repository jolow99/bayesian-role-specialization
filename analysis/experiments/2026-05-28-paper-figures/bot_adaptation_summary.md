# Section 3 — Adaptation to stubborn bot teammates

**Data scope.** 5 exports, bot rounds, clean games only: **204 human bot-round observations** across 15 treatments, **102 unique participants**. Bot positions and human stats are resolved per CLAUDE.md → "Bot Round Ground Truth".

## Headline numbers

- Aggregate stat-optimal play: **60%** of stages.
- Aggregate deviate-optimal play: **26%** of stages.

## Behavior types per participant

| Type | Criterion | N | % |
|------|-----------|--:|--:|
| **Stat-adherent** | stat_rate ≥ 0.70 | 46 | 45% |
| **Mixed/Explorer** | neither threshold met | 36 | 35% |
| **Deviator** | dev_rate ≥ 0.50 | 20 | 20% |

## Per-treatment role-choice fractions

| Treatment | N stages | N rounds | Stat-opt role | Dev-opt role | %F | %T | %M |
|-----------|--------:|--------:|:-------------:|:------------:|---:|---:|---:|
| `114_222_222__MFF_FMT` | 69 | 14 | M | F | 41% | 12% | 48% |
| `114_222_222__MFF_FTT` | 71 | 15 | M | F | 23% | 6% | 72% |
| `114_222_222__MFF_TMM` | 64 | 13 | M | T | 45% | 11% | 44% |
| `114_222_222__MFT_FTT` | 56 | 13 | M | F | 25% | 5% | 70% |
| `141_222_222__TFF_FFT` | 75 | 15 | T | F | 23% | 59% | 19% |
| `141_222_222__TFF_FMT` | 51 | 13 | T | F | 29% | 49% | 22% |
| `141_222_222__TFM_FFT` | 70 | 14 | T | F | 19% | 73% | 9% |
| `141_222_222__TFM_FMT` | 58 | 13 | T | F | 19% | 74% | 7% |
| `141_222_222__TMM_FFT` | 63 | 13 | T | F | 30% | 44% | 25% |
| `411_222_222__FFF_MMT` | 39 | 13 | F | M | 64% | 5% | 31% |
| `411_222_222__FFT_TMM` | 59 | 15 | F | T | 63% | 15% | 22% |
| `411_222_222__FMM_MFT` | 47 | 13 | F | M | 60% | 6% | 34% |
| `411_222_222__FMT_MFT` | 51 | 13 | F | M | 43% | 12% | 45% |
| `411_222_222__FMT_MTT` | 50 | 14 | F | M | 64% | 2% | 34% |
| `411_222_222__FMT_TMM` | 41 | 13 | F | T | 73% | 12% | 15% |

## Interpretation

Bot rounds pit the human's stat-suggested role against the true deviate-optimal role. **Some humans clearly adapt** — Deviators play deviate-optimal ≥ 50% of the time despite their stats suggesting a different role. **Others refuse to deviate** — Stat-adherents play stat-optimal ≥ 70% of the time. The middle group flips between strategies. This individual-difference pattern is what the paper's model needs to explain in Section 3.