# Bot-round adaptation — can humans adapt to stubborn teammates?

Scope: 204 clean bot rounds (5 exports), 102 participants. In each bot round, 2 fixed-strategy bots play their deviate-optimal roles; the human's stat-optimal role always differs from their deviate-optimal role, so adapting means abandoning the stat-suggested default. All CIs are percentile cluster bootstraps over participants (10,000 resamples).

## Headline numbers (participant means, 05-28 conventions)

- Stat-optimal play: **60%** of stages.
- Deviate-optimal play: **26%** of stages.

## Behavior types per participant

| Type | Criterion | N | % |
|------|-----------|--:|--:|
| **Stat-adherent** | stat_rate ≥ 0.70 | 46 | 45% |
| **Mixed/Explorer** | neither threshold met | 36 | 35% |
| **Deviator** | dev_rate ≥ 0.50 | 20 | 20% |

## Adaptation by stage (Figure 1)

Row unit = (bot round, stage). Cluster-bootstrap 95% CIs over participants.

| Stage | n | P(deviate-opt) | 95% CI | P(stat-opt) | 95% CI | P(other) | 95% CI |
|---|--:|--:|---|--:|---|--:|---|
| 1 | 204 | 0.225 | [0.162, 0.294] | 0.642 | [0.564, 0.721] | 0.132 | [0.083, 0.186] |
| 2 | 204 | 0.196 | [0.132, 0.265] | 0.657 | [0.578, 0.735] | 0.147 | [0.098, 0.196] |
| 3 | 201 | 0.244 | [0.183, 0.307] | 0.642 | [0.569, 0.713] | 0.114 | [0.075, 0.158] |
| 4 | 142 | 0.310 | [0.234, 0.388] | 0.500 | [0.412, 0.587] | 0.190 | [0.129, 0.255] |
| 5 | 113 | 0.381 | [0.283, 0.479] | 0.451 | [0.353, 0.554] | 0.168 | [0.102, 0.241] |

## Symmetry breakdown

Team stat-profile symmetry class (`SYMMETRIC_PROFILES`): `last_two` = the two bots share symmetric 222 stats while the human has a distinct profile; `all` would mean the human is 222 as well (fully symmetrical).

| Symmetry class | n stage-rows | n participants | P(deviate-opt) | 95% CI | P(stat-opt) | 95% CI |
|---|--:|--:|--:|---|--:|---|
| `last_two` | 864 | 102 | 0.257 | [0.209, 0.308] | 0.597 | [0.535, 0.658] |

## Interpretation

Stat-optimal play starts dominant and falls across stages while deviate-optimal play roughly doubles — evidence that a meaningful fraction of humans integrate their teammates' observed behavior and adapt away from their stat-suggested default. The per-participant split shows this is driven by individual differences (Deviators adapt, Stat-adherents never do) rather than uniform partial adaptation.
