# Bot-Round Behavioral Analysis (04-23 vs 03-18 reference)

Per-stage learning curves for the human player only. Each bot round has 2 fixed-strategy AI bots; the human's stat-suggested role differs from the deviate-optimal slot they should fill (see CLAUDE.md → Bot Round Ground Truth).

## Round outcomes

| Export | n rounds | WIN | LOSE | TIMEOUT |
|---|---:|---:|---:|---:|
| 04-23 | 50 | 22 (44%) | 14 (28%) | 14 (28%) |
| 03-18 | 54 | 14 (26%) | 27 (50%) | 13 (24%) |

## Did the human deviate at all?

| Export | rounds | ever deviated | deviated on final stage |
|---|---:|---:|---:|
| 04-23 | 50 | 29 (58%) | 17 (34%) |
| 03-18 | 54 | 20 (37%) | 9 (17%) |

## Per-stage human role choice

### 04-23

| Stage | n | % deviate-opt | % stat-opt | % other |
|---:|---:|---:|---:|---:|
| 1 | 49 | 22% | 63% | 14% |
| 2 | 49 | 16% | 65% | 18% |
| 3 | 49 | 27% | 65% | 8% |
| 4 | 37 | 30% | 57% | 14% |
| 5 | 24 | 29% | 50% | 21% |

### 03-18

| Stage | n | % deviate-opt | % stat-opt | % other |
|---:|---:|---:|---:|---:|
| 1 | 54 | 9% | 74% | 17% |
| 2 | 54 | 13% | 67% | 20% |
| 3 | 52 | 23% | 65% | 12% |
| 4 | 29 | 10% | 59% | 31% |
| 5 | 23 | 26% | 39% | 35% |

## Per-stage inference accuracy (human → bot)

### 04-23

| Stage | n inferences | % correct |
|---:|---:|---:|
| 2 | 98 | 72% |
| 3 | 98 | 50% |
| 4 | 74 | 70% |
| 5 | 48 | 67% |

### 03-18

| Stage | n inferences | % correct |
|---:|---:|---:|
| 2 | 108 | 66% |
| 3 | 102 | 50% |
| 4 | 58 | 57% |
| 5 | 44 | 70% |

Chance baseline for inference = 33%.
