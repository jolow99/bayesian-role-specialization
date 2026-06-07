# Symmetry-breaking in human rounds

Scope: 204 clean human team-rounds (5 exports). Row unit = (team-round, player pair, stage); clash = both players of the pair chose the same role. Identical-stat pairs come from the 114/141/411_222_222 profiles (one 222/222 pair each) and 222_222_222 (fully symmetric, 3 pairs); 411_141_114 rounds contribute stat-distinct pairs only. All CIs are percentile cluster bootstraps over games (10,000 resamples) — the same 3 participants play every round of a game.

## Clash rate by stage (the figure)

| Stage | n identical | P(clash) identical | 95% CI | n distinct | P(clash) distinct | 95% CI |
|---|--:|--:|---|--:|--:|---|
| 1 | 242 | 0.388 | [0.328, 0.453] | 370 | 0.319 | [0.263, 0.378] |
| 2 | 236 | 0.352 | [0.297, 0.408] | 367 | 0.256 | [0.209, 0.302] |
| 3 | 220 | 0.355 | [0.287, 0.426] | 336 | 0.262 | [0.215, 0.309] |
| 4 | 133 | 0.323 | [0.243, 0.412] | 192 | 0.240 | [0.187, 0.287] |
| 5 | 44 | 0.250 | [0.170, 0.342] | 37 | 0.162 | [0.056, 0.268] |

Pooled over all stages, identical-stat pairs clash **+0.083** [+0.033, +0.136] more often than stat-distinct pairs (cluster-bootstrap difference, game clusters).

## Trajectory-level stats

### Identical-stat pairs (all) (n = 242 pair-rounds)

- P(clash at stage 1): **0.388** [0.327, 0.453]
- P(split at final stage): **0.649** [0.573, 0.721]
- mirror switches (simultaneous same-role switches): 70 total
- of 94 pairs clashing at stage 1, 71 split at some point (mean stages to first split 1.38)

### — of which fully-symmetric rounds (222_222_222) (n = 120 pair-rounds)

- P(clash at stage 1): **0.442** [0.341, 0.547]
- P(split at final stage): **0.600** [0.496, 0.692]
- mirror switches (simultaneous same-role switches): 39 total
- of 53 pairs clashing at stage 1, 40 split at some point (mean stages to first split 1.43)

### — of which one-pair rounds (H_222_222) (n = 122 pair-rounds)

- P(clash at stage 1): **0.336** [0.262, 0.407]
- P(split at final stage): **0.697** [0.613, 0.778]
- mirror switches (simultaneous same-role switches): 31 total
- of 41 pairs clashing at stage 1, 31 split at some point (mean stages to first split 1.32)

### Stat-distinct pairs (control) (n = 370 pair-rounds)

- P(clash at stage 1): **0.319** [0.262, 0.377]
- P(split at final stage): **0.724** [0.686, 0.763]
- mirror switches (simultaneous same-role switches): 45 total
- of 118 pairs clashing at stage 1, 90 split at some point (mean stages to first split 1.28)

## Interpretation

Symmetry has a real but modest cost, and humans resolve it behaviorally. Identical-stat pairs clash consistently more than the stat-distinct control at every stage (pooled difference +0.083), most at stage 1 of fully-symmetric 222_222_222 rounds (0.44), where stats give no tie-breaking signal at all. The stage-1 clash rate sits near chance rather than far above it — players already randomize/diversify somewhat from the start. Crucially, clashes do not persist: ~3/4 of identical pairs that start clashing split within ~1.4 stages on average, and both pair types' clash rates decline across the round — consistent with players using teammates' observed behavior (not stats) to settle who takes the contested role. Mirror switches (both players switching into the same role simultaneously) do occur (70 among identical pairs) — the symmetry-breaking failure mode the qualitative flip-flop figure illustrates.
