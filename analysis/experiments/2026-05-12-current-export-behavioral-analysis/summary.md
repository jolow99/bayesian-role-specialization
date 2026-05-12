# Current Export Behavioral Analysis

Descriptive behavioral analysis over the current exports in `analysis/data/exports/`.
No model parameters were fit in this analysis.

## Scope

| export | games | player_rounds | human_player_rounds | bot_player_rounds | dropout_games | dropout_player_rounds |
| --- | --- | --- | --- | --- | --- | --- |
| 04-23 | 10 | 240 | 180 | 60 | 3 | 40 |
| 04-27a | 11 | 264 | 198 | 66 | 5 | 56 |
| 04-27b | 9 | 216 | 162 | 54 | 3 | 24 |
| 05-11a | 10 | 240 | 180 | 60 | 2 | 16 |
| 05-11b | 10 | 240 | 180 | 60 | 3 | 32 |
| All | 50 | 1200 | 900 | 300 | 16 | 168 |

## Human Rounds

| export | complete_clean_team_rounds | WIN | LOSE | TIMEOUT | win_rate |
| --- | --- | --- | --- | --- | --- |
| 04-23 | 32 | 30 | 2 | 0 | 94% |
| 04-27a | 35 | 29 | 5 | 1 | 83% |
| 04-27b | 31 | 26 | 2 | 3 | 84% |
| 05-11a | 47 | 42 | 3 | 2 | 89% |
| 05-11b | 42 | 34 | 6 | 2 | 81% |
| All | 187 | 161 | 18 | 8 | 86% |

## Bot Rounds

| export | usable_bot_rounds | WIN | LOSE | TIMEOUT | win_rate | ever_deviated_rate | final_deviated_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 04-23 | 50 | 22 | 14 | 14 | 44% | 58% | 34% |
| 04-27a | 52 | 22 | 11 | 19 | 42% | 48% | 33% |
| 04-27b | 48 | 10 | 21 | 17 | 21% | 46% | 31% |
| 05-11a | 56 | 21 | 12 | 23 | 38% | 57% | 34% |
| 05-11b | 52 | 14 | 28 | 10 | 27% | 50% | 29% |
| All | 258 | 89 | 86 | 83 | 34% | 52% | 32% |

## Figures

- `figures/outcome_win_rates.png`
- `figures/human_role_behavior_by_stage.png`
- `figures/bot_role_choice_by_stage.png`
- `figures/inference_accuracy_by_stage.png`
- `figures/dropout_games.png`

## Tables

- `tables/scope_summary.csv`
- `tables/human_outcomes.csv`
- `tables/human_stage_behavior.csv`
- `tables/human_inference.csv`
- `tables/bot_outcomes.csv`
- `tables/bot_stage_behavior.csv`
- `tables/bot_inference.csv`

## Interpretation

### Dropout exclusion (sanity check)

- 50 games across 5 exports; **16 games (32%)** had at least one dropout
  player-round and are excluded from every behavioral metric below
  (`pr.is_dropout` flagged via the v3 `isDropout` column).
- 168 of 1200 player-rounds (14%) carried `is_dropout=True`. The new 05-11a
  has the cleanest data (only 2/10 games affected, 16 dropout player-rounds);
  04-27a is the noisiest (5/11 games).
- Human metrics use the strictest filter: team-round is kept only if no
  player in the team was a dropout AND no stage in that team was
  `is_bot`/auto-submitted. That yields **187 clean human team-rounds** out
  of a possible 300 (62% retention). Bot metrics drop only the dropout
  player-rounds.

### Human rounds — players cooperate well, but stat-optimal play is barely above chance

- **Win rate is high and stable across exports:** 81–94%, pooled 86%
  (161 W / 18 L / 8 T over 187 clean team-rounds). The two new exports sit
  in the middle of the range (05-11a 89%, 05-11b 81%), so adding them does
  not change the headline cooperation finding.
- **Stat-optimal rate hovers around 50% at every stage** (chance = 33%):
  pooled stage-1 = 50%, stage-5 = 42%. Players are nudged by their stats but
  are far from locking in. The combination "high win rate + low stat-optimal
  rate" implies many teams reach a *working* role split that is not the
  stat-suggested one.
- **Switch rate stays high (34–46%)** across stages, including late stages
  with high stakes. This is more thrashing than a Bayesian learner should
  produce; it explains the strong fit of stickiness-aware models in the
  metric-comparison experiment.
- **Inference accuracy is well above chance but decays:** stage-2 = 68%,
  stage-3 = 62%, stage-4 = 59%, stage-5 = 61% (chance = 33%). Decay is
  consistent with stat-optimal anchoring at stage 2 (easy because most
  teammates start near stat-optimal) and noisier inferences later, once
  switching has scrambled the signal.

### Bot rounds — the deviation-from-stat test is hard

- **Win rate collapses to 34%** (vs 86% in human rounds), with substantial
  spread by export (04-23 44% → 04-27b 21%). The new 05-11b (27%) sits at
  the low end, 05-11a (38%) near the mean. Even on the easiest export
  (04-23), bot rounds win less than half the time — the deviate-optimal
  configuration is genuinely demanding.
- **Only 52% of humans *ever* take the deviate-optimal role, and only 32%
  end the round there.** A stubborn-stat-optimal player would generate
  ~0% deviate / 100% stat; a perfectly responsive player would be near
  100% / 100%. The split says about half the humans never figure out (or
  never accept) the deviation in 5 stages.
- **The deviation rate does climb with stage,** which is the qualitative
  signature of learning: 23% → 20% → 24% → 30% → 37%. The corresponding
  stat-optimal rate falls from 62% to 42%. By stage 5, deviation has nearly
  caught up to stat-optimal, but this is *also* the smallest-sample stage
  (n=140 choices pooled).
- **Bot-round inference accuracy is noisier than human-round inference**
  (62%, 48%, 63%, 59% across stages 2–5) and dips at stage 3. The
  improvement at stages 4–5 plausibly reflects players locking in on the
  fixed bot strategies once they have observed enough.

### Bottom line

Treatment conditions look comparable enough that pooling is defensible:
human-round win rate, switch rate, and inference accuracy all sit within a
narrow band across the five exports, and the new 05-11 exports do not
introduce a behavioral discontinuity. The substantive findings from the
3-export version hold up at n=187 clean human team-rounds and n=258 clean
bot rounds: players cooperate, but their role choice is closer to a noisy
walk over plausible roles than a tight follow of stat-optimal, and most
humans struggle to fully deviate when bots force it.
