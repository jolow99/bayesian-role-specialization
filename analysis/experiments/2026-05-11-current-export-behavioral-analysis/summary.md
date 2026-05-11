# Current Export Behavioral Analysis

Descriptive behavioral analysis over the current exports in `analysis/data/exports/`.
No model parameters were fit in this analysis.

## Scope

| export | games | player_rounds | human_player_rounds | bot_player_rounds | dropout_games | dropout_player_rounds |
| --- | --- | --- | --- | --- | --- | --- |
| 04-23 | 10 | 240 | 180 | 60 | 3 | 40 |
| 04-27a | 11 | 264 | 198 | 66 | 5 | 56 |
| 04-27b | 9 | 216 | 162 | 54 | 3 | 24 |
| All | 30 | 720 | 540 | 180 | 11 | 120 |

## Human Rounds

| export | complete_clean_team_rounds | WIN | LOSE | TIMEOUT | win_rate |
| --- | --- | --- | --- | --- | --- |
| 04-23 | 32 | 30 | 2 | 0 | 94% |
| 04-27a | 35 | 29 | 5 | 1 | 83% |
| 04-27b | 31 | 26 | 2 | 3 | 84% |
| All | 98 | 85 | 9 | 4 | 87% |

## Bot Rounds

| export | usable_bot_rounds | WIN | LOSE | TIMEOUT | win_rate | ever_deviated_rate | final_deviated_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 04-23 | 50 | 22 | 14 | 14 | 44% | 58% | 34% |
| 04-27a | 52 | 22 | 11 | 19 | 42% | 48% | 33% |
| 04-27b | 48 | 10 | 21 | 17 | 21% | 46% | 31% |
| All | 150 | 54 | 46 | 50 | 36% | 51% | 33% |

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
