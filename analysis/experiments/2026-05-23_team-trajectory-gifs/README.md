# Team / Human Trajectory GIFs

Animated, per-environment trajectory GIFs covering both round types of
the experiment:

* **Human rounds** (`figures/`) — 15 environments × ≥ 12 teams each.
  Shows how full human teams converge (or don't) on the optimal role
  combo alongside Bayesian-Belief / -Value / -Walk predictions.
* **Bot rounds** (`figures_bot/`) — 15 treatments × ≥ 13 humans each.
  Shows whether the human managed to *deviate* from their stat-optimal
  role (the lure their stats suggest) to the deviate-optimal role (the
  one the stubborn bots need them to fill).

## Inputs

* All five current treatment-condition exports in `data/exports/` (loaded
  via `pipeline.load_human_team_records` for human rounds and
  `bot_pipeline.load_bot_round_records` for bot rounds).
* Clean games only (any game with a dropout player-round is excluded).
* The trajectory engine + stage-1 / stage-2 tuned parameters from the
  `2026-05-12-current-export-metric-comparison` experiment so the GIFs
  match the published metric numbers exactly:

  | Param                | Value                |
  |----------------------|----------------------|
  | `tau_prior`          | 4.638                |
  | `epsilon`            | 0.0624               |
  | memory strategy      | `drift_prior_0.500`  |
  | `tau_softmax` (Value)| 13.716               |
  | `tau_softmax` (Walk) | 7.207                |
  | `epsilon_switch`     | 0.559                |

## Output

* `figures/<env_id>.gif` — human-round GIFs (15 envs)
* `figures_bot/<treatment_id>.gif` — bot-round GIFs (15 treatments,
  `<stat_profile>__<optimalDeviateRolesId>`)
* `frames/<env_id>/`, `frames_bot/<treatment_id>/` — per-stage PNGs
  (gitignored)

## What a human-round frame shows

```
Title strip  : env id  •  stat profile  •  optimal combo  •  team count
HP sparkline : team / enemy HP as fraction-of-max across every turn
               (yellow band = current stage; red ▼ = enemy attack)
Per-team grid: rows = teams, columns = stages; each cell is a 3-tile
               glyph coloured by role (F red, T blue, M green).
               Future stages are faded; current stage is highlighted.
Distribution : horizontal bars = empirical frequency of each canonical
               combo among teams at this stage; coloured dots = each
               model's predicted probability for that combo.
               ★ marks the optimal combo.
```

Symmetric stat profiles (`222_222_222`, `411_222_222`, `114_222_222`,
`141_222_222`) get canonicalised combos so that interchangeable players
collapse into a single row in the distribution chart.

## What a bot-round frame shows

```
Title strip   : stat profile  ·  optimalDeviateRolesId
                "human stat-opt X  →  dev-opt Y    bots play [Z, W]    n=K"
HP sparkline  : same range-framed turn-level chart as human rounds
Per-human grid: rows = humans, columns = stages. Each cell is ONE
                role-coloured tile (the human's pick at that stage).
                Green solid border = chose deviate-optimal (succeeded).
                Brown dashed border = stuck on stat-optimal (failed).
                No border = chose neither.
Distribution  : 3 horizontal bars (F / T / M) with empirical counts;
                stat-opt row tagged in brown, dev-opt row tagged in
                green; coloured dots overlay each model's predicted
                probability for the human's role at this stage.
```

### Bot-round caveats

* The model marginals are taken at the human's **in-game position**
  (`pr.player_id`), per `CLAUDE.md` → "Bot Round Ground Truth", not
  position-averaged.
* Bot rounds use a wider variety of (`team_max_hp`, `enemy_max_hp`,
  `boss_damage`) combinations than the human rounds, and the
  precomputed value matrices in `data/human_envs_value_matrices/` only
  cover 3 of them. So:
    - **Bayesian-Belief** runs on all 15 bot treatments (it never reads
      the value matrix).
    - **Bayesian-Value / -Walk** only appear on 4 of the 15 bot
      treatments (the ones whose HP/boss config matches an available
      matrix): `114_222_222__MFF_FMT`, `411_222_222__FFF_MMT`,
      `411_222_222__FMM_MFT`, `411_222_222__FMT_MFT`. To enable them on
      every bot treatment, generate the missing value matrices for the
      bot-round HP / boss combos.
* The trajectory engine replays each round using the actual roles the
  human chose plus the fixed bot roles (`botPlayers[i].strategy.role`,
  permuted to in-game positions per `CLAUDE.md`). Posteriors therefore
  reflect what an idealised observer would believe given those
  observations.

## How to regenerate

```bash
cd analysis

# Human-round GIFs (15 files, ~3 minutes)
.venv/bin/python experiments/2026-05-23_team-trajectory-gifs/build_gifs.py

# Bot-round GIFs (15 files, ~2 minutes)
.venv/bin/python experiments/2026-05-23_team-trajectory-gifs/build_bot_gifs.py
```

## Design notes (Tufte-style)

* **Comparisons in every frame**: empirical bars vs three model dots;
  one team / human vs the others; current stage vs faded past/future;
  human's actual pick vs both the stat-optimal lure and the
  deviate-optimal target (bot rounds).
* **Layering instead of separation**: bars, model dots, and the optimal
  reference all sit on the same horizontal axis so the eye compares
  them in-place. Borders on the bot-round grid annotate each cell
  in-place instead of needing a separate "success" column.
* **Range-framed HP**: the y-axis is shown as `0 / ½ / max` rather than
  unit-mismatched raw HP, so team and enemy curves are directly
  comparable despite very different scales.
* **Micro/macro**: at a glance the grid shows convergence; up close,
  individual rows show whether convergence was monotone or noisy.

## File map

```
2026-05-23_team-trajectory-gifs/
├── README.md
├── build_gifs.py         # human-round GIF driver
├── build_bot_gifs.py     # bot-round GIF driver
├── bot_pipeline.py       # bot-round records + trajectory engine
├── bot_models.py         # per-human-position model predictions
├── figures/              # human-round GIFs (15 files)
├── figures_bot/          # bot-round GIFs (15 files)
├── frames/               # per-stage PNGs (human; gitignored)
└── frames_bot/           # per-stage PNGs (bot; gitignored)
```
