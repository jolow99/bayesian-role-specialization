# Per-Team / Per-Human Trajectory GIFs

Sibling experiment to `2026-05-23_team-trajectory-gifs/`. Where the
earlier one *averaged* HP / role-choice / model-prediction across teams
within an environment, this one flips the lens: each GIF follows a
**single** team-round (human round) or **single** player-round (bot
round), so every element of the frame reflects that one game.

## Why a second pass?

Aggregating across teams was useful for the population-level "did
models capture the empirical distribution" question, but it conflates
heterogeneous trajectories — the averaged HP line in particular isn't
any real game's HP, and the averaged model marginal blurs the fact that
each team has a *different* posterior because each team observed a
different sequence of role choices. These per-team GIFs let you ask
"what happened in this specific game, and what did each model think
*about this game* at each stage?"

## Output

| Folder         | Count | Content                                                        |
|----------------|-------|----------------------------------------------------------------|
| `figures/`     | 204   | One GIF per clean human team-round                             |
| `figures_bot/` | 204   | One GIF per bot player-round                                   |

File naming:
- Human: `<env_id>__<game_id_short>_r<round_number>.gif`
- Bot: `<treatment_id>__<game_id_short>_r<round_number>.gif`

`<game_id_short>` is the last 6 characters of the Empirica game ID,
which is enough to uniquely identify it within an env.

## What a human-round frame shows

```
Title strip   : team id  •  env id  •  stats  •  optimal  •  stage s of S
HP sparkline  : THIS team's actual HP across every turn
Role strip    : 1 row × stages; each cell = 3-tile role combo
                (P1·P2·P3). Current stage emphasised; future faded.
Predictions   : per-stage panel; one row per canonical combo that has
                meaningful mass under any model OR was actually chosen.
                Three horizontal bars per combo (one per model) show
                each model's predicted probability. ★ tags the combo
                this team picked; ◆ tags the env-optimal combo.
```

The combo set on the y-axis is fixed across the GIF (union of top
combos at each stage + chosen + optimal), so the rows don't reshuffle
frame-to-frame.

## What a bot-round frame shows

```
Title strip   : human id  •  treatment  •  "stat-opt X → dev-opt Y
                                            bots play [Z, W]"
HP sparkline  : THIS human's actual HP across every turn
Role strip    : 1 row × stages; each cell = the human's role tile.
                Green solid border = chose deviate-optimal (succeeded).
                Brown dashed border = stuck on stat-optimal (failed).
                Current stage emphasised; future faded.
Predictions   : 3 rows (F / T / M). Three horizontal bars per row
                (one per model) for the human's role probability under
                that model. dev-opt row tagged green, stat-opt brown;
                ★ tags the role this human picked at this stage.
```

Same value-matrix caveat as the aggregate version: Bayesian-Belief
runs on every bot record; Bayesian-Value / -Walk only render when the
HP/boss config has a matching value matrix in
`data/human_envs_value_matrices/` (4 of the 15 treatments).

## How to regenerate

```bash
cd analysis

# Human single-team GIFs (~5 min)
.venv/bin/python experiments/2026-05-24_single-team-gifs/build_team_gifs.py

# Bot single-human GIFs (~5 min)
.venv/bin/python experiments/2026-05-24_single-team-gifs/build_bot_team_gifs.py
```

## File map

```
2026-05-24_single-team-gifs/
├── README.md
├── build_team_gifs.py      # one GIF per human team-round
├── build_bot_team_gifs.py  # one GIF per bot player-round
├── figures/                # 204 human-round GIFs
├── figures_bot/            # 204 bot-round GIFs
├── frames/                 # per-stage PNGs (human; gitignored)
└── frames_bot/             # per-stage PNGs (bot; gitignored)
```

## Design notes (Tufte-style)

* **Comparison**: each frame lets you compare three models against one
  another *and* against the team's actual pick. The grouped horizontal
  bars put all three model predictions on the same baseline, so a
  reader's eye can run vertically (across combos) or horizontally
  (across models at one combo).
* **Stable y-axis**: combo ordering on the y-axis is fixed across the
  GIF, so the animation only changes bar lengths — exactly the
  variation the viewer should track. Mixing in row-reordering would
  burn cognitive load on noise.
* **Layering**: chosen-combo highlight is a low-alpha yellow band
  *behind* the bars; the bars stay readable while the highlight gently
  draws the eye. Optimal combo gets a diamond tag so it never competes
  visually with the chosen-combo star.
