# Bayesian Role Specialization Experiment

## Git Conventions
- Do NOT add Co-Authored-By trailers to commits.

## Overview
A multiplayer experiment (Empirica framework) studying how players learn and specialize into roles through Bayesian inference. Teams of 3 fight a boss enemy across rounds, choosing roles each stage.

## Roles & Actions
- **Fighter** (index 0, short: F) → action: ATTACK
- **Tank** (index 1, short: T) → action: BLOCK
- **Medic** (index 2, short: M) → action: HEAL

## Round Types
- **Human rounds**: All 3 players are real humans cooperating
- **Bot rounds**: 1 human + 2 AI bots with fixed-strategy roles. Tests whether the human can deviate from their natural role.

## Key Concepts

### Stat-Optimal Role
The role suggested by a player's stat profile (STR/DEF/SUP). E.g., high STR → Fighter. This is what the player would "naturally" pick based on their stats.

### Deviate-Optimal Role
The TRUE optimal role the human should play in a bot round. Different from stat-optimal — the whole point of bot rounds is to test whether players can deviate from their stat-suggested role to the actual optimal.

### Config ID Format: `optimalDeviateRolesId`
Format: `ABC_XYZ` (e.g., `FMT_TMM`)
- **First group** (`ABC`): stat-optimal roles — **1st letter = human**, 2nd & 3rd = bots
- **Second group** (`XYZ`): deviate-optimal roles — **1st letter = human**, 2nd & 3rd = bots
- The bots are hard-coded to play their **deviate-optimal** roles (2nd & 3rd letters of the *second* group). Verified empirically: `botPlayers[i].strategy.role == deviateRoles[i+1]` in 104/104 bot rounds. The human then has to deviate from stat-optimal to fill the remaining deviate-optimal slot.
- **IMPORTANT**: Index 0 in `optimalRoles[]`, `deviateRoles[]`, and the groups inside `optimalDeviateRolesId` / `statProfileId` is ALWAYS the human (logical order). This is **not** the same as the human's in-game position — see "Bot Round Ground Truth" below.

### Stat Profile ID
Format: `STR_DEF_SUP` per player (e.g., `411_222_222`). In bot rounds the first group is the human (logical index 0); in human rounds the first group is the player at in-game position 0. Bot rounds always use symmetric bot stats (`H_222_222`).

### Bot Round Ground Truth

**CRITICAL for any code that computes Bayesian posteriors or inference ground truth on bot rounds.** The config fields are logical-order (human first), but everything else — `stage.inferred_roles` keys, turn actions, `gamePlayerId` — is keyed by **in-game position**. Getting the mapping wrong silently corrupts posteriors and compares inferences against the wrong targets.

- **Human's in-game position** = `pr.player_id` (aka `gamePlayerId`). Do **not** use `config.humanRole` — it is a stored config value that does not reliably match the Empirica-assigned game position (only matches in ~27% of cases).
- **Bot in-game positions** = `sorted({0, 1, 2} - {pr.player_id})`. `botPlayers[0]` goes to the **lower** non-human position, `botPlayers[1]` to the **higher**. Verified via stable late-game inference agreement (A_ascending 70% vs A_reversed 22% vs cyclic ~50%, chance = 33%).
- **Bot role** = `botPlayers[i]["strategy"]["role"]` — already an `int` in {0, 1, 2}. Use it directly; do **not** look it up in `GAME_ROLE_TO_IDX` (that dict is keyed by role-name strings, so the lookup silently fails and produces an empty bot-role map).
- **`botPlayers` has no `position` / `playerIndex` field.** Each entry is just `{"strategy": {"type": "fixed", "role": <int>}}`. Order in the list is the only position signal.
- **Permute `player_stats` from logical to position order** before passing to `utility_based_prior` / `game_step` in bot rounds:
  ```python
  pid = pr.player_id
  others = sorted(i for i in range(3) if i != pid)
  logical_stats = np.array([[int(c) for c in part]
                            for part in rnd.stat_profile_id.split('_')])
  player_stats = np.zeros((3, 3), dtype=int)
  player_stats[pid]       = logical_stats[0]
  player_stats[others[0]] = logical_stats[1]
  player_stats[others[1]] = logical_stats[2]
  bot_role_map = {
      others[0]: int(pr.round.config['botPlayers'][0]['strategy']['role']),
      others[1]: int(pr.round.config['botPlayers'][1]['strategy']['role']),
  }
  ```
- **Inference targets** in bot rounds: `stage.inferred_roles` keys are 0-indexed in-game positions, and the human's own position (`pr.player_id`) is always absent. Compare each target against `bot_role_map[target_pos]` (constant across stages — bots never switch).

## Data Structure

### Source: `player.csv`
One row per player per game. Key columns:
- `gameID`, `gamePlayerId`, `participantID`
- `gameSummary` — JSON string with all round/stage/turn data
- `isDropout`, `droppedOutAtRound`, `droppedOutAtStage`, `bufferTimeRemaining`

### gameSummary Structure
```
gameSummary.rounds[] → each round:
  roundNumber, roundType ("human" | "bot"), outcome ("WIN" | "LOSE" | "TIMEOUT")
  config:
    optimalDeviateRolesId, statProfileId
    optimalRoles[3], deviateRoles[3]  ← index 0 = human
    humanRole                          ← stored config value, NOT reliably the in-game position — use gamePlayerId (pr.player_id) instead
    botPlayers[2]                      ← bot rounds only, each is {strategy:{type,role:<int>}}; no position field
    enemyIntentSequence                ← "1"=attack, "0"=no attack per turn
  stages[] → each stage:
    stage (number), role (chosen role name, UPPERCASE)
    inferredRoles (string like "P2: F, P3: T" or null for stage 1)
    isBot (whether this was auto-submitted by dropout replacement)
    turns[] → each turn:
      turn, action (ATTACK/BLOCK/HEAL), teamHealth, enemyHealth
```

### Inference Format
String: `"P2: F, P3: T"` — player infers other players' roles using single-letter codes (F/T/M). Parse with regex `P(\d+):\s*([FTM])`. Player indices are 1-based (P1=position 0).

**IMPORTANT — Inference timing**: Inferences made at stage N are about what happened in stage N-1. When checking correctness, compare the inference against the target's role in the **previous** stage, not the current one. In turn-level data, use `turns[ti-1]` (previous turn, which was in the old stage) not `turns[ti]` (current turn, already in new stage).

### Dropout Handling
- Dropout players get bot replacement: `getOptimalRoleForDropout()` auto-submits roles
- `autoSubmitted` flag on stage indicates bot replacement
- `isBot` field in gameSummary uses `roleEntry.autoSubmitted`
- Buffer time system: 300s shared across all rounds. `if (newBuffer < 1)` triggers dropout (epsilon to avoid float issues)

## Analysis Workspace (`analysis/`)

### Directory Structure

```
analysis/
├── shared/             # Python package — constants, data loading, inference, evaluation
│   ├── __init__.py     # Exports ANALYSIS_ROOT, DATA_ROOT, EXPORTS_DIR, ENVS_DIR, HUMAN_ENVS_DIR
│   ├── constants.py    # Role/action maps, colors, game params, known dropout IDs
│   ├── parsing.py      # Inference string parsing, config ID parsing, canonical combos
│   ├── data_loading.py # Format-aware loader from player.csv gameSummary
│   ├── env_loading.py  # MDP env config loading (values.npy + config.py, JAX-free)
│   ├── inference.py    # Bayesian update, priors, action model, game mechanics, softmax
│   └── evaluation.py   # Model-agnostic: run_predictions, compute_pearson, log-likelihood
├── data/               # gitignored
│   ├── exports/        # Timestamped Empirica CSV exports
│   ├── envs/           # ~7500 MDP environment configs
│   └── human_envs_value_matrices/  # 8 role-combo env configs
├── experiments/        # Date-prefixed experiment folders (tracked in git)
│   ├── 2026-03-24_combined_pilot_viz/
│   └── 2026-03-24_model_benchmarks/
├── legacy/             # Old notebooks (gitignored)
├── pyproject.toml
└── .venv/
```

### Setup

```bash
cd analysis
uv sync
uv pip install -e .    # installs shared/ as editable package
```

In Jupyter, select the `.venv` kernel.

### Creating a New Experiment

1. Create a folder: `analysis/experiments/YYYY-MM-DD_descriptive-name/`
2. Add notebook(s) and/or scripts inside it
3. Import from shared:

```python
from shared import EXPORTS_DIR
from shared.constants import ROLE_MAP, ROLE_NAMES, ROLE_SHORT, ROLE_COLORS
from shared.data_loading import load_all_exports, to_dataframe
from shared.inference import utility_based_prior, bayesian_update
from shared.evaluation import run_predictions, compute_pearson, extract_metrics
```

4. Save outputs (figures, tables) inside the experiment folder
5. Commit the experiment folder to git

### What Goes in `shared/` vs Experiment Folders

**`shared/` — edit only when adding genuinely reusable utilities:**

| Module | What it provides | When to edit |
|--------|------------------|--------------|
| `constants.py` | `ROLE_MAP`, `ROLE_NAMES`, `ROLE_SHORT`, `ROLE_COLORS`, `ACTION_SYMBOLS`, `ALL_ROLE_COMBOS`, `DROPOUT_GAME_IDS`, `SYMMETRIC_PROFILES`, etc. | New role/action mappings, new known dropout games, new symmetric profiles |
| `parsing.py` | `parse_inferred_roles()`, `parse_deviate_roles()`, `parse_stat_optimal_roles()`, `canonical_combo()`, `get_canonical_combos()` | New string formats to parse |
| `data_loading.py` | `load_export()`, `load_all_exports()`, `to_dataframe()`, `detect_format()`, `PlayerRound`/`RoundRecord`/`StageRecord` dataclasses | New export format versions, new fields in dataclasses |
| `env_loading.py` | `load_env_config()`, `get_env_dir()`, `make_env_loader()` | New env config formats |
| `inference.py` | `utility_based_prior()`, `uniform_prior()`, `bayesian_update()`, `action_prob()`, `preferred_action()`, `game_step()`, `softmax_role_dist()`, `combo_marginal()` | Bug fixes only — this is the shared inference engine used by all models |
| `evaluation.py` | `run_predictions()`, `compute_pearson()`, `compute_log_likelihood()`, `extract_metrics()` | New evaluation metrics |

**Experiment folders — where model-specific and analysis-specific code lives:**
- Baseline prediction functions (Random, Random Walk, Copy Others, etc.)
- Model wrappers (e.g., `make_bayesian_belief()`, `run_bayesian_belief()`)
- Parameter sweeps
- Visualization code (plotting is analysis-specific, not shared)
- Summary tables, figures, interpretation

**Rule of thumb:** If two experiments need the same function, move it to `shared/`. If only one experiment uses it, keep it in the experiment folder.

### Shared Package API Reference

#### `shared.data_loading`

```python
# Load a single export
records = load_export("data/exports/bayesian-role-specialization-2026-03-18-15-47-09")

# Load all supported exports (auto-discovers, skips v1 format)
records = load_all_exports()

# Load specific exports
records = load_all_exports(data_dirs=[EXPORTS_DIR / "...-03-06-...", EXPORTS_DIR / "...-03-18-..."])

# Flatten to DataFrame for quick exploration
df = to_dataframe(records)
```

Returns `list[PlayerRound]`. Each `PlayerRound` has:
- `export_name`, `game_id`, `player_id`, `participant_id`, `is_dropout`
- `round`: `RoundRecord` with `round_number`, `round_type`, `outcome`, `config`, `player_stats`, `stages`, `optimal_roles`, `deviate_roles`, `stat_profile_id`, `enemy_intent_sequence`
- Each stage: `StageRecord` with `stage`, `role`, `role_idx`, `inferred_roles` (already parsed dict), `is_bot`, `turns`

**Format support:** v2 (Feb 13) and v3 (Mar 6, Mar 18) exports. v1 exports (Jan 25, Jan 28) lack `gameSummary` and are skipped with a message.

#### `shared.inference`

```python
# Build prior from player stats
prior = utility_based_prior(player_stats, tau=2.0)  # (3,3,3) array

# Update posterior after observing actions
posterior = bayesian_update(prior, actions=[0,1,2], intent=1, team_hp=10, team_max_hp=15, epsilon=0.2)

# Get preferred action for a role
action = preferred_action(role=F, intent=1, team_hp=10, team_max_hp=15)

# Advance game state
new_team_hp, new_enemy_hp = game_step(intent, team_hp, enemy_hp, actions, player_stats, boss_damage, team_max_hp)

# Softmax over expected values (used by bayesian-value model)
role_dist = softmax_role_dist(agent_i=0, intent=1, team_hp=10, enemy_hp=20, prior=prior, values=values, tau=0.1)
```

#### `shared.evaluation`

```python
# Run any model on team-round records
# predict_fn(record) -> list[{predicted_dist, human_combo, model_marginal}]
results = run_predictions(records, predict_fn)

# Compute Pearson correlation
corrs = compute_pearson(results)
metrics = extract_metrics(corrs)  # {'combo_r': float, 'marg_r': float}

# Log-likelihood
ll = compute_log_likelihood(results)
```

#### `shared.env_loading`

```python
# Load one env
env = load_env_config("data/envs/139")
# Returns: {values, player_stats, boss_damage, team_max_hp, enemy_max_hp}

# Cached loader (prefers human_envs_value_matrices/, falls back to envs/)
loader = make_env_loader()
env = loader(role_combo="FTM")
env = loader(env_id=139)
```

### Model Names

| Model | Description |
|-------|-------------|
| **Bayesian-Value** | Softmax over expected values given posterior beliefs. Formerly "Bayesian (finetuned)". |
| **Bayesian-Belief** | Marginalizes posterior directly for role predictions. Formerly "Posterior Sampling". |

### Data Loading: `load_team_rounds` vs `load_all_exports`

Two data loading paths exist:

| Function | Source | Returns | Use case |
|----------|--------|---------|----------|
| `shared.data_loading.load_all_exports()` | `shared/` | `list[PlayerRound]` — per-player, includes bot rounds, typed dataclasses | Human behavior analysis, visualization, inference accuracy |
| `oms.load_team_rounds()` | `computational_model/analysis/online_model_sim.py` | `list[dict]` — per-team, human rounds only, includes `env_config` with values.npy | Model comparison (needs value matrices for softmax/optimal baselines) |

For model benchmarks, use `oms.load_team_rounds()` because it attaches `env_config` (value matrices). For behavioral analysis, use `shared.data_loading.load_all_exports()`.

## Analysis Conventions

### Key Questions
1. **Human-round gameplay reasonable?** — Win rate, stat-optimal role adherence, inference accuracy
2. **Bot-round gameplay reasonable?** — Win rate, deviation to deviate-optimal vs stubborn stat-optimal, inference accuracy

### Plot Structure (per round)
Three-panel figure (figsize 14x8 or 16x12):
1. **Top**: Role choices over turns (y-axis: Fighter/Tank/Medic). Players colored P1=red, P2=blue, P3=green. Lines connecting roles across turns. Action labels (A/B/H) above points. Per-player optimal role as colored dashed reference line. Red shading = enemy attacks. Player stats (STR/DEF/SUP) in legend.
2. **Middle**: Inferences. Human rounds: 6 rows (P1→P2, P1→P3, P2→P1, P2→P3, P3→P1, P3→P2). Bot rounds: rows for Human→each bot. Green = correct, red = incorrect. Role letter labels with colored boxes.
3. **Bottom**: Health bars (team blue, enemy red) per turn, including turn 0 (starting HP).

### Summary Tables
Use markdown tables with: Metric | Value | Notes/Interpretation. Bold key numbers. Include chance-level baselines for inference (33.3%).

### Clean vs Dropout Games
Always separate analysis for clean games (no dropouts) vs all games. Flag dropout games in plots with `[DROPOUT GAME]` tag.
