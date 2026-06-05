"""Shared scaffolding for the 2026-06-05 paper-figures-v2 experiment.

Everything is built on the 2026-05-12-current-export-metric-comparison
pipeline (same 5 exports, clean-teams filter, Stage-1 inference engine).
This module only adds:

  * Stage-1 fitted-parameter loading (tau_prior, epsilon, memory strategy).
  * Observer-aware inference queries (the pipeline's Stage-1 queries drop
    the observer id, which tasks 1 and 3/4 need).
  * Apple-Color-Emoji rasterization for the qualitative v2 figures
    (matplotlib's Agg backend cannot render color emoji; Pillow can, at
    the font's native strike size of 160 px).
  * Per-stage value ranking of the played joint combo (same expected-value
    convention as 2026-05-28-paper-figures/section2_best_response.py:
    eap-weighted mix of intent-0/intent-1 value slices at start-of-stage HP).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent / "2026-05-12-current-export-metric-comparison"
sys.path.insert(0, str(PIPELINE_DIR))

from pipeline import (  # noqa: E402
    EXPORT_DIRS, MemoryStrategy, compute_posteriors, discover_dropout_games,
    filter_clean_prs, load_human_inference_records, load_human_team_records,
    stage1_evaluate, strategy_from_params,
)
from shared.constants import (  # noqa: E402
    ROLE_CHAR_TO_IDX, ROLE_NAMES, ROLE_SHORT, TURNS_PER_STAGE,
)
from shared.data_loading import load_all_exports  # noqa: E402
from shared.inference import game_step, preferred_action  # noqa: E402

STAGE1_PARAMS_PATH = (PIPELINE_DIR / "stage1_inference"
                      / "best_inference_params.json")
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Paper role palette (matches shared.constants.ROLE_COLORS).
ROLE_COLORS_IDX = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71"}

# Role emoji. The game UI's role icons are 🤺/💂/👩🏻‍⚕️
# (human_experiment/client/src/constants.js) but the medic is a ZWJ sequence
# Pillow can't shape without Raqm, so we use the sword/shield/pill set —
# ⚔️ and 🛡️ are the game's own action icons, which map 1:1 onto roles.
ROLE_EMOJI = {0: "⚔️", 1: "🛡️", 2: "💊"}


def load_stage1():
    """Fitted Stage-1 params + the MemoryStrategy object."""
    with open(STAGE1_PARAMS_PATH) as f:
        s1 = json.load(f)
    strat = strategy_from_params(
        s1.get("memory_strategy", "full"), s1.get("window"),
        s1.get("drift_delta", 0.0))
    return s1, strat


# ──────────────────────────────────────────────────────────────────────
# Clean human teams with observer-aware inference queries
# ──────────────────────────────────────────────────────────────────────

def load_clean_human_teams(verbose: bool = True):
    """{(export, game_id, round_number): [pr0, pr1, pr2]} for clean teams."""
    all_prs = load_all_exports(data_dirs=EXPORT_DIRS)
    dropout_games = discover_dropout_games(all_prs)
    clean = filter_clean_prs(all_prs, dropout_games)

    human_teams = defaultdict(list)
    for pr in clean:
        if pr.round.round_type == "human":
            human_teams[(pr.export_name, pr.game_id,
                         pr.round.round_number)].append(pr)
    teams = {k: sorted(v, key=lambda p: p.player_id)
             for k, v in human_teams.items() if len(v) == 3}
    if verbose:
        print(f"[common] {len(teams)} clean human team-rounds")
    return teams


def prepare_team(team_prs):
    """Mirror of pipeline._prepare_human_team, but queries keep the observer.

    Returns the same dict compute_posteriors() expects, with
    ``queries`` = (observer_pos, stage_idx, target_pos, guessed_role,
    true_prev_role).
    """
    rnd = team_prs[0].round
    config = rnd.config
    parts = rnd.stat_profile_id.split("_")
    player_stats = np.array([[int(c) for c in p] for p in parts], dtype=float)

    player_roles = {pr.player_id: [s.role_idx for s in pr.round.stages]
                    for pr in team_prs}
    n_stages = max(len(v) for v in player_roles.values())
    role_seq = []
    for s in range(n_stages):
        roles = [0, 0, 0]
        for pid, rs in player_roles.items():
            if s < len(rs):
                roles[pid] = rs[s]
        role_seq.append(roles)

    queries = []
    for pr in team_prs:
        for si, stage in enumerate(pr.round.stages):
            if si == 0 or not stage.inferred_roles:
                continue
            for target_pos, guessed in stage.inferred_roles.items():
                if (target_pos not in player_roles
                        or si - 1 >= len(player_roles[target_pos])):
                    continue
                true_prev = player_roles[target_pos][si - 1]
                queries.append((pr.player_id, si, target_pos, guessed,
                                true_prev))

    return {
        "player_stats": player_stats,
        "boss_damage": config.get("bossDamage", 2),
        "team_max_hp": config.get("maxTeamHealth", 15),
        "enemy_max_hp": config.get("maxEnemyHealth", 30),
        "eis": rnd.enemy_intent_sequence,
        "role_seq": role_seq,
        "queries": queries,
    }


def target_marginal(posterior, target_pos):
    """Normalized marginal over one player's role from a (3,3,3) joint."""
    marg = np.sum(posterior, axis=tuple(j for j in range(3) if j != target_pos))
    t = marg.sum()
    return marg / t if t > 0 else np.ones(3) / 3.0


# ──────────────────────────────────────────────────────────────────────
# Value ranking of the played combo (section2 conventions)
# ──────────────────────────────────────────────────────────────────────

def replay_state_per_stage(record):
    """Yield (stage_idx, team_hp, enemy_hp) at the START of each stage."""
    env = record["env_config"]
    player_stats = env["player_stats"]
    boss_damage = env["boss_damage"]
    team_max_hp = env["team_max_hp"]

    team_hp = float(team_max_hp)
    enemy_hp = float(env["enemy_max_hp"])
    lds = record["lds"]
    turn_idx = 0
    for s, combo in enumerate(record["stage_roles"]):
        if team_hp <= 0 or enemy_hp <= 0 or turn_idx >= len(lds):
            return
        yield s, team_hp, enemy_hp
        roles = [ROLE_CHAR_TO_IDX[c] for c in combo]
        for _ in range(TURNS_PER_STAGE):
            if turn_idx >= len(lds) or team_hp <= 0 or enemy_hp <= 0:
                break
            intent = int(lds[turn_idx])
            actions = [preferred_action(roles[i], intent, team_hp, team_max_hp)
                       for i in range(3)]
            team_hp, enemy_hp = game_step(intent, team_hp, enemy_hp, actions,
                                          player_stats, boss_damage,
                                          team_max_hp)
            turn_idx += 1


def stage_value_vector(record, thp_f, ehp_f):
    """eap-weighted expected value of all 27 combos at a stage-start state."""
    env = record["env_config"]
    values = env["values"]
    lds = record["lds"]
    eap = sum(lds) / len(lds) if lds else 0.5
    thp = min(int(thp_f), values.shape[2] - 1)
    ehp = min(int(ehp_f), values.shape[3] - 1)
    if thp < 0 or ehp < 0:
        return None
    return (1.0 - eap) * values[:, 0, thp, ehp] + eap * values[:, 1, thp, ehp]


def combo_to_idx(combo: str) -> int:
    return (ROLE_CHAR_TO_IDX[combo[0]] * 9 + ROLE_CHAR_TO_IDX[combo[1]] * 3
            + ROLE_CHAR_TO_IDX[combo[2]])


def idx_to_combo(idx: int) -> str:
    return ROLE_SHORT[idx // 9] + ROLE_SHORT[(idx // 3) % 3] + ROLE_SHORT[idx % 3]


def compute_rank_rows(records):
    """One row per team-stage: rank of the played combo among all 27."""
    rows = []
    for ri, rec in enumerate(records):
        for s, thp_f, ehp_f in replay_state_per_stage(rec):
            vals = stage_value_vector(rec, thp_f, ehp_f)
            if vals is None:
                continue
            chosen = rec["stage_roles"][s]
            order = np.argsort(-vals)
            rank = int(np.where(order == combo_to_idx(chosen))[0][0]) + 1
            rows.append({
                "record_idx": ri,
                "game_id": rec["game_id"],
                "round_number": int(rec["round_number"]),
                "stage": int(s),
                "rank": rank,
                "chosen_combo": chosen,
                "optimal_combo": idx_to_combo(int(order[0])),
                "env_id": rec["env_id"],
                "stat_profile": rec["stat_profile"],
            })
    return rows


# ──────────────────────────────────────────────────────────────────────
# Emoji rasterization (Apple Color Emoji via Pillow)
# ──────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=32)
def emoji_array(emoji: str):
    """RGBA numpy array of an emoji, tightly cropped. None if unavailable."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(
            "/System/Library/Fonts/Apple Color Emoji.ttc", 160)
    except Exception:
        return None
    img = Image.new("RGBA", (320, 320), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((160, 160), emoji, font=font, embedded_color=True, anchor="mm")
    arr = np.asarray(img)
    ys, xs = np.where(arr[..., 3] > 0)
    if len(ys) == 0:
        return None
    return arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
