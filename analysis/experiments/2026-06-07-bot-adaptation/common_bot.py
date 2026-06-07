"""Shared scaffolding for the 2026-06-07 bot-adaptation experiment.

Question: can humans adapt to suboptimal, stubborn, or symmetrical cases?
Bot rounds put 1 human with 2 fixed-strategy ("stubborn") bots that play
their deviate-optimal roles; the human must abandon their stat-optimal
role to fill the remaining deviate-optimal slot.

Built on the same stack as 2026-06-07-epistemic-rationality:

  * Stage-1 fitted Bayesian observer (tau_prior, epsilon, memory drift)
    from the 2026-05-25 full pipeline — asserted byte-identical to the
    05-12 fit that 06-05's common.py loads.
  * compute_posteriors from the 05-12 pipeline, fed a POSITION-ORDER
    role_seq (human's logged roles at pid, constant bot roles at the bot
    positions) — it reconstructs all players' actions via
    preferred_action, which is exactly the bots' fixed behavior.
  * Game-UI role emoji 🤺 / 💂 / 👩🏻‍⚕️ rasterized via Apple Color Emoji.
    The medic is a ZWJ sequence that needs Raqm shaping — Pillow is built
    from source against Homebrew libraqm (pyproject [tool.uv]
    no-binary-package). _emoji_is_shaped() fails loudly if that breaks,
    routing to assets/role_medic.png when present.

Bot-round ground truth (CLAUDE.md): ALL positional facts come from
shared.data_loading.build_bot_round_layout — never config.humanRole.
Logical index 0 of optimal_roles / deviate_roles is the human.
Reports at stage s pair with posteriors[s] (belief from stages 0..s-1);
bots never switch, so report correctness = guessed == bot_role_map[pos].
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
V2_DIR = SCRIPT_DIR.parent / "2026-06-05-paper-figures-v2"
PIPELINE_DIR = SCRIPT_DIR.parent / "2026-05-12-current-export-metric-comparison"
sys.path.insert(0, str(V2_DIR))
sys.path.insert(0, str(PIPELINE_DIR))

from common import emoji_array, load_stage1, target_marginal  # noqa: E402,F401
from pipeline import (  # noqa: E402
    EXPORT_DIRS, compute_posteriors, discover_dropout_games, filter_clean_prs,
)
from shared.constants import SYMMETRIC_PROFILES  # noqa: E402
from shared.data_loading import (  # noqa: E402
    build_bot_round_layout, load_all_exports,
)
from shared.parsing import parse_stat_optimal_roles  # noqa: E402

# Stage-1 source of truth: the 05-25 full pipeline (same convention as
# 2026-06-07-epistemic-rationality). common.py loads the 05-12 fit — the
# two are byte-identical, asserted below so any re-fit fails loudly here.
FULL_PIPELINE_STAGE1 = (SCRIPT_DIR.parent / "2026-05-25-full-pipeline"
                        / "stage1_inference" / "best_inference_params.json")

OUT_DIR = SCRIPT_DIR / "stuff to incorporate"
OUT_DIR.mkdir(exist_ok=True)

# Game-UI role icons (human_experiment/client/src/constants.js).
ROLE_EMOJI_GAME = {0: "🤺", 1: "💂", 2: "👩🏻‍⚕️"}


def load_stage1_canonical():
    """05-25 Stage-1 params + MemoryStrategy, cross-checked vs common.py."""
    with open(FULL_PIPELINE_STAGE1) as f:
        s1_canon = json.load(f)
    s1, strat = load_stage1()
    for k in ("tau_prior", "epsilon", "memory_strategy"):
        assert s1_canon[k] == s1[k], (
            f"Stage-1 param mismatch on '{k}': 05-25 full pipeline has "
            f"{s1_canon[k]!r}, common.py (05-12) has {s1[k]!r} — "
            f"re-point common.py or update this experiment.")
    assert abs(s1_canon["tau_prior"] - 4.6385) < 1e-3, s1_canon["tau_prior"]
    assert abs(s1_canon["epsilon"] - 0.0624) < 1e-3, s1_canon["epsilon"]
    assert s1_canon["memory_strategy"] == "drift_prior_0.500", \
        s1_canon["memory_strategy"]
    return s1_canon, strat


# ──────────────────────────────────────────────────────────────────────
# Data: clean bot player-rounds → normalized records
# ──────────────────────────────────────────────────────────────────────

def load_clean_bot_prs(verbose: bool = True):
    """Clean-game bot-round PlayerRounds from the 5 current exports."""
    all_prs = load_all_exports(data_dirs=EXPORT_DIRS)
    dropout_games = discover_dropout_games(all_prs)
    clean = filter_clean_prs(all_prs, dropout_games)
    bot_prs = [pr for pr in clean if pr.round.round_type == "bot"
               and pr.round.config.get("optimalDeviateRolesId")
               and pr.round.stages]
    if verbose:
        n_participants = len({pr.participant_id for pr in bot_prs})
        print(f"[common_bot] {len(bot_prs)} clean bot rounds, "
              f"{n_participants} participants")
    return bot_prs


def bot_round_record(pr):
    """Normalize one bot-round PlayerRound into a flat dict.

    Positional truth via build_bot_round_layout; logical-order config only
    for the human's stat-optimal / deviate-optimal roles (index 0 = human).
    Human actions/HP are LOGGED (stage.turns); bot actions are NOT logged
    and must be reconstructed (see storyboard_v3).
    """
    layout = build_bot_round_layout(pr)
    rnd = pr.round
    dev_id = rnd.config["optimalDeviateRolesId"]

    human_stat_opt = int(rnd.optimal_roles[0])
    human_dev_opt = int(rnd.deviate_roles[0])
    # optimalRoles[] and the first group of optimalDeviateRolesId encode the
    # same thing — fail loudly if they ever disagree.
    assert human_stat_opt == int(parse_stat_optimal_roles(dev_id)[0]), \
        f"optimalRoles[0] disagrees with {dev_id!r} for game {pr.game_id}"
    assert human_stat_opt != human_dev_opt, (
        f"stat-opt == dev-opt ({human_stat_opt}) in game {pr.game_id} "
        f"r{rnd.round_number} — bot rounds are designed to differ")

    human_roles = [int(s.role_idx) for s in rnd.stages]
    stage_turns = [list(s.turns) for s in rnd.stages]
    n_turns = sum(len(t) for t in stage_turns)
    # enemyIntentSequence is longer than the played round — clip.
    turn_intent = [int(c) for c in rnd.enemy_intent_sequence[:n_turns]]

    # Reports made at stage s (about stage s-1); keys = in-game bot
    # positions; the human's own position is never present.
    inferred = {si: dict(stage.inferred_roles)
                for si, stage in enumerate(rnd.stages)
                if stage.inferred_roles}

    return {
        "export_name": pr.export_name,
        "game_id": pr.game_id,
        "participant_id": pr.participant_id,
        "round_number": int(rnd.round_number),
        "outcome": rnd.outcome,
        "treatment_id": f"{rnd.stat_profile_id}__{dev_id}",
        "stat_profile_id": rnd.stat_profile_id,
        "symmetry": SYMMETRIC_PROFILES.get(rnd.stat_profile_id),
        "pid": layout.pid,
        "others": list(layout.others),
        "bot_role_map": dict(layout.bot_role_map),
        "player_stats": layout.player_stats.astype(float),
        "human_stat_opt": human_stat_opt,
        "human_dev_opt": human_dev_opt,
        "human_roles": human_roles,
        "stage_turns": stage_turns,
        "turn_intent": turn_intent,
        "inferred": inferred,
        "team_max_hp": int(rnd.config.get("maxTeamHealth", 15)),
        "enemy_max_hp": int(rnd.config.get("maxEnemyHealth", 30)),
        "boss_damage": float(rnd.config.get("bossDamage", 2)),
        "eis": rnd.enemy_intent_sequence,
    }


def load_bot_records(verbose: bool = True):
    return [bot_round_record(pr) for pr in load_clean_bot_prs(verbose)]


# ──────────────────────────────────────────────────────────────────────
# Bayesian observer posteriors on a bot round
# ──────────────────────────────────────────────────────────────────────

def bot_posteriors(rec, s1, strat):
    """posteriors[s] = start-of-stage-s belief; pairs with reports made
    at stage s. role_seq is POSITION order: human's logged roles at pid,
    constant bot roles at the bot positions."""
    role_seq = []
    for r in rec["human_roles"]:
        roles = [0, 0, 0]
        roles[rec["pid"]] = r
        for pos, br in rec["bot_role_map"].items():
            roles[pos] = br
        role_seq.append(roles)
    data = {
        "player_stats": rec["player_stats"],
        "boss_damage": rec["boss_damage"],
        "team_max_hp": rec["team_max_hp"],
        "enemy_max_hp": rec["enemy_max_hp"],
        "eis": rec["eis"],
        "role_seq": role_seq,
        "queries": [],
    }
    return compute_posteriors(data, s1["tau_prior"], s1["epsilon"], strat)


# ──────────────────────────────────────────────────────────────────────
# Emoji glyphs (game-UI role icons)
# ──────────────────────────────────────────────────────────────────────

def _pad_square(arr):
    """Pad an RGBA crop to square so imshow extents don't distort it."""
    h, w = arr.shape[:2]
    side = max(h, w)
    out = np.zeros((side, side, 4), dtype=arr.dtype)
    y0, x0 = (side - h) // 2, (side - w) // 2
    out[y0:y0 + h, x0:x0 + w] = arr
    return out


def _emoji_is_shaped(arr) -> bool:
    """ZWJ sequences render as side-by-side glyphs (~2x wide) when text
    shaping is unavailable; a shaped single glyph is roughly square."""
    h, w = arr.shape[:2]
    return w / h <= 1.6


def emoji_glyph(role_idx: int):
    """Square RGBA array for a role's game-UI emoji.

    Prefers live rasterization (needs Raqm-enabled Pillow for the medic
    ZWJ sequence); falls back to a committed assets/ PNG; raises if both
    are unavailable rather than rendering a broken double-width glyph.
    """
    arr = emoji_array(ROLE_EMOJI_GAME[role_idx])
    if arr is not None and _emoji_is_shaped(arr):
        return _pad_square(arr)
    asset = SCRIPT_DIR / "assets" / ("role_medic.png" if role_idx == 2
                                     else f"role_{role_idx}.png")
    if asset.exists():
        from PIL import Image
        return _pad_square(np.asarray(Image.open(asset).convert("RGBA")))
    raise RuntimeError(
        f"Emoji {ROLE_EMOJI_GAME[role_idx]!r} did not shape into a single "
        f"glyph and no fallback asset exists at {asset}. Rebuild Pillow "
        "against libraqm (see analysis/pyproject.toml [tool.uv]) or commit "
        "a pre-rendered PNG.")
