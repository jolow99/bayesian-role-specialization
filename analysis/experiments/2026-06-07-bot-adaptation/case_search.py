"""Pick the illustrative successful-adaptation bot round for storyboard v3.

Gate (the qualitative story we want to show):
  * stage-1 role == the human's stat-optimal role (starts on the default)
  * there is a switch stage k >= 1 such that role == deviate-optimal for
    ALL stages s >= k (clean switch, stable tail — no flip-flopping back)
  * round outcome == WIN
  * >= 3 stages (>= 4 preferred via the n_stages score term)

Score (higher = clearer storyboard):
  + 12 per stable dev-optimal tail stage
  + 10 if the switch lands at stage 2-3 (visible pre-switch AND tail)
  +  1 per stage played
  +  8 per correct pre-switch bot inference (reports at stages 1..k;
       bots never switch, so correct = guessed == bot_role_map[pos]) —
       evidence the human's switch followed accurate beliefs.

Writes the top ~8 to case_candidates.md. The top candidate gets pinned
as CASE_GAME_ID / CASE_ROUND in storyboard_v3.py (reproducible).

Run from analysis/:
    uv run python experiments/2026-06-07-bot-adaptation/case_search.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common_bot import load_bot_records  # noqa: E402
from shared.constants import ROLE_SHORT  # noqa: E402

OUT_PATH = SCRIPT_DIR / "case_candidates.md"


def find_switch_stage(roles, dev_opt):
    """Smallest k >= 1 with roles[s] == dev_opt for all s >= k, or None."""
    n = len(roles)
    k = n
    while k > 0 and roles[k - 1] == dev_opt:
        k -= 1
    if k == 0 or k == n:        # dev from the start / never ends on dev
        return None
    return k


def score_case(rec):
    roles = rec["human_roles"]
    n = len(roles)
    if n < 3 or rec["outcome"] != "WIN":
        return None
    if roles[0] != rec["human_stat_opt"]:
        return None
    k = find_switch_stage(roles, rec["human_dev_opt"])
    if k is None:
        return None

    # Correct pre-switch bot inferences: reports made at stages 1..k
    # (0-indexed si in rec["inferred"]) about the constant bot roles.
    n_correct = n_reports = 0
    for si, guesses in rec["inferred"].items():
        if si > k:
            continue
        for pos, guessed in guesses.items():
            n_reports += 1
            if guessed == rec["bot_role_map"][pos]:
                n_correct += 1

    tail = n - k
    score = (tail * 12
             + (10 if k in (1, 2) else 0)   # switch at stage 2-3 (1-based)
             + n
             + 8 * n_correct)
    return {"score": score, "switch_stage": k, "tail": tail,
            "n_stages": n, "n_correct_pre": n_correct,
            "n_reports_pre": n_reports}


def main():
    records = load_bot_records()
    candidates = []
    for rec in records:
        s = score_case(rec)
        if s is not None:
            candidates.append({**rec, **s})
    candidates.sort(key=lambda c: -c["score"])

    lines = [
        "# Successful-adaptation case candidates (storyboard v3)",
        "",
        f"Gate: starts stat-optimal, clean stable switch to "
        f"deviate-optimal, WIN, >= 3 stages. {len(candidates)} of "
        f"{len(records)} clean bot rounds qualify.",
        "",
    ]
    for i, c in enumerate(candidates[:8], 1):
        traj = " → ".join(ROLE_SHORT[r] for r in c["human_roles"])
        lines += [
            f"{i}. `{c['game_id']}` r{c['round_number']} · participant "
            f"`{c['participant_id']}` · `{c['treatment_id']}` · "
            f"score {c['score']}",
            f"   - trajectory {traj} · switch at stage "
            f"{c['switch_stage'] + 1} · {c['tail']}-stage dev tail · "
            f"{c['n_correct_pre']}/{c['n_reports_pre']} correct "
            f"pre-switch bot inferences · human at position {c['pid']} · "
            f"stat-opt {ROLE_SHORT[c['human_stat_opt']]}, dev-opt "
            f"{ROLE_SHORT[c['human_dev_opt']]}",
        ]
    # Pin rule for storyboard v3: among candidates whose switch lands at
    # stage 2-3 with a >= 2-stage stable tail (the "observe, then commit"
    # story the storyboard illustrates), take the highest-scoring one.
    # The raw top scorer can be a final-stage switch (1-stage tail) ranked
    # up by its many pre-switch inferences — correct per the formula, but
    # a weak illustration of *stable* adaptation.
    pinnable = [c for c in candidates
                if c["switch_stage"] in (1, 2) and c["tail"] >= 2]
    pin = pinnable[0]
    lines += [
        "## Pinned for storyboard_v3",
        "",
        f"`{pin['game_id']}` r{pin['round_number']} participant "
        f"`{pin['participant_id']}` — highest-scoring candidate with a "
        "stage 2-3 switch and a >= 2-stage stable tail. (game_id + "
        "round_number alone is ambiguous: every player in a game has "
        "their own bot rounds, so the same (game, round) can be a bot "
        "round for up to 3 different humans.)",
        "",
    ]
    text = "\n".join(lines)
    OUT_PATH.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
