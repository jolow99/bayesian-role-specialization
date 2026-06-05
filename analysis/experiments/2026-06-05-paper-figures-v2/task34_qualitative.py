"""Tasks 3 & 4 — render the redesigned qualitative figures.

Figure 3 (qualitative_best_respond_v2.png): the MFT round — stats
114_222_222, game 01KQ6YDA7NHWHSTQNZB82F8H1E round 2. Team plays FTT in
stage 1, then locks onto the value-optimal MFT for stages 2-4 and wins.

Figure 4 (qualitative_flip_flop_v2.png): the TFF round — stats
141_222_222, game 01KRBT2X2GA4BYS41KYT2R4QJ6 round 5. The two
identical-stat players (P2, P3, both 222) mirror each other (three-medic
stage 2) before the team converges to TFM and wins.
"""

from __future__ import annotations

from common import (
    FIGURES_DIR, load_clean_human_teams, load_human_team_records,
)
from qualitative_v2 import render_round_v2

ROUNDS = {
    "qualitative_best_respond_v2.png": {
        "game_id": "01KQ6YDA7NHWHSTQNZB82F8H1E",
        "round_number": 2,
        "mirror_pair": None,
        "title": "Best response: locking onto the optimal combo (MFT)\n"
                 "stats 114 / 222 / 222 · round 2 · WIN",
        "caption": None,
    },
    "qualitative_flip_flop_v2.png": {
        "game_id": "01KRBT2X2GA4BYS41KYT2R4QJ6",
        "round_number": 5,
        "mirror_pair": (1, 2),   # identical-stat players P2, P3 (both 222)
        "title": "Symmetry-breaking failure and recovery\n"
                 "stats 141 / 222 / 222 · round 5 · WIN",
        "caption": "P2 and P3 have identical stats (2/2/2). In stage 2 the "
                   "team collapses into three medics (“mirror”) "
                   "before P2 and P3 split into complementary roles and the "
                   "team converges.",
    },
    # Task-5 runner-up for the flip-flop story: far more dramatic mirroring
    # (the identical-stat pair moves in lock-step T→M→F→T for four straight
    # stages) but the round TIMEOUTs and never truly converges. Rendered as
    # an alternative so the team can pick.
    "qualitative_flip_flop_v2_alt.png": {
        "game_id": "01KRBKSTM48HJWYZ0J4SRBRN0Z",
        "round_number": 3,
        "mirror_pair": (1, 2),
        "title": "Symmetry-breaking failure (alternative example)\n"
                 "stats 222 / 222 / 222 · round 3 · TIMEOUT",
        "caption": "P2 and P3 (identical stats) switch in lock-step — "
                   "T→M→F→T over four stages — splitting only in the final "
                   "stage; the round times out before the team converges.",
    },
}


def main():
    teams = load_clean_human_teams()
    records = load_human_team_records(verbose=False)
    rec_by_key = {(r["game_id"], r["round_number"]): r for r in records}

    for fname, spec in ROUNDS.items():
        key = (spec["game_id"], spec["round_number"])
        team_prs = next((v for k, v in teams.items()
                         if (k[1], k[2]) == key), None)
        record = rec_by_key.get(key)
        if team_prs is None or record is None:
            print(f"  !! round not found: {key}")
            continue
        opt, played = render_round_v2(
            team_prs, record, FIGURES_DIR / fname,
            title=spec["title"], mirror_pair=spec["mirror_pair"],
            caption=spec["caption"])
        print(f"  {fname}: played {' '.join(played)} | "
              f"optimal {' '.join(str(o) for o in opt)}")


if __name__ == "__main__":
    main()
