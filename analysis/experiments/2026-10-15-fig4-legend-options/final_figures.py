"""Final D5-layout figures -> final/ (names match the paper repo's figures)."""
import grouped_variants as gv
import legend_options as lo
from legend_options import tm

CASES = [("2F8H1E", 2, "R3_team_case"), ("11B0J4", 6, "R3_team_case_11B0J4_r6"),
         ("11B0J4", 8, "R3_team_case_11B0J4_r8"), ("K72DV0", 5, "R3_team_case_K72DV0_r5"),
         ("RBRN0Z", 8, "R3_team_case_RBRN0Z_r8")]
if __name__ == "__main__":
    for s, r, name in CASES:
        rec, post, preds = lo.prepare_case(s, r)
        print(name, rec["n_stages"], "stages")
        gv.render_variant(f"final/{name}", gv.D5, rec, post, preds)
