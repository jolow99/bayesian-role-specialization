"""2026-10-22 — paper figures and tables after the 10-20 lapse refit.

1. Fig. 4 + SI Figs. S1–S4 (R3 team cases): re-rendered with the refit Bayesian-Walk-BR
   parameters through the 10-15 final6pt code. Old-vs-new probability on each chosen role
   goes to case_probabilities.md for the captions.
2. Fig. 5 (R4): individual fits over 11 models. Top-k (agg-LL fit k = 27, identical to
   Random) and Bayesian-Threshold-BR (agg-LL fit at δ_BR = −70, identical to Bayesian-BR)
   collapse onto other models, so they are dropped from the individual comparison.
3. Table 2 rows and SI Table S4 rows (tables.tex), from the refit's results.json.

Run (a few minutes) from analysis/:
    uv run python experiments/2026-10-22-paper-refit-updates/make_paper_updates.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXP = SCRIPT_DIR.parent
REFIT = EXP / "2026-10-20-lapse-refit-all-models"
FIG4 = EXP / "2026-10-15-fig4-legend-options"
for d in ("2026-05-28-paper-figures", "2026-05-25-full-pipeline",
          "2026-10-20-tolerant-threshold-ps", "2026-10-15-fig4-legend-options"):
    sys.path.insert(0, str(EXP / d))

import final6pt as f6  # noqa: E402
import tolerant_threshold_ps as tps  # noqa: E402
from shared.constants import ROLE_CHAR_TO_IDX, ROLE_SHORT  # noqa: E402

FIG_DIR = SCRIPT_DIR / "stuff to incorporate"
FIG_DIR.mkdir(exist_ok=True)
REFIT_RES = json.load(open(REFIT / "results.json"))

# ── 1. R3 team cases with the refit Walk-BR parameters ───────────────

tm, lo = f6.tm, f6.lo
_orig_load = tm.load_best_model_params
OLD = _orig_load()[0]
NEW = REFIT_RES["params_by_model"]["Bayesian Walk"]
ACTIVE = {"params": OLD}


def _load_params():
    _, s1, res = _orig_load()
    return ACTIVE["params"], s1, res


tm.load_best_model_params = _load_params

_traj_cache = {}
_orig_build = tm.build_trajectories


def _build_cached(s1):
    key = json.dumps(s1, sort_keys=True, default=str)
    if key not in _traj_cache:
        _traj_cache[key] = _orig_build(s1)
    return _traj_cache[key]


tm.build_trajectories = _build_cached


def _role_idx(r):
    return ROLE_CHAR_TO_IDX[r] if isinstance(r, str) else int(r)


def chosen_rows(preds_old, preds_new):
    """Per stage and player: role chosen, its probability under old / new params, and
    whether it was the model's most probable role."""
    rows = []
    for s, (po, pn) in enumerate(zip(preds_old, preds_new)):
        for i in range(3):
            r = _role_idx(pn["combo"][i])
            rows.append(dict(stage=s + 1, player=i + 1, role=ROLE_SHORT[r],
                             old=float(po["pred"][i][r]), new=float(pn["pred"][i][r]),
                             top_old=int(np.argmax(po["pred"][i])) == r,
                             top_new=int(np.argmax(pn["pred"][i])) == r,
                             dist_old=[float(x) for x in po["pred"][i]],
                             dist_new=[float(x) for x in pn["pred"][i]]))
    return rows


def render_cases():
    cases = {}
    for suffix, rnd, name in f6.CASES:
        s_print = f6.S09 if name == "R3_team_case" else f6.S10
        ACTIVE["params"] = OLD
        rec, post, preds_old = lo.prepare_case(suffix, rnd)
        # the old parameters must reproduce the paper figure pixel for pixel
        with tempfile.TemporaryDirectory() as tmp:
            f6.HERE = Path(tmp)
            f6.render6(name, rec, post, preds_old, s_print)
            assert np.array_equal(mpimg.imread(Path(tmp) / f"{name}.png"),
                                  mpimg.imread(FIG4 / "final6pt" / f"{name}.png")), name
        ACTIVE["params"] = NEW
        rec, post, preds_new = lo.prepare_case(suffix, rnd)
        f6.HERE = FIG_DIR
        f6.render6(name, rec, post, preds_new, s_print)
        cases[name] = chosen_rows(preds_old, preds_new)
    return cases


def write_case_md(cases):
    L = ["# Probability on the chosen role, old vs refit Walk-BR parameters", "",
         f"old: τ_v = {OLD['tau_softmax']:.2f}, ε_s = {1 - OLD['epsilon_switch']:.3f}  ·  "
         f"new: τ_v = {NEW['tau_softmax']:.2f}, ε_s = {1 - NEW['epsilon_switch']:.3f}", "",
         "`*` = the model's most probable role. Distributions are (F, T, M).", ""]
    for name, rows in cases.items():
        L += [f"## {name}", "", "| Stage | Player | Role | old | new | old dist | new dist |",
              "|--:|--:|:-:|--:|--:|---|---|"]
        for r in rows:
            mark = lambda p, t: f"{p:.2f}{'*' if t else ''}"
            L.append(f"| {r['stage']} | P{r['player']} | {r['role']} | {mark(r['old'], r['top_old'])} | "
                     f"{mark(r['new'], r['top_new'])} | {', '.join(f'{x:.2f}' for x in r['dist_old'])} | "
                     f"{', '.join(f'{x:.2f}' for x in r['dist_new'])} |")
        stages = sorted({r["stage"] for r in rows})
        for key in ("old", "new"):
            means = [np.mean([r[key] for r in rows if r["stage"] == s]) for s in stages]
            L.append("")
            L.append(f"{key}: stage means {', '.join(f'{m:.2f}' for m in means)}; "
                     f"overall mean {np.mean([r[key] for r in rows]):.2f}; "
                     f"most probable on {sum(r['top_' + key] for r in rows)} of {len(rows)}")
        L.append("")
    (SCRIPT_DIR / "case_probabilities.md").write_text("\n".join(L))


# ── 2. R4 individual fits over 11 models ─────────────────────────────

DROPPED = ("Top-7", "Bayesian Threshold")
ORDER = [m for m in tps.MODEL_ORDER if m not in DROPPED]
BAYES = [m for m in tps.BAYES if m not in DROPPED]
BASE = [m for m in tps.BASE if m not in DROPPED]
VALUE_BASE = ("Optimal", "Random-to-Optimal")


def plot_r4(post):
    rc = {"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
          "font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 6,
          "ytick.labelsize": 6, "axes.linewidth": 0.6}
    with plt.rc_context(rc):
        dom = {p: max(ORDER, key=lambda m: v[m]) for p, v in post.items()}
        idx = {m: i for i, m in enumerate(ORDER)}
        pids = sorted(post, key=lambda p: (idx[dom[p]], -post[p][dom[p]]))
        fig, ax = plt.subplots(figsize=(3.42, 2.95))
        x = np.arange(len(pids)); bottoms = np.zeros(len(pids))
        for m in ORDER:
            vals = np.array([post[p][m] for p in pids])
            ax.bar(x, vals, bottom=bottoms, width=1.0, color=tps.PALETTE[m],
                   edgecolor="white", linewidth=0.35)
            bottoms += vals
        assert np.allclose(bottoms, 1.0, atol=1e-9)
        ax.set_xlim(-0.5, len(pids) - 0.5); ax.set_ylim(0, 1)
        ax.set_xlabel(f"Participants (n = {len(pids)}, sorted by best-fitting model)")
        ax.set_ylabel("P(model | participant's choices)")
        ax.set_xticks([]); ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.spines[["top", "right"]].set_visible(False)
        proxy = lambda m: mpatches.Patch(facecolor=tps.PALETTE[m], edgecolor="white", linewidth=0.35)
        fig.tight_layout(pad=0.3, rect=(0, 0.23, 1, 1))
        kw = dict(loc="upper center", ncols=2, fontsize=5, title_fontsize=5.5, frameon=False,
                  handlelength=1.2, handleheight=1.1, columnspacing=0.5, labelspacing=0.35,
                  borderpad=0.2)
        l1 = fig.legend([proxy(m) for m in BAYES], [tps.dn(m) for m in BAYES],
                        title="Bayesian models", bbox_to_anchor=(0.25, 0.22), **kw)
        l2 = fig.legend([proxy(m) for m in BASE], [tps.dn(m) for m in BASE],
                        title="Non-Bayesian baselines", bbox_to_anchor=(0.79, 0.22), **kw)
        for l in (l1, l2):
            l._legend_box.align = "left"
        for ext, k in (("png", {"dpi": 300}), ("pdf", {})):
            fig.savefig(FIG_DIR / f"R4_individual_fitting.{ext}", bbox_inches="tight",
                        facecolor="white", **k)
        plt.close(fig)
    return dom


def individual():
    lls = REFIT_RES["individual_ll"]
    pids = list(lls["Random"])
    # why the two models are dropped: they score every participant like another model
    assert all(lls["Top-7"][p] == lls["Random"][p] for p in pids)
    assert max(abs(lls["Bayesian Threshold"][p] - lls["Bayesian-Value"][p]) for p in pids) < 1e-3
    post = tps.posteriors_from_ll(lls, ORDER)
    dom = plot_r4(post)
    counts = Counter(dom.values())
    n = len(pids)
    groups = {"Bayesian": sum(counts[m] for m in BAYES),
              "value-based baselines": sum(counts[m] for m in VALUE_BASE)}
    groups["noise / contrarian baselines"] = n - sum(groups.values())
    leading = sum(max(v.values()) > 0.5 for v in post.values())
    return dict(n=n, counts={m: counts.get(m, 0) for m in ORDER}, groups=groups,
                leading_over_half=leading)


# ── 3. Table 2 and SI Table S4 rows ──────────────────────────────────

def num(x, bold=False):
    s = f"{abs(x):.3f}"
    if bold:
        return f"$\\mathbf{{-{s}}}$" if x < 0 else f"\\textbf{{{s}}}"
    return f"$-${s}" if x < 0 else s


def p2(x):
    s = f"{x:.2f}"
    return f"{{-}}{s[1:]}" if s.startswith("-") else s


MAIN_ROWS = [("Bayesian-BR", "Bayesian-Value"), ("Bayesian-Walk-BR", "Bayesian Walk"),
             ("Bayesian-Threshold-BR", "Bayesian Threshold"), ("Bayesian-PS", "Bayesian-Belief"),
             ("Bayesian-Walk-PS", "Bayesian Walk-PS"), ("Bayesian-Threshold-PS", "Bayesian Thresh-PS"),
             None,
             ("Optimal", "Optimal"), ("Top-$k$ ($k=27$)", "Top-7"),
             ("Random-to-Optimal", "Random-to-Optimal"), ("Copy Others", "Copy Others"),
             ("Contradict Others", "Contradict Others"), ("Random Walk", "Random Walk"),
             ("Random", "Random")]
METRICS = ("combo_r", "marg_r", "agg_ll", "mean_ll")


def si_params(m, fit):
    if m == "Bayesian Walk-PS":
        return f"$\\varepsilon_s{{=}}{p2(1 - fit['epsilon_switch'])}$"
    if m == "Bayesian Thresh-PS":
        return f"$\\delta_{{\\mathrm{{PS}}}}{{=}}{p2(fit['delta'])}$"
    if m == "Bayesian-Value":
        return f"$\\tau_v{{=}}{p2(fit['tau_softmax'])}$"
    if m == "Bayesian Walk":
        return (f"$\\tau_v{{=}}{p2(fit['tau_softmax'])}$, "
                f"$\\varepsilon_s{{=}}{p2(1 - fit['epsilon_switch'])}$")
    if m == "Bayesian Threshold":
        return (f"$\\tau_v{{=}}{p2(fit['tau_softmax'])}$, "
                f"$\\delta_{{\\mathrm{{BR}}}}{{=}}{p2(fit['delta'])}$")
    raise KeyError(m)


SI_MODELS = [("Bayesian-Walk-PS", "Bayesian Walk-PS"), ("Bayesian-Threshold-PS", "Bayesian Thresh-PS"),
             ("Bayesian-BR", "Bayesian-Value"), ("Bayesian-Walk-BR", "Bayesian Walk"),
             ("Bayesian-Threshold-BR", "Bayesian Threshold")]
OBJ_LABEL = {"combo_r": "combo-$r$", "agg_ll": "agg-LL", "mean_ll": "mean-LL"}


def write_tables():
    mb = REFIT_RES["metrics_by_model"]
    best = {k: max(mb[m][k] for _, m in filter(None, MAIN_ROWS)) for k in METRICS}
    L = ["% main-text Table 2 rows (agg-LL fits, lambda = 0.01 lapse, no floor)"]
    for row in MAIN_ROWS:
        if row is None:
            L.append("\\midrule")
            continue
        label, m = row
        cells = [num(mb[m][k], mb[m][k] == best[k]) for k in METRICS]
        L.append(f"{label:<22}& " + " & ".join(cells) + " \\\\")
    L += ["", "% SI Table S4 rows (fitting objective x model)"]
    b = mb["Bayesian-Belief"]
    L.append("Bayesian-PS & (all) & none & " + " & ".join(num(b[k]) for k in METRICS) + " \\\\")
    for label, m in SI_MODELS:
        L.append("\\midrule")
        for j, obj in enumerate(("combo_r", "agg_ll", "mean_ll")):
            ev = REFIT_RES["per_objective_eval"][m][obj]
            head = label if j == 0 else ""
            L.append(f"{head} & {OBJ_LABEL[obj]} & {si_params(m, REFIT_RES['fits'][m][obj])} & "
                     + " & ".join(num(ev[k]) for k in METRICS) + " \\\\")
    (SCRIPT_DIR / "tables.tex").write_text("\n".join(L) + "\n")


def main():
    cases = render_cases()
    write_case_md(cases)
    ind = individual()
    write_tables()
    L = ["# Individual fits over 11 models (Fig. 5)", "",
         f"Dropped: {', '.join(tps.dn(m) for m in DROPPED)} (collapse onto Random / Bayesian-BR). "
         f"λ = {REFIT_RES['lapse']} lapse on each player's marginal, refit (agg-LL) parameters.", "",
         "| Model | Dominant for |", "|---|--:|"]
    L += [f"| {tps.dn(m)} | {c} |" for m, c in ind["counts"].items()]
    L += ["", "| Group | n | % |", "|---|--:|--:|"]
    L += [f"| {g} | {c} | {100 * c / ind['n']:.0f}% |" for g, c in ind["groups"].items()]
    L += ["", f"Leading model posterior > 0.5: **{ind['leading_over_half']} of {ind['n']}**"]
    (SCRIPT_DIR / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
