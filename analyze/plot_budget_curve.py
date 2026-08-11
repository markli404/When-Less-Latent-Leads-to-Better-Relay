"""What the KV budget actually does, and whether OBF tracks it.

Two rows, one column per benchmark that has runs.

  Row 1  accuracy against budget, log-2 x. Headwise and H-OBF as lines with a
         band covering their seed range. Full and `gonly` as horizontal
         references: those are the two ends of the axis, since Full evicts
         nothing and gonly relays nothing.

  Row 2  the decomposition the sweep exists for. `gap` is Full minus headwise,
         which is what eviction destroyed at that budget. `recovery` is H-OBF
         minus headwise, which is what backfilling put back. The recovery law
         says the second should track the first. Budget is the only clean way
         to move the gap on purpose, so this panel is the controlled version of
         the correlation we report over scattered cells.

Bars in row 2 are per-seed points, not just the mean: a recovery of +0.3 pp
whose three seeds straddle zero is not the same result as one whose three seeds
agree, and at this effect size the difference decides whether there is anything
to say.

Usage:
    python analyze/plot_budget_curve.py
    python analyze/plot_budget_curve.py --refresh     # re-pull wandb first
    python analyze/plot_budget_curve.py --out figures/
"""

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_config import DATASET_LABELS, METHOD_COLORS, apply_plot_style  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SUMMARY_DIR = REPO / "results" / "wandb_summary"
SWEEP_CSV = SUMMARY_DIR / "budget_sweep.csv"

MODEL = "Qwen/Qwen3-4B"
GPU = "NVIDIA A100-PCIE-40GB"
BUDGETS = [4, 8, 16, 32, 64]
OBF_RANK = "2"
ORDER = ["gsm8k", "arc_challenge", "medqa", "humanevalplus"]


def sweep_cells():
    """(task, method, budget) -> list of per-seed accuracies, from the sweep only.

    The sweep is single-source on purpose, so nothing here is merged with the
    older projects. Only `gonly` is borrowed, below, and it is a flat line.
    """
    out = defaultdict(list)
    if not SWEEP_CSV.exists():
        return out
    for r in csv.DictReader(open(SWEEP_CSV)):
        if r.get("config.model_name") != MODEL or not r.get("summary.accuracy"):
            continue
        if (r.get("config.inject_mode") or "uniform") != "uniform":
            continue
        comp = r.get("config.compressor") or ""
        if comp.startswith("hobf"):
            if r.get("config.pca_rank") != OBF_RANK:
                continue
            comp = "hobf"
        try:
            budget = int(r.get("config.kv_budget") or 32)
        except ValueError:
            continue
        out[(r["config.task"], comp, budget)].append(float(r["summary.accuracy"]) * 100)
    return out


def gonly_floor():
    """gonly per task, from the main projects. It has no budget, so it is a line."""
    out, seen = defaultdict(list), set()
    for path in sorted(SUMMARY_DIR.glob("*.csv")):
        if ".bak" in path.name or path.name == SWEEP_CSV.name:
            continue
        for r in csv.DictReader(open(path)):
            rid = r.get("id") or r.get("name")
            if not rid or rid in seen:
                continue
            seen.add(rid)
            if (r.get("config.model_name") != MODEL or r.get("metadata.gpu") != GPU
                    or r.get("config.compressor") != "gonly" or not r.get("summary.accuracy")):
                continue
            out[r["config.task"]].append(float(r["summary.accuracy"]) * 100)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-pull the budget_sweep project first")
    ap.add_argument("--out", default=str(REPO / "figures"))
    args = ap.parse_args()

    if args.refresh:
        from utils.wandb_summary import update_project_summary
        info = update_project_summary("budget_sweep")
        print(f"pulled {info['saved_runs']} runs")

    cells = sweep_cells()
    floor = gonly_floor()
    tasks = [t for t in ORDER if any(k[0] == t for k in cells)]
    if not tasks:
        print(f"No runs in {SWEEP_CSV}. Nothing to plot.")
        return

    apply_plot_style()
    fig, axes = plt.subplots(2, len(tasks), figsize=(3.6 * len(tasks), 6.4), squeeze=False)

    for col, task in enumerate(tasks):
        full = cells.get((task, "full", 32), [])
        full_mean = st.mean(full) if full else None

        ax = axes[0][col]
        for method, label in (("headwise", "Headwise"), ("hobf", "H-OBF r2")):
            xs, mid, lo, hi = [], [], [], []
            for b in BUDGETS:
                v = cells.get((task, method, b))
                if not v:
                    continue
                xs.append(b); mid.append(st.mean(v)); lo.append(min(v)); hi.append(max(v))
            if not xs:
                continue
            c = METHOD_COLORS.get(method, None)
            ax.plot(xs, mid, marker="o", ms=4, lw=1.8, color=c, label=label, zorder=3)
            ax.fill_between(xs, lo, hi, color=c, alpha=0.18, lw=0, zorder=2)
        if full_mean is not None:
            ax.axhline(full_mean, ls="--", lw=1.2, color=METHOD_COLORS.get("full"),
                       label="Full relay", zorder=1)
        if floor.get(task):
            ax.axhline(st.mean(floor[task]), ls=":", lw=1.2, color=METHOD_COLORS.get("gonly"),
                       label="No relay", zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS); ax.set_xticklabels(BUDGETS)
        ax.set_title(DATASET_LABELS.get(task, task))
        if col == 0:
            ax.set_ylabel("Accuracy (%)")
            ax.legend(fontsize=7, frameon=False, loc="lower right")

        # Row 2: what eviction destroyed, and what backfilling returned.
        ax = axes[1][col]
        gx, gy, rx, ry, pts = [], [], [], [], []
        for b in BUDGETS:
            h = cells.get((task, "headwise", b)); o = cells.get((task, "hobf", b))
            if h and full_mean is not None:
                gx.append(b); gy.append(full_mean - st.mean(h))
            if h and o:
                rx.append(b); ry.append(st.mean(o) - st.mean(h))
                # pair the seeds by rank; run ids are not aligned across arms
                pts += [(b, oo - hh) for hh, oo in zip(sorted(h), sorted(o))]
        ax.axhline(0, lw=0.8, color="0.6", zorder=1)
        if gx:
            ax.plot(gx, gy, marker="s", ms=4, lw=1.8, color="0.35",
                    label="gap: Full - headwise", zorder=3)
        if rx:
            ax.plot(rx, ry, marker="o", ms=4, lw=1.8, color=METHOD_COLORS.get("hobf"),
                    label="recovery: H-OBF - headwise", zorder=3)
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=9, alpha=0.5,
                       color=METHOD_COLORS.get("hobf"), zorder=4, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS); ax.set_xticklabels(BUDGETS)
        ax.set_xlabel("KV budget (tokens kept)")
        if col == 0:
            ax.set_ylabel("Accuracy difference (pp)")
            ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    out = Path(args.out) / "budget_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")

    for task in tasks:
        full = cells.get((task, "full", 32), [])
        fm = st.mean(full) if full else float("nan")
        print(f"\n{task}   Full {fm:.2f}   no-relay {st.mean(floor[task]):.2f}"
              if floor.get(task) else f"\n{task}   Full {fm:.2f}")
        print(f"  {'B':<5}{'headwise':>10}{'H-OBF':>9}{'gap':>8}{'recovery':>10}{'seeds':>7}")
        for b in BUDGETS:
            h = cells.get((task, "headwise", b)); o = cells.get((task, "hobf", b))
            if not h:
                continue
            mh = st.mean(h)
            mo = st.mean(o) if o else float("nan")
            print(f"  {b:<5}{mh:>10.2f}{mo:>9.2f}{fm - mh:>8.2f}{mo - mh:>+10.2f}{len(h):>7}")


if __name__ == "__main__":
    main()
