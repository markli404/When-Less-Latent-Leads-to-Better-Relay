"""Plot the eviction-gap to OBF-recovery relationship.

Two panels — left: headwise selector, right: layerwise selector.
  x = Full minus eviction baseline   (how much the eviction step destroyed)
  y = OBF minus the same baseline    (how much OBF put back)

One point per (model, dataset). NO rank is selected: y is OBF's recovery
AVERAGED over every rank in the sweep, and the whisker spans the min and max
across those ranks. Two alternatives were rejected. Picking the best rank per
cell inflates noisy cells more than quiet ones and manufactures correlation.
Picking one fixed rank invites the question of which rank, and the answer moves
the correlation around (headwise r runs 0.56 to 0.93 across ranks). Pooling
every (cell, rank) pair as a separate point is worse still: the gap on the x
axis is identical across a cell's ranks, so the extra points are replicates and
the p-value they produce is inflated by roughly the number of ranks.

Accuracy at each cell is the mean over that model's three seeds (4B: 4/44/444,
8B: 8/88/888), A100 only.

The two 30-problem AIME sets are drawn hollow and excluded from the fit: one
problem moves them 3.3 pp, so their per-cell gap is dominated by seed noise and
is not a reliable estimate of the eviction damage. The console also prints the
fit WITH them included, as a robustness check.
"""

from pathlib import Path
import math
import statistics as st

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_config import DATASET_LABELS, METHOD_COLORS, apply_plot_style
from summary_utils import load_summary_records


SCRIPT_DIR = Path(__file__).resolve().parent
SAVE_DIR = SCRIPT_DIR.parent / "figures"
SAVE_NAME = "gap_recovery.png"

# Explicit source list: load_summary_records() rglobs the summary dir, which
# would also pull in the stale sweeps under results/wandb_summary/old_data/.
SUMMARY_CSVS = [
    SCRIPT_DIR.parent / "results" / "wandb_summary" / "main_experiment.csv",
    SCRIPT_DIR.parent / "results" / "wandb_summary" / "pca_rank_sweep.csv",
]

# One card per scale, which is the campaign standard. Both axes here are
# within-cell differences (Full minus baseline, OBF minus baseline), so the
# card cancels out of each point and mixing scales does not mix hardware into
# the fit.
MODEL_GPU = {
    "Qwen/Qwen3-4B": "NVIDIA A100-PCIE-40GB",
    "Qwen/Qwen3-8B": "NVIDIA A100-PCIE-40GB",
    "Qwen/Qwen3-14B": "NVIDIA H200 NVL",
}
# The paper fits the relationship on 4B and 8B, where every cell carries three
# seeds. Set INCLUDE_14B to True to add the 14B column once it has the same
# seed coverage. For reference, at 21 cells headwise goes r 0.854 -> 0.822 with
# p improving and layerwise 0.702 -> 0.593, and headwise reproduces on the 14B
# cells alone at r = 0.85. Layerwise on 14B comes out at r = -0.11, which is
# range restriction rather than a failure: its gaps there span 2.61 pp against
# 7.24 at 4B, so the x axis barely varies. Report that, not the coefficient.
INCLUDE_14B = False
MODELS = (["Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]
          + (["Qwen/Qwen3-14B"] if INCLUDE_14B else []))
MODEL_MARKERS = {"Qwen/Qwen3-4B": "o", "Qwen/Qwen3-8B": "s", "Qwen/Qwen3-14B": "^"}
# Marker shape alone does not separate three scales once twenty-one points pile
# up near the origin under their labels, so each scale also gets its own value
# of the panel hue, light to dark as the backbone grows.
MODEL_TINT = {"Qwen/Qwen3-4B": 0.42, "Qwen/Qwen3-8B": 0.0, "Qwen/Qwen3-14B": -0.42}
MODEL_SIZES = {"Qwen/Qwen3-4B": 95, "Qwen/Qwen3-8B": 88, "Qwen/Qwen3-14B": 115}


def tint(color, k):
    """Blend `color` toward white for k > 0 and toward black for k < 0."""
    r, g, b = mcolors.to_rgb(color)
    if k >= 0:
        return tuple(c + (1.0 - c) * k for c in (r, g, b))
    return tuple(c * (1.0 + k) for c in (r, g, b))
MODEL_LABELS = {"Qwen/Qwen3-4B": "Qwen3-4B", "Qwen/Qwen3-8B": "Qwen3-8B",
                "Qwen/Qwen3-14B": "Qwen3-14B"}

# (selector, obf compressor, eviction compressor)
PANELS = [
    ("Headwise", "hobf", "headwise"),
    ("Layerwise", "lobf", "layerwise"),
]

MIN_SAMPLES_FOR_FIT = 164  # exclude the two 30-problem AIME sets from the fit


def pearson(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def t_sf(t, df):
    """Two-sided survival function of Student's t, via the incomplete beta."""
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5

    def betacf(a, b, x):
        tiny = 1e-30
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        d = 1.0 / (d if abs(d) > tiny else tiny)
        h = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            c = 1.0 + aa / c
            d = 1.0 / (d if abs(d) > tiny else tiny)
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            c = 1.0 + aa / c
            d = 1.0 / (d if abs(d) > tiny else tiny)
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 3e-9:
                break
        return h

    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1 - x)
    )
    if x < (a + 1) / (a + b + 2):
        return front * betacf(a, b, x) / a
    return 1.0 - front * betacf(b, a, 1 - x) / b


def p_value(r, n):
    df = n - 2
    if df < 1 or abs(r) >= 1:
        return float("nan")
    return t_sf(abs(r) * math.sqrt(df / (1 - r * r)), df)


def cell_means(records):
    """(model, dataset, compressor, rank) -> mean accuracy in percent.

    Also returns the DISTINCT seed count per cell, so the console can show which
    cells rest on the full campaign and which do not.
    """
    buckets = {}
    seeds = {}
    sizes = {}
    for rec in records:
        if rec.get("state") != "finished":
            continue
        model = rec.get("model")
        gpu = rec.get("gpu_name") or ""
        # Empty metadata rows are a logging gap, not another card; they belong
        # to the campaign they were launched in.
        if gpu and gpu != MODEL_GPU.get(model):
            continue
        # inject_mode is part of a cell's identity: the attention-weighted
        # ablation logs as compressor "lobf" at rank 4, the exact key of the
        # L-OBF baseline. Neither source CSV carries those runs today, but any
        # aggregation on (model, dataset, compressor, rank) needs the guard.
        if (rec.get("config.inject_mode") or "uniform") != "uniform":
            continue
        acc = rec.get("accuracy")
        if acc is None:
            continue
        key = (
            model,
            rec.get("dataset"),
            rec.get("compressor"),
            rec.get("config.pca_rank"),
        )
        buckets.setdefault(key, []).append(float(acc) * 100.0)
        seeds.setdefault(key, set()).add(str(rec.get("seed")))
        n = rec.get("summary.max_samples")
        if n is not None:
            sizes.setdefault(rec.get("dataset"), float(n))
    return ({k: st.mean(v) for k, v in buckets.items()},
            sizes,
            {k: len(v) for k, v in seeds.items()})


def over_ranks(means, model, dataset, compressor):
    return [v for (m, d, c, _), v in means.items()
            if m == model and d == dataset and c == compressor]


def collect(means, sizes, obf, eviction):
    """One record per (model, dataset): the eviction gap, and OBF's recovery
    averaged over every swept rank (with the min/max across ranks kept for the
    whisker). No rank is selected, so there is no rank to defend."""
    points = []
    datasets = sorted({d for (_, d, _, _) in means})
    for model in MODELS:
        for dataset in datasets:
            full = over_ranks(means, model, dataset, "full")
            base = over_ranks(means, model, dataset, eviction)
            gains = over_ranks(means, model, dataset, obf)
            if not full or not base or not gains:
                continue
            baseline = st.mean(base)
            recovery = [v - baseline for v in gains]
            points.append(
                dict(
                    model=model,
                    dataset=dataset,
                    n=sizes.get(dataset, float("nan")),
                    gap=st.mean(full) - baseline,
                    gain=st.mean(recovery),
                    gain_lo=min(recovery),
                    gain_hi=max(recovery),
                    n_ranks=len(recovery),
                )
            )
    return points


def place_label(ax, x, y, text, color, placed):
    """Greedy label placement: first candidate offset that clears earlier ones."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    fx, fy = (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)
    w, h = 0.019 * len(text) + 0.012, 0.042
    candidates = [(0.020, 0.012), (0.020, -0.050), (-w - 0.020, 0.012),
                  (-w - 0.020, -0.050), (0.020, 0.055), (0.020, -0.095),
                  (-w - 0.020, 0.055), (-w - 0.020, -0.095)]
    for i, (dx, dy) in enumerate(candidates):
        box = (fx + dx, fy + dy, fx + dx + w, fy + dy + h)
        if box[0] < -0.01 or box[2] > 1.01 or box[1] < -0.01 or box[3] > 1.01:
            continue
        if any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3]
               for b in placed):
            continue
        placed.append(box)
        _emit(ax, fx, fy, fx + dx, fy + dy + 0.008, text, color, leader=i >= 2)
        return
    placed.append((fx + 0.02, fy + 0.012, fx + 0.02 + w, fy + 0.054))
    _emit(ax, fx, fy, fx + 0.02, fy + 0.020, text, color, leader=True)


def _emit(ax, fx, fy, tx, ty, text, color, leader):
    """Draw the label at (tx, ty) in axes fraction, with a leader line if far."""
    if leader:
        ax.annotate(
            text, xy=(fx, fy), xycoords="axes fraction",
            xytext=(tx, ty), textcoords="axes fraction",
            fontsize=7.5, color="#3A3A3A", zorder=4,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.7,
                            alpha=0.55, shrinkA=1, shrinkB=5),
        )
    else:
        ax.annotate(text, (tx, ty), xycoords="axes fraction",
                    fontsize=7.5, color="#3A3A3A", zorder=4)


def draw_panel(ax, title, points, color, lo, hi):
    fit = [p for p in points if p["n"] >= MIN_SAMPLES_FOR_FIT]
    small = [p for p in points if p["n"] < MIN_SAMPLES_FOR_FIT]

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.axhline(0, color="#B0B0B0", lw=0.8, zorder=1)
    ax.axvline(0, color="#B0B0B0", lw=0.8, zorder=1)
    ax.plot([lo, hi], [lo, hi], ls=":", color="#909090", lw=1.2, zorder=1)

    xs = [p["gap"] for p in fit]
    ys = [p["gain"] for p in fit]
    r = pearson(xs, ys)
    slope = r * st.pstdev(ys) / st.pstdev(xs)
    intercept = st.mean(ys) - slope * st.mean(xs)
    ax.plot([lo, hi], [intercept + slope * lo, intercept + slope * hi],
            color=color, lw=2.0, alpha=0.85, zorder=2)

    # Slope reported once, in the top-right corner.
    ax.text(0.96, 0.955, f"slope {slope:+.2f}", color=color, fontsize=11.5,
            fontweight="bold", transform=ax.transAxes, va="top", ha="right",
            zorder=5)

    placed = [(0.0, 0.84, 0.42, 1.0)]  # reserve the stats box in the top-left
    ordered = sorted(fit, key=lambda p: -abs(p["gap"])) + small
    for p in ordered:
        in_fit = p["n"] >= MIN_SAMPLES_FOR_FIT
        c = tint(color, MODEL_TINT[p["model"]])
        # Whisker: how far the recovery moves across the swept ranks. The marker
        # is the mean the fit uses; the whisker shows what averaging hides.
        ax.plot([p["gap"], p["gap"]], [p["gain_lo"], p["gain_hi"]],
                color=c, lw=1.1, alpha=0.45 if in_fit else 0.3, zorder=2,
                solid_capstyle="butt")
        ax.scatter(
            p["gap"], p["gain"],
            marker=MODEL_MARKERS[p["model"]], s=MODEL_SIZES[p["model"]],
            facecolor=c if in_fit else "white",
            edgecolor=c, linewidths=1.6,
            alpha=0.92 if in_fit else 0.75, zorder=3,
        )
        place_label(ax, p["gap"], p["gain"],
                    DATASET_LABELS.get(p["dataset"], p["dataset"]), c, placed)

    p_val = p_value(r, len(fit))
    star = ("***" if p_val < 1e-3 else "**" if p_val < 1e-2
            else "*" if p_val < 5e-2 else "n.s.")
    ax.set_title(title, pad=10, fontweight="bold")
    ax.set_xlabel("Eviction damage:  Full − baseline  (pp)")
    ax.text(
        0.04, 0.955,
        f"r = {r:+.2f} {star}\n(p = {p_val:.3f}, N = {len(fit)})",
        transform=ax.transAxes, va="top", ha="left", fontsize=11, zorder=5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#CCCCCC", alpha=0.95),
    )
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_aspect("equal", adjustable="box")


def report_seeds(nseeds):
    """Seeds behind every (model, dataset, compressor, rank) cell this fit uses.

    Each point in the figure is a cell mean, and the fit weights every point
    equally, so a one-seed cell carries the same weight as a three-seed one.
    Print the distribution rather than leaving that implicit.
    """
    from collections import Counter
    per_comp = {}
    for (model, ds, comp, rank), n in nseeds.items():
        per_comp.setdefault(comp, Counter())[n] += 1
    print("seeds per cell, by compressor (cells at each seed count):")
    for comp in sorted(per_comp):
        dist = ", ".join(f"n={k}: {v} cells" for k, v in sorted(per_comp[comp].items()))
        print(f"  {comp:<12}{dist}")
    thin = sorted(f"{m.split('-')[-1]}/{ds}/{c}@r{r}" for (m, ds, c, r), n in nseeds.items() if n < 3)
    print(f"  cells below 3 seeds: {len(thin)}" + (f"\n    {', '.join(thin)}" if thin else ""))
    print()


def report(panel_points):
    """Print the numbers the paper quotes, both with and without AIME.

    The AIME-included row is the robustness check: the relationship weakens but
    the headwise selector stays significant, which is the honest way to answer
    "did you drop the two benchmarks that disagreed with you".
    """
    print(f"{'panel':<11}{'AIME':<10}{'N':>4}{'r':>9}{'slope':>9}{'p':>10}")
    for (label, _, _), points in zip(PANELS, panel_points):
        for tag, subset in (
            ("excluded", [p for p in points if p["n"] >= MIN_SAMPLES_FOR_FIT]),
            ("included", points),
        ):
            xs = [p["gap"] for p in subset]
            ys = [p["gain"] for p in subset]
            r = pearson(xs, ys)
            slope = r * st.pstdev(ys) / st.pstdev(xs)
            print(f"{label:<11}{tag:<10}{len(subset):>4}{r:>+9.3f}"
                  f"{slope:>+9.3f}{p_value(r, len(subset)):>10.4f}")
    print()

    # Per-scale, AIME excluded. This is the part that separates the two
    # selectors: the headwise relationship reappears at every scale on its own,
    # the layerwise one does not survive at 14B.
    print(f"{'panel':<11}{'scale':<10}{'N':>4}{'r':>9}{'slope':>9}{'p':>10}")
    for (label, _, _), points in zip(PANELS, panel_points):
        for model in MODELS:
            subset = [p for p in points
                      if p["model"] == model and p["n"] >= MIN_SAMPLES_FOR_FIT]
            if len(subset) < 3:
                continue
            xs = [p["gap"] for p in subset]
            ys = [p["gain"] for p in subset]
            r = pearson(xs, ys)
            slope = r * st.pstdev(ys) / st.pstdev(xs)
            print(f"{label:<11}{MODEL_LABELS[model].replace('Qwen3-',''):<10}"
                  f"{len(subset):>4}{r:>+9.3f}{slope:>+9.3f}"
                  f"{p_value(r, len(subset)):>10.4f}")
    print()


def main():
    plt.rcParams.update(apply_plot_style())
    means, sizes, nseeds = cell_means(load_summary_records(csv_paths=SUMMARY_CSVS))

    panel_points = [collect(means, sizes, obf, ev) for _, obf, ev in PANELS]
    report(panel_points)
    report_seeds(nseeds)
    # Shared limits across both panels so the two selectors are visually comparable.
    span = [v for pts in panel_points for p in pts for v in (p["gap"], p["gain"])]
    lo, hi = min(span) - 1.0, max(span) + 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.4))
    subtitles = [
        "Headwise eviction, recovered by H-OBF",
        "Layerwise eviction, recovered by L-OBF",
    ]
    for ax, (_, obf, _), points, sub in zip(axes, PANELS, panel_points, subtitles):
        draw_panel(ax, sub, points, METHOD_COLORS[obf], lo, hi)
    axes[0].set_ylabel("OBF recovery:  OBF − baseline  (pp)")

    legend_base = METHOD_COLORS[PANELS[0][1]]
    handles = [
        Line2D([], [], ls="", marker=MODEL_MARKERS[m], markersize=9.5,
               markerfacecolor=tint(legend_base, MODEL_TINT[m]),
               markeredgecolor=tint(legend_base, MODEL_TINT[m]),
               label=MODEL_LABELS[m])
        for m in MODELS
    ]
    handles += [
        Line2D([], [], ls="", marker="o", markersize=9, markerfacecolor="white",
               markeredgecolor="#4C4C4C", label="AIME (n=30, excluded from fit)"),
        Line2D([], [], color="#4C4C4C", lw=1.1, alpha=0.55,
               label="min to max across ranks"),
        Line2D([], [], ls=":", color="#909090", lw=1.2,
               label="full recovery (y = x)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.01), fontsize=11)
    fig.subplots_adjust(bottom=0.20, top=0.94, left=0.07, right=0.98, wspace=0.14)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVE_DIR / SAVE_NAME
    fig.savefig(save_path, dpi=300)
    print(f"saved {save_path}")


if __name__ == "__main__":
    main()
