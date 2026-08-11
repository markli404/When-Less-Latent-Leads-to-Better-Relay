"""Plot PCA-rank sweep accuracy curves.

Single panel — x = pca_rank, y = accuracy (%), one line per (dataset,
compressor). Auto-fetches/refreshes the project's wandb summary CSV on every
run.

Sweep semantics:
- For each (dataset, compressor=hobf) we pull pca_rank ∈ {2, 4, 8, 16, 32}.
- For each (dataset, compressor=lobf) we likewise pull the same ranks.
- pca_rank=0 (the leftmost x-axis tick) is the no-OBF baseline:
    * for HOBF curve: rank=0 point comes from compressor=`headwise`
    * for LOBF curve: rank=0 point comes from compressor=`layerwise`
- Accuracy at every (dataset, compressor, rank) cell is averaged across the
  three seeds 4, 44, 444.

To plot a different project, edit ``PROJECT_NAME`` below.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_config import DATASET_LABELS, apply_plot_style, canonicalize_records, expand_method_aliases
from summary_utils import filter_summary_records, load_summary_records
from utils.wandb_summary import update_project_summary


SCRIPT_DIR = Path(__file__).resolve().parent
SAVE_DIR = SCRIPT_DIR.parent / "figures"
SAVE_NAME = "pca_rank_sweep.png"

PROJECT_NAME = "pca_rank_sweep"
MODEL_NAME = "Qwen/Qwen3-4B"
SEEDS = [4, 44, 444]
LATENT_STEPS = 40

# Display compressors (curves on the plot).
COMPRESSORS = ["lobf", "hobf"]
# rank=0 baselines: each baseline compressor maps to (display_compressor, 0).
# Records logged with compressor=headwise are treated as "hobf at rank=0".
BASELINE_COMPRESSORS = {
    "headwise": ("hobf", 0),
    "layerwise": ("lobf", 0),
}
PCA_RANK_ORDER = [0, 2, 4, 8, 16, 32]
BASELINE_P = 0  # delta annotations are computed vs rank=0 (layerwise / headwise no-OBF baseline)
DATASETS = ["aime2024", "aime2025", "gpqa", "arc_challenge", "arc_easy", "humanevalplus", "mbppplus", "medqa", "gsm8k"]

FIG_SIZE = (14.0, 5.6)  # two subplots side by side
LINE_WIDTH = 1.8
MARKER_SIZE = 6.2
DATASET_COLORS = {
    "aime2024": "#009E73",
    "aime2025": "#117733",
    "gpqa": "#56B4E9",
    "arc_challenge": "#D55E00",
    "arc_easy": "#E69F00",
    "humanevalplus": "#CC79A7",
    "mbppplus": "#882255",
    "medqa": "#0072B2",
    "gsm8k": "#882E72",
}
COMPRESSOR_STYLES = {
    "lobf": {"linestyle": "-", "marker": "o", "label": "L-OBF"},
    "hobf": {"linestyle": "--", "marker": "s", "label": "H-OBF"},
}
ANNOTATION_FONT_SIZE = 8


def _is_a100_run(record):
    """True iff the wandb run was executed on an NVIDIA A100 GPU.

    Checks both ``metadata.gpu_nvidia`` (JSON-ish blob like
    ``[{'name': 'NVIDIA A100-PCIE-40GB', ...}]``) and ``metadata.gpu``
    (plain string). Anything else (RTX 6000 / Blackwell / unknown) is
    excluded, so the plot only averages over the homogeneous A100 hardware
    and is free of cross-GPU bf16 sampling noise.
    """
    gpu_blob = ""
    for key in ("metadata.gpu_nvidia", "metadata.gpu"):
        v = record.get(key)
        if v:
            gpu_blob += str(v)
    return "A100" in gpu_blob


def remap_record_to_display(record):
    """Return (display_compressor, display_rank) for a record, applying the
    baseline compressor remap. Returns ``None`` if the record has no usable
    compressor field.
    """
    compressor = record.get("compressor")
    if compressor is None:
        return None
    if compressor in BASELINE_COMPRESSORS:
        return BASELINE_COMPRESSORS[compressor]
    p_value = record.get("p_value")
    if p_value is None:
        return None
    return (compressor, p_value)


def aggregate_records(records):
    """Group by (dataset, display_compressor, display_rank); average accuracy
    across all matching records (typically across seeds 4 / 44 / 444).
    """
    grouped = {}
    for record in records:
        dataset = record.get("dataset")
        accuracy = record.get("accuracy")
        if dataset is None or accuracy is None:
            continue
        mapped = remap_record_to_display(record)
        if mapped is None:
            continue
        compressor, p_value = mapped
        key = (dataset, compressor, p_value)
        grouped.setdefault(key, []).append(record)

    aggregated = {}
    duplicates = []
    for key, rows in grouped.items():
        accuracies = [float(row["accuracy"]) for row in rows if np.isfinite(row.get("accuracy"))]
        if not accuracies:
            continue
        seeds_seen = sorted(
            {row.get("seed") for row in rows if row.get("seed") is not None},
            key=lambda s: (s is None, s),
        )
        evrs = [
            float(row["summary.pca_evr_mean"])
            for row in rows
            if row.get("summary.pca_evr_mean") is not None
            and np.isfinite(row.get("summary.pca_evr_mean"))
        ]
        merged = dict(rows[-1])
        merged["accuracy"] = float(np.mean(accuracies))
        merged["pca_evr_mean"] = float(np.mean(evrs)) if evrs else None
        merged["n_runs"] = len(accuracies)
        merged["seeds_seen"] = seeds_seen
        aggregated[key] = merged
        if len(rows) > 1:
            duplicates.append((key, len(rows), seeds_seen))
    return aggregated, duplicates


def annotate_delta_vs_baseline(ax, x_values, y_values, baseline_value, color):
    for x_value, y_value in zip(x_values, y_values):
        delta = y_value - baseline_value
        # Always place the delta label above the point, regardless of sign.
        ax.annotate(
            f"{delta:+.1f}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE,
            color=color,
        )


def _baseline_accuracy(aggregated, dataset, compressor):
    """Accuracy of the no-OBF baseline (rank=0 = layerwise/headwise)."""
    rec = aggregated.get((dataset, compressor, 0))
    if rec is None:
        return None
    acc = rec.get("accuracy")
    return acc if (acc is not None and np.isfinite(acc)) else None


def _draw_curve(ax, xs, ys, above_mask, color, marker, linestyle):
    """Line through (xs, ys) with per-point markers: filled when above the
    baseline (above_mask True), hollow (white face) when below."""
    ax.plot(xs, ys, color=color, linewidth=LINE_WIDTH, linestyle=linestyle, alpha=0.95, zorder=2)
    for x, y, above in zip(xs, ys, above_mask):
        ax.plot(
            [x], [y],
            marker=marker,
            markersize=MARKER_SIZE,
            markerfacecolor=(color if above else "white"),
            markeredgecolor=color,
            markeredgewidth=1.4,
            linestyle="none",
            zorder=3,
        )


def plot_panel(ax, aggregated, datasets, compressor):
    """Absolute accuracy vs PCA rank, one curve per dataset.

    Marker fill encodes vs the no-OBF baseline: filled = at/above baseline,
    hollow = below. Each point is annotated with the |Δ| magnitude (pp), no sign.
    """
    any_line = False
    style = COMPRESSOR_STYLES.get(compressor, {})
    marker = style.get("marker", "o")
    linestyle = style.get("linestyle", "-")
    for dataset in datasets:
        color = DATASET_COLORS.get(dataset, "#666666")
        base = _baseline_accuracy(aggregated, dataset, compressor)
        x_values = []
        y_values = []
        above_mask = []
        for p_rank in PCA_RANK_ORDER:
            record = aggregated.get((dataset, compressor, p_rank))
            if record is None:
                continue
            value = record.get("accuracy")
            if value is None or not np.isfinite(value):
                continue
            x_values.append(p_rank)
            y_values.append(value * 100)
            above_mask.append(base is None or value >= base)

        if not x_values:
            continue

        any_line = True
        _draw_curve(ax, x_values, y_values, above_mask, color, marker, linestyle)

        # Annotate |Δ| vs baseline (no sign) above each point.
        if base is not None and np.isfinite(base):
            for x, y in zip(x_values, y_values):
                delta = abs(y - base * 100)
                ax.annotate(
                    f"{delta:.1f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    va="bottom",
                    fontsize=ANNOTATION_FONT_SIZE,
                    color=color,
                )

    if not any_line:
        return False

    ax.set_xticks(PCA_RANK_ORDER)
    ax.set_xlim(min(PCA_RANK_ORDER) - 1, max(PCA_RANK_ORDER) + 1)
    ax.set_xlabel("PCA Rank")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.grid(axis="x", color="#F2F2F2", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return True


def plot_panel_evr(ax, aggregated, datasets, compressor):
    """Accuracy vs explained-variance-ratio (EVR) for a single compressor.

    x = pca_evr_mean (% of total variance the selected rank captures),
    y = accuracy. rank=0 baselines are skipped (no OBF → no EVR).
    """
    any_line = False
    style = COMPRESSOR_STYLES.get(compressor, {})
    for dataset in datasets:
        color = DATASET_COLORS.get(dataset, "#666666")
        base = _baseline_accuracy(aggregated, dataset, compressor)
        if base is None:
            continue
        # EVR=0 anchor = no-OBF baseline (layerwise/headwise) → delta 0.
        pts = [(0.0, 0.0)]
        for p_rank in PCA_RANK_ORDER:
            if p_rank == 0:
                continue  # baseline already added as the EVR=0 anchor
            record = aggregated.get((dataset, compressor, p_rank))
            if record is None:
                continue
            evr = record.get("pca_evr_mean")
            acc = record.get("accuracy")
            if evr is None or not np.isfinite(evr) or acc is None or not np.isfinite(acc):
                continue
            pts.append((evr * 100.0, (acc - base) * 100.0))  # delta vs baseline
        if len(pts) <= 1:
            continue
        pts.sort()  # ascending EVR
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        above_mask = [y >= 0 for y in ys]  # at/above baseline (Δ ≥ 0) → filled
        any_line = True
        _draw_curve(
            ax, xs, ys, above_mask, color,
            style.get("marker", "o"), style.get("linestyle", "-"),
        )

    if not any_line:
        return False

    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Explained Variance (%)")
    ax.set_ylabel(r"$\Delta$ Accuracy vs no-OBF (pp)")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.grid(axis="x", color="#F2F2F2", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return True


def build_legend(fig, datasets):
    """Figure legend: dataset colors + the two marker encodings.

    Encoding 1 (shape): circle = L-OBF, square = H-OBF (COMPRESSOR_STYLES).
    Encoding 2 (fill): solid marker = at/above the no-OBF baseline,
    hollow marker = below it.
    """
    enc = "#444444"
    handles = [
        Line2D(
            [0], [0],
            color=DATASET_COLORS.get(dataset, "#666666"),
            linewidth=2.0,
            label=DATASET_LABELS.get(dataset, dataset),
        )
        for dataset in datasets
    ]
    # Shape encoding: one neutral-gray handle per compressor marker.
    shape_handles = [
        Line2D(
            [0], [0],
            color=enc, linestyle="none",
            marker=COMPRESSOR_STYLES[c]["marker"], markersize=MARKER_SIZE,
            markerfacecolor=enc, markeredgecolor=enc,
            label=COMPRESSOR_STYLES[c]["label"],
        )
        for c in COMPRESSORS if c in COMPRESSOR_STYLES
    ]
    # Fill encoding: solid vs hollow (use a circle for both).
    fill_handles = [
        Line2D(
            [0], [0],
            color=enc, linestyle="none",
            marker="o", markersize=MARKER_SIZE,
            markerfacecolor=enc, markeredgecolor=enc,
            label=r"$\geq$ no-OBF baseline",
        ),
        Line2D(
            [0], [0],
            color=enc, linestyle="none",
            marker="o", markersize=MARKER_SIZE,
            markerfacecolor="white", markeredgecolor=enc, markeredgewidth=1.4,
            label=r"$<$ no-OBF baseline",
        ),
    ]
    # Lay out exactly two rows. NOTE: matplotlib fills legends COLUMN-major, so
    # we build the two target rows then interleave (row1[i], row2[i], ...) so the
    # column-major fill reproduces the intended rows:
    #   row 1:  ds[0..4]            L-OBF   solid(≥ baseline)
    #   row 2:  ds[5..8] + blank    H-OBF   hollow(< baseline)
    def _blank():
        return Line2D([0], [0], color="none", marker="none", linestyle="none", label=" ")

    half = (len(handles) + 1) // 2           # 5 for 9 datasets
    row1 = handles[:half] + [shape_handles[0], fill_handles[0]]
    row2 = handles[half:]
    row2 = row2 + [_blank() for _ in range(half - len(row2))] + [shape_handles[1], fill_handles[1]]
    ncol = len(row1)                         # datasets + shape + fill
    all_handles = [h for pair in zip(row1, row2) for h in pair]
    labels = [h.get_label() for h in all_handles]
    fig.legend(
        all_handles,
        labels,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=0.9,
        handlelength=2.0,
        handletextpad=0.5,
    )


def main():
    plt.rcParams.update(apply_plot_style())

    update_result = update_project_summary(
        PROJECT_NAME,
        states=("finished",),
    )

    # Pull the union of {sweep compressors} ∪ {baseline compressors} so the
    # rank=0 records (headwise / layerwise) survive the filter.
    fetch_compressors = expand_method_aliases(
        list(COMPRESSORS) + list(BASELINE_COMPRESSORS.keys())
    )

    records = load_summary_records(csv_paths=[update_result["output_path"]])
    filtered = filter_summary_records(
        records,
        datasets=DATASETS,
        compressors=fetch_compressors,
        models=[MODEL_NAME],
        seeds=SEEDS,
        states=["finished"],
        predicate=lambda record: (
            record.get("config.latent_steps") == LATENT_STEPS
            and _is_a100_run(record)
        ),
    )
    filtered = canonicalize_records(filtered)
    aggregated, duplicates = aggregate_records(filtered)

    if not aggregated:
        raise ValueError(
            f"No matching A100 PCA-rank sweep rows in project '{PROJECT_NAME}' "
            f"after filtering (model={MODEL_NAME}, seeds={SEEDS}, "
            f"latent_steps={LATENT_STEPS})."
        )

    datasets = [d for d in DATASETS if any(key[0] == d for key in aggregated)]
    compressors = [c for c in COMPRESSORS if any(key[1] == c for key in aggregated)]
    if not datasets or not compressors:
        raise ValueError("No plottable rows after filtering by datasets/compressors.")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # 2 x N grid:
    #   top row    = accuracy vs PCA rank
    #   bottom row = accuracy vs explained-variance-ratio (EVR)
    # columns = compressors (L-OBF, H-OBF). y-axis (accuracy) shared per column.
    ncols = len(compressors)
    fig, axes = plt.subplots(
        2, ncols, figsize=(FIG_SIZE[0], FIG_SIZE[1] * 1.75),
        constrained_layout=False, sharey="row",
    )
    axes = np.atleast_2d(axes)
    if ncols == 1:
        axes = axes.reshape(2, 1)

    plotted_any = False
    for col, compressor in enumerate(compressors):
        ax_top = axes[0][col]
        ax_bot = axes[1][col]
        ok_top = plot_panel(ax_top, aggregated, datasets, compressor)
        ok_bot = plot_panel_evr(ax_bot, aggregated, datasets, compressor)
        if ok_top:
            plotted_any = True
            ax_top.set_title(
                COMPRESSOR_STYLES.get(compressor, {}).get("label", compressor),
                fontweight="bold",
            )
        else:
            ax_top.axis("off")
        if not ok_bot:
            ax_bot.axis("off")

    if not plotted_any:
        raise ValueError("No plottable PCA-rank sweep lines were found.")

    # Only the leftmost column keeps the y-label (sharey hides other ticklabels).
    for row in range(2):
        for col in range(1, ncols):
            axes[row][col].set_ylabel("")

    build_legend(fig, datasets)
    fig.subplots_adjust(bottom=0.15, top=0.95, left=0.07, right=0.98, wspace=0.10, hspace=0.28)

    save_path = SAVE_DIR / SAVE_NAME
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    saved_paths = [save_path]

    print(f"Project path:           {update_result['project_path']}")
    print(f"Project CSV:            {update_result['output_path']}")
    print(f"Fetched runs:           {update_result['fetched_records']}")
    print(f"Reused cached runs:     {update_result['reused_records']}")
    print(f"Matched rows (filter):  {len(filtered)}")
    print(f"Aggregated cells:       {len(aggregated)}")
    print(f"Seeds requested:        {SEEDS}")
    print(f"Datasets plotted:       {datasets}")
    print(f"Compressors plotted:    {compressors}")
    if duplicates:
        print("Cells averaged across multiple records:")
        for (dataset, compressor, p_value), count, seeds_seen in duplicates:
            print(
                f"  dataset={dataset} compressor={compressor} p={p_value} "
                f"n={count} seeds={seeds_seen}"
            )

    sparse = [
        (key, info["n_runs"], info.get("seeds_seen", []))
        for key, info in aggregated.items()
        if info["n_runs"] < len(SEEDS)
    ]
    if sparse:
        print(f"Cells with fewer than {len(SEEDS)} seed runs (incomplete sweep):")
        for (dataset, compressor, p_value), count, seeds_seen in sparse:
            print(
                f"  dataset={dataset} compressor={compressor} p={p_value} "
                f"n={count} seeds={seeds_seen}"
            )

    for p in saved_paths:
        print(f"Saved plot to:          {p}")


if __name__ == "__main__":
    main()
