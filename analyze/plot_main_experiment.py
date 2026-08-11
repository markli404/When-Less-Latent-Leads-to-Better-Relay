"""Main-experiment grouped bar chart — Qwen3-4B.

Running this script produces:
    figures/main_experiment_4b.png

Plots accuracy of {full, gonly, layerwise, headwise, lobf, hobf} on every
dataset plus an "Average" column. Auto-fetches/refreshes both the main_experiment
and pca_rank_sweep wandb project CSVs before plotting; lobf/hobf bars use the
best-over-pca_rank accuracy from the sweep.

The 8B figure is produced by the companion script plot_main_experiment_8b.py,
which reads its lobf {2,4,8,32} rank spread from main_experiment itself and skips
the pca_rank_sweep fetch. Shared drawing/aggregation logic lives here and is
imported by that script — edit styling in this file and both figures update.
"""

import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_config import (
    DATASET_ORDER,
    DATASET_LABELS,
    METHOD_COLORS,
    METHOD_LABELS,
    apply_plot_style,
    canonicalize_records,
    expand_method_aliases,
)
from summary_utils import filter_summary_records, load_summary_records
from utils.wandb_summary import update_project_summary


SCRIPT_DIR = Path(__file__).resolve().parent
SAVE_DIR = SCRIPT_DIR.parent / "figures"

PROJECT_NAME = "main_experiment"
LATENT_STEPS = 40
METHOD_ORDER = ["full", "gonly", "layerwise", "headwise", "lobf", "hobf"]

# For these methods, the bar value is replaced by the BEST-over-pca_rank
# accuracy from the pca_rank_sweep project (A100 only, mean over seeds), instead
# of the single fixed-rank value logged in main_experiment.
SWEEP_PROJECT = "pca_rank_sweep"
SWEEP_BEST_METHODS = {"lobf", "hobf"}

FIG_SIZE = (12.8, 6.0)


# ---------------------------------------------------------------------------
# Project-name fuzzy matching (kept identical to the previous implementation)
# ---------------------------------------------------------------------------

def normalize_project_token(text):
    text = str(text).strip().lower()
    if text.endswith("_merged"):
        text = text[: -len("_merged")]
    if text.endswith(".csv"):
        text = text[: -len(".csv")]
    return text


def record_project_tokens(record):
    tokens = []
    for key in ("project", "config.project_name", "config.project", "summary.project_name"):
        value = record.get(key)
        if value:
            tokens.append(str(value))
    source_csv = record.get("source_csv")
    if source_csv:
        stem = Path(str(source_csv)).stem
        tokens.append(stem)
        tokens.append(normalize_project_token(stem))
    return tokens


def filter_records_by_project(records, project_name):
    project_token = normalize_project_token(project_name)
    filtered = []
    for record in records:
        tokens = {normalize_project_token(token) for token in record_project_tokens(record)}
        if any(
            token == project_token or token.startswith(project_token) or project_token.startswith(token)
            for token in tokens
        ):
            filtered.append(record)
    return filtered


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_records(records):
    grouped = {}
    for record in records:
        dataset = record.get("dataset")
        method = record.get("compressor")
        accuracy = record.get("accuracy")
        if dataset is None or method is None or accuracy is None:
            continue
        key = (dataset, method)
        grouped.setdefault(key, []).append(record)

    aggregated = {}
    duplicates = []
    for key, rows in grouped.items():
        accuracies = [float(row["accuracy"]) for row in rows if np.isfinite(row.get("accuracy"))]
        if not accuracies:
            continue
        merged = dict(rows[-1])
        merged["accuracy"] = float(np.mean(accuracies))
        merged["n_runs"] = len(accuracies)
        aggregated[key] = merged
        if len(rows) > 1:
            duplicates.append((key[0], key[1], len(rows)))
    return aggregated, duplicates


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def annotate_bars(ax, bars, values=None):
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        text_value = height if values is None else values[idx]
        if not np.isfinite(text_value):
            continue
        ax.annotate(
            f"{text_value:+.1f}" if values is not None else f"{height:.1f}",
            (bar.get_x() + bar.get_width() / 2, height),
            textcoords="offset points",
            xytext=(0, 1),
            ha="center",
            fontsize=7,
            color="#333333",
        )


def set_adaptive_ylim(ax, aggregated, datasets, methods):
    values = []
    for dataset in datasets:
        for method in methods:
            record = aggregated.get((dataset, method))
            accuracy = record.get("accuracy") if record is not None else np.nan
            value = accuracy * 100 if np.isfinite(accuracy) else np.nan
            if np.isfinite(value):
                values.append(value)

    if not values:
        return

    min_val = min(values)
    max_val = max(values)
    lower = max(0, 5 * np.floor((min_val - 4) / 5))
    upper = min(100, 5 * np.ceil((max_val + 4) / 5))
    if upper - lower < 20:
        center = 0.5 * (lower + upper)
        lower = max(0, 5 * np.floor((center - 10) / 5))
        upper = min(100, 5 * np.ceil((center + 10) / 5))
    ax.set_ylim(lower, upper)


def draw_grouped_bar_chart(ax, datasets, methods, aggregated):
    panel_datasets = list(datasets) + ["Average"]
    x = np.arange(len(panel_datasets))
    width = 0.82 / len(methods)
    full_values = {}
    for dataset in datasets:
        record = aggregated.get((dataset, "full"))
        if record is not None and np.isfinite(record.get("accuracy")):
            full_values[dataset] = record.get("accuracy") * 100

    for idx, method in enumerate(methods):
        values = []
        label_values = []
        dataset_values = []
        dataset_label_values = []
        for dataset in datasets:
            record = aggregated.get((dataset, method))
            accuracy = record.get("accuracy") if record is not None else np.nan
            value = accuracy * 100 if np.isfinite(accuracy) else np.nan
            dataset_values.append(value)
            if method != "full" and np.isfinite(value):
                full_value = full_values.get(dataset, np.nan)
                dataset_label_values.append(value - full_value if np.isfinite(full_value) else np.nan)
            else:
                dataset_label_values.append(np.nan)

        finite_vals = [v for v in dataset_values if np.isfinite(v)]
        values.extend(dataset_values)
        label_values.extend(dataset_label_values)
        avg_value = float(np.mean(finite_vals)) if finite_vals else np.nan
        values.append(avg_value)
        if method != "full" and np.isfinite(avg_value):
            avg_full_vals = [
                full_values[d]
                for d in datasets
                if d in full_values and np.isfinite(aggregated.get((d, method), {}).get("accuracy", np.nan))
            ]
            avg_full = float(np.mean(avg_full_vals)) if avg_full_vals else np.nan
            label_values.append(avg_value - avg_full if np.isfinite(avg_full) else np.nan)
        else:
            label_values.append(np.nan)

        offset = (idx - (len(methods) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "#666666"),
            edgecolor="white",
            linewidth=0.7,
        )
        if method != "full":
            annotate_bars(ax, bars, label_values)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [DATASET_LABELS.get(d, d) for d in datasets] + ["Average"],
        rotation=18,
        ha="right",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("Accuracy (%)")
    set_adaptive_ylim(ax, aggregated, datasets, methods)
    ax.axvline(len(datasets) - 0.5, color="#888888", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.04)


# ---------------------------------------------------------------------------
# Per-model plot
# ---------------------------------------------------------------------------

def _is_a100_run(record):
    """True iff the wandb run was executed on an NVIDIA A100 GPU.

    Checks both ``metadata.gpu_nvidia`` (modern field, often a JSON-ish blob
    like ``[{'name': 'NVIDIA A100-PCIE-40GB', ...}]``) and ``metadata.gpu``
    (older field, a plain string like ``NVIDIA A100-PCIE-40GB``). Anything
    else (RTX 6000, Blackwell, etc.) is excluded.
    """
    gpu_blob = ""
    for key in ("metadata.gpu_nvidia", "metadata.gpu"):
        v = record.get(key)
        if v:
            gpu_blob += str(v)
    return "A100" in gpu_blob


def compute_sweep_best(sweep_records, model_name, gpu_filter=None):
    """Best-over-pca_rank accuracy per (dataset, method) from the rank sweep.

    For each method in SWEEP_BEST_METHODS: average accuracy over seeds within
    each pca_rank, then take the best (highest) rank's mean.
    ``gpu_filter``: hardware predicate; defaults to A100-only (the 4B/8B
    protocol). The 14B wrapper passes an A100-or-H200 filter.
    Returns {(dataset, method): {"accuracy", "rank", "n_seeds"}}.
    """
    gpu_filter = gpu_filter or _is_a100_run
    filtered = filter_summary_records(
        sweep_records,
        compressors=expand_method_aliases(list(SWEEP_BEST_METHODS)),
        states=["finished"],
        predicate=lambda r: (
            r.get("config.latent_steps") == LATENT_STEPS
            and r.get("config.model_name") == model_name
            and gpu_filter(r)
        ),
    )
    filtered = canonicalize_records(filtered)

    # (dataset, method, rank) -> [accuracy per seed]
    by_rank = defaultdict(list)
    for r in filtered:
        ds = r.get("dataset")
        m = r.get("compressor")
        p = r.get("p_value")
        acc = r.get("accuracy")
        if None in (ds, m, p, acc):
            continue
        if not np.isfinite(acc):
            continue
        by_rank[(ds, m, p)].append(float(acc))

    # mean over seeds per rank, then best rank per (dataset, method)
    best = {}
    for (ds, m, p), accs in by_rank.items():
        mean = float(np.mean(accs))
        key = (ds, m)
        if key not in best or mean > best[key]["accuracy"]:
            best[key] = {"accuracy": mean, "rank": p, "n_seeds": len(accs)}
    return best


def plot_one_model(records, model_name, save_name, sweep_records=None, gpu_filter=None):
    """Filter ``records`` to ``model_name`` and the hardware filter, write one PNG.

    ``sweep_records``: if given, the bar for each method in SWEEP_BEST_METHODS
    is replaced by its best-over-pca_rank accuracy from the rank sweep.
    ``gpu_filter``: hardware predicate; defaults to A100-only (4B/8B protocol).
    The 14B wrapper passes an H200-only filter.

    Returns a small dict of stats for logging.
    """
    gpu_filter = gpu_filter or _is_a100_run
    filtered = filter_summary_records(
        records,
        compressors=expand_method_aliases(METHOD_ORDER),
        states=["finished"],
        predicate=lambda r: (
            r.get("config.latent_steps") == LATENT_STEPS
            and r.get("config.model_name") == model_name
            and gpu_filter(r)
        ),
    )
    filtered = filter_records_by_project(filtered, PROJECT_NAME)
    filtered = canonicalize_records(filtered)

    aggregated, duplicates = aggregate_records(filtered)

    # Override SWEEP_BEST_METHODS bars with the best-over-pca_rank value.
    # Source pool = pca_rank_sweep ∪ main_experiment, filtered to this model:
    #   - 4B: the rank spread lives in pca_rank_sweep.
    #   - 8B: the rank spread (lobf {2,4,8,32}) lives in main_experiment itself,
    #     so we must include `records` here or the lobf bar would collapse to the
    #     mean-over-ranks that aggregate_records() produces.
    # best-rank = per (dataset, method): average accuracy over seeds at each rank,
    # then take the rank with the highest mean.
    sweep_overrides = []
    best_rank_pool = list(records)
    if sweep_records:
        best_rank_pool = list(sweep_records) + best_rank_pool
    if SWEEP_BEST_METHODS:
        sweep_best = compute_sweep_best(best_rank_pool, model_name, gpu_filter=gpu_filter)
        for (ds, m), info in sweep_best.items():
            if m not in SWEEP_BEST_METHODS:
                continue
            if (ds, m) in aggregated:
                merged = dict(aggregated[(ds, m)])
                merged["accuracy"] = info["accuracy"]
                merged["sweep_best_rank"] = info["rank"]
                merged["n_runs"] = info["n_seeds"]
                aggregated[(ds, m)] = merged
            else:
                # dataset/method present in sweep but not in main aggregate —
                # add it so the bar still shows the swept best.
                aggregated[(ds, m)] = {
                    "accuracy": info["accuracy"],
                    "sweep_best_rank": info["rank"],
                    "n_runs": info["n_seeds"],
                }
            sweep_overrides.append((ds, m, info["rank"], info["accuracy"]))

    datasets = [d for d in DATASET_ORDER if any((d, m) in aggregated for m in METHOD_ORDER)]
    methods = [m for m in METHOD_ORDER if any((d, m) in aggregated for d in datasets)]
    if not datasets or not methods:
        raise ValueError(
            f"No matching A100 rows for model='{model_name}' in project '{PROJECT_NAME}'."
        )

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    draw_grouped_bar_chart(ax, datasets=datasets, methods=methods, aggregated=aggregated)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            ncol=min(len(methods), 6),
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
        )
    fig.subplots_adjust(bottom=0.18, top=0.94)

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = SAVE_DIR / save_name
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    return {
        "model_name": model_name,
        "save_path": save_path,
        "matched": len(filtered),
        "aggregated": len(aggregated),
        "datasets": datasets,
        "methods": methods,
        "duplicates": duplicates,
        "sweep_overrides": sweep_overrides,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(models):
    """``models`` is a list of (model_name, save_name) tuples."""
    plt.rcParams.update(apply_plot_style())

    update_result = update_project_summary(
        PROJECT_NAME,
        states=("finished",),
    )
    records = load_summary_records(csv_paths=[update_result["output_path"]])

    print(f"Project:                {PROJECT_NAME}")
    print(f"Project CSV:            {update_result['output_path']}")
    print(f"Fetched runs:           {update_result['fetched_records']}")
    print(f"Reused cached runs:     {update_result['reused_records']}")
    print(f"Loaded summary records: {len(records)}")

    # Rank-sweep records: used to replace SWEEP_BEST_METHODS bars with their
    # best-over-pca_rank value. If the sweep project can't be fetched, fall back
    # to the fixed-rank main_experiment bars.
    sweep_records = None
    if SWEEP_BEST_METHODS:
        try:
            sweep_update = update_project_summary(SWEEP_PROJECT, states=("finished",))
            sweep_records = load_summary_records(csv_paths=[sweep_update["output_path"]])
            print(f"Sweep project:          {SWEEP_PROJECT} ({sweep_update['output_path']})")
            print(f"Sweep records:          {len(sweep_records)}")
            print(f"Best-rank override for: {sorted(SWEEP_BEST_METHODS)}")
        except Exception as exc:
            print(f"[warn] could not load sweep '{SWEEP_PROJECT}': {exc}")
            print(f"[warn] {sorted(SWEEP_BEST_METHODS)} bars fall back to fixed-rank main values.")

    for model_name, save_name in models:
        try:
            stats = plot_one_model(records, model_name, save_name, sweep_records=sweep_records)
        except Exception as exc:
            print(f"[skip] {model_name}: {exc}")
            continue
        print(f"--- {model_name} ---")
        print(f"  Matched after filter:   {stats['matched']}")
        print(f"  Aggregated cells:       {stats['aggregated']}")
        print(f"  Datasets:               {stats['datasets']}")
        print(f"  Methods:                {stats['methods']}")
        if stats["duplicates"]:
            print("  Duplicate cells averaged:")
            for dataset, method, count in stats["duplicates"]:
                print(f"    {dataset} | {method} | n={count}")
        if stats.get("sweep_overrides"):
            print("  Sweep best-rank overrides (dataset, method, best_rank, acc):")
            for ds, m, rank, acc in sorted(stats["sweep_overrides"]):
                print(f"    {ds} | {m} | rank={rank} | acc={acc:.4f}")
        print(f"  Saved plot to:          {stats['save_path']}")


if __name__ == "__main__":
    # 4B only. Its lobf/hobf best-rank lives in the pca_rank_sweep project, which
    # main() fetches. The 8B figure is produced by plot_main_experiment_8b.py,
    # which reads its rank spread from main_experiment itself and skips the
    # (slow) pca_rank_sweep fetch.
    main([
        ("Qwen/Qwen3-4B",  "main_experiment_4b.png"),
    ])
