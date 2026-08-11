import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_config import (
    DATASET_LABELS,
    DATASET_ORDER,
    METHOD_COLORS,
    METHOD_LABELS,
    apply_plot_style,
    canonicalize_records,
)
from summary_utils import filter_summary_records, load_summary_records
from utils.wandb_summary import update_project_summary


SCRIPT_DIR = Path(__file__).resolve().parent
SAVE_DIR = SCRIPT_DIR.parent / "figures"

# Source the main-experiment runs the same way plot_main_experiment.py does:
# auto-fetch / refresh the project's wandb summary CSV on every invocation.
PROJECT_NAME = "main_experiment"

# Fallback CSV path for offline use — the historical "main experiment" dump
# that lived alongside the active CSVs before they got reorganised.
FALLBACK_CSV_PATHS = [
    SCRIPT_DIR.parent / "results" / "wandb_summary" / "main experiment.csv",
]

SEEDS = [4, 44, 444]
P_VALUE = 2
LATENT_STEPS = 40
METHODS = ["full", "gonly", "layerwise", "headwise", "lobf", "hobf"]

# For OBF methods the tradeoff figure uses the SAME best-over-pca_rank run as the
# accuracy/efficiency tables: A100 only, mean accuracy over seeds, best rank per
# (dataset, method). These runs come from the pca_rank_sweep project (main_experiment
# only logs one fixed rank each). Non-OBF methods stay sourced from main_experiment.
SWEEP_PROJECT = "pca_rank_sweep"
OBF_METHODS = {"lobf", "hobf"}
DATASETS = [d for d in DATASET_ORDER if d in {"gsm8k", "aime2024", "aime2025", "gpqa", "medqa", "arc_easy", "arc_challenge", "mbppplus", "humanevalplus"}]

SCATTER_PLOTS = [
    ("token_usage", "Token Usage", "accuracy_vs_token_usage.png"),
    ("avg_text_inference_time_s", "Text Inference Time (s)", "accuracy_vs_text_inference_time.png"),
    ("avg_latent_inference_time_s", "Latent Inference Time (s)", "accuracy_vs_latent_inference_time.png"),
]

DUAL_AXIS_SAVE_NAME = "text_inference_time_vs_token_usage.png"

# Font sizes (bumped per request).
SUBPLOT_TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 18
SCATTER_TITLE_FONTSIZE = 18


def _is_a100_run(record):
    """True iff the run was executed on an A100 GPU (checks both metadata fields)."""
    blob = ""
    for key in ("metadata.gpu_nvidia", "metadata.gpu"):
        v = record.get(key)
        if v:
            blob += str(v)
    return "A100" in blob


def dataset_label_with_device_mark(dataset, gpu_name=None):
    if dataset == "average":
        return "Average"
    return DATASET_LABELS.get(dataset, dataset)


def compute_best_ranks(sweep_records):
    """Best-over-pca_rank per (dataset, OBF method), by mean accuracy over seeds.

    Mirrors the accuracy-table selection so every figure/table uses the same rank.
    Returns {(dataset, method): best_rank}.
    """
    by_rank = defaultdict(list)  # (dataset, method, rank) -> [accuracy per seed]
    for r in sweep_records:
        m = r.get("compressor")
        if m not in OBF_METHODS:
            continue
        ds, rank, acc = r.get("dataset"), r.get("p_value"), r.get("accuracy")
        if None in (ds, rank, acc) or not np.isfinite(acc):
            continue
        by_rank[(ds, m, rank)].append(float(acc))

    best = {}  # (dataset, method) -> {"rank", "acc"}
    for (ds, m, rank), accs in by_rank.items():
        mean = float(np.mean(accs))
        key = (ds, m)
        if key not in best or mean > best[key]["acc"]:
            best[key] = {"rank": rank, "acc": mean}
    return {key: info["rank"] for key, info in best.items()}


def select_best_rank_obf_records(sweep_records, best_ranks):
    """Keep only the OBF sweep runs whose rank == the best rank for that cell."""
    out = []
    for r in sweep_records:
        m = r.get("compressor")
        if m not in OBF_METHODS:
            continue
        key = (r.get("dataset"), m)
        if key in best_ranks and r.get("p_value") == best_ranks[key]:
            out.append(r)
    return out


def average_across_seeds(records, required_metric_keys):
    """Average required metrics and accuracy across seeds for each (dataset, method)."""
    required_metric_keys = list(required_metric_keys)
    grouped = defaultdict(list)
    for record in records:
        if record.get("accuracy") is None:
            continue
        if any(record.get(key) is None for key in required_metric_keys):
            continue
        grouped[(record.get("dataset"), record.get("compressor"))].append(record)

    averaged = {}
    for key, bucket in grouped.items():
        avg_record = {
            "dataset": key[0],
            "compressor": key[1],
            "accuracy": float(np.mean([r["accuracy"] for r in bucket])),
        }
        for metric_key in required_metric_keys:
            avg_record[metric_key] = float(np.mean([r[metric_key] for r in bucket]))
        avg_record["_n_seeds"] = len(bucket)
        averaged[key] = avg_record
    return averaged


def compute_pearson_r(xs, ys):
    if len(xs) < 2 or len(ys) < 2:
        return None
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def plot_tradeoff(records, metric_key, x_label, save_name):
    deduped = average_across_seeds(records, [metric_key])

    datasets = [dataset for dataset in DATASETS if any((dataset, method) in deduped for method in METHODS)]
    if not datasets:
        raise ValueError(f"No data available for metric: {metric_key}")

    panel_datasets = datasets + ["average"]
    n_cols = 5
    n_rows = math.ceil(len(panel_datasets) / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, 3.6 * n_rows),
        squeeze=False,
        constrained_layout=False,
    )
    axes = axes.flatten()

    handles = {}

    for ax, dataset in zip(axes, panel_datasets):
        panel_idx = panel_datasets.index(dataset)
        row_idx = panel_idx // n_cols
        col_idx = panel_idx % n_cols
        panel_xs = []
        panel_ys = []
        for method in METHODS:
            color = METHOD_COLORS.get(method, "#666666")
            label = METHOD_LABELS.get(method, method)

            if dataset == "average":
                # Average panel: scale Full's cross-dataset mean by the
                # per-dataset mean ratio so datasets with different scales
                # contribute equally on the metric axis. Accuracy is already
                # in uniform units, so direct mean is fine.
                full_metric_per_d = {
                    d: float(deduped[(d, "full")].get(metric_key))
                    for d in datasets
                    if (d, "full") in deduped
                    and deduped[(d, "full")].get(metric_key) not in (None, 0)
                }
                full_metric_avg = (
                    float(np.mean(list(full_metric_per_d.values())))
                    if full_metric_per_d else 0.0
                )
                ratios = []
                ys = []
                for source_dataset in datasets:
                    record = deduped.get((source_dataset, method))
                    full_value = full_metric_per_d.get(source_dataset)
                    if record is None or full_value in (None, 0):
                        continue
                    ratios.append(float(record.get(metric_key)) / full_value)
                    ys.append(float(record.get("accuracy")) * 100)
                if not ratios or not ys:
                    continue
                x = full_metric_avg * float(np.mean(ratios))
                y = float(np.mean(ys))
            else:
                record = deduped.get((dataset, method))
                if record is None:
                    continue
                x = float(record.get(metric_key))
                y = float(record.get("accuracy")) * 100

            ax.scatter(x, y, color=color, s=60)
            panel_xs.append(x)
            panel_ys.append(y)
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8.5,
                color=color,
            )
            handles[method] = plt.Line2D([], [], linestyle="", marker="o", color=color, label=label)

        r_value = compute_pearson_r(panel_xs, panel_ys)
        if len(panel_xs) >= 2 and len(panel_ys) >= 2:
            x_arr = np.asarray(panel_xs, dtype=float)
            y_arr = np.asarray(panel_ys, dtype=float)
            if not np.allclose(x_arr, x_arr[0]):
                slope, intercept = np.polyfit(x_arr, y_arr, 1)
                x_fit = np.linspace(float(x_arr.min()), float(x_arr.max()), 100)
                y_fit = slope * x_fit + intercept
                ax.plot(
                    x_fit,
                    y_fit,
                    linestyle="--",
                    linewidth=1.2,
                    color="#777777",
                    alpha=0.85,
                    zorder=1,
                )

        title = dataset_label_with_device_mark(dataset)
        if r_value is not None:
            title = f"{title}\n$r={r_value:.2f}$"
        ax.set_title(
            title,
            fontweight="bold",
            fontsize=SCATTER_TITLE_FONTSIZE,
        )
        if row_idx == n_rows - 1:
            ax.set_xlabel(x_label)
        else:
            ax.set_xlabel("")
        if col_idx == 0:
            ax.set_ylabel("Accuracy (%)")
        else:
            ax.set_ylabel("")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(panel_datasets):]:
        ax.axis("off")

    if handles:
        fig.legend(
            [handles[m] for m in METHODS if m in handles],
            [METHOD_LABELS.get(m, m) for m in METHODS if m in handles],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(6, len(handles)),
            frameon=False,
        )

    fig.subplots_adjust(bottom=0.16, top=0.90, wspace=0.28, hspace=0.32)

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = SAVE_DIR / save_name
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"\nMetric: {metric_key}")
    for dataset in datasets:
        print(f"  {dataset} (cross-seed average)")
        missing = [method for method in METHODS if (dataset, method) not in deduped]
        if missing:
            print(f"    missing={missing}")
    print(f"Saved plot to: {save_path}")


def plot_dual_axis(records):
    metric_keys = ["token_usage", "avg_text_inference_time_s"]
    deduped = average_across_seeds(records, metric_keys)

    datasets = [dataset for dataset in DATASETS if any((dataset, method) in deduped for method in METHODS)]
    if not datasets:
        raise ValueError("No data available for dual-axis plot.")

    panel_datasets = datasets + ["average"]
    n_cols = 5
    n_rows = math.ceil(len(panel_datasets) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 3.9 * n_rows),
        squeeze=False,
        constrained_layout=False,
    )
    axes = axes.flatten()

    value_cache = {}
    all_token_vals = []
    all_time_vals = []
    for dataset in datasets:
        value_cache[dataset] = {}
        for method in METHODS:
            record = deduped.get((dataset, method))
            if record is None:
                continue
            token_val = float(record["token_usage"])
            time_val = float(record["avg_text_inference_time_s"])
            value_cache[dataset][method] = (token_val, time_val)
            all_token_vals.append(token_val)
            all_time_vals.append(time_val)

    # Normalise every value to a PERCENTAGE of that dataset's Full (Full=100%),
    # so datasets with very different absolute scales (e.g. aime2024 ~25 tok/s
    # vs gsm8k ~80) are comparable. Each panel then shows "% of Full", and the
    # "average" panel is the mean of these per-dataset percentages.
    pct_cache = {}
    for dataset in datasets:
        full_vals = value_cache.get(dataset, {}).get("full")
        pct_cache[dataset] = {}
        if not full_vals or full_vals[0] == 0 or full_vals[1] == 0:
            continue
        for method, (tok, tm) in value_cache[dataset].items():
            pct_cache[dataset][method] = (tok / full_vals[0] * 100.0, tm / full_vals[1] * 100.0)
    value_cache = pct_cache

    avg_values = {}
    for method in METHODS:
        toks = [value_cache[d][method][0] for d in datasets if method in value_cache.get(d, {})]
        tms = [value_cache[d][method][1] for d in datasets if method in value_cache.get(d, {})]
        if toks and tms:
            avg_values[method] = (float(np.mean(toks)), float(np.mean(tms)))

    bar_width = 0.34
    metric_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#777777", edgecolor="none", alpha=0.80, label="Token Usage"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#444444", hatch="///", linewidth=1.0, label="Text Inference Time"),
    ]

    for ax, dataset in zip(axes, panel_datasets):
        panel_idx = panel_datasets.index(dataset)
        row_idx = panel_idx // n_cols
        col_idx = panel_idx % n_cols
        if dataset == "average":
            method_values = avg_values
        else:
            method_values = value_cache.get(dataset, {})

        methods_present = [method for method in METHODS if method in method_values]
        x = np.arange(len(methods_present))

        token_vals = [method_values[method][0] for method in methods_present]
        time_vals = [method_values[method][1] for method in methods_present]
        colors = [METHOD_COLORS.get(method, "#666666") for method in methods_present]

        ax2 = ax.twinx()

        ax.bar(
            x - bar_width / 2,
            token_vals,
            width=bar_width,
            color=colors,
            alpha=0.78,
            edgecolor="none",
        )
        ax2.bar(
            x + bar_width / 2,
            time_vals,
            width=bar_width,
            color="white",
            edgecolor=colors,
            hatch="///",
            linewidth=1.1,
        )

        ax.set_title(
            dataset_label_with_device_mark(dataset),
            fontweight="bold",
            fontsize=SUBPLOT_TITLE_FONTSIZE,
        )
        # x-axis: keep tick positions for layout, drop the per-method text labels
        # (methods are now identified via the figure-level legend below).
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", length=0)
        if col_idx == 0:
            ax.set_ylabel("Token Usage (% of Full)")
        else:
            ax.set_ylabel("")
        if col_idx == n_cols - 1:
            ax2.set_ylabel("Text Time (% of Full)")
        else:
            ax2.set_ylabel("")

        if "full" in methods_present:
            full_idx = methods_present.index("full")
            full_token = token_vals[full_idx]
            full_time = time_vals[full_idx]

            # Both axes are now "% of Full" (full_token == full_time == 100).
            # Share one symmetric range so token% and time% bars are visually
            # comparable at equal heights.
            spread = 0.0
            for v in token_vals + time_vals:
                spread = max(spread, abs(v - 100.0))
            radius = max(spread * 1.15, 12.0)  # at least ±12% window
            ax.set_ylim(100.0 - radius, 100.0 + radius)
            ax2.set_ylim(100.0 - radius, 100.0 + radius)
            ax.axhline(100.0, color="#888888", linestyle="--", linewidth=1.0, alpha=0.8)

            if dataset == "average":
                for idx, method in enumerate(methods_present):
                    if method == "full":
                        continue
                    token_pct = (token_vals[idx] / full_token - 1.0) * 100.0 if full_token else np.nan
                    time_pct = (time_vals[idx] / full_time - 1.0) * 100.0 if full_time else np.nan
                    if np.isfinite(token_pct):
                        ax.annotate(
                            f"{token_pct:+.0f}%",
                            (x[idx] - bar_width / 2, token_vals[idx]),
                            textcoords="offset points",
                            xytext=(0, 3),
                            ha="center",
                            fontsize=8,
                            color="#333333",
                        )
                    if np.isfinite(time_pct):
                        ax2.annotate(
                            f"{time_pct:+.0f}%",
                            (x[idx] + bar_width / 2, time_vals[idx]),
                            textcoords="offset points",
                            xytext=(0, 3),
                            ha="center",
                            fontsize=8,
                            color="#333333",
                        )
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)

    for ax in axes[len(panel_datasets):]:
        ax.axis("off")

    # Unified legend: method colors first (replaces the removed x-axis labels),
    # then the two bar-pattern indicators distinguishing token vs. time bars.
    method_handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=METHOD_COLORS.get(method, "#666666"),
            edgecolor="none",
            label=METHOD_LABELS.get(method, method),
        )
        for method in METHODS
    ]
    legend_handles = method_handles + metric_handles
    legend_labels = [h.get_label() for h in legend_handles]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        columnspacing=1.2,
        handlelength=1.6,
        handletextpad=0.5,
    )

    fig.subplots_adjust(bottom=0.18, top=0.92, wspace=0.28, hspace=0.34)

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = SAVE_DIR / DUAL_AXIS_SAVE_NAME
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("\nMetric: dual_axis_token_usage_and_text_time")
    for dataset in datasets:
        print(f"  {dataset} (cross-seed average)")
        missing = [method for method in METHODS if (dataset, method) not in deduped]
        if missing:
            print(f"    missing={missing}")
    print(f"Saved plot to: {save_path}")


def _resolve_csv_paths():
    """Try wandb auto-fetch first; fall back to a local CSV if wandb is unavailable."""
    try:
        update_result = update_project_summary(PROJECT_NAME, states=("finished",))
        print(f"Updated project '{PROJECT_NAME}' -> {update_result['output_path']}")
        print(f"  fetched={update_result['fetched_records']}  reused={update_result['reused_records']}")
        return [update_result["output_path"]]
    except Exception as exc:
        print(f"[warn] wandb auto-fetch failed for '{PROJECT_NAME}': {exc}")
        for path in FALLBACK_CSV_PATHS:
            if Path(path).exists():
                print(f"[warn] using fallback CSV: {path}")
                return [Path(path)]
        raise FileNotFoundError(
            "No wandb fetch and no fallback CSV. Tried: "
            + ", ".join(str(p) for p in FALLBACK_CSV_PATHS)
        )


def main():
    plt.rcParams.update(apply_plot_style())

    csv_paths = _resolve_csv_paths()
    records = canonicalize_records(load_summary_records(csv_paths=csv_paths))
    # Non-OBF methods (full/gonly/layerwise/headwise) come from main_experiment.
    # A100 only. No p_value filter needed here.
    non_obf = filter_summary_records(
        records,
        datasets=DATASETS,
        compressors=[m for m in METHODS if m not in OBF_METHODS],
        seeds=SEEDS,
        states=["finished"],
        predicate=lambda r: (
            r.get("config.latent_steps") == LATENT_STEPS
            and _is_a100_run(r)
        ),
    )

    # OBF methods (lobf/hobf) come from the pca_rank_sweep project, using the SAME
    # best-over-pca_rank run as the accuracy/efficiency tables (口径统一).
    obf = []
    try:
        sweep_update = update_project_summary(SWEEP_PROJECT, states=("finished",))
        sweep_records = canonicalize_records(
            load_summary_records(csv_paths=[sweep_update["output_path"]])
        )
        sweep_a100 = filter_summary_records(
            sweep_records,
            datasets=DATASETS,
            compressors=list(OBF_METHODS),
            seeds=SEEDS,
            states=["finished"],
            predicate=lambda r: (
                r.get("config.latent_steps") == LATENT_STEPS
                and _is_a100_run(r)
            ),
        )
        best_ranks = compute_best_ranks(sweep_a100)
        obf = select_best_rank_obf_records(sweep_a100, best_ranks)
        print(f"Sweep project '{SWEEP_PROJECT}': {len(sweep_records)} records")
        print("Best-rank per OBF cell (dataset, method, rank):")
        for (ds, m), rank in sorted(best_ranks.items()):
            print(f"    {ds} | {m} | rank={rank}")
    except Exception as exc:
        print(f"[warn] could not load sweep '{SWEEP_PROJECT}': {exc}")
        print("[warn] falling back to fixed-rank lobf/hobf from main_experiment.")
        obf = filter_summary_records(
            records,
            datasets=DATASETS,
            compressors=list(OBF_METHODS),
            seeds=SEEDS,
            states=["finished"],
            predicate=lambda r: (
                r.get("config.latent_steps") == LATENT_STEPS
                and _is_a100_run(r)
            ),
        )

    filtered = list(non_obf) + list(obf)

    print(f"Loaded summary records: {len(records)}")
    print(f"Matched: non-OBF (main)={len(non_obf)}, OBF best-rank (sweep)={len(obf)}, total={len(filtered)}")

    for metric_key, x_label, save_name in SCATTER_PLOTS:
        plot_tradeoff(filtered, metric_key, x_label, save_name)
    plot_dual_axis(filtered)


if __name__ == "__main__":
    main()
