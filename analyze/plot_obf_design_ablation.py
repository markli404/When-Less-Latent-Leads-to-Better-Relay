"""OBF design-ablation figure (wandb project: obf_ablation).

Grouped bars of accuracy per dataset for every ablation variant, against the
full L-OBF reference (pca_rank=4, uniform injection) pulled from the existing
main_experiment project.

Variants (one campaign seed; A100 only):
  g1 components : lobf_naive, lobf_no_proj, lobf_no_scale, lobf_max_p
  g2 injection  : lobf + inject_mode=attn ("lobf_attn"), lobf_per_token
  g3 rank       : lobf_evr (tau = 0.35 via pca_rank=35)

Bars are annotated with the delta vs the L-OBF reference. Output:
    figures/obf_design_ablation.png
"""

import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_config import DATASET_LABELS, apply_plot_style, canonicalize_records
from summary_utils import filter_summary_records, load_summary_records
from utils.wandb_summary import update_project_summary

SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_DIR = SCRIPT_DIR.parent / "results" / "wandb_summary"
SAVE_PATH_COMPONENTS = SCRIPT_DIR.parent / "figures" / "obf_design_ablation_components.png"
SAVE_PATH_ALTERNATIVES = SCRIPT_DIR.parent / "figures" / "obf_design_ablation_alternatives.png"

PROJECT = "obf_ablation"
MODEL = "Qwen/Qwen3-4B"
# Average over every campaign seed that exists; partial coverage is
# reported in the console so cells are never read as equal evidence.
SEEDS = [4, 44, 444]
LATENT_STEPS = 40
REF_RANK = 4
DATASETS = ["gsm8k", "arc_challenge", "medqa", "mbppplus", "humanevalplus"]

# Two groups: components (remove/simplify a step) and alternatives (replace a
# step with a different implementation). Each figure includes the "ref" bar.
REF_VARIANT = ("ref", f"L-OBF (r={REF_RANK}, ref)", "#0072B2")
COMPONENT_VARIANTS = [
    REF_VARIANT,
    ("lobf_no_proj",   "No Projection",              "#8C564B"),
    ("lobf_max_p",     "Max-P",                      "#E69F00"),
    ("lobf_no_scale",  "No Scaling",                 "#CC79A7"),
    ("lobf_naive",     "Naive Aggregation",          "#9C9C9C"),
]
ALTERNATIVE_VARIANTS = [
    REF_VARIANT,
    ("lobf_attn",      "Attn-inject",                "#56B4E9"),
    ("lobf_per_token", "Per-token",                  "#8FBBD9"),
    ("lobf_evr",       r"EVR-adaptive ($\tau$=0.35)", "#009E73"),
]
# Union used for the console table and shared y-axis reasoning.
VARIANTS = COMPONENT_VARIANTS + [v for v in ALTERNATIVE_VARIANTS if v is not REF_VARIANT]


def _is_a100_run(record):
    blob = ""
    for key in ("metadata.gpu_nvidia", "metadata.gpu"):
        v = record.get(key)
        if v:
            blob += str(v)
    return "A100" in blob


def variant_of(record):
    comp = record.get("compressor")
    if comp == "lobf":
        # in this project plain lobf rows are the attn-injection ablation
        return "lobf_attn"
    return comp


def load_project_records():
    try:
        res = update_project_summary(PROJECT, states=("finished",))
        csv_path = res["output_path"]
    except Exception as exc:
        csv_path = SUMMARY_DIR / f"{PROJECT}.csv"
        print(f"[warn] wandb fetch failed ({exc}); using local {csv_path}")
        if not Path(csv_path).exists():
            return []
    records = load_summary_records(csv_paths=[csv_path])
    filtered = filter_summary_records(
        records,
        states=["finished"],
        seeds=SEEDS,
        predicate=lambda r: (
            r.get("config.model_name") == MODEL
            and r.get("config.latent_steps") == LATENT_STEPS
            and _is_a100_run(r)
        ),
    )
    return canonicalize_records(filtered)


def load_reference():
    """L-OBF @ REF_RANK, uniform, A100, same seed — from the existing projects."""
    # main_experiment only: pca_rank_sweep holds the same L-OBF r=4 runs and would
    # report a three-seed reference cell as six.
    csvs = [SUMMARY_DIR / "main_experiment.csv"]
    records = load_summary_records(csv_paths=[p for p in csvs if p.exists()])
    filtered = filter_summary_records(
        records,
        compressors=["lobf"],
        p_values=[REF_RANK],
        seeds=SEEDS,
        states=["finished"],
        predicate=lambda r: (
            r.get("config.model_name") == MODEL
            and r.get("config.latent_steps") == LATENT_STEPS
            and _is_a100_run(r)
            and (r.get("config.inject_mode") in (None, "", "uniform"))
        ),
    )
    acc = defaultdict(list)
    seen = defaultdict(set)
    for r in canonicalize_records(filtered):
        if r.get("dataset") in DATASETS and r.get("accuracy") is not None:
            acc[r["dataset"]].append(float(r["accuracy"]) * 100)
            seen[r["dataset"]].add(str(r.get("seed")))
    # Second return value counts DISTINCT seeds, not rows: the reference is read
    # from two projects, so the same run can appear twice and a three-seed cell
    # would otherwise report six.
    return ({ds: sum(v) / len(v) for ds, v in acc.items()},
            {ds: len(seen[ds]) for ds in acc})


def main():
    plt.rcParams.update(apply_plot_style())

    ref, ref_n = load_reference()
    records = load_project_records()
    if not records:
        print("No ablation data yet (sweep still running?) - nothing to plot.")
        return

    acc = defaultdict(list)
    seen_seeds = defaultdict(set)
    for r in records:
        ds = r.get("dataset")
        if ds in DATASETS and r.get("accuracy") is not None:
            acc[(ds, variant_of(r))].append(float(r["accuracy"]) * 100)
            seen_seeds[(ds, variant_of(r))].add(str(r.get("seed")))
    cell = {k: sum(v) / len(v) for k, v in acc.items()}
    cell_n = {k: len(seen_seeds[k]) for k in acc}

    datasets = [d for d in DATASETS if any((d, v) in cell for v, _, _ in VARIANTS[1:]) or d in ref]

    # Colorblind-safe palette for benchmarks (constant across the two figures).
    DATASET_COLORS = {
        "gsm8k":         "#0072B2",  # blue
        "arc_challenge": "#009E73",  # green
        "medqa":         "#D55E00",  # vermillion
        "mbppplus":      "#CC79A7",  # pink
        "humanevalplus": "#E69F00",  # amber
    }

    def render(variants, save_path):
        """One subplot per variant. Each subplot shows the delta versus the
        L-OBF reference across benchmarks (bar height = variant - L-OBF ref).
        The L-OBF reference is excluded because its delta is always zero. All
        subplots share a y-axis so a reader can compare the depth of the drops
        directly."""
        plot_variants = [(vk, lbl, c) for (vk, lbl, c) in variants if vk != "ref"]
        n = len(plot_variants)

        # Shared y-axis over all deltas across variants and benchmarks.
        all_deltas = []
        for vk, _, _ in plot_variants:
            for ds in datasets:
                v = cell.get((ds, vk))
                if v is not None and ds in ref:
                    all_deltas.append(v - ref[ds])
        if all_deltas:
            lo, hi = min(all_deltas), max(all_deltas)
            pad = max(0.5, 0.08 * max(abs(lo), abs(hi)))
            ylim = (lo - pad, hi + pad)
        else:
            ylim = None

        # All subplots in a single row.
        nrows, ncols = 1, n
        figsize = (3.4 * n, 4.2)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
        axes = np.array(axes).reshape(-1)

        n_ds = len(datasets)
        x = np.arange(n_ds + 1)  # last slot is the average
        for i, (vk, lbl, method_color) in enumerate(plot_variants):
            ax = axes[i]
            deltas = []
            for ds in datasets:
                v = cell.get((ds, vk))
                deltas.append((v - ref[ds]) if (v is not None and ds in ref) else np.nan)
            avg = float(np.nanmean(deltas)) if any(np.isfinite(d) for d in deltas) else np.nan
            deltas_with_avg = deltas + [avg]
            xticklabels = [DATASET_LABELS.get(d, d) for d in datasets] + ["Avg"]
            bars = ax.bar(x, deltas_with_avg, width=0.72, color=method_color,
                          edgecolor="white", linewidth=0.6)
            for b, d in zip(bars, deltas_with_avg):
                if np.isfinite(d):
                    va = "bottom" if d >= 0 else "top"
                    dy = 2 if d >= 0 else -2
                    ax.annotate(f"{d:+.1f}",
                                (b.get_x() + b.get_width() / 2, b.get_height()),
                                textcoords="offset points", xytext=(0, dy),
                                ha="center", va=va, fontsize=8, color="#333333")
            ax.axhline(0, color="#333333", linewidth=0.9)
            # Vertical divider between per-benchmark bars and the average bar.
            ax.axvline(n_ds - 0.5, color="#333333", linewidth=0.9, linestyle="--")
            ax.set_title(lbl, fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(xticklabels, fontsize=9, rotation=25, ha="right")
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Unused axes (none expected but be safe).
        for j in range(len(plot_variants), len(axes)):
            axes[j].axis("off")

        # Shared y-axis label on the leftmost column.
        for k in range(nrows):
            axes[k * ncols].set_ylabel("$\\Delta$ accuracy vs L-OBF (pp)", fontsize=10)

        fig.tight_layout()
        os.makedirs(save_path.parent, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved plot to: {save_path}")

    render(COMPONENT_VARIANTS, SAVE_PATH_COMPONENTS)
    render(ALTERNATIVE_VARIANTS, SAVE_PATH_ALTERNATIVES)

    # Seeds per cell. Every bar in this figure is a mean over however many seeds
    # that cell happens to have, so print the count next to the value rather than
    # letting a one-seed bar read like a three-seed one.
    print(f"\naccuracy (%) with seed count per cell   [seed pool {SEEDS}]")
    print(f"  {'variant':<20}" + "".join(f"{DATASET_LABELS.get(d, d):>16}" for d in datasets))
    for vk, label, _ in VARIANTS:
        row = f"  {label.split(chr(40))[0].strip()[:20]:<20}"
        for ds in datasets:
            v = ref.get(ds) if vk == "ref" else cell.get((ds, vk))
            n = ref_n.get(ds, 0) if vk == "ref" else cell_n.get((ds, vk), 0)
            row += f"{v:>11.2f}(n{n})" if v is not None else f"{'-':>16}"
        print(row)
    thin = sorted({f"{vk}/{ds}" for (ds, vk), n in cell_n.items() if n < len(SEEDS)})
    print(f"\n  cells below {len(SEEDS)} seeds: {len(thin)}" + (f" -> {', '.join(thin)}" if thin else ""))


if __name__ == "__main__":
    main()
