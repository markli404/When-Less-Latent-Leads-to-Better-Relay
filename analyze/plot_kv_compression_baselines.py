"""External KV-compression baselines (wandb project: kv_baselines).

Grouped bars of accuracy per dataset comparing the new external baselines
against the free references (Full / Layerwise / L-OBF@r4 from the existing
projects, matched A100 + seed):

  references : full, layerwise, lobf (r=4, uniform)
  new        : lmerge (token merging), full_quant b8 / b4, lobf_quant b8

Every (dataset, variant) cell is averaged over whatever campaign seeds exist
(4 / 44 / 444); the console prints the per-cell seed count so partially covered
cells are visible. Deltas annotated vs Full. The console also prints
avg_communication_MB per variant (the quantized runs log TRUE n-bit payloads),
which is the cost side of this comparison. Output:
    figures/kv_compression_baselines.png
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
SAVE_PATH = SCRIPT_DIR.parent / "figures" / "kv_compression_baselines.png"

PROJECT = "kv_baselines"
MODEL = "Qwen/Qwen3-4B"
# Average over every campaign seed that exists; cells with partial seed
# coverage are flagged in the console table (n=) so they are never read as
# equal-evidence comparisons.
SEEDS = [4, 44, 444]
LATENT_STEPS = 40
REF_RANK = 4
DATASETS = ["medqa", "gsm8k", "arc_challenge", "mbppplus", "humanevalplus"]

# (variant_key, label, color); "ref:*" come from the existing projects.
VARIANTS = [
    ("ref:full",       "Full",            "#4C4C4C"),
    ("ref:layerwise",  "Layerwise",       "#56B4E9"),
    ("ref:lobf",       f"L-OBF (r={REF_RANK})", "#0072B2"),
    ("lmerge",         "Cache Merging",   "#E69F00"),
    ("full_quant_q8",  "Full+int8",       "#BBBBBB"),
    ("full_quant_q4",  "Full+int4",       "#8C8C8C"),
    ("lobf_quant_q8",  "L-OBF+int8",      "#009E73"),
]


def _is_a100_run(record):
    blob = ""
    for key in ("metadata.gpu_nvidia", "metadata.gpu"):
        v = record.get(key)
        if v:
            blob += str(v)
    return "A100" in blob


def qbits(record):
    v = record.get("config.quant_bits")
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def variant_of(record):
    comp = record.get("compressor")
    if comp in ("full_quant", "lobf_quant"):
        b = qbits(record)
        return f"{comp}_q{b}" if b else comp
    return comp


def base_filter(records):
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


def cell_means(records, key_fn, fields=("accuracy", "avg_communication_MB")):
    vals = defaultdict(lambda: defaultdict(list))
    seen_seeds = defaultdict(set)
    for r in records:
        ds = r.get("dataset")
        if ds not in DATASETS:
            continue
        k = key_fn(r)
        seen_seeds[(ds, k)].add(r.get("seed"))
        for f in fields:
            v = r.get(f) if f != "avg_communication_MB" else r.get("summary.avg_communication_MB")
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            vals[(ds, k)][f].append(v)
    out = {}
    for key, d in vals.items():
        out[key] = {f: (sum(v) / len(v)) for f, v in d.items() if v}
        out[key]["n_seeds"] = len(seen_seeds[key])
    return out


def main():
    plt.rcParams.update(apply_plot_style())

    # References from existing projects.
    # Reference rows come from main_experiment ONLY. pca_rank_sweep carries the
    # same L-OBF r=4 runs, so including it double-counted seeds without changing
    # any value; the paper figures quote main_experiment provenance.
    ref_csvs = [p for p in (SUMMARY_DIR / "main_experiment.csv",) if p.exists()]
    ref_records = base_filter(load_summary_records(csv_paths=ref_csvs))
    ref_records = [
        r for r in ref_records
        if (r.get("compressor") in ("full", "layerwise"))
        or (r.get("compressor") == "lobf"
            and r.get("p_value") == REF_RANK
            and r.get("config.inject_mode") in (None, "", "uniform"))
    ]
    ref_cells = cell_means(ref_records, lambda r: f"ref:{r.get('compressor')}")

    # The external baselines.
    try:
        res = update_project_summary(PROJECT, states=("finished",))
        csv_path = res["output_path"]
    except Exception as exc:
        csv_path = SUMMARY_DIR / f"{PROJECT}.csv"
        print(f"[warn] wandb fetch failed ({exc}); using local {csv_path}")
    new_records = base_filter(load_summary_records(csv_paths=[csv_path])) if Path(csv_path).exists() else []
    new_cells = cell_means(new_records, variant_of)
    if not new_cells:
        print("No baseline data yet (sweep still running?) - plotting references only.")

    cells = {**ref_cells, **new_cells}
    full_acc = {ds: cells.get((ds, "ref:full"), {}).get("accuracy") for ds in DATASETS}

    datasets = [d for d in DATASETS if any((d, vk) in cells for vk, _, _ in VARIANTS)]
    x = np.arange(len(datasets))
    width = 0.9 / len(VARIANTS)

    fig, ax = plt.subplots(figsize=(13.0, 5.2))
    for i, (vk, label, color) in enumerate(VARIANTS):
        vals, deltas = [], []
        for ds in datasets:
            a = cells.get((ds, vk), {}).get("accuracy")
            vals.append(a * 100 if a is not None else np.nan)
            f = full_acc.get(ds)
            deltas.append((a - f) * 100 if (a is not None and f is not None and vk != "ref:full") else np.nan)
        offset = (i - (len(VARIANTS) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width, label=label, color=color,
                      edgecolor="white", linewidth=0.6)
        for b, d in zip(bars, deltas):
            if np.isfinite(b.get_height()) and np.isfinite(d):
                ax.annotate(f"{d:+.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                            textcoords="offset points", xytext=(0, 2), ha="center",
                            fontsize=6.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    finite = [v.get("accuracy") * 100 for v in cells.values() if v.get("accuracy") is not None]
    if finite:
        ax.set_ylim(max(0, min(finite) - 6), min(100, max(finite) + 4))
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.2, top=0.95)

    os.makedirs(SAVE_PATH.parent, exist_ok=True)
    fig.savefig(SAVE_PATH, dpi=300)
    plt.close(fig)
    print(f"Saved plot to: {SAVE_PATH}")

    # Seed coverage: which cells rest on how many seeds (uneven coverage must be
    # visible before any of these numbers is quoted).
    print("\naccuracy (%) with seed count per cell:")
    header = "  " + " " * 18 + "".join(f"{DATASET_LABELS.get(d, d):>16s}" for d in datasets)
    print(header)
    for vk, label, _ in VARIANTS:
        row = f"  {label:18s}"
        for ds in datasets:
            c = cells.get((ds, vk))
            if c and c.get("accuracy") is not None:
                row += f"{c['accuracy'] * 100:10.2f}(n{c.get('n_seeds', 0)})"
            else:
                row += f"{'-':>16s}"
        print(row)

    # Cost side: mean communication per variant (true n-bit for quant runs).
    print("\navg_communication_MB (mean over datasets present):")
    for vk, label, _ in VARIANTS:
        cs = [cells[(ds, vk)].get("avg_communication_MB") for ds in datasets
              if (ds, vk) in cells and cells[(ds, vk)].get("avg_communication_MB") is not None]
        if cs:
            print(f"  {label:16s}: {sum(cs)/len(cs):8.1f} MB  (n={len(cs)} datasets)")


if __name__ == "__main__":
    main()
