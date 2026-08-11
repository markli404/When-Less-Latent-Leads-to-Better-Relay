"""Direct memory/performance trade-off curves over the kv_budget sweep.

Accuracy against retained KV memory, against peak GPU
memory, and accuracy vs end-to-end latency under different KV budgets"),
A100-ONLY throughout (homogeneous hardware — no cross-device bf16/timing
confound). kv_budget in {4,8,16,32,64}; headwise + H-OBF @ pca_rank=2.

Data provenance: CURRENT-protocol projects only (pca_rank_sweep +
main_experiment). The legacy seed-444 budget sweep (old_data) was DISCARDED
2026-07-24: its Full runs disagreed with the current protocol by up to 26 pp
at the same seed, contaminating the Full reference and making B=8/16/64
points incomparable with the B=32 points. Until the budget sweep is re-run
under the current protocol (project name below), only B=32 exists and this
figure renders single points per method rather than curves.

Grid: rows = x-axis metric, cols = datasets (gsm8k, medqa, humanevalplus,
arc_challenge). y = accuracy (%). One line per method, points at B=4/8/16/32/64
(annotated), Full-relay reference (A100) as a star.

Row 3 is accuracy vs AVERAGE device GPU memory (GB): the mean of wandb's
system.gpu.0.memoryAllocatedBytes over each run's FULL evaluation
(fetch_gpu_memory_avg.py pulls the whole sampled series and averages it, which
is far more stable than the summary last-value). Caveat to state in the caption:
at 4B / batch=1 device memory is dominated by the ~8 GB of weights, so the
per-method KV difference (sub-GB) is small relative to the total — the memory
advantage of compression shows up in the relayed-KV footprint and as serving
concurrency (see the cost analysis / W2 argument), not in single-sample device
totals.

Output: figures/budget_tradeoff_curves.png
"""

import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from plot_config import (
    DATASET_LABELS,
    METHOD_COLORS,
    METHOD_LABELS,
    apply_plot_style,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_DIR = SCRIPT_DIR.parent / "results" / "wandb_summary"
# Current-protocol sources only. Add the fresh budget-sweep project CSV here
# once it is run (e.g. SUMMARY_DIR / "budget_sweep.csv").
CSV_SOURCES = [
    SUMMARY_DIR / "budget_sweep.csv",     # B in {4, 8, 16, 64}
    SUMMARY_DIR / "main_experiment.csv",  # B = 32, the middle point, 3 seeds
]
SAVE_PATH = SCRIPT_DIR.parent / "figures" / "budget_tradeoff_curves.png"
# Time-averaged device GPU memory over each run's full evaluation (produced by
# fetch_gpu_memory_avg.py). Maps run_id -> mean bytes. Falls back to the summary
# last-value column if this file is absent.
GPU_AVG_CSV = SCRIPT_DIR.parent / "results" / "wandb_summary" / "gpu_memory_avg.csv"


def load_gpu_avg():
    if not GPU_AVG_CSV.exists():
        return None
    out = {}
    with open(GPU_AVG_CSV) as f:
        for r in csv.DictReader(f):
            try:
                out[r["run_id"]] = float(r["mean_gpu_bytes"])
            except (TypeError, ValueError, KeyError):
                continue
    return out

DATASETS = ["gsm8k", "medqa", "humanevalplus", "arc_challenge"]
METHODS = ["headwise", "hobf"]        # the two lines drawn per dataset
# The sweep runs hobf_fast, not the exact class: the two agree to 7e-5 on the
# operator and the fast one is what anybody would deploy. Both names are
# accepted and collapse onto one "hobf" line below.
ACCEPTED = {"headwise", "hobf", "hobf_fast"}
# 4 and 8 are where the curve moves. The prompt being compressed is 524-1012
# tokens, so B=32 already keeps 3-6% of it, and 16 through 64 measured flat.
# 64 is kept as the point showing that more budget does not close the gap.
BUDGETS = [4, 8, 16, 32, 64]
OBF_RANK = "2"          # the paper rank, so the new points match the B=32 ones
MODEL = "Qwen/Qwen3-4B"
SEEDS = {"4", "44", "444"}   # budget_sweep campaign seeds; cells average whatever exists.

# Peak relayed-KV memory is ANALYTIC, not read from the noisy `peak_overhead`
# log (whose K+V/scale convention differs across projects). For Qwen3-4B the
# per-relayed-token KV size is exact:
#   layers x kv_heads x head_dim x 2(K,V) x 2 bytes(bf16)
#   = 36 x 8 x 128 x 2 x 2 = 147,456 bytes = 144 KiB / token.
# A budget-B eviction method holds exactly B tokens; Full holds its measured
# relayed length L (avg_communication_MB already encodes L, so we derive Full's
# peak from the same source for consistency).
KV_BYTES_PER_TOKEN = 36 * 8 * 128 * 2 * 2       # = 147456 bytes
KV_MB_PER_TOKEN = KV_BYTES_PER_TOKEN / (1024 ** 2)   # ~0.1406 MiB/token


def peak_kv_mb_budget(budget):
    """Deterministic peak relayed-KV footprint (MB) for a budget-B method."""
    return budget * KV_MB_PER_TOKEN


def is_a100(row):
    blob = (row.get("metadata.gpu_nvidia", "") or "") + (row.get("metadata.gpu", "") or "")
    return "A100" in blob

LINE_WIDTH = 1.8
MARKER_SIZE = 5


def fnum(row, col):
    try:
        return float(row.get(col, ""))
    except (TypeError, ValueError):
        return None


def load_rows():
    """Merge all A100 finished rows across the source projects (dedup by run id)."""
    rows, seen = [], set()
    for path in CSV_SOURCES:
        if not path.exists():
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                if r.get("state") != "finished" or not is_a100(r):
                    continue
                rid = r.get("id") or r.get("name")
                if rid in seen:
                    continue
                seen.add(rid)
                rows.append(r)
    return rows




def collect(rows, gpu_avg=None):
    """(dataset, method, budget) -> averaged {acc, comm, tps, peak} over dup runs."""
    grouped = defaultdict(list)
    for r in rows:
        if r.get("config.model_name") != MODEL:
            continue
        comp = r.get("config.compressor")
        ds = r.get("config.task")
        if ds not in DATASETS or comp not in ACCEPTED:
            continue
        try:
            budget = int(r.get("config.kv_budget", ""))
        except (TypeError, ValueError):
            continue
        if budget not in BUDGETS:
            continue
        if r.get("config.seed") not in SEEDS:
            continue
        # keep the appendix protocol: OBF rows at the swept rank only
        if comp.startswith("hobf") and r.get("config.pca_rank") != OBF_RANK:
            continue
        # inject_mode is part of a cell's identity, not a tag: the attention-
        # weighted ablation logs under the same compressor and rank as the
        # baseline, so without this it lands inside the curve.
        if (r.get("config.inject_mode") or "uniform") != "uniform":
            continue
        # The exact and fast classes are the same operator; collapse them so a
        # dataset's points form one line instead of two half-empty ones.
        if comp.startswith("hobf"):
            comp = "hobf"
        grouped[(ds, comp, budget)].append(r)

    # One run per (cell, seed): if a configuration was run more than once, keep
    # the LATEST by summary._timestamp and drop the rest. Averaging duplicates
    # would silently mix a superseded run into the number, which is exactly what
    # a re-run is meant to replace. Distinct seeds are then averaged, since a
    # cell is a seed mean by definition.
    out = {}
    for key, rs in grouped.items():
        newest = {}
        for r in rs:
            seed = r.get("config.seed")
            ts = fnum(r, "summary._timestamp") or 0.0
            if seed not in newest or ts > newest[seed][0]:
                newest[seed] = (ts, r)
        kept = [r for _, r in newest.values()]
        dropped = len(rs) - len(kept)
        if dropped:
            print(f"  {key}: kept {len(kept)} seed(s), dropped {dropped} superseded run(s)")
        out[key] = _agg(kept, gpu_avg)
        out[key]["n_seeds"] = len(kept)
    return out


# Averaged summary columns. "peak" is the REAL device GPU memory (GB): the
# whole-evaluation time-average from gpu_memory_avg.csv when present, else the
# summary last-value column as a fallback.
METRIC_COLS = [("acc", "summary.accuracy"),
               ("comm", "summary.avg_communication_MB"),
               ("tps", "summary.time_per_sample_sec"),
               ("peak_bytes", "system.system.gpu.0.memoryAllocatedBytes")]


def _agg(rs, gpu_avg=None):
    rec = {}
    for name, col in METRIC_COLS:
        vals = [v for v in (fnum(r, col) for r in rs) if v is not None]
        rec[name] = sum(vals) / len(vals) if vals else None
    # Prefer the time-averaged device memory, which is the whole-evaluation mean
    # per run, and fall back to the summary last-value column PER RUN rather than
    # per file. gpu_memory_avg.csv is fetched separately and always lags the
    # newest runs, so an all-or-nothing fallback silently blanks the memory panel
    # for exactly the runs one is waiting on.
    gv, n_fallback = [], 0
    for r in rs:
        if gpu_avg is not None and r.get("id") in gpu_avg:
            gv.append(gpu_avg[r["id"]])
        else:
            v = fnum(r, "system.system.gpu.0.memoryAllocatedBytes")
            if v is not None:
                gv.append(v)
                n_fallback += 1
    rec["peak"] = (sum(gv) / len(gv)) / 1e9 if gv else None
    rec["peak_is_partial_fallback"] = n_fallback
    rec["n"] = len(rs)
    return rec


def collect_full(rows, gpu_avg=None):
    """Full-relay reference per dataset (same protocol: latent_steps=40, seed)."""
    grouped = defaultdict(list)
    for r in rows:
        if r.get("config.model_name") != MODEL or r.get("config.compressor") != "full":
            continue
        if r.get("config.latent_steps") not in ("40", ""):
            continue
        if r.get("config.seed") not in SEEDS:
            continue
        ds = r.get("config.task")
        if ds in DATASETS:
            grouped[ds].append(r)
    return {ds: _agg(rs, gpu_avg) for ds, rs in grouped.items()}


def draw_panel(ax, data, full_ref, ds, xkey, xlabel, logx=False):
    drew = False
    for m in METHODS:
        pts = []
        for b in BUDGETS:
            rec = data.get((ds, m, b))
            if rec and rec.get(xkey) is not None and rec.get("acc") is not None:
                pts.append((rec[xkey], rec["acc"] * 100, b))
        if not pts:
            continue
        pts.sort()
        xs, ys, bs = zip(*pts)
        ax.plot(xs, ys, marker="o", markersize=MARKER_SIZE, linewidth=LINE_WIDTH,
                color=METHOD_COLORS.get(m, "#666666"), label=METHOD_LABELS.get(m, m), alpha=0.95)
        for x, y, b in pts:
            ax.annotate(f"B={b}", (x, y), textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=6, color=METHOD_COLORS.get(m, "#666666"))
        drew = True
    ref = full_ref.get(ds)
    if ref and ref.get(xkey) is not None and ref.get("acc") is not None:
        ax.plot([ref[xkey]], [ref["acc"] * 100], marker="*", markersize=13,
                color=METHOD_COLORS.get("full", "#333333"), linestyle="none",
                label=METHOD_LABELS.get("full", "Full"), zorder=5)
        ax.axhline(ref["acc"] * 100, color="#999999", linestyle=":", linewidth=0.9, alpha=0.7)
    if logx:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.grid(color="#E9E9E9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return drew


def main():
    plt.rcParams.update(apply_plot_style())
    rows = load_rows()
    gpu_avg = load_gpu_avg()
    if gpu_avg is None:
        print('[note] gpu_memory_avg.csv missing — run fetch_gpu_memory_avg.py for whole-eval average; using summary last-value for now.')
    data = collect(rows, gpu_avg)
    full_ref = collect_full(rows, gpu_avg)

    axes_spec = [
        ("comm", "Relayed KV per sample (MB)", True),
        ("tps", "End-to-end latency (s/sample)", False),
        ("peak", "Average GPU memory (GB)", False),
    ]

    n_rows, n_cols = len(axes_spec), len(DATASETS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.9 * n_rows), squeeze=False)

    for j, ds in enumerate(DATASETS):
        axes[0][j].set_title(DATASET_LABELS.get(ds, ds), fontsize=11, fontweight="bold")
        for i, (xkey, xlabel, logx) in enumerate(axes_spec):
            draw_panel(axes[i][j], data, full_ref, ds, xkey, xlabel, logx=logx)
            if j == 0:
                axes[i][j].set_ylabel("Accuracy (%)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    seen, hl = set(), []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            hl.append((h, l))
    fig.legend([h for h, _ in hl], [l for _, l in hl], loc="lower center",
               ncol=len(hl), frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(bottom=0.13 if n_rows > 2 else 0.17, top=0.94,
                        left=0.06, right=0.99, wspace=0.22, hspace=0.42)

    os.makedirs(SAVE_PATH.parent, exist_ok=True)
    fig.savefig(SAVE_PATH, dpi=300)
    plt.close(fig)
    print(f"Saved plot to: {SAVE_PATH}")
    n_cells = sum(1 for k in data if data[k].get('acc') is not None)
    print(f"cells plotted (A100-only): {n_cells}/{len(DATASETS)*len(METHODS)*len(BUDGETS)}")
    # per-cell A100 run counts, so single-run budgets are explicit for the caption
    print("A100 runs per cell (dataset x method x budget):")
    for ds in DATASETS:
        for m in METHODS:
            ns = [str(data.get((ds, m, b), {}).get("n", 0)) for b in BUDGETS]
            print(f"  {ds:14s} {m:10s} B{'/'.join(map(str,BUDGETS))} = {'/'.join(ns)}")


if __name__ == "__main__":
    main()
