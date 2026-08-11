"""What the fast path actually bought, and what it may have cost.

Three panels, one figure, per selector family:

  1. Compression time per sample, exact against fast, one bar pair per benchmark.
     Two fast bars are drawn: the wall clock the pipeline records, and the same
     call with the analysis blocks excluded. The exact classes still compute the
     paper's figures inside their wall clock, so only the second is a like-for-
     like comparison; the first is what a reader of the paper's column sees.

  2. Speedup per benchmark, both ratios, with the mean as a line. This is where
     you see whether the ratio measured on one benchmark transfers to the rest.

  3. Accuracy delta, fast minus exact, against the seed noise of the exact path.
     This panel exists because numerical agreement is not accuracy agreement:
     the injected vector matches the exact one to ~1e-4, but sampling is
     stochastic, so a perturbation that small can still flip a token and
     cascade. The shaded band is +/- 1 SD of the exact path across its own
     seeds, computed per benchmark. A fast bar inside the band is noise; a
     column of bars all on one side of zero is not, however small each one is.

Usage:
    python analyze/plot_fast_speedup.py
    python analyze/plot_fast_speedup.py --family lobf --out figures/
    python analyze/plot_fast_speedup.py --refresh        # re-pull wandb first

The fast runs are matched to the exact runs on (task, seed) at the paper's
fixed configuration: L-OBF r=4 against lobf_fast r=4, H-OBF r=2 against
hobf_fast r=2. Cells where either side is missing are skipped and reported.
"""

import argparse
import csv
import itertools
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plot_config import DATASET_LABELS, DATASET_ORDER, apply_plot_style

_ROWS_CACHE = {}

REPO = Path(__file__).resolve().parent.parent
SUMMARY_DIR = REPO / "results" / "wandb_summary"

# The exact path and the fast path that replaces it, at the paper's fixed rank.
FAMILIES = {
    "lobf": {
        "exact": ("lobf", "4"),
        "fast": ("lobf_fast", "4"),
        "label": "L-OBF",
    },
    "hobf": {
        "exact": ("hobf", "2"),
        "fast": ("hobf_fast", "2"),
        "label": "H-OBF",
    },
}

# Runs on the retired cards are excluded everywhere: mixing them puts a
# placement difference inside a timing ratio.
RETIRED_GPUS = {
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
}

FIELDS = {
    "acc": "summary.accuracy",
    "total": "summary.avg_compression_time_s",
    "core": "summary.avg_compression_core_s",
    "e2e": "summary.time_per_sample_sec",
}


def load(model="Qwen/Qwen3-4B", gpu="NVIDIA A100-PCIE-40GB"):
    """One record per run, deduplicated by run id across every summary CSV.

    The same run appears in more than one export, and counting it twice shifts
    a cell mean by a few hundredths. Dedup by id, not by cell key.
    """
    rows, seen = [], set()
    for path in sorted(SUMMARY_DIR.glob("*.csv")):
        if path.name.endswith(".bak") or ".bak-" in path.name:
            continue
        try:
            with open(path) as fh:
                for r in csv.DictReader(fh):
                    rid = r.get("id") or r.get("name")
                    if not rid or rid in seen:
                        continue
                    if r.get("config.model_name") != model:
                        continue
                    g = r.get("metadata.gpu") or ""
                    if g in RETIRED_GPUS:
                        continue
                    # An empty gpu field is a logging gap, not another card:
                    # those runs belong to the campaigns they were launched in.
                    if g and gpu and g != gpu:
                        continue
                    seen.add(rid)
                    rows.append(r)
        except (OSError, csv.Error):
            continue
    return rows


def _stamp(r):
    try:
        return float(r.get("summary._timestamp"))
    except (TypeError, ValueError):
        return 0.0


def cells(rows, compressors, rank, near=None):
    """(task, seed) -> field dict, for one compressor family at one rank.

    A cell can hold several runs. Two launched in the same campaign reproduce
    each other bit for bit, so duplicates there are harmless. Runs launched
    weeks apart do NOT: on this project the same (benchmark, method, rank, seed,
    A100) drifts by a median of about 0.9 pp between campaigns, which is the
    same size as the cross-GPU effect. So when comparing a fast run against its
    exact counterpart, take the exact run closest in time to it. Comparing a
    July fast run against a June exact run measures the drift, not the operator.
    """
    names = (compressors,) if isinstance(compressors, str) else tuple(compressors)
    out, chosen = {}, {}
    for r in rows:
        if r.get("config.compressor") not in names:
            continue
        if rank is not None and r.get("config.pca_rank") != rank:
            continue
        # The attention-weighted injection ablation logs as compressor "lobf" at
        # rank 4, which is exactly the exact-path key. Without this line those
        # runs land in the L-OBF baseline and the fast-vs-exact delta comes out
        # 1.1 pp too kind. inject_mode is part of a cell's identity, not a tag.
        if (r.get("config.inject_mode") or "uniform") != "uniform":
            continue
        key = (r["config.task"], r.get("config.seed"))
        t = _stamp(r)
        if key in chosen:
            if near is None:
                # No reference time: keep the most recent.
                if t <= chosen[key]:
                    continue
            else:
                ref = near.get(key)
                if ref is None:
                    if t <= chosen[key]:
                        continue
                elif abs(t - ref) >= abs(chosen[key] - ref):
                    continue
        rec = {}
        for k, col in FIELDS.items():
            v = r.get(col)
            if v not in (None, ""):
                try:
                    rec[k] = float(v)
                except ValueError:
                    pass
        if "acc" in rec:
            rec["acc"] *= 100.0
        rec["_t"] = t
        out[key] = rec
        chosen[key] = t
    return out


def seed_sd(exact, task):
    """SD of the exact path across its own seeds on this benchmark.

    This is the bar every accuracy delta has to clear. Fewer than two seeds
    means we cannot say anything, so return None rather than zero.
    """
    vals = [v["acc"] for (t, _), v in exact.items() if t == task and "acc" in v]
    return st.stdev(vals) if len(vals) >= 2 else None


def collect(rows, family):
    spec = FAMILIES[family]
    fast = cells(rows, spec["fast"][0], spec["fast"][1])
    # Pick, for each cell, the exact run nearest in time to the fast run.
    exact = cells(rows, spec["exact"][0], spec["exact"][1],
                  near={k: v["_t"] for k, v in fast.items()})

    paired, missing = [], []
    tasks = [t for t in DATASET_ORDER if any(k[0] == t for k in fast)]
    for task in tasks:
        seeds = sorted({s for (t, s) in fast if t == task})
        for seed in seeds:
            f = fast.get((task, seed), {})
            e = exact.get((task, seed), {})
            if not f.get("total") or not e.get("total"):
                missing.append((task, seed, "no matched exact run" if not e else "no fast timing"))
                continue
            paired.append(
                {
                    "task": task,
                    "seed": seed,
                    "e_total": e["total"],
                    "e_acc": e.get("acc"),
                    "e_e2e": e.get("e2e"),
                    "f_total": f["total"],
                    # With the analysis code removed the class reports one
                    # number; older runs report both and the gap is the analysis.
                    "f_core": f.get("core", f["total"]),
                    "f_acc": f.get("acc"),
                    "f_e2e": f.get("e2e"),
                    "_ft": f.get("_t", 0.0),
                    "sd": seed_sd(exact, task),
                }
            )
    return paired, missing


def aggregate(paired):
    """Per-benchmark means over seeds. One number per bar, not three."""
    by_task = defaultdict(list)
    for p in paired:
        by_task[p["task"]].append(p)

    def _mean(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return st.mean(vals) if vals else None

    out = []
    for t in [t for t in DATASET_ORDER if t in by_task]:
        rows = by_task[t]
        out.append({
            "task": t,
            "n": len(rows),
            # Only the time the pipeline reports. The classes no longer compute
            # the paper's diagnostics inside the call, so the reported number
            # IS the deployable one and the old core column is redundant.
            "e_total": _mean(rows, "e_total"),
            "f_total": _mean(rows, "f_total"),
            "e_acc": _mean(rows, "e_acc"),
            "f_acc": _mean(rows, "f_acc"),
            # SD of the exact path across its OWN seeds on this benchmark. It
            # is already per-task, so take it rather than average it.
            "sd": rows[0].get("sd"),
        })
    return out


EXACT_COLOR = "#444444"
FAST_COLOR = "#0072B2"
WORSE_COLOR = "#D55E00"
BETTER_COLOR = "#009E73"


def _shared_legend(fig):
    """Two entries, once. Both panels use the same pair of colours and the
    family is already in each panel title, so a per-family legend would repeat
    the same two swatches four times."""
    handles = [Patch(color=EXACT_COLOR, label="exact SVD"),
               Patch(color=FAST_COLOR, label="fast, subspace iteration")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=10.5)


def plot_time(agg_by_family, out_dir):
    """One row, one panel per family: what a compression call costs.

    The speedup is written on each pair rather than given its own panel. It is
    a ratio of the two bars directly above it, so a reader can check it against
    the picture instead of matching benchmarks across two axes.
    """
    fams = [f for f in FAMILIES if agg_by_family.get(f)]
    fig, axes = plt.subplots(1, len(fams), figsize=(6.0 * len(fams), 4.6), squeeze=False)
    w = 0.36
    for ax, fam in zip(axes[0], fams):
        agg = agg_by_family[fam]
        label = FAMILIES[fam]["label"]
        xs = np.arange(len(agg))
        ax.bar(xs - w / 2, [a["e_total"] for a in agg], w, color=EXACT_COLOR)
        ax.bar(xs + w / 2, [a["f_total"] for a in agg], w, color=FAST_COLOR)
        top = max(a["e_total"] for a in agg)
        for i, a in enumerate(agg):
            ax.text(i, a["e_total"] + top * 0.03, f"{a['e_total'] / a['f_total']:.1f}x",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                    color=FAST_COLOR)
            ax.text(i + w / 2, a["f_total"] + top * 0.01, f"{a['f_total']:.2f}",
                    ha="center", va="bottom", fontsize=8, color="#333333")
        mean_x = st.mean(a["e_total"] / a["f_total"] for a in agg)
        ax.set_title(f"{label}: {st.mean(a['f_total'] for a in agg):.2f} s per call, "
                     f"{mean_x:.1f}x faster on average")
        ax.set_ylabel("compression, s per sample")
        ax.set_ylim(0, top * 1.22)
        ax.set_xticks(xs)
        ax.set_xticklabels([DATASET_LABELS.get(a["task"], a["task"]) for a in agg],
                           rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _shared_legend(fig)
    path = Path(out_dir) / "fast_compression_time.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_accuracy(agg_by_family, out_dir):
    """Fast minus exact, against the noise the exact path makes on its own.

    Absolute accuracy was tried and does not work: the scores run from 55 to 90
    percent, so the two bars are the same height and the whole effect lives in
    a text label. The difference is the quantity, and the grey band is what
    stops it from being read as larger than it is. A bar inside the band moved
    less than the exact path moves between its own seeds.

    Why a band and not an error bar: the injected vector matches the exact one
    to 7e-5, which cannot change an answer directly. What it can do at
    temperature 0.6 is flip one sampled token, after which the continuation
    forks. So the fast run is a fresh draw rather than a nudged one, and seed
    spread is the right yardstick.
    """
    fams = [f for f in FAMILIES if agg_by_family.get(f)]
    fig, axes = plt.subplots(1, len(fams), figsize=(6.0 * len(fams), 4.6), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        agg = [a for a in agg_by_family[fam]
               if a["e_acc"] is not None and a["f_acc"] is not None]
        if not agg:
            ax.set_title(f"{FAMILIES[fam]['label']}: no matched accuracy pairs")
            continue
        label = FAMILIES[fam]["label"]
        xs = np.arange(len(agg))
        d = [a["f_acc"] - a["e_acc"] for a in agg]

        for i, a in enumerate(agg):
            if a["sd"]:
                ax.add_patch(plt.Rectangle((i - 0.46, -a["sd"]), 0.92, 2 * a["sd"],
                                           color="#999999", alpha=0.28, lw=0, zorder=1))
        ax.bar(xs, d, 0.62, zorder=3,
               color=[WORSE_COLOR if x < 0 else BETTER_COLOR for x in d])
        ax.axhline(0, color="black", lw=0.9, zorder=2)
        m = st.mean(d)
        ax.axhline(m, color=WORSE_COLOR if m < 0 else BETTER_COLOR, ls="--", lw=1.2,
                   zorder=2)

        span = max(max(abs(x) for x in d),
                   max((a["sd"] or 0) for a in agg)) * 1.35 + 0.3
        for i, x in enumerate(d):
            ax.text(i, x + (0.06 * span if x >= 0 else -0.06 * span), f"{x:+.2f}",
                    ha="center", va="bottom" if x >= 0 else "top", fontsize=9)
        inside = sum(1 for a, x in zip(agg, d) if a["sd"] and abs(x) <= a["sd"])
        ax.set_title(f"{label}: {m:+.2f} pp on average, "
                     f"{inside} of {len(d)} inside the seed noise")
        ax.set_ylabel("fast minus exact (pp)")
        ax.set_ylim(-span, span)
        ax.set_xticks(xs)
        ax.set_xticklabels([DATASET_LABELS.get(a["task"], a["task"]) for a in agg],
                           rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    handles = [Patch(color=BETTER_COLOR, label="fast scores higher"),
               Patch(color=WORSE_COLOR, label="fast scores lower"),
               Patch(color="#999999", alpha=0.28,
                     label="seed noise of the exact path (+/- 1 SD)")]
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=10.5)
    path = Path(out_dir) / "fast_accuracy.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def seed_pair_control(paired, family):
    """The exact path against itself, one seed versus another.

    A 1e-4 difference in the injected vector does not perturb an answer, it
    occasionally flips a sampled token and the continuation diverges from there.
    So the fast run is a fresh draw, not a nudged one, and the question is
    whether it lands further from the exact run than two seeds of the exact run
    land from each other. Same benchmarks, same one-observation-per-cell.
    """
    spec = FAMILIES[family]
    rows = _ROWS_CACHE.get("rows")
    if rows is None:
        return []
    exact = cells(rows, spec["exact"][0], spec["exact"][1],
                  near={(p["task"], p["seed"]): p["_ft"] for p in paired})
    tasks = [p["task"] for p in paired]
    seeds = sorted({s for (t, s) in exact if t in tasks})
    out = []
    for a, b in itertools.combinations(seeds, 2):
        d = [exact[(t, a)]["acc"] - exact[(t, b)]["acc"]
             for t in tasks
             if (t, a) in exact and (t, b) in exact
             and "acc" in exact[(t, a)] and "acc" in exact[(t, b)]]
        if len(d) == len(tasks):
            out.append((st.mean(d), sum(1 for x in d if x < 0)))
    return out


def report(paired, missing, family):
    spec = FAMILIES[family]
    print(f"\n=== {spec['label']} : fast against exact, matched on (task, seed) ===")
    if not paired:
        print("  no matched pairs yet")
        for t, s, why in missing:
            print(f"    skipped {t} seed={s}: {why}")
        return
    hdr = (f"{'benchmark':<16}{'seed':>5}{'exact s':>9}{'fast s':>8}{'core s':>8}"
           f"{'x rep':>7}{'x core':>8}{'acc d':>8}{'sd':>6}")
    print(hdr)
    print("-" * len(hdr))
    for p in paired:
        d = (p["f_acc"] - p["e_acc"]) if (p["f_acc"] is not None and p["e_acc"] is not None) else None
        print(f"{DATASET_LABELS.get(p['task'], p['task']):<16}{p['seed']:>5}"
              f"{p['e_total']:9.2f}{p['f_total']:8.2f}{p['f_core']:8.2f}"
              f"{p['e_total']/p['f_total']:7.2f}{p['e_total']/p['f_core']:8.2f}"
              f"{'' if d is None else format(d, '+8.2f')}"
              f"{'' if not p['sd'] else format(p['sd'], '6.2f')}")
    core = [p["e_total"] / p["f_core"] for p in paired]
    print(f"\n  deployable compression {st.mean(p['f_core'] for p in paired):.2f} s, "
          f"speedup {st.mean(core):.2f}x (per benchmark {min(core):.2f} to {max(core):.2f})")

    seed_ctrl = seed_pair_control(paired, family)
    if seed_ctrl:
        mags = sorted(abs(m) for m, _ in seed_ctrl)
        print(f"\n  Control: the exact path against ITSELF, one seed versus another, on the same")
        print(f"  benchmarks and with the same single observation per cell. This is the honest")
        print(f"  yardstick, because a 1e-4 perturbation of the injected vector does not change")
        print(f"  the answer, it occasionally flips a sampled token and the continuation forks.")
        for m, neg in seed_ctrl:
            print(f"    seed pair: mean {m:+6.2f} pp, {neg} of {len(paired)} below zero")
        print(f"    |mean| across seed pairs: {mags[0]:.2f} to {mags[-1]:.2f} pp")

    d = [p["f_acc"] - p["e_acc"] for p in paired if p["f_acc"] is not None and p["e_acc"] is not None]
    if len(d) >= 2:
        sds = [p["sd"] for p in paired if p["sd"]]
        floor = st.mean(sds) if sds else None
        m = st.mean(d)
        line = f"  accuracy {m:+.2f} pp, {sum(1 for x in d if x < 0)} of {len(d)} below zero"
        if floor:
            # Two independent single-seed draws differ with sd*sqrt(2); the mean
            # of n such differences with sd*sqrt(2/n).
            se = floor * (2 / len(d)) ** 0.5
            line += f", against a seed noise floor of {floor:.2f} pp (|t| = {abs(m)/se:.2f})"
        print(line)
        if seed_ctrl:
            mags = [abs(mm) for mm, _ in seed_ctrl]
            beaten = sum(1 for mm in mags if mm >= abs(m))
            print(f"  {beaten} of {len(mags)} seed pairs move at least as much as the fast path does.")
            if beaten == 0:
                print("  NOTE: the fast path moved further than the exact path moves between its own")
                print("        seeds. Sampling does not explain this on its own. Check the operator")
                print("        (check_gram_equivalence.py --alt fast) before calling it noise.")
            else:
                print("  This is inside the range the exact path covers on its own, i.e. sampling.")
    for t, s, why in missing:
        print(f"  skipped {t} seed={s}: {why}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=list(FAMILIES) + ["both"], default="both")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--gpu", default="NVIDIA A100-PCIE-40GB",
                    help="empty string to allow any card, which mixes placement into the ratio")
    ap.add_argument("--out", default=str(REPO / "figures"))
    ap.add_argument("--refresh", action="store_true",
                    help="re-pull the wandb summaries before plotting")
    args = ap.parse_args()

    if args.refresh:
        try:
            from utils.wandb_summary import update_project_summary
            for proj in ("fast_path", "main_experiment"):
                update_project_summary(proj)
        except Exception as exc:  # the plot is still worth making from cache
            print(f"  refresh failed ({exc}); using the cached summaries")

    apply_plot_style()
    rows = load(args.model, args.gpu)
    _ROWS_CACHE['rows'] = rows
    fams = list(FAMILIES) if args.family == "both" else [args.family]
    agg_by_family = {}
    for fam in fams:
        paired, missing = collect(rows, fam)
        report(paired, missing, fam)
        if paired:
            agg_by_family[fam] = aggregate(paired)
    if agg_by_family:
        print(f"\n  wrote {plot_time(agg_by_family, args.out)}")
        print(f"  wrote {plot_accuracy(agg_by_family, args.out)}")


if __name__ == "__main__":
    main()
