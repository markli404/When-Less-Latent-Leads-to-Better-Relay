"""Main-experiment grouped bar chart — Qwen3-8B only.

Produces figures/main_experiment_8b.png.

Split out from plot_main_experiment.py (which handles 4B) for speed: the 8B
rank sweep for lobf {2,4,8,32} lives in the main_experiment project itself, so
this script does NOT fetch the large pca_rank_sweep project. best-rank for
lobf/hobf is computed from main_experiment: for each (dataset, method), average
accuracy over seeds at every rank, then keep the rank with the highest mean
(handled inside plot_one_model via compute_sweep_best on the main pool).

All drawing/aggregation logic is reused from plot_main_experiment.py — edit
styling there and it applies to both figures.
"""

from plot_config import apply_plot_style
from plot_main_experiment import PROJECT_NAME, plot_one_model
from summary_utils import load_summary_records
from utils.wandb_summary import update_project_summary

import matplotlib.pyplot as plt


MODEL_NAME = "Qwen/Qwen3-8B"
SAVE_NAME = "main_experiment_8b.png"


def main():
    plt.rcParams.update(apply_plot_style())

    update_result = update_project_summary(PROJECT_NAME, states=("finished",))
    records = load_summary_records(csv_paths=[update_result["output_path"]])

    print(f"Project:                {PROJECT_NAME}")
    print(f"Project CSV:            {update_result['output_path']}")
    print(f"Fetched runs:           {update_result['fetched_records']}")
    print(f"Loaded summary records: {len(records)}")

    # sweep_records=None → best-rank pool is the main_experiment records only,
    # which for 8B contain the lobf {2,4,8,32} spread. No pca_rank_sweep fetch.
    try:
        stats = plot_one_model(records, MODEL_NAME, SAVE_NAME, sweep_records=None)
    except Exception as exc:
        print(f"[skip] {MODEL_NAME}: {exc}")
        return

    print(f"--- {MODEL_NAME} ---")
    print(f"  Matched after filter:   {stats['matched']}")
    print(f"  Aggregated cells:       {stats['aggregated']}")
    print(f"  Datasets:               {stats['datasets']}")
    print(f"  Methods:                {stats['methods']}")
    if stats.get("sweep_overrides"):
        print("  Best-rank overrides (dataset, method, best_rank, acc):")
        for ds, m, rank, acc in sorted(stats["sweep_overrides"]):
            print(f"    {ds} | {m} | rank={rank} | acc={acc:.4f}")
    print(f"  Saved plot to:          {stats['save_path']}")


if __name__ == "__main__":
    main()
