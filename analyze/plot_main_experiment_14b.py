"""Main-experiment grouped bar chart — Qwen3-14B only.

Produces figures/main_experiment_14b.png.

Same thin-wrapper pattern as plot_main_experiment_8b.py: the 14B rank spread
(lobf/hobf x {2,4,8,32}) lives in the main_experiment project itself, so no
pca_rank_sweep fetch is needed; best-rank for lobf/hobf is computed from the
main pool inside plot_one_model.

Hardware: 14B runs on H200 (seed=14, cenvalarc.gpu) and partially on A100
(seed=141414, cancelled early). For now this figure is H200-ONLY — the partial
A100 data is excluded to keep the hardware homogeneous. Flip GPU_FILTER when
the A100 campaign completes and a mixed/other protocol is decided.

All drawing/aggregation logic is reused from plot_main_experiment.py.
"""

from plot_config import apply_plot_style
from plot_main_experiment import PROJECT_NAME, plot_one_model
from summary_utils import load_summary_records
from utils.wandb_summary import update_project_summary

import matplotlib.pyplot as plt


MODEL_NAME = "Qwen/Qwen3-14B"
SAVE_NAME = "main_experiment_14b.png"


def _is_h200_run(record):
    blob = ""
    for key in ("metadata.gpu_nvidia", "metadata.gpu"):
        v = record.get(key)
        if v:
            blob += str(v)
    return "H200" in blob


def main():
    plt.rcParams.update(apply_plot_style())

    update_result = update_project_summary(PROJECT_NAME, states=("finished",))
    records = load_summary_records(csv_paths=[update_result["output_path"]])

    print(f"Project:                {PROJECT_NAME}")
    print(f"Project CSV:            {update_result['output_path']}")
    print(f"Fetched runs:           {update_result['fetched_records']}")
    print(f"Loaded summary records: {len(records)}")
    print("Hardware filter:        H200 only (A100 seed=141414 partial data excluded)")

    try:
        stats = plot_one_model(records, MODEL_NAME, SAVE_NAME,
                               sweep_records=None, gpu_filter=_is_h200_run)
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
