#!/usr/bin/env python3
import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from plot_config import DATASET_LABELS


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_DIR = REPO_DIR / "results" / "wandb_summary"
DEFAULT_SEEDS = (4, 44, 444)
DEFAULT_EFFICIENCY_SEED = 4
DEFAULT_OUTPUT_PATH = REPO_DIR / "results" / "qwen4b_tables.tex"
DEFAULT_MAIN_EXPERIMENT_CSV = DEFAULT_SUMMARY_DIR / "main_experiment.csv"
DEFAULT_SWEEP_CSV = DEFAULT_SUMMARY_DIR / "pca_rank_sweep.csv"

# Accuracy is reported A100-only (homogeneous hardware; avoids cross-GPU bf16
# sampling noise). For OBF variants we report a single fixed pca_rank per
# method (default: L-OBF=4, H-OBF=2), avoiding test-set hyperparameter tuning.
A100_SUBSTR = "A100"
PCA_RANK_CANDIDATES = (2, 4, 8, 16, 32)
FIXED_RANK_BY_METHOD = {"L-OBF": 4, "H-OBF": 2}

DATASETS = [
    ("gsm8k", "GSM8K"),
    ("aime2024", "AIME24"),
    ("aime2025", "AIME25"),
    ("gpqa", "GPQA"),
    ("medqa", "MedQA"),
    ("arc_easy", "ARC-E"),
    ("arc_challenge", "ARC-C"),
    ("mbppplus", "MBPP+"),
    ("humanevalplus", "HumanEval+"),
]

DATASET_GROUPS = [
    ("Math", ("gsm8k", "aime2024", "aime2025")),
    ("Expert QA", ("gpqa", "medqa")),
    ("Commonsense QA", ("arc_easy", "arc_challenge")),
    ("Coding", ("mbppplus", "humanevalplus")),
]


@dataclass(frozen=True)
class MethodSpec:
    label: str
    compressors: Tuple[str, ...]


METHODS = [
    MethodSpec("Full", ("full",)),
    MethodSpec("L", ("layerwise",)),
    MethodSpec("L-OBF", ("lobf", "lobf_metric")),
    MethodSpec("H", ("headwise",)),
    MethodSpec("H-OBF", ("hobf", "hobf_metric")),
    MethodSpec("Gen", ("gonly",)),
]
RELAY_METHOD_LABELS = ("L", "L-OBF", "H", "H-OBF")


CellKey = Tuple[str, str]
SeedRows = Dict[int, Dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX accuracy table from W&B merged summary CSVs."
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=DEFAULT_SUMMARY_DIR,
        help="Directory containing LatentMAS_<seed>_merged.csv files.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="*",
        default=None,
        help="Optional explicit merged summary CSV paths. Defaults to one CSV per seed.",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--latent-steps", type=int, default=40)
    parser.add_argument("--pca-rank", type=int, default=2,
                        help="Fallback pca_rank for OBF rows when blank in CSV; best-rank selection overrides this.")
    parser.add_argument(
        "--a100-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only runs executed on A100 GPUs (matches the plot scripts). "
             "Use --no-a100-only to include all hardware.",
    )
    parser.add_argument("--kv-budget", type=int, default=32)
    parser.add_argument("--relay-rounds", type=int, default=3)
    parser.add_argument("--sink-size", type=int, default=4,
                        help="Number of sink tokens retained once across the rollout (matches --sink_size in run.py).")
    parser.add_argument("--output-folder", default="results")
    parser.add_argument("--no-table-env", action="store_true", help="Print only tabular body.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output .tex path containing both the accuracy and efficiency tables.",
    )
    parser.add_argument(
        "--efficiency-seed",
        type=int,
        default=DEFAULT_EFFICIENCY_SEED,
        help="Single seed used to report efficiency numbers (no cross-seed averaging).",
    )
    parser.add_argument(
        "--efficiency-gpu-filter",
        default="A100",
        help=(
            "Substring match against metadata.gpu_nvidia when picking the efficiency-seed run "
            "for each (method, dataset) cell. Ensures all reported timing numbers come from the "
            "same GPU architecture. Pass an empty string to disable."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Average over available seeds instead of requiring every requested seed.",
    )
    parser.add_argument("--print-table", action="store_true", help="Print the generated table to stdout.")
    parser.add_argument("--verbose", action="store_true", help="Print selected runs and missing cells.")
    parser.add_argument(
        "--strict-explicit-config",
        action="store_true",
        help="Require explicit kv_budget/pca_rank fields instead of accepting old blank defaults.",
    )
    return parser.parse_args()


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: Optional[str]) -> Optional[int]:
    number = parse_float(value)
    if number is None:
        return None
    return int(number)


def row_value(row: Dict[str, str], key: str) -> str:
    return row.get(key, "")


def blank_or_int(row: Dict[str, str], key: str, default: int, strict: bool) -> Optional[int]:
    value = row_value(row, key)
    if value == "":
        return None if strict else default
    return parse_int(value)


def default_csv_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    if DEFAULT_MAIN_EXPERIMENT_CSV.exists():
        paths.append(DEFAULT_MAIN_EXPERIMENT_CSV)
    else:
        paths.extend(args.summary_dir / f"LatentMAS_{seed}_merged.csv" for seed in args.seeds)
    # Rank-sweep CSV supplies the OBF best-rank candidates.
    if DEFAULT_SWEEP_CSV.exists():
        paths.append(DEFAULT_SWEEP_CSV)
    return paths


def read_rows(csv_paths: Sequence[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            print(f"WARNING: missing CSV: {csv_path}", file=sys.stderr)
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                copied = dict(row)
                copied["_source_csv"] = str(csv_path)
                rows.append(copied)
    return rows


def is_obf_compressor(compressor: str) -> bool:
    return compressor in {"lobf", "lobf_metric", "hobf", "hobf_metric"}


def is_a100_row(row: Dict[str, str]) -> bool:
    blob = (row_value(row, "metadata.gpu_nvidia") or "") + (row_value(row, "metadata.gpu") or "")
    return A100_SUBSTR in blob


def all_supported_compressors() -> set:
    return {compressor for method in METHODS for compressor in method.compressors}


def row_matches_base_filters(row: Dict[str, str], args: argparse.Namespace) -> bool:
    if row_value(row, "config.model_name") != args.model_name:
        return False
    allowed_seeds = set(args.seeds) | {args.efficiency_seed}
    if parse_int(row_value(row, "config.seed")) not in allowed_seeds:
        return False
    if parse_int(row_value(row, "config.latent_steps")) != args.latent_steps:
        return False
    if parse_int(row_value(row, "config.max_samples")) != -1:
        return False
    if row_value(row, "state") and row_value(row, "state") != "finished":
        return False
    if row_value(row, "config.output_folder") != args.output_folder:
        return False
    if getattr(args, "a100_only", True) and not is_a100_row(row):
        return False

    compressor = row_value(row, "config.compressor")
    if compressor not in all_supported_compressors():
        return False

    kv_budget = blank_or_int(row, "config.kv_budget", args.kv_budget, args.strict_explicit_config)
    if kv_budget != args.kv_budget:
        return False

    if is_obf_compressor(compressor):
        # Accept every swept rank; the actual rank per (method, dataset) is
        # chosen later by best-over-rank selection. (Non-OBF methods ignore
        # pca_rank entirely.)
        pca_rank = blank_or_int(row, "config.pca_rank", args.pca_rank, args.strict_explicit_config)
        if pca_rank not in PCA_RANK_CANDIDATES:
            return False

    return True


def method_exactness(method: MethodSpec, row: Dict[str, str]) -> int:
    compressor = row_value(row, "config.compressor")
    if method.label == "L-OBF":
        return 1 if compressor == "lobf" else 0
    if method.label == "H-OBF":
        return 1 if compressor == "hobf" else 0
    return 1


def selection_score(method: MethodSpec, row: Dict[str, str]) -> Tuple[int, int, float]:
    explicit_budget = 1 if row_value(row, "config.kv_budget") != "" else 0
    timestamp = parse_float(row_value(row, "summary._timestamp")) or 0.0
    return (method_exactness(method, row), explicit_budget, timestamp)


def _seed_rows_at_rank(
    filtered: Sequence[Dict[str, str]],
    method: MethodSpec,
    dataset: str,
    seed_pool: Sequence[int],
    rank: Optional[int],
    args: argparse.Namespace,
) -> SeedRows:
    """Best row per seed for a (method, dataset), optionally restricted to one
    pca_rank. rank=None means no rank restriction (non-OBF methods)."""
    seed_rows: SeedRows = {}
    for seed in seed_pool:
        candidates = []
        for row in filtered:
            if parse_int(row_value(row, "config.seed")) != seed:
                continue
            if row_value(row, "config.task") != dataset:
                continue
            if row_value(row, "config.compressor") not in method.compressors:
                continue
            if rank is not None:
                row_rank = blank_or_int(row, "config.pca_rank", args.pca_rank, args.strict_explicit_config)
                if row_rank != rank:
                    continue
            candidates.append(row)
        if candidates:
            seed_rows[seed] = max(candidates, key=lambda row: selection_score(method, row))
    return seed_rows


def select_seed_rows(rows: Sequence[Dict[str, str]], args: argparse.Namespace) -> Dict[CellKey, SeedRows]:
    filtered = [row for row in rows if row_matches_base_filters(row, args)]
    selected: Dict[CellKey, SeedRows] = {}

    seed_pool = list(dict.fromkeys(list(args.seeds) + [args.efficiency_seed]))
    for dataset, _ in DATASETS:
        for method in METHODS:
            key = (method.label, dataset)
            fixed_rank = FIXED_RANK_BY_METHOD.get(method.label)
            selected[key] = _seed_rows_at_rank(filtered, method, dataset, seed_pool, fixed_rank, args)
            if args.verbose and fixed_rank is not None:
                print(
                    f"  [fixed-rank] {method.label} {dataset}: rank={fixed_rank}",
                    file=sys.stderr,
                )

    return selected


def cell_complete(seed_rows: SeedRows, args: argparse.Namespace) -> bool:
    return args.allow_partial or all(seed in seed_rows for seed in args.seeds)


def accuracy_values(seed_rows: SeedRows, args: argparse.Namespace) -> List[float]:
    if not cell_complete(seed_rows, args):
        return []
    values = []
    for seed in args.seeds:
        row = seed_rows.get(seed)
        if row is None:
            continue
        accuracy = parse_float(row_value(row, "summary.accuracy"))
        if accuracy is not None:
            values.append(accuracy * 100.0)
    return values


def mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def fmt_accuracy(seed_rows: SeedRows, best: Optional[float], args: argparse.Namespace) -> str:
    value = mean(accuracy_values(seed_rows, args))
    if value is None:
        return "--"
    text = f"{value:.2f}"
    if best is not None and abs(value - best) < 1e-9:
        return rf"\textbf{{{text}}}"
    return text


def accuracy_mean(seed_rows: SeedRows, args: argparse.Namespace) -> Optional[float]:
    return mean(accuracy_values(seed_rows, args))


def fmt_improvement(
    method_rows: SeedRows,
    full_rows: SeedRows,
    args: argparse.Namespace,
) -> str:
    method_value = accuracy_mean(method_rows, args)
    full_value = accuracy_mean(full_rows, args)
    if method_value is None or full_value is None:
        return "--"
    if abs(method_value - full_value) < 1e-9:
        return "0.00"
    return f"{method_value - full_value:+.2f}"


def fmt_delta_percent(method_value: Optional[float], full_value: Optional[float], decimals: int = 1) -> str:
    if method_value is None or full_value is None or abs(full_value) < 1e-12:
        return "--"
    delta = (method_value - full_value) / full_value * 100.0
    if abs(delta) < 1e-9:
        return f"{0:.{decimals}f}"
    return f"{delta:+.{decimals}f}"


def communication_values(seed_rows: SeedRows, args: argparse.Namespace) -> List[float]:
    if not cell_complete(seed_rows, args):
        return []
    values = []
    for seed in args.seeds:
        row = seed_rows.get(seed)
        if row is None:
            continue
        communication_mb = parse_float(row_value(row, "summary.avg_communication_MB"))
        if communication_mb is not None:
            values.append(communication_mb)
    return values


def communication_ratio_cell(
    selected: Dict[CellKey, SeedRows],
    dataset: str,
    args: argparse.Namespace,
) -> str:
    full_comm = mean(communication_values(selected[("Full", dataset)], args))
    relay_comms = [
        mean(communication_values(selected[(method_label, dataset)], args))
        for method_label in RELAY_METHOD_LABELS
    ]
    relay_comms = [value for value in relay_comms if value is not None]
    relay_comm = mean(relay_comms)
    if full_comm is None or full_comm <= 0 or relay_comm is None:
        return "--"
    rho = relay_comm / full_comm * 100.0
    return f"{full_comm:.1f}/{rho:.1f}"


def token_usage_values(seed_rows: SeedRows, args: argparse.Namespace) -> List[float]:
    if not cell_complete(seed_rows, args):
        return []
    values = []
    for seed in args.seeds:
        row = seed_rows.get(seed)
        if row is None:
            continue
        token_usage = parse_float(row_value(row, "summary.token_usage"))
        if token_usage is not None:
            values.append(token_usage)
    return values


def token_usage_cell(selected: Dict[CellKey, SeedRows], dataset: str, args: argparse.Namespace) -> str:
    token_usage = mean(token_usage_values(selected[("Full", dataset)], args))
    if token_usage is None:
        return "--"
    return f"{round(token_usage):.0f}"


def fmt_token_usage(seed_rows: SeedRows, args: argparse.Namespace) -> str:
    token_usage = token_usage_mean(seed_rows, args)
    if token_usage is None:
        return "--"
    return f"{round(token_usage):.0f}"


def token_usage_mean(seed_rows: SeedRows, args: argparse.Namespace) -> Optional[float]:
    return mean(token_usage_values(seed_rows, args))


def text_inference_time_values(seed_rows: SeedRows, args: argparse.Namespace) -> List[float]:
    if not cell_complete(seed_rows, args):
        return []
    values = []
    for seed in args.seeds:
        row = seed_rows.get(seed)
        if row is None:
            continue
        text_time = parse_float(row_value(row, "summary.avg_text_inference_time_s"))
        if text_time is not None:
            values.append(text_time)
    return values


def fmt_text_inference_time(seed_rows: SeedRows, args: argparse.Namespace) -> str:
    text_time = text_inference_time_mean(seed_rows, args)
    if text_time is None:
        return "--"
    return f"{text_time:.2f}"


def text_inference_time_mean(seed_rows: SeedRows, args: argparse.Namespace) -> Optional[float]:
    return mean(text_inference_time_values(seed_rows, args))


def tokens_per_second_mean(seed_rows: SeedRows, args: argparse.Namespace) -> Optional[float]:
    token_usage = token_usage_mean(seed_rows, args)
    text_time = text_inference_time_mean(seed_rows, args)
    if token_usage is None or text_time is None or text_time <= 0:
        return None
    return token_usage / text_time


def fmt_tokens_per_second(seed_rows: SeedRows, args: argparse.Namespace) -> str:
    tokens_per_second = tokens_per_second_mean(seed_rows, args)
    if tokens_per_second is None:
        return "--"
    return f"{tokens_per_second:.1f}"


def best_by_dataset(selected: Dict[CellKey, SeedRows], args: argparse.Namespace) -> Dict[str, Optional[float]]:
    best: Dict[str, Optional[float]] = {}
    for dataset, _ in DATASETS:
        values = []
        for method in METHODS:
            value = mean(accuracy_values(selected[(method.label, dataset)], args))
            if value is not None:
                values.append(value)
        best[dataset] = max(values) if values else None
    return best


def second_best_by_dataset(
    selected: Dict[CellKey, SeedRows], args: argparse.Namespace
) -> Dict[str, Optional[float]]:
    """Second-highest DISTINCT accuracy per dataset (for \\underline)."""
    second: Dict[str, Optional[float]] = {}
    for dataset, _ in DATASETS:
        values = []
        for method in METHODS:
            value = mean(accuracy_values(selected[(method.label, dataset)], args))
            if value is not None:
                values.append(value)
        # distinct values at display precision, descending; index 1 = runner-up
        distinct = sorted({round(v, 2) for v in values}, reverse=True)
        second[dataset] = distinct[1] if len(distinct) >= 2 else None
    return second


def grouped_datasets() -> List[Tuple[str, str]]:
    label_by_dataset = dict(DATASETS)
    ordered = []
    seen = set()
    for _, datasets in DATASET_GROUPS:
        for dataset in datasets:
            if dataset not in label_by_dataset:
                raise ValueError(f"Unknown dataset in DATASET_GROUPS: {dataset}")
            ordered.append((dataset, label_by_dataset[dataset]))
            seen.add(dataset)

    missing = [dataset for dataset, _ in DATASETS if dataset not in seen]
    if missing:
        raise ValueError(f"DATASET_GROUPS missing datasets: {missing}")
    return ordered


def combined_tabular_column_spec() -> str:
    group_specs = ["cc" * len(datasets) for _, datasets in DATASET_GROUPS]
    return "ll" + "|".join(group_specs)


def combined_header_rows() -> List[str]:
    rows = []
    header_cells = ["Method", "Metric"]
    column = 3
    cmidrules = []
    for group_index, (group_label, datasets) in enumerate(DATASET_GROUPS):
        span = len(datasets) * 2
        alignment = "c|" if group_index < len(DATASET_GROUPS) - 1 else "c"
        header_cells.append(rf"\multicolumn{{{span}}}{{{alignment}}}{{{group_label}}}")
        cmidrules.append(rf"\cmidrule(lr){{{column}-{column + span - 1}}}")
        column += span

    rows.append(" & ".join(header_cells) + r" \\")
    rows.append(" ".join(cmidrules))
    dataset_cells = []
    for group_index, (_, group_datasets) in enumerate(DATASET_GROUPS):
        for dataset_index, dataset in enumerate(group_datasets):
            is_group_end = dataset_index == len(group_datasets) - 1
            alignment = "c|" if group_index < len(DATASET_GROUPS) - 1 and is_group_end else "c"
            dataset_cells.append(rf"\multicolumn{{2}}{{{alignment}}}{{{dict(DATASETS)[dataset]}}}")
    rows.append(" & & " + " & ".join(dataset_cells) + r" \\")
    rows.append(" & & " + " & ".join(["Value & Imp."] * len(grouped_datasets())) + r" \\")
    return rows


def communication_row(selected: Dict[CellKey, SeedRows], args: argparse.Namespace) -> str:
    datasets = grouped_datasets()
    communication_cells = [
        communication_ratio_cell(selected, dataset, args)
        for dataset, _ in datasets
    ]
    communication_spans = []
    for group_index, (_, group_datasets) in enumerate(DATASET_GROUPS):
        for dataset_index, dataset in enumerate(group_datasets):
            is_group_end = dataset_index == len(group_datasets) - 1
            alignment = "c|" if group_index < len(DATASET_GROUPS) - 1 and is_group_end else "c"
            value = communication_cells[[d for d, _ in datasets].index(dataset)]
            communication_spans.append(rf"\multicolumn{{2}}{{{alignment}}}{{{value}}}")
    return r"\multicolumn{2}{l}{$C/\rho$ (\%)} & " + " & ".join(communication_spans) + r" \\"


def fmt_token_time(seed_rows: SeedRows, args: argparse.Namespace) -> str:
    token_value = token_usage_mean(seed_rows, args)
    time_value = text_inference_time_mean(seed_rows, args)
    if token_value is None or time_value is None:
        return "--"
    return f"{round(token_value):.0f}/{time_value:.2f}"


def fmt_accuracy_value(
    value: Optional[float],
    best: Optional[float],
    second: Optional[float] = None,
) -> str:
    if value is None:
        return "--"
    text = f"{value:.2f}"
    # Compare at display precision (2 decimals) so the bold/underline marks
    # match exactly what the reader sees.
    v = round(value, 2)
    if best is not None and v == round(best, 2):
        return rf"\textbf{{{text}}}"
    if second is not None and v == round(second, 2):
        return rf"\underline{{{text}}}"
    return text


def single_seed_float(seed_rows: SeedRows, seed: int, key: str) -> Optional[float]:
    row = seed_rows.get(seed)
    if row is None:
        return None
    return parse_float(row_value(row, key))


def cell_metric_from_record(
    record: Optional[Dict[str, Any]], key: str
) -> Optional[float]:
    if record is None:
        return None
    value = record.get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def single_seed_accuracy(seed_rows: SeedRows, seed: int) -> Optional[float]:
    value = single_seed_float(seed_rows, seed, "summary.accuracy")
    return value * 100.0 if value is not None else None


def single_seed_token_usage(seed_rows: SeedRows, seed: int) -> Optional[float]:
    return single_seed_float(seed_rows, seed, "summary.token_usage")


def single_seed_text_time(seed_rows: SeedRows, seed: int) -> Optional[float]:
    return single_seed_float(seed_rows, seed, "summary.avg_text_inference_time_s")


def row_gpu_matches(row: Optional[Dict[str, str]], gpu_filter: str) -> bool:
    if not gpu_filter:
        return True
    if row is None:
        return False
    gpu_name = row_value(row, "metadata.gpu_nvidia")
    return gpu_filter.lower() in gpu_name.lower()


def single_seed_token_usage_gpu(seed_rows: SeedRows, seed: int, gpu_filter: str) -> Optional[float]:
    row = seed_rows.get(seed)
    if not row_gpu_matches(row, gpu_filter):
        return None
    return single_seed_token_usage(seed_rows, seed)


def single_seed_text_time_gpu(seed_rows: SeedRows, seed: int, gpu_filter: str) -> Optional[float]:
    row = seed_rows.get(seed)
    if not row_gpu_matches(row, gpu_filter):
        return None
    return single_seed_text_time(seed_rows, seed)


def single_seed_communication(seed_rows: SeedRows, seed: int) -> Optional[float]:
    return single_seed_float(seed_rows, seed, "summary.avg_communication_MB")


def accuracy_tabular_column_spec() -> str:
    group_specs = ["c" * len(datasets) for _, datasets in DATASET_GROUPS]
    return "l|" + "|".join(group_specs)


def accuracy_header_rows() -> List[str]:
    rows = []
    header_cells = ["Method"]
    column = 2
    cmidrules = []
    for group_index, (group_label, group_datasets) in enumerate(DATASET_GROUPS):
        span = len(group_datasets)
        header_cells.append(rf"\multicolumn{{{span}}}{{c{'|' if group_index < len(DATASET_GROUPS) - 1 else ''}}}{{{group_label}}}")
        cmidrules.append(rf"\cmidrule(lr){{{column}-{column + span - 1}}}")
        column += span
    rows.append(" & ".join(header_cells) + r" \\")
    rows.append(" ".join(cmidrules))

    dataset_cells = []
    for group_index, (_, group_datasets) in enumerate(DATASET_GROUPS):
        for dataset_index, dataset in enumerate(group_datasets):
            dataset_cells.append(dict(DATASETS)[dataset])
    rows.append(" & " + " & ".join(dataset_cells) + r" \\")
    return rows


def prompt_len_values(seed_rows: SeedRows, args: argparse.Namespace) -> List[float]:
    if not cell_complete(seed_rows, args):
        return []
    values = []
    for seed in args.seeds:
        row = seed_rows.get(seed)
        if row is None:
            continue
        value = parse_float(row_value(row, "summary.prompt_len"))
        if value is not None:
            values.append(value)
    return values


def fmt_L_rho_cell(seed_rows: SeedRows, args: argparse.Namespace) -> str:
    """Render the "L / rho" cell for one dataset.

    L   is the mean real prompt length summed across all non-judger agents
        per sample, then averaged over samples (matches summary.prompt_len).
    rho is the prompt-side KV compression ratio, defined as the ratio of
        prompt KV retained at the end of a full multi-agent rollout to the
        prompt KV fed in across all rounds:

            rho = (B * N_agents + s) / L,

        where B is kv_budget retained per agent, N_agents is the number of
        non-judger agents (sink is kept once across the rollout, not per
        agent, because the compressor sets _has_kept_sink=True after the
        first compression), and s is sink_size.
    """
    L = mean(prompt_len_values(seed_rows, args))
    if L is None or L <= 0:
        return "--"
    numerator = args.kv_budget * args.relay_rounds + args.sink_size
    rho_percent = numerator / L * 100.0
    return f"{L:.0f} / {rho_percent:.1f}"


def make_accuracy_table(selected: Dict[CellKey, SeedRows], args: argparse.Namespace) -> str:
    datasets = grouped_datasets()
    best = best_by_dataset(selected, args)
    second = second_best_by_dataset(selected, args)

    rows = []
    rows.append(r"\begin{table*}[t]")
    rows.append(r"% === AUTOGEN-ACCURACY BEGIN === (do not edit by hand; replace only between the AUTOGEN markers)")
    rows.append(r"\centering")
    rows.append(r"\footnotesize")
    rows.append(r"\setlength{\tabcolsep}{4pt}")
    rows.append(r"\renewcommand{\arraystretch}{1.08}")
    rows.append(r"\resizebox{\textwidth}{!}{%")
    rows.append(rf"\begin{{tabular}}{{{accuracy_tabular_column_spec()}}}")
    rows.append(r"\toprule")
    rows.extend(accuracy_header_rows())
    rows.append(r"\midrule")

    # L / rho row: L = mean real prompt_len (Full-KV run),
    # rho = kv_budget / L as percent.
    l_rho_cells = [
        fmt_L_rho_cell(selected[("Full", dataset)], args) for dataset, _ in datasets
    ]
    rows.append(r"$L$ / $\rho$ (\%) & " + " & ".join(l_rho_cells) + r" \\")
    rows.append(r"\midrule")

    for method in METHODS:
        cells = []
        for dataset, _ in datasets:
            seed_rows = selected[(method.label, dataset)]
            accuracy_value = accuracy_mean(seed_rows, args)
            cells.append(fmt_accuracy_value(accuracy_value, best[dataset], second[dataset]))
        rows.append(method.label + " & " + " & ".join(cells) + r" \\")

    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}%")
    rows.append(r"}")
    rows.append(r"% === AUTOGEN-ACCURACY END ===")
    rows.append(r"\end{table*}")
    return "\n".join(rows)


def make_combined_table(selected: Dict[CellKey, SeedRows], args: argparse.Namespace) -> str:
    datasets = grouped_datasets()
    best = best_by_dataset(selected, args)
    min_tokens_by_dataset = {}
    min_times_by_dataset = {}
    for dataset, _ in datasets:
        min_tokens_by_dataset[dataset] = min_available(
            [token_usage_mean(selected[(method.label, dataset)], args) for method in METHODS]
        )
        min_times_by_dataset[dataset] = min_available(
            [text_inference_time_mean(selected[(method.label, dataset)], args) for method in METHODS]
        )

    rows = []
    rows.append(r"\begin{table*}[t]")
    rows.append(r"\centering")
    rows.append(r"\scriptsize")
    rows.append(r"\setlength{\tabcolsep}{2.6pt}")
    rows.append(r"\renewcommand{\arraystretch}{1.08}")
    rows.append(r"\resizebox{\textwidth}{!}{%")
    rows.append(rf"\begin{{tabular}}{{{combined_tabular_column_spec()}}}")
    rows.append(r"\toprule")
    rows.extend(combined_header_rows())
    rows.append(r"\midrule")
    rows.append(communication_row(selected, args))
    rows.append(r"\midrule")

    for method in METHODS:
        accuracy_cells = []
        token_cells = []
        time_cells = []
        for dataset, _ in datasets:
            seed_rows = selected[(method.label, dataset)]
            full_rows = selected[("Full", dataset)]

            accuracy_value = accuracy_mean(seed_rows, args)
            full_accuracy = accuracy_mean(full_rows, args)
            token_value = token_usage_mean(seed_rows, args)
            full_token = token_usage_mean(full_rows, args)
            time_value = text_inference_time_mean(seed_rows, args)
            full_time = text_inference_time_mean(full_rows, args)

            accuracy_bold = best[dataset] is not None and accuracy_value is not None and abs(accuracy_value - best[dataset]) < 1e-9
            min_token = min_tokens_by_dataset[dataset]
            min_time = min_times_by_dataset[dataset]
            token_bold = min_token is not None and token_value is not None and abs(token_value - min_token) < 1e-9
            time_bold = min_time is not None and time_value is not None and abs(time_value - min_time) < 1e-9

            accuracy_cells.append(fmt_accuracy_value(accuracy_value, best[dataset]))
            accuracy_cells.append(
                "--"
                if method.label == "Full"
                else maybe_bold(fmt_absolute_delta(accuracy_value, full_accuracy), accuracy_bold)
            )
            token_cells.append(fmt_metric_value(token_value, decimals=0, bold=token_bold))
            token_cells.append(
                "--"
                if method.label == "Full"
                else maybe_bold(fmt_relative_delta(token_value, full_token), token_bold)
            )
            time_cells.append(fmt_metric_value(time_value, decimals=0, bold=time_bold))
            time_cells.append(
                "--"
                if method.label == "Full"
                else maybe_bold(fmt_relative_delta(time_value, full_time), time_bold)
            )
        rows.append(method.label + " & Acc. & " + " & ".join(accuracy_cells) + r" \\")
        rows.append(r" & Tok. & " + " & ".join(token_cells) + r" \\")
        rows.append(r" & Time & " + " & ".join(time_cells) + r" \\")
        if method != METHODS[-1]:
            rows.append(r"\midrule")

    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}%")
    rows.append(r"}")

    seed_text = ", ".join(str(seed) for seed in args.seeds)
    averaging_text = (
        f"Numbers are averaged over seeds {seed_text}."
        if not args.allow_partial
        else f"Numbers are averaged over available runs from seeds {seed_text}."
    )
    rows.append(
        rf"\caption{{\textbf{{{args.model_name.split('/')[-1]} accuracy and efficiency across datasets.}} "
        rf"Higher accuracy is better. Tok. reports average generated token usage and Time reports average text "
        rf"inference time in seconds. Imp. reports relative percentage change from Full without the percent sign. "
        rf"The $C/\rho$ row reports Full average "
        rf"communication in MB and the average communication ratio of L/L-OBF/H/H-OBF relative to Full, "
        rf"where lower is better. L/L-OBF/H/H-OBF use a relay budget of "
        rf"$B={args.kv_budget}$ prompt tokens per round; all methods use $p={args.latent_steps}$ "
        rf"latent steps per round. {averaging_text} Best result on each benchmark is in "
        rf"\textbf{{bold}}.}}"
    )
    rows.append(rf"\label{{{args.label}}}")
    rows.append(r"\end{table*}")
    return "\n".join(rows)


def efficiency_tabular_column_spec() -> str:
    n_datasets = sum(len(ds) for _, ds in DATASET_GROUPS)
    return "l|" + "|".join(["c"] * n_datasets)


def fmt_relative_delta(method_value: Optional[float], full_value: Optional[float]) -> str:
    if method_value is None or full_value is None or abs(full_value) < 1e-12:
        return "--"
    delta = (method_value - full_value) / full_value * 100.0
    sign = "+" if delta >= 0 else "-"
    magnitude = abs(delta)
    padding = r"\phantom{0}" if magnitude < 10 else ""
    return rf"{sign}{padding}{magnitude:.2f}"


def fmt_absolute_delta(method_value: Optional[float], full_value: Optional[float]) -> str:
    if method_value is None or full_value is None:
        return "--"
    delta = method_value - full_value
    sign = "+" if delta >= 0 else "-"
    magnitude = abs(delta)
    padding = r"\phantom{0}" if magnitude < 10 else ""
    return rf"{sign}{padding}{magnitude:.2f}"


def maybe_bold(text: str, bold: bool) -> str:
    if bold and text != "--":
        return rf"\textbf{{{text}}}"
    return text


def fmt_metric_value(value: Optional[float], *, decimals: int, bold: bool) -> str:
    if value is None:
        return "--"
    text = f"{value:.{decimals}f}"
    if bold:
        return rf"\textbf{{{text}}}"
    return text


def min_available(values: Sequence[Optional[float]]) -> Optional[float]:
    available = [value for value in values if value is not None]
    return min(available) if available else None


def single_seed_communication_cell(
    selected: Dict[CellKey, SeedRows],
    dataset: str,
    args: argparse.Namespace,
) -> str:
    seed = args.efficiency_seed
    full_comm = single_seed_communication(selected[("Full", dataset)], seed)
    relay_comms = [
        single_seed_communication(selected[(method_label, dataset)], seed)
        for method_label in RELAY_METHOD_LABELS
    ]
    relay_comms = [value for value in relay_comms if value is not None]
    relay_comm = mean(relay_comms)
    if full_comm is None or full_comm <= 0 or relay_comm is None:
        return "--"
    rho = relay_comm / full_comm * 100.0
    return f"{full_comm:.1f}/{rho:.1f}"


def make_efficiency_table(
    selected: Dict[CellKey, SeedRows],
    args: argparse.Namespace,
    csv_paths: Sequence[Path],
) -> str:
    datasets = grouped_datasets()
    rows = []
    rows.append(r"\begin{table*}[t]")
    rows.append(r"% === AUTOGEN-EFFICIENCY BEGIN === (do not edit by hand; replace only between the AUTOGEN markers)")
    rows.append(r"\centering")
    rows.append(r"\scriptsize")
    rows.append(r"\setlength{\tabcolsep}{3pt}")
    rows.append(r"\renewcommand{\arraystretch}{1.08}")
    rows.append(r"\resizebox{\textwidth}{!}{%")
    rows.append(rf"\begin{{tabular}}{{{accuracy_tabular_column_spec()}}}")
    rows.append(r"\toprule")
    rows.extend(accuracy_header_rows())
    rows.append(r"\midrule")

    def cross_seed_tok_per_sec(seed_rows: SeedRows) -> Optional[float]:
        per_seed = []
        for seed in args.seeds:
            row = seed_rows.get(seed)
            if row is None:
                continue
            token_value = parse_float(row_value(row, "summary.token_usage"))
            time_value = parse_float(row_value(row, "summary.avg_text_inference_time_s"))
            if token_value is not None and time_value is not None and time_value > 0:
                per_seed.append(token_value / time_value)
        if not per_seed:
            return None
        return sum(per_seed) / len(per_seed)

    # Per-dataset Tok/s maxima for bolding.
    max_tokens_per_second_by_dataset = {}
    for dataset, _ in datasets:
        values = []
        for method in METHODS:
            tps = cross_seed_tok_per_sec(selected[(method.label, dataset)])
            if tps is not None:
                values.append(tps)
        max_tokens_per_second_by_dataset[dataset] = max(values) if values else None

    for method in METHODS:
        cells = []
        for dataset, _ in datasets:
            tokens_per_second = cross_seed_tok_per_sec(selected[(method.label, dataset)])
            best_value = max_tokens_per_second_by_dataset[dataset]
            bold = best_value is not None and tokens_per_second is not None and abs(tokens_per_second - best_value) < 1e-9
            cells.append(fmt_metric_value(tokens_per_second, decimals=1, bold=bold))
        rows.append(method.label + " & " + " & ".join(cells) + r" \\")

    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}%")
    rows.append(r"}")
    rows.append(r"% === AUTOGEN-EFFICIENCY END ===")
    rows.append(r"\end{table*}")
    return "\n".join(rows)


def make_table(selected: Dict[CellKey, SeedRows], args: argparse.Namespace) -> str:
    # Legacy dispatcher retained for callers that still rely on a single-table interface.
    return make_combined_table(selected, args)


def display_config(row: Dict[str, str], key: str, assumed: int, relevant: bool) -> str:
    if not relevant:
        return "n/a"
    value = row_value(row, key)
    if value == "":
        return f"{assumed} (assumed; blank in CSV)"
    return value


def warn_selection(selected: Dict[CellKey, SeedRows], args: argparse.Namespace) -> None:
    print("Selected runs:", file=sys.stderr)
    for dataset, _ in DATASETS:
        for method in METHODS:
            seed_rows = selected[(method.label, dataset)]
            missing = [seed for seed in args.seeds if seed not in seed_rows]
            values = accuracy_values(seed_rows, args)
            if missing:
                message = f"  MISSING dataset={dataset} method={method.label} seeds={missing}"
                if args.allow_partial and seed_rows:
                    message += " (using available seeds)"
                print(message, file=sys.stderr)

            if not seed_rows:
                continue

            mean_value = mean(values)
            mean_text = f"{mean_value:.2f}" if mean_value is not None else "n/a"
            print(f"  dataset={dataset} method={method.label} mean_acc={mean_text}", file=sys.stderr)
            for seed in args.seeds:
                row = seed_rows.get(seed)
                if row is None:
                    continue
                accuracy = parse_float(row_value(row, "summary.accuracy"))
                compressor = row_value(row, "config.compressor")
                kv_relevant = compressor not in {"full"}
                pca_relevant = is_obf_compressor(compressor)
                print(
                    "    "
                    f"seed={seed} "
                    f"acc={accuracy * 100.0:.2f} "
                    f"compressor={compressor} "
                    f"kv={display_config(row, 'config.kv_budget', args.kv_budget, kv_relevant)} "
                    f"pca={display_config(row, 'config.pca_rank', args.pca_rank, pca_relevant)} "
                    f"name={row_value(row, 'name')}",
                    file=sys.stderr,
                )


def main() -> None:
    args = parse_args()
    csv_paths = args.csv if args.csv is not None else default_csv_paths(args)
    rows = read_rows(csv_paths)
    selected = select_seed_rows(rows, args)
    if args.verbose:
        warn_selection(selected, args)

    accuracy_table = make_accuracy_table(selected, args)
    efficiency_table = make_efficiency_table(selected, args, csv_paths)
    combined_output = accuracy_table + "\n\n" + efficiency_table

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(combined_output + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    if args.print_table or not args.output:
        print(combined_output)


if __name__ == "__main__":
    main()
