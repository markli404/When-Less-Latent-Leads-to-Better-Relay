#!/usr/bin/env python3
"""Generate a LaTeX table ranking pca_rank values by accuracy per dataset.

Three columns: Dataset | L-OBF ranking | H-OBF ranking
Cell shows the pca_rank values sorted best-to-worst (left = best, right = worst).
All runs are averaged over seeds {4, 44, 444} on A100 hardware only.
"""

import csv
from collections import defaultdict
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_DIR / "results" / "wandb_summary" / "pca_rank_sweep.csv"
DEFAULT_OUTPUT = REPO_DIR / "results" / "pca_rank_ranking_table.tex"

SEEDS = {4, 44, 444}
PCA_RANKS = [2, 4, 8, 16, 32]
COMPRESSORS = ["lobf", "hobf"]
A100_SUBSTR = "A100"

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


def is_a100(row):
    gpu_blob = str(row.get("metadata.gpu_nvidia", "")) + str(row.get("metadata.gpu", ""))
    return A100_SUBSTR in gpu_blob


def parse_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_records(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        yield from reader


def build_ranking(csv_path):
    """Return (accuracy_by_key, missing_cells).

    accuracy_by_key[(dataset, compressor, rank)] = mean accuracy across seeds.
    """
    grouped = defaultdict(list)
    for row in load_records(csv_path):
        if not is_a100(row):
            continue
        seed = parse_int(row.get("config.seed"))
        if seed not in SEEDS:
            continue
        compressor = row.get("config.compressor", "")
        if compressor not in COMPRESSORS:
            continue
        rank = parse_int(row.get("config.pca_rank"))
        if rank not in PCA_RANKS:
            continue
        dataset = row.get("config.task", "")
        if dataset == "":
            continue
        accuracy = parse_float(row.get("summary.accuracy"))
        if accuracy is None:
            continue
        grouped[(dataset, compressor, rank)].append(accuracy)

    averaged = {}
    missing = []
    for dataset, _ in DATASETS:
        for compressor in COMPRESSORS:
            for rank in PCA_RANKS:
                key = (dataset, compressor, rank)
                values = grouped.get(key, [])
                if not values:
                    missing.append(key)
                    continue
                averaged[key] = sum(values) / len(values)
    return averaged, missing


def sorted_ranks(averaged, dataset, compressor):
    """Return ranks sorted best-to-worst by accuracy."""
    pairs = []
    for rank in PCA_RANKS:
        acc = averaged.get((dataset, compressor, rank))
        if acc is None:
            continue
        pairs.append((rank, acc))
    pairs.sort(key=lambda x: -x[1])
    return [r for r, _ in pairs]


def emit_ranking_block(averaged):
    """Emit the AUTOGEN-PCARANKING content: dataset x compressor rankings.

    Excludes the outer \\begin{table}/\\end{table}, \\caption, and \\label.
    """
    lines = [
        r"    \centering",
        r"    \small",
        r"    \begin{tabular}{lcc}",
        r"    \toprule",
        r"    Dataset & L-OBF & H-OBF \\",
        r"    \midrule",
    ]
    for dataset, label in DATASETS:
        cells = [label]
        for compressor in COMPRESSORS:
            ranks = sorted_ranks(averaged, dataset, compressor)
            # Wrap in math mode so ">" renders correctly (T1 text mode maps
            # ">" to an unrelated glyph like an inverted question mark).
            cells.append("$" + " > ".join(str(r) for r in ranks) + "$" if ranks else "--")
        lines.append(f"    {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}")
    return "\n".join(lines) + "\n"


def compute_scores(averaged):
    """Return {compressor: {rank: total_score}} using a 5-4-3-2-1 scheme.

    For each dataset, ranks are ordered best-to-worst by accuracy and receive
    5, 4, 3, 2, 1 points respectively; scores accumulate across all datasets.
    """
    scores = {c: {r: 0 for r in PCA_RANKS} for c in COMPRESSORS}
    n_ranks = len(PCA_RANKS)  # 5
    for dataset, _ in DATASETS:
        for compressor in COMPRESSORS:
            ordering = sorted_ranks(averaged, dataset, compressor)
            for position, rank in enumerate(ordering):
                scores[compressor][rank] += (n_ranks - position)
    return scores


def emit_scoring_block(averaged):
    """Emit the AUTOGEN-PCASCORING content: aggregated scores per rank.

    Rows are sorted by combined score descending. Each column's top value is
    bolded via \\textbf{}.
    """
    scores = compute_scores(averaged)
    per_compressor = {c: scores[c] for c in COMPRESSORS}
    combined = {r: sum(per_compressor[c][r] for c in COMPRESSORS) for r in PCA_RANKS}
    best_lobf = max(per_compressor["lobf"].values())
    best_hobf = max(per_compressor["hobf"].values())
    best_combined = max(combined.values())
    ranks_sorted = sorted(PCA_RANKS, key=lambda r: -combined[r])

    def cell(value, is_best):
        return f"\\textbf{{{value}}}" if is_best else f"{value}"

    lines = [
        r"    \centering",
        r"    \small",
        r"    \begin{tabular}{cccc}",
        r"    \toprule",
        r"    \texttt{pca\_rank} & L-OBF & H-OBF & Combined \\",
        r"    \midrule",
    ]
    for rank in ranks_sorted:
        l = per_compressor["lobf"][rank]
        h = per_compressor["hobf"][rank]
        c = combined[rank]
        lines.append(
            f"    {rank} & "
            f"{cell(l, l == best_lobf)} & "
            f"{cell(h, h == best_hobf)} & "
            f"{cell(c, c == best_combined)} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}")
    return "\n".join(lines) + "\n"


def main():
    averaged, missing = build_ranking(DEFAULT_CSV)
    ranking_block = emit_ranking_block(averaged)
    scoring_block = emit_scoring_block(averaged)

    print("=" * 70)
    print("AUTOGEN-PCARANKING (paste between markers of tab:pca_rank_ordering):")
    print("=" * 70)
    print(ranking_block)
    print("=" * 70)
    print("AUTOGEN-PCASCORING (paste between markers of tab:pca_rank_scoring):")
    print("=" * 70)
    print(scoring_block)

    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined_output = (
        "% AUTOGEN-PCARANKING\n"
        + ranking_block
        + "\n% AUTOGEN-PCASCORING\n"
        + scoring_block
    )
    DEFAULT_OUTPUT.write_text(combined_output)
    print(f"Saved both blocks to: {DEFAULT_OUTPUT}")
    if missing:
        print(f"\nWarning: {len(missing)} (dataset, compressor, rank) cells had no A100 data:")
        for key in missing:
            print(f"  {key}")


if __name__ == "__main__":
    main()
