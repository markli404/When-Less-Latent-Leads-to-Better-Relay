DATASET_ORDER = [
    "gsm8k",
    "aime2024",
    "aime2025",
    "arc_easy",
    "arc_challenge",
    "medqa",
    "gpqa",
    "mbppplus",
    "humanevalplus",
]

DATASET_LABELS = {
    "gsm8k": "GSM8K",
    "medqa": "MedQA",
    "mbppplus": "MBPP+",
    "humanevalplus": "HumanEval+",
    "arc_easy": "ARC-E",
    "arc_challenge": "ARC-C",
    "aime2024": "AIME24",
    "aime2025": "AIME25",
    "gpqa": "GPQA",
}

COMPRESSOR_ALIASES = {
    "lobf": ["lobf", "lobf_metric"],
    "hobf": ["hobf", "hobf_metric"],
}

METHOD_LABELS = {
    "full": "Full",
    "gonly": "Gen",
    "layerwise": "Layerwise",
    "headwise": "Headwise",
    "lobf": "L-OBF",
    "lobf_metric": "L-OBF",
    "hobf": "H-OBF",
    "hobf_metric": "H-OBF",
    "hobf_no_scale": "(a) No Scaling",
    "hobf_no_proj": "(b) No Projection",
    "hobf_max_p": "(c) Max-P",
    "hobf_naive": "(d) Naive Aggregation",
    "lobf_no_scale": "L-OBF No Scaling",
    "lobf_no_proj": "L-OBF No Projection",
    "lobf_max_p": "L-OBF Max-P",
    "lobf_naive": "L-OBF Naive Aggregation",
    # Design ablations and comparison baselines
    "lobf_evr": "L-OBF (EVR-adaptive)",
    "lobf_fast": "L-OBF (fast)",
    "hobf_fast": "H-OBF (fast)",
    "lobf_per_token": "L-OBF (per-token)",
    "lmerge": "Token Merging",
    "full_quant": "Full + Quant",
    "lobf_quant": "L-OBF + Quant",
}

# Shared, presentation-friendly palette for method comparison plots.
# Blue family: baseline / layerwise / L-OBF comparisons
# Orange family: baseline / headwise / H-OBF comparisons
METHOD_COLORS = {
    "full": "#4C4C4C",
    "gonly": "#9C9C9C",
    "layerwise": "#56B4E9",
    "lobf": "#0072B2",
    "lobf_metric": "#0072B2",
    "headwise": "#E69F00",
    "hobf": "#D55E00",
    "hobf_metric": "#D55E00",
    "hobf_no_scale": "#CC79A7",
    "hobf_no_proj": "#8C564B",
    "hobf_max_p": "#009E73",
    "hobf_naive": "#F0A3FF",
    "lobf_no_scale": "#6BAED6",
    "lobf_no_proj": "#3182BD",
    "lobf_max_p": "#41AB5D",
    "lobf_naive": "#08519C",
    # Design ablations and comparison baselines
    "lobf_evr": "#0072B2",
    "lobf_fast": "#0072B2",
    "hobf_fast": "#D55E00",
    "lobf_per_token": "#3182BD",
    "lmerge": "#E69F00",
    "full_quant": "#9C9C9C",
    "lobf_quant": "#009E73",
}


def apply_plot_style():
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#666666",
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }


def expand_method_aliases(methods):
    expanded = []
    for method in methods:
        expanded.extend(COMPRESSOR_ALIASES.get(method, [method]))
    seen = []
    for method in expanded:
        if method not in seen:
            seen.append(method)
    return seen


def canonicalize_method_name(method):
    for canonical, aliases in COMPRESSOR_ALIASES.items():
        if method in aliases:
            return canonical
    return method


def canonicalize_records(records):
    normalized = []
    for record in records:
        copied = dict(record)
        copied["compressor"] = canonicalize_method_name(record.get("compressor"))
        normalized.append(copied)
    return normalized
