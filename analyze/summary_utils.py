import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY_DIR = SCRIPT_DIR.parent / "results" / "wandb_summary"
DEFAULT_P_VALUE = 8


def as_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def parse_scalar(value):
    if value in (None, ""):
        return None
    if isinstance(value, (int, float, bool)):
        return value

    text = str(value).strip()
    if text == "":
        return None

    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        if "." not in text and "e" not in lowered:
            return int(text)
        return float(text)
    except ValueError:
        return text


def find_summary_csvs(summary_dir=DEFAULT_SUMMARY_DIR):
    summary_dir = Path(summary_dir)
    if not summary_dir.exists():
        return []
    return sorted(path for path in summary_dir.rglob("*.csv") if path.is_file())


def read_csv_rows(csv_path):
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def get_first_present(row, keys):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def extract_p_value(row):
    p_value = parse_scalar(row.get("config.pca_rank"))
    if p_value is not None:
        return p_value
    return DEFAULT_P_VALUE


def normalize_summary_row(row, source_csv):
    normalized = {key: parse_scalar(value) for key, value in row.items()}
    normalized["source_csv"] = str(Path(source_csv))

    normalized["dataset"] = get_first_present(
        row, ["config.task", "summary.task", "task"]
    )
    normalized["compressor"] = get_first_present(
        row, ["config.compressor", "summary.compressor", "compressor"]
    )
    normalized["model"] = get_first_present(
        row, ["config.model_name", "summary.model", "model"]
    )
    normalized["device"] = get_first_present(
        row, ["config.device", "summary.device", "device"]
    )
    normalized["gpu_name"] = get_first_present(
        row,
        [
            "system.GPU type",
            "system.gpu_type",
            "metadata.gpu",
            "metadata.gpu_name",
            "metadata.gpu_type",
            "metadata.cuda",
            "metadata.cuda_gpu",
            "metadata.cuda.gpu",
            "metadata.cuda.gpu_name",
            "metadata.cuda.gpu_type",
            "metadata.system.gpu",
            "metadata.system.gpu_name",
            "metadata.system.gpu_type",
            "config.gpu_name",
            "config.gpu",
            "summary.gpu_name",
            "summary.gpu",
            "system.gpu",
            "system.gpu_name",
            "gpu_name",
            "gpu",
        ],
    )
    normalized["seed"] = parse_scalar(
        get_first_present(row, ["config.seed", "summary.seed", "seed"])
    )
    normalized["method"] = get_first_present(
        row, ["config.method", "summary.method", "method"]
    )
    normalized["split"] = get_first_present(
        row, ["config.split", "summary.split", "split"]
    )
    normalized["project"] = get_first_present(row, ["project", "project_path"])
    normalized["run_name"] = get_first_present(row, ["name", "run_name"])
    normalized["state"] = get_first_present(row, ["state"])
    normalized["p_value"] = extract_p_value(row)
    normalized["accuracy"] = parse_scalar(row.get("summary.accuracy"))
    normalized["token_usage"] = parse_scalar(row.get("summary.token_usage"))
    normalized["avg_text_inference_time_s"] = parse_scalar(
        row.get("summary.avg_text_inference_time_s")
    )
    normalized["avg_latent_inference_time_s"] = parse_scalar(
        row.get("summary.avg_latent_inference_time_s")
    )
    normalized["avg_compression_time_s"] = parse_scalar(
        row.get("summary.avg_compression_time_s")
    )
    return normalized


def load_summary_records(summary_dir=DEFAULT_SUMMARY_DIR, csv_paths=None):
    if csv_paths is None:
        csv_paths = find_summary_csvs(summary_dir)

    records = []
    for csv_path in csv_paths:
        for row in read_csv_rows(csv_path):
            records.append(normalize_summary_row(row, csv_path))
    return records


def value_matches(value, expected):
    expected_values = as_list(expected)
    if expected_values is None:
        return True
    return value in expected_values


def filter_summary_records(
    records,
    datasets=None,
    compressors=None,
    models=None,
    p_values=None,
    devices=None,
    seeds=None,
    methods=None,
    projects=None,
    states=None,
    predicate=None,
):
    filtered = []
    for record in records:
        if not value_matches(record.get("dataset"), datasets):
            continue
        if not value_matches(record.get("compressor"), compressors):
            continue
        if not value_matches(record.get("model"), models):
            continue
        if not value_matches(record.get("p_value"), p_values):
            continue
        if not value_matches(record.get("device"), devices):
            continue
        if not value_matches(record.get("seed"), seeds):
            continue
        if not value_matches(record.get("method"), methods):
            continue
        if not value_matches(record.get("project"), projects):
            continue
        if not value_matches(record.get("state"), states):
            continue
        if predicate is not None and not predicate(record):
            continue
        filtered.append(record)
    return filtered


def available_values(records, key):
    return sorted({record.get(key) for record in records if record.get(key) is not None})


if __name__ == "__main__":
    records = load_summary_records()
    print(f"Loaded records: {len(records)}")
    print(f"CSV files: {len(find_summary_csvs())}")
    print(f"Datasets: {available_values(records, 'dataset')}")
    print(f"Compressors: {available_values(records, 'compressor')}")
    print(f"Models: {available_values(records, 'model')}")
    print(f"P values: {available_values(records, 'p_value')}")
    print(f"Devices: {available_values(records, 'device')}")
    print(f"GPU names: {available_values(records, 'gpu_name')}")
