import csv
import os
import json
import tempfile
from pathlib import Path

try:
    import wandb
except ImportError:
    wandb = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent.parent / "results" / "wandb_summary"
DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY")  # set this, or pass --entity


def normalize_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def project_file_stem(project_name):
    return str(project_name).strip().split("/")[-1]


def project_csv_path(project_name, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    return output_dir / f"{project_file_stem(project_name)}.csv"


def legacy_project_csv_path(project_name, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    return output_dir / f"{project_file_stem(project_name)}_merged.csv"


def resolve_project_path(project_name, entity=None):
    project_name = str(project_name).strip()
    if "/" in project_name:
        return project_name
    if entity is None:
        entity = DEFAULT_ENTITY
    return f"{entity.rstrip('/')}/{project_name}"


def flatten_dict(data, prefix=""):
    flat = {}
    if not isinstance(data, dict):
        return flat
    for key, value in data.items():
        key = str(key)
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, full_key))
        else:
            flat[full_key] = value
    return flat


def safe_summary_dict(run):
    try:
        run.load_full_data()
    except Exception:
        pass
    try:
        summary = getattr(run, "summary_metrics", None)
        if isinstance(summary, dict):
            return summary
    except Exception:
        pass
    return {}


def safe_config_dict(run):
    try:
        run.load_full_data()
    except Exception:
        pass
    try:
        raw = getattr(run, "rawconfig", None)
        if isinstance(raw, dict) and raw:
            return {k: v for k, v in raw.items() if not str(k).startswith("_")}
    except Exception:
        pass
    try:
        config = getattr(run, "config", None)
        if isinstance(config, dict):
            return {k: v for k, v in config.items() if not str(k).startswith("_")}
    except Exception:
        pass
    return {}


def safe_system_dict(run):
    try:
        run.load_full_data()
    except Exception:
        pass

    for attr_name in ("metadata", "_attrs", "_json_dict"):
        try:
            raw = getattr(run, attr_name, None)
        except Exception:
            raw = None
        if not isinstance(raw, dict):
            continue
        for key in ("system", "systemMetrics", "system_metrics"):
            value = raw.get(key)
            if isinstance(value, dict) and value:
                return value
    return {}


def safe_metadata_dict(run):
    try:
        raw_metadata = getattr(run, "metadata", None)
    except Exception:
        raw_metadata = None
    if isinstance(raw_metadata, dict) and raw_metadata:
        flat = flatten_dict(raw_metadata)
        selected = {}
        for key, value in flat.items():
            lowered = key.lower()
            if any(
                token in lowered
                for token in (
                    "gpu",
                    "cuda",
                    "cpu",
                    "memory",
                    "host",
                    "hostname",
                    "os",
                    "platform",
                    "processor",
                )
            ):
                selected[key] = value
        if selected:
            return selected

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_file = run.file("wandb-metadata.json")
            downloaded = metadata_file.download(root=tmp_dir, replace=True)
            metadata_path = Path(downloaded.name)
            if not metadata_path.exists():
                metadata_path = Path(tmp_dir) / "wandb-metadata.json"
            if not metadata_path.exists():
                return {}
            with metadata_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
    except Exception:
        return {}

    flat = flatten_dict(raw)
    selected = {}
    for key, value in flat.items():
        lowered = key.lower()
        if any(
            token in lowered
            for token in (
                "gpu",
                "cuda",
                "cpu",
                "memory",
                "host",
                "hostname",
                "os",
                "platform",
                "processor",
            )
        ):
            selected[key] = value
    return selected


def collect_base_record(run, project_path):
    return {
        "project": project_file_stem(project_path),
        "project_path": project_path,
        "name": run.name,
        "id": run.id,
        "state": run.state,
        "url": run.url,
    }


def collect_merged_record(run, project_path):
    record = collect_base_record(run, project_path)
    summary = safe_summary_dict(run)
    config = safe_config_dict(run)
    system = safe_system_dict(run)
    metadata = safe_metadata_dict(run)
    for key, value in summary.items():
        record[f"summary.{key}"] = normalize_value(value)
    for key, value in config.items():
        record[f"config.{key}"] = normalize_value(value)
    for key, value in system.items():
        record[f"system.{key}"] = normalize_value(value)
    for key, value in metadata.items():
        record[f"metadata.{key}"] = normalize_value(value)
    record["_summary_count"] = len(summary)
    record["_config_count"] = len(config)
    record["_system_count"] = len(system)
    record["_metadata_count"] = len(metadata)
    return record


def write_csv(records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in records for key in record.keys()})
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def read_existing_records(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    existing = {}
    for row in rows:
        run_id = row.get("id")
        if run_id:
            existing[run_id] = row
    return existing


def _matches_name(run_name, name_contains):
    run_name = run_name or ""
    if isinstance(name_contains, str):
        return name_contains in run_name
    if isinstance(name_contains, (list, tuple, set)):
        return all(part in run_name for part in name_contains)
    return True


def _keep_run(run_stub, states, name_contains):
    if states and run_stub.state not in states:
        return False
    if not _matches_name(run_stub.name, name_contains):
        return False
    return True


def update_project_summary(
    project_name,
    *,
    entity=None,
    output_dir=DEFAULT_OUTPUT_DIR,
    states=("finished",),
    name_contains=None,
    max_runs=None,
):
    if wandb is None:
        raise ImportError("wandb is not installed. Run `pip install wandb` first.")

    project_path = resolve_project_path(project_name, entity=entity)
    output_dir = Path(output_dir)
    output_path = project_csv_path(project_path, output_dir=output_dir)
    legacy_path = legacy_project_csv_path(project_path, output_dir=output_dir)
    existing_records = read_existing_records(output_path)
    if not existing_records and legacy_path.exists():
        existing_records = read_existing_records(legacy_path)

    api = wandb.Api()
    records_by_id = {}
    reused_records = 0
    fetched_records = 0
    skipped_non_finished = 0
    scanned_runs = 0

    for run_stub in api.runs(project_path):
        scanned_runs += 1
        if not _keep_run(run_stub, states=states, name_contains=name_contains):
            if states and run_stub.state not in states:
                skipped_non_finished += 1
            continue

        existing_record = existing_records.get(run_stub.id)
        if existing_record and existing_record.get("state") == run_stub.state:
            records_by_id[run_stub.id] = existing_record
            reused_records += 1
        else:
            run = api.run(f"{project_path}/{run_stub.id}")
            records_by_id[run_stub.id] = collect_merged_record(run, project_path)
            fetched_records += 1

        if max_runs is not None and len(records_by_id) >= max_runs:
            break

    records = sorted(records_by_id.values(), key=lambda row: (row.get("name") or "", row.get("id") or ""))
    write_csv(records, output_path)
    return {
        "project_path": project_path,
        "output_path": output_path,
        "saved_runs": len(records),
        "reused_records": reused_records,
        "fetched_records": fetched_records,
        "skipped_non_finished": skipped_non_finished,
        "scanned_runs": scanned_runs,
    }
