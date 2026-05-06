"""System-performance diagnostics for inference jobs."""

import logging
import re
from datetime import datetime
from pathlib import Path

LOG = logging.getLogger(__name__)

_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_JOB_ID = re.compile(r"srun: job (\d+) queued")
_CHECKPOINT_SIZE = re.compile(r"Checkpoint size: ([\d.]+) GiB")
_N_STEPS = re.compile(r"Forecasting (\d+) steps")
_STEP_TIME = re.compile(r"Forecast\. Model call \d+:.+?: (\d+) seconds\.")

# Human-readable names for wide-format columns used in the dashboard
SYSMETRICS_COLS = {
    "wall_time_s": "Wall Time (s)",
    "gpu_hours": "GPU Hours",
    "mean_step_time_s": "Mean Step Time (s)",
    "max_step_time_s": "Max Step Time (s)",
    "checkpoint_size_gib": "Checkpoint Size (GiB)",
    "n_steps": "No. Steps",
}


def parse_single_log(log_path: str) -> dict:
    """Return raw metric values extracted from one inference log file."""
    job_id = None
    first_ts = last_ts = None
    checkpoint_gib = None
    n_steps = None
    step_times: list[int] = []

    with open(log_path) as fh:
        for line in fh:
            if job_id is None:
                m = _JOB_ID.search(line)
                if m:
                    job_id = m.group(1)

            m = _TIMESTAMP.match(line)
            if m:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

            if checkpoint_gib is None:
                m = _CHECKPOINT_SIZE.search(line)
                if m:
                    checkpoint_gib = float(m.group(1))

            if n_steps is None:
                m = _N_STEPS.search(line)
                if m:
                    n_steps = int(m.group(1))

            m = _STEP_TIME.search(line)
            if m:
                step_times.append(int(m.group(1)))

    wall_time_s = (
        (last_ts - first_ts).total_seconds()
        if first_ts is not None and last_ts is not None
        else None
    )
    return {
        "job_id": job_id,
        "wall_time_s": wall_time_s,
        "n_steps": n_steps if n_steps is not None else len(step_times),
        "mean_step_time_s": (
            round(sum(step_times) / len(step_times), 2) if step_times else None
        ),
        "max_step_time_s": max(step_times) if step_times else None,
        "checkpoint_size_gib": checkpoint_gib,
    }


def parse_logs(
    log_files: list[str],
    label_map: dict[str, str],
    gpu_map: dict[str, int],
    log_dir: str,
) -> list[dict]:
    """Parse inference log files and return one record per (run, init_time).

    Parameters
    ----------
    log_files : paths to .log files to parse.
    label_map : {run_id: human-readable label} — supplied by Snakemake rule params.
    gpu_map   : {run_id: n_gpu} — GPU count used for each run.
    log_dir   : root of the inference_execute logs directory; used to derive run_id.
    """
    log_dir_path = Path(log_dir)
    records: list[dict] = []

    for log_file in log_files:
        log_path = Path(log_file)
        if not log_path.exists():
            LOG.warning("Log file not found, skipping: %s", log_file)
            continue

        # Derive run_id and init_time from the path.
        # Relative path structure: "{run_id}-{init_time}.log"
        # init_time is always 12 digits (YYYYMMDDHHM).
        try:
            stem = str(log_path.relative_to(log_dir_path).with_suffix(""))
            init_time_str = stem[-12:]
            run_id = stem[:-13]  # strip trailing "-YYYYMMDDHHM"
        except Exception:
            LOG.warning("Cannot derive run_id from path, skipping: %s", log_file)
            continue

        label = label_map.get(run_id, run_id)
        n_gpu = int(gpu_map.get(run_id, 1))

        try:
            raw = parse_single_log(str(log_path))
        except Exception as exc:
            LOG.warning("Failed to parse %s: %s", log_file, exc)
            continue

        wall_s = raw.get("wall_time_s")
        gpu_hours = round(wall_s / 3600 * n_gpu, 4) if wall_s is not None else None

        try:
            init_iso = datetime.strptime(init_time_str, "%Y%m%d%H%M").isoformat()
        except ValueError:
            init_iso = init_time_str

        records.append(
            {
                "source": label,
                "run_id": run_id,
                "init_time": init_iso,
                "n_gpu": n_gpu,
                "gpu_hours": gpu_hours,
                **raw,
            }
        )

    LOG.info("Parsed %d log files → %d records", len(log_files), len(records))
    return records


def melt_for_dashboard(records: list[dict]) -> tuple[str, list[str]]:
    """Convert wide-format system metrics records to long format for Vega-Lite.

    Returns (json_string, sorted_source_list).
    """
    import json

    long_records = []
    for r in records:
        base = {k: r.get(k) for k in ("source", "init_time", "n_gpu", "job_id")}
        for col, label in SYSMETRICS_COLS.items():
            if r.get(col) is not None:
                long_records.append({**base, "metric": label, "value": r[col]})

    sources = sorted({r["source"] for r in records})
    return json.dumps(long_records), sources
