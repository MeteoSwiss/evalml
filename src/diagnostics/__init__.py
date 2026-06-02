"""System-performance diagnostics for inference jobs."""

import logging
import math
from datetime import datetime
from pathlib import Path

LOG = logging.getLogger(__name__)

# Columns exposed as distribution metrics in the dashboard
SYSMETRICS_COLS = {
    "wall_time_s": "Wall Time (s)",
    "gpu_hours": "GPU Hours",
    "max_rss_mb": "Peak CPU Memory (MB)",
    "gpu_util_mean": "Mean GPU Util (%)",
    "gpu_util_max": "Peak GPU Util (%)",
    "gpu_mem_used_mean": "Mean GPU Memory (MiB)",
    "gpu_mem_used_max": "Peak GPU Memory (MiB)",
    "gpu_power_mean": "Mean GPU Power (W)",
}


def _parse_elapsed(s: str) -> float | None:
    """Parse sacct elapsed 'D-HH:MM:SS' or 'HH:MM:SS' → seconds."""
    s = s.strip()
    try:
        if "-" in s:
            days, rest = s.split("-", 1)
            h, m, sec = rest.split(":")
            return int(days) * 86400 + int(h) * 3600 + int(m) * 60 + int(sec)
        h, m, sec = s.split(":")
        return int(h) * 3600 + int(m) * 60 + int(sec)
    except Exception:
        return None


def _parse_memory(s: str) -> float | None:
    """Parse sacct memory string ('2048K', '1.5M', '0.5G') → MB."""
    s = s.strip()
    if not s or s == "0":
        return None
    try:
        if s.endswith("K"):
            return float(s[:-1]) / 1024
        if s.endswith("M"):
            return float(s[:-1])
        if s.endswith("G"):
            return float(s[:-1]) * 1024
        return float(s) / (1024 * 1024)
    except ValueError:
        return None


def parse_sacct_log(log_path: str) -> dict:
    """Parse slurm_metrics.log (sacct --parsable2 output) → metric dict.

    Extracts wall_time_s from the parent job record (no dot in JobID) and
    max_rss_mb as the maximum MaxRSS across all step records.
    """
    path = Path(log_path)
    if not path.exists() or path.stat().st_size == 0:
        return {}

    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        return {}

    headers = [h.strip() for h in lines[0].split("|")]
    rows = []
    for line in lines[1:]:
        fields = [f.strip() for f in line.split("|")]
        if len(fields) >= len(headers):
            rows.append(dict(zip(headers, fields)))

    if not rows:
        return {}

    result: dict = {}

    # Elapsed from the parent job record (no dot in JobID, no 'batch'/'extern' suffix)
    for row in rows:
        job_id = row.get("JobID", "")
        if "." not in job_id and not job_id.endswith(("batch", "extern")):
            val = _parse_elapsed(row.get("Elapsed", ""))
            if val is not None:
                result["wall_time_s"] = val
            break

    # MaxRSS: take the maximum across all step rows (parent job entry is 0)
    rss_vals = [
        v
        for row in rows
        if (v := _parse_memory(row.get("MaxRSS", ""))) is not None and v > 0
    ]
    if rss_vals:
        result["max_rss_mb"] = round(max(rss_vals), 1)

    return result


def _parse_dmon_headers(header_line: str) -> dict[str, int]:
    """Return {col_name: data_row_index} from a '# gpu …' nvidia-smi dmon header.

    The first token after '#' is 'gpu' — a row-type label, not a data column —
    so actual data indices are offset by -1 relative to the token positions.
    """
    tokens = header_line.lstrip("#").split()
    return {name: i - 1 for i, name in enumerate(tokens) if i > 0}


def parse_gpu_metrics_log(log_path: str) -> dict:
    """Parse gpu_metrics.log (nvidia-smi dmon -o DT) → GPU utilisation/memory/power dict."""
    path = Path(log_path)
    if not path.exists() or path.stat().st_size == 0:
        return {}

    col_idx: dict[str, int] = {}
    data_rows: list[list[float]] = []

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            tokens = line.lstrip("#").split()
            # Column-name header contains 'sm' or 'Idx'; unit header contains '%' or 'W'
            if "sm" in tokens or "Idx" in tokens:
                col_idx = _parse_dmon_headers(line)
            continue
        fields = line.split()
        if len(fields) < 5 or not col_idx:
            continue
        nums: list[float] = []
        for f in fields:
            try:
                nums.append(float(f))
            except ValueError:
                nums.append(float("nan"))
        data_rows.append(nums)

    if not data_rows or not col_idx:
        return {}

    def _vals(col: str) -> list[float]:
        idx = col_idx.get(col)
        if idx is None:
            return []
        return [
            r[idx]
            for r in data_rows
            if idx < len(r) and math.isfinite(r[idx])
        ]

    result: dict = {}

    sm = _vals("sm")
    if sm:
        result["gpu_util_mean"] = round(sum(sm) / len(sm), 1)
        result["gpu_util_max"] = round(max(sm), 1)

    fb = _vals("fb")
    if fb:
        result["gpu_mem_used_mean"] = round(sum(fb) / len(fb), 1)
        result["gpu_mem_used_max"] = round(max(fb), 1)

    pwr = _vals("pwr")
    if pwr:
        result["gpu_power_mean"] = round(sum(pwr) / len(pwr), 1)

    return result


def parse_run_metrics(workdir: str) -> dict:
    """Read sacct and GPU metric files from a run's workdir and merge them."""
    wd = Path(workdir)
    result: dict = {}

    result.update(parse_sacct_log(str(wd / "slurm_metrics.log")))
    result.update(parse_gpu_metrics_log(str(wd / "gpu_metrics.log")))

    job_id_file = wd / "slurm_job_id"
    if job_id_file.exists():
        result["job_id"] = job_id_file.read_text().strip()

    return result


def parse_logs(
    run_info: list[dict],
    label_map: dict[str, str],
    gpu_map: dict[str, int],
) -> list[dict]:
    """Parse Slurm metric files and return one record per (run, init_time).

    Parameters
    ----------
    run_info  : list of dicts with keys 'workdir', 'run_id', 'init_time'.
    label_map : {run_id: human-readable label}.
    gpu_map   : {run_id: n_gpu}.
    """
    records: list[dict] = []

    for spec in run_info:
        workdir = spec["workdir"]
        run_id = spec["run_id"]
        init_time_str = spec["init_time"]

        wd = Path(workdir)
        if not wd.exists():
            LOG.warning("Workdir not found, skipping: %s", workdir)
            continue

        label = label_map.get(run_id, run_id)
        n_gpu = int(gpu_map.get(run_id, 1))
        model_type = run_id.split("-")[0]

        try:
            raw = parse_run_metrics(workdir)
        except Exception as exc:
            LOG.warning("Failed to parse metrics for %s: %s", workdir, exc)
            continue

        if not raw:
            LOG.warning("No metrics found for %s, skipping", workdir)
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
                "model_type": model_type,
                "init_time": init_iso,
                "n_gpu": n_gpu,
                "gpu_hours": gpu_hours,
                **raw,
            }
        )

    LOG.info("Parsed %d run specs → %d records", len(run_info), len(records))
    return records


def melt_for_dashboard(records: list[dict]) -> tuple[str, list[str], list[str]]:
    """Convert wide-format system metrics records to long format for Vega-Lite.

    Returns (json_string, sorted_source_list, sorted_model_type_list).
    """
    import json

    long_records = []
    for r in records:
        base = {
            k: r.get(k)
            for k in ("source", "model_type", "init_time", "n_gpu", "job_id")
        }
        for col, label in SYSMETRICS_COLS.items():
            if r.get(col) is not None:
                long_records.append({**base, "metric": label, "value": r[col]})

    sources = sorted({r["source"] for r in records})
    model_types = sorted({r.get("model_type", "unknown") for r in records})
    return json.dumps(long_records), sources, model_types
