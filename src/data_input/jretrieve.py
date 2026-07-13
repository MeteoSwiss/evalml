"""Subprocess wrapper around ``jretrievedwh.py`` for retrieving SwissMetNet
(SMN) surface observations from the MeteoSwiss data warehouse (DWH).

Requires ``jretrievedwh.py`` (resolved via $PATH, $OPR_HOME, or the hardcoded
fallback path) and credentials set as ``JRETRIEVE_CLIENT_ID`` /
``JRETRIEVE_CLIENT_SECRET`` in the environment or in a ``.env`` file in the
working directory.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

BINARY_NAME = "jretrievedwh.py"
HARDCODED_BINARY_PATH = "/oprusers/osm/opr.inn/bin/jretrievedwh.py"
DEFAULT_META_FIELDS: tuple[str, ...] = ("lat", "lon", "elev", "name", "nat_abbr")
DEFAULT_GROUP = "SwissMetNet"
CATALOG_TIME_RANGE_START = datetime(1900, 1, 1)
CATALOG_TIME_RANGE_END = datetime(2100, 12, 31, 23, 59)


class JretrieveError(RuntimeError):
    """Raised when jretrievedwh.py fails or returns malformed output."""


def _resolve_binary() -> str:
    path = shutil.which(BINARY_NAME)
    if path is not None:
        return path
    if os.path.isfile(HARDCODED_BINARY_PATH):
        return HARDCODED_BINARY_PATH
    raise JretrieveError(
        f"{BINARY_NAME} not found on $PATH or at {HARDCODED_BINARY_PATH}."
    )


def _build_env(stage: str) -> dict[str, str]:
    if stage != "prod":
        raise ValueError(f"Only 'prod' stage is supported, got {stage!r}.")
    conf_dir = str(Path(__file__).parents[2])  # project root
    conf_name = ".jretrievedwh-conf.prod.py"
    env = os.environ.copy()
    env["JRETRIEVE_CONF_DIR"] = conf_dir
    env["JRETRIEVE_CONF_NAME"] = conf_name
    return env


def _check_credentials(conf_dir: Path) -> str | None:
    """Return a descriptive error string if jretrieve credentials are missing."""
    client_id = os.environ.get("JRETRIEVE_CLIENT_ID")
    client_secret = os.environ.get("JRETRIEVE_CLIENT_SECRET")

    dotenv_path = conf_dir / ".env"
    dotenv_exists = dotenv_path.is_file()

    if not (client_id and client_secret) and dotenv_exists:
        dotenv: dict[str, str] = {}
        try:
            with open(dotenv_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    dotenv[key.strip()] = value.strip().strip('"').strip("'")
        except OSError:
            pass
        client_id = client_id or dotenv.get("JRETRIEVE_CLIENT_ID")
        client_secret = client_secret or dotenv.get("JRETRIEVE_CLIENT_SECRET")

    if client_id and client_secret:
        return None

    missing = [
        name
        for name, val in (
            ("JRETRIEVE_CLIENT_ID", client_id),
            ("JRETRIEVE_CLIENT_SECRET", client_secret),
        )
        if not val
    ]
    lines = [
        f"Missing jretrieve credentials: {', '.join(missing)}.",
        "Credentials must be supplied in one of two ways:",
        f"  1. Set {' and '.join(missing)} as environment variables.",
        f"  2. Add them to {dotenv_path}",
    ]
    if dotenv_exists:
        lines.append(
            f"     (.env file exists but does not contain {' or '.join(missing)})"
        )
    else:
        lines.append("     (.env file not found — create it with the missing keys)")
    return "\n".join(lines)


def check_prerequisites(stage: str = "prod") -> None:
    """Fail-fast validation that the jretrievedwh environment is usable.

    Checks the binary is reachable and credentials are available. Raises a
    single ``JretrieveError`` listing *all* problems found, so a misconfigured
    environment is reported up front rather than hours into a verification job.
    """
    problems: list[str] = []
    if stage != "prod":
        problems.append(f"Only 'prod' stage is supported, got {stage!r}.")
    try:
        _resolve_binary()
    except JretrieveError as e:
        problems.append(str(e))
    conf_dir = Path(__file__).parents[2]
    conf_path = conf_dir / ".jretrievedwh-conf.prod.py"
    if not conf_path.is_file():
        problems.append(f"jretrieve conf file not found: {conf_path}")
    cred_problem = _check_credentials(conf_dir)
    if cred_problem:
        problems.append(cred_problem)
    if problems:
        raise JretrieveError(
            "jretrievedwh prerequisites not met:\n  - " + "\n  - ".join(problems)
        )


def _fmt_time(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M")


def _stations_to_argv(stations: dict[str, Any]) -> list[str]:
    """Translate a station selection dict to jretrieve CLI args.

    Exactly one of {group, locations, bbox} must be set.
    """
    keys = [k for k in ("group", "locations", "bbox") if stations.get(k) is not None]
    if len(keys) != 1:
        raise ValueError(
            f"stations must specify exactly one of group/locations/bbox, got {keys}"
        )
    key = keys[0]
    val = stations[key]
    if key == "group":
        return ["-a", f"stn_group_id,{val}"]
    if key == "locations":
        if isinstance(val, str):
            val = [v for v in val.split(",") if v]
        if not isinstance(val, Sequence):
            raise ValueError("stations.locations must be a list of nat_abbr strings.")
        return ["-i", "nat_abbr," + ",".join(str(v) for v in val)]
    if key == "bbox":
        if isinstance(val, str):
            val = [v for v in val.split(",") if v]
        if len(val) != 4:
            raise ValueError("stations.bbox must be [minlat, maxlat, minlon, maxlon].")
        return ["-l", ",".join(str(v) for v in val)]
    raise AssertionError("unreachable")


def parse_selection(root: Any) -> tuple[dict[str, Any], str, str]:
    """Parse a truth-root marker into (stations, stage, seq_type).

    Examples (slash-free so they survive ``Path()`` normalisation):
      ``jretrievedwh:SwissMetNet``                       -> group
      ``jretrievedwh:group=SwissMetNet;stage=devt``
      ``jretrievedwh:locations=ARO,KLO``
      ``jretrievedwh:bbox=45.8,47.8,5.9,10.5``
    """
    _, _, rest = str(root).partition(":")
    rest = rest.strip()
    stations: dict[str, Any] = {}
    stage = "prod"
    seq_type = "surface"
    for i, part in enumerate([p for p in rest.split(";") if p]):
        if "=" not in part:
            if i == 0:
                stations["group"] = part
                continue
            raise ValueError(f"Invalid jretrieve selector fragment: {part!r}")
        key, _, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if key in ("group", "locations", "bbox"):
            stations[key] = value
        elif key == "stage":
            stage = value
        elif key == "seq_type":
            seq_type = value
        else:
            raise ValueError(f"Unknown jretrieve selector key: {key!r}")
    if not stations:
        stations = {"group": DEFAULT_GROUP}
    return stations, stage, seq_type


def _run(argv: list[str], env: dict[str, str], timeout_s: int) -> str:
    try:
        proc = subprocess.run(
            argv,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise JretrieveError(
            f"jretrieve timed out after {timeout_s}s: {' '.join(argv)}"
        ) from e
    if proc.returncode != 0:
        raise JretrieveError(
            f"jretrieve exited with {proc.returncode}\nargv: {argv}\n"
            f"stderr: {proc.stderr.strip()}\nstdout (head): {proc.stdout[:500]}"
        )
    if proc.stdout.lstrip().startswith("ERROR"):
        raise JretrieveError(f"jretrieve returned error: {proc.stdout.strip()[:500]}")
    return proc.stdout


def _run_with_retry(argv, env, timeout_s, attempts=3) -> str:
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _run(argv, env=env, timeout_s=timeout_s)
        except JretrieveError as e:
            last_err = e
            if attempt == attempts:
                break
            backoff = 2**attempt
            LOG.warning(
                "jretrieve attempt %d/%d failed (%s); retrying in %ds",
                attempt,
                attempts,
                e,
                backoff,
            )
            time.sleep(backoff)
    assert last_err is not None
    raise last_err


def _parse_csv(csv_text: str) -> pd.DataFrame:
    csv_text = csv_text.strip()
    if not csv_text:
        return pd.DataFrame()
    return pd.read_csv(StringIO(csv_text), sep=";")


def fetch_meta(
    *,
    stations,
    params,
    seq_type="surface",
    stage="prod",
    meta_fields=DEFAULT_META_FIELDS,
    timeout_s=300,
) -> pd.DataFrame:
    """Fetch the station catalog (rows per station x parameter x period) over a
    fixed wide time range so the response is deterministic."""
    if not params:
        raise ValueError("params must be non-empty.")
    argv = [
        _resolve_binary(),
        "-s",
        seq_type,
        "-n",
        ",".join(params),
        "-t",
        f"{_fmt_time(CATALOG_TIME_RANGE_START)},{_fmt_time(CATALOG_TIME_RANGE_END)}",
        "--meta-info",
        ",".join(meta_fields),
        "--format",
        "csv",
        *_stations_to_argv(stations),
    ]
    LOG.info("jretrieve meta: %s", " ".join(argv))
    df = _parse_csv(_run_with_retry(argv, env=_build_env(stage), timeout_s=timeout_s))
    if df.empty:
        raise JretrieveError("jretrieve meta-info returned no rows.")
    return df


def fetch_data(
    *,
    stations,
    params,
    start,
    end,
    increment_minutes=60,
    seq_type="surface",
    stage="prod",
    use_limitation: int | None = None,
    timeout_s=600,
) -> pd.DataFrame:
    """Fetch observation data; columns: station (int), termin (YYYYMMDDhhmmss),
    one column per requested short name."""
    if not params:
        raise ValueError("params must be non-empty.")
    argv = [
        _resolve_binary(),
        "-s",
        seq_type,
        "-n",
        ",".join(params),
        "-t",
        f"{_fmt_time(start)},{_fmt_time(end)},{int(increment_minutes)}",
        "--format",
        "csv",
        *_stations_to_argv(stations),
    ]
    if use_limitation is not None:
        argv += ["--use-limitation", str(use_limitation)]
    LOG.info("jretrieve data: %s", " ".join(argv))
    return _parse_csv(_run_with_retry(argv, env=_build_env(stage), timeout_s=timeout_s))


@dataclass(frozen=True)
class StationCatalog:
    """Stable, nat_abbr-sorted station catalog used as the cell axis."""

    nat_abbr: np.ndarray
    station_id: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    elevation: np.ndarray
    name: np.ndarray

    @property
    def n(self) -> int:
        return len(self.nat_abbr)

    @classmethod
    def from_meta(cls, meta: pd.DataFrame) -> "StationCatalog":
        per_station = (
            meta.sort_values(["nat_abbr", "parameter", "op_since"], kind="stable")
            .drop_duplicates(subset=["station"], keep="first")
            .sort_values("nat_abbr", kind="stable")
            .reset_index(drop=True)
        )
        return cls(
            nat_abbr=per_station["nat_abbr"].to_numpy(dtype=object),
            station_id=per_station["station"].to_numpy(dtype=np.int64),
            latitude=per_station["latitude"].to_numpy(dtype=np.float64),
            longitude=per_station["longitude"].to_numpy(dtype=np.float64),
            elevation=per_station["elev"].to_numpy(dtype=np.float64),
            name=per_station["stn_name"].to_numpy(dtype=object),
        )
