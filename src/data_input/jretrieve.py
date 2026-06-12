"""Subprocess wrapper around ``jretrievedwh.py`` for retrieving SwissMetNet
(SMN) surface observations from the MeteoSwiss data warehouse (DWH).

Ported/adapted from MeteoSwiss/anemoi-plugins-meteoswiss (add-synop-dwh-source).
Requires ``jretrievedwh.py`` on $PATH and $OPR_HOME set with a readable
``.jretrievedwh-conf.<stage>.py`` conf file.
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
from typing import Any, Sequence

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

BINARY_NAME = "jretrievedwh.py"
VALID_STAGES = {"prod", "depl", "devt"}
DEFAULT_META_FIELDS: tuple[str, ...] = ("lat", "lon", "elev", "name", "nat_abbr")
DEFAULT_GROUP = "SwissMetNet"
CATALOG_TIME_RANGE_START = datetime(1900, 1, 1)
CATALOG_TIME_RANGE_END = datetime(2100, 12, 31, 23, 59)


class JretrieveError(RuntimeError):
    """Raised when jretrievedwh.py fails or returns malformed output."""


def _resolve_binary() -> str:
    path = shutil.which(BINARY_NAME)
    if path is None:
        raise JretrieveError(
            f"{BINARY_NAME} not found in $PATH. "
            "Make sure /oprusers/osm/opr.inn/bin (or equivalent) is on your PATH."
        )
    return path


def _build_env(stage: str) -> dict[str, str]:
    if stage not in VALID_STAGES:
        raise ValueError(
            f"Invalid stage {stage!r}. Must be one of {sorted(VALID_STAGES)}."
        )
    opr_home = os.environ.get("OPR_HOME")
    if not opr_home:
        raise JretrieveError("OPR_HOME is not set; cannot locate jretrieve conf file.")
    conf_name = f".jretrievedwh-conf.{stage}.py"
    conf_path = os.path.join(opr_home, conf_name)
    if not os.path.isfile(conf_path):
        raise JretrieveError(f"jretrieve conf file not found: {conf_path}")
    if not os.access(conf_path, os.R_OK):
        raise JretrieveError(f"jretrieve conf file not readable: {conf_path}")
    env = os.environ.copy()
    env["JRETRIEVE_CONF_DIR"] = opr_home
    env["JRETRIEVE_CONF_NAME"] = conf_name
    return env


def check_prerequisites(stage: str = "prod") -> None:
    """Fail-fast validation that the jretrievedwh environment is usable.

    Verifies the CLI is on $PATH, $OPR_HOME is set, and the conf file for
    ``stage`` exists and is readable. Raises a single ``JretrieveError`` listing
    *all* problems found, so a misconfigured environment is reported up front
    (e.g. at workflow launch) instead of hours later inside the verification job.
    """
    problems: list[str] = []
    if shutil.which(BINARY_NAME) is None:
        problems.append(
            f"{BINARY_NAME} not found in $PATH "
            "(e.g. add /oprusers/osm/opr.inn/bin to $PATH)."
        )
    opr_home = os.environ.get("OPR_HOME")
    if not opr_home:
        problems.append("$OPR_HOME is not set.")
    elif stage not in VALID_STAGES:
        problems.append(
            f"Invalid stage {stage!r}; must be one of {sorted(VALID_STAGES)}."
        )
    else:
        conf_path = os.path.join(opr_home, f".jretrievedwh-conf.{stage}.py")
        if not os.path.isfile(conf_path):
            problems.append(f"jretrieve conf file not found: {conf_path}")
        elif not os.access(conf_path, os.R_OK):
            problems.append(f"jretrieve conf file not readable: {conf_path}")
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
        return ["-a", f"stn_group,{val}"]
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
