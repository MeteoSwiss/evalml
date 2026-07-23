"""Capture real jretrieve output and write it to fixture CSV files.

Run once locally with DWH credentials available to refresh the fixtures used by
the mock_jretrieve pytest fixture:

    python tests/integration/capture_jretrieve_fixtures.py

The script intercepts the raw CSV strings returned by _run_with_retry and
writes them to tests/integration/fixtures/jretrieve/{meta,data}.csv.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import sys

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from data_input import jretrieve as jr
import data_input

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "jretrieve"
ROOT = "jretrievedwh:1,2"
REFTIME = datetime(2024, 8, 1, 0, 0)
STEPS = [0, 6, 12]
PARAMS = ["T_2M", "SP_10M", "TOT_PREC6"]

captured: dict[str, str] = {}
_original_run_with_retry = jr._run_with_retry


def _capturing_run_with_retry(argv, env, timeout_s, attempts=3):
    result = _original_run_with_retry(
        argv, env=env, timeout_s=timeout_s, attempts=attempts
    )
    call_type = "meta" if "--meta-info" in argv else "data"
    captured[call_type] = result
    return result


def main():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    with patch.object(jr, "_run_with_retry", _capturing_run_with_retry):
        data_input.load_obs_data_from_jretrieve(ROOT, REFTIME, STEPS, PARAMS)

    for call_type, csv_text in captured.items():
        out = FIXTURE_DIR / f"{call_type}.csv"
        out.write_text(csv_text)
        print(f"written: {out} ({len(csv_text.splitlines())} lines)")

    if len(captured) < 2:
        missing = {"meta", "data"} - captured.keys()
        print(f"WARNING: missing captures for: {missing}")


if __name__ == "__main__":
    main()
