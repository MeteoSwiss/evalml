import csv
import glob
import math
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent / "configs" / "sal_small.yaml"

# This test reads the baseline archive and the gridded truth live from
# /store_new (no inference / fixture: the ICON-CH2-CTRL baseline is static
# archive data). Read the input paths from the config so the test and the config
# can never disagree, and skip cleanly when they are absent (e.g. off balfrin)
# rather than failing deep inside snakemake -- same pattern as
# test_scoremaps_small.py.
_CFG = yaml.safe_load(CONFIG.read_text())
BASELINE_ROOT = Path(
    next(r["baseline"] for r in _CFG["runs"] if "baseline" in r)["root"]
)
TRUTH_ROOT = Path(_CFG["truth"]["root"])
PARAMS = _CFG["experiment"]["sal"]["params"]
LEADTIMES = _CFG["experiment"]["sal"]["leadtimes"]
# Taken from the config rather than hardcoded: the config points at a tree of its
# own so the exactly-one-CSV assertion below cannot be tripped by unrelated
# experiments sharing output/.
OUT_ROOT = PROJECT_ROOT / _CFG["locations"]["output_root"]

DATETIME_FMT = "%Y%m%d%H%M"
EXPECTED_REFTIMES = [
    t.strftime(DATETIME_FMT)
    for t in pd.date_range(
        _CFG["dates"]["start"], _CFG["dates"]["end"], freq=_CFG["dates"]["frequency"]
    )
]
EXPECTED_COLUMNS = ["reftime", "S", "A", "L", "fcst_mean", "truth_mean"]

# verification.sal.sal_raster() over GRID_EXTENT at 0.01 lat x 0.0145 lon.
# Pinned here so a silent change to the scoring raster fails this test too.
EXPECTED_GRID_CELLS = "851x1311"

# --- Tolerances -------------------------------------------------------------
# The amplitude identity below is exact in practice (both sides are float64
# reductions of the *same* arrays), so this is a formatting/round-trip margin,
# not a physical one.
AMPLITUDE_RTOL = 1e-12
# The truth cross-check compares the same accumulation window reached via
# different (reftime, leadtime) pairs. Observed agreement is ~5e-12 relative
# but not bit-exact -- the zarr is accumulated chunk-wise, so the summation
# order differs between paths. 1e-9 leaves headroom without admitting any
# plausible wiring error.
TRUTH_RTOL = 1e-9


def _read_sal_csv(path: Path) -> tuple[list[str], list[dict]]:
    """Split a SAL CSV into its commented metadata header and its data rows."""
    lines = path.read_text().splitlines()
    header = [line[2:] for line in lines if line.startswith("# ")]
    rows = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    return header, rows


def _find_sal_csvs(param: str, leadtime: int) -> list[str]:
    return sorted(
        glob.glob(str(OUT_ROOT / f"data/baselines/*/sal/{param}_{leadtime}_*.csv"))
    )


def _check_header(header: list[str], param: str, leadtime: int, path: str) -> None:
    joined = "\n".join(header)
    for expected in (
        f"param: {param}",
        f"step_h: {leadtime}",
        f"grid_cells: {EXPECTED_GRID_CELLS}",
        f"n_init: {len(EXPECTED_REFTIMES)}",
        str(BASELINE_ROOT),
    ):
        assert expected in joined, (
            f"{path}: metadata header is missing {expected!r}. Header was:\n{joined}"
        )


def _check_rows(rows: list[dict], param: str, leadtime: int, path: str) -> None:
    """Reference-free per-row checks on one SAL CSV.

    Exact SAL numerics are unit-tested (tests/unit/test_sal.py), so this checks
    end-to-end wiring and internal consistency rather than re-deriving values.
    """
    assert [r["reftime"] for r in rows] == EXPECTED_REFTIMES, (
        f"{path}: rows do not match the config's init times "
        f"(got {[r['reftime'] for r in rows]}, expected {EXPECTED_REFTIMES})"
    )

    n_scored = 0
    for row in rows:
        where = f"{path} {row['reftime']}"
        s, a, ell = (float(row[k]) for k in ("S", "A", "L"))
        fcst_mean, truth_mean = float(row["fcst_mean"]), float(row["truth_mean"])

        # Domain-mean precipitation is finite and non-negative by construction;
        # NaNs in the native fields are remapped to 0 (= no precipitation).
        for name, value in (("fcst_mean", fcst_mean), ("truth_mean", truth_mean)):
            assert math.isfinite(value) and value >= 0, (
                f"{where}: {name}={value} is not a finite, non-negative mean"
            )

        # A is the normalised difference of the domain means and does NOT depend
        # on object detection, so it must be finite whenever both fields are wet
        # -- and it must equal the identity recomputed from this row's own mean
        # columns. This pins A to the same pair of fields the means were taken
        # from, so e.g. scoring a different lead time or a different slice than
        # the one reported cannot pass unnoticed.
        if fcst_mean > 0 and truth_mean > 0:
            assert math.isfinite(a), (
                f"{where}: A is NaN although both fields are wet "
                f"(fcst_mean={fcst_mean}, truth_mean={truth_mean})"
            )
            identity = (fcst_mean - truth_mean) / (0.5 * (fcst_mean + truth_mean))
            assert math.isclose(a, identity, rel_tol=AMPLITUDE_RTOL, abs_tol=1e-12), (
                f"{where}: A={a} != (fcst_mean - truth_mean) / "
                f"(0.5 * (fcst_mean + truth_mean))={identity}"
            )

        # Bounds hold by construction (Wernli et al. 2008); NaN marks a window
        # with no detectable objects and is a documented outcome, not a failure.
        for name, value, low, high in (
            ("S", s, -2.0, 2.0),
            ("A", a, -2.0, 2.0),
            ("L", ell, 0.0, 2.0),
        ):
            if math.isfinite(value):
                assert low <= value <= high, (
                    f"{where}: {name}={value} outside [{low}, {high}]"
                )

        if math.isfinite(s) and math.isfinite(ell):
            n_scored += 1

    # The all-NaN guard: object detection actually ran for at least one init.
    # Without this, a run that silently scored nothing but wrote a well-formed
    # CSV would pass every check above. If a future date/lead time is genuinely
    # dry across the whole domain, pick a wetter one rather than relaxing this.
    assert n_scored > 0, (
        f"{path}: S/L are NaN for every init -- no objects were detected in any "
        f"window, so this file carries no SAL information. Either the forecast "
        f"or the truth field is empty, or the period is dry."
    )


def _check_truth_depends_only_on_valid_time(
    rows_by_key: dict[tuple[str, int], list[dict]],
) -> None:
    """The truth column must be a function of valid time alone.

    For a P-hour accumulated param the truth window is [valid - P, valid] with
    valid = reftime + leadtime, so any two (reftime, leadtime) pairs reaching
    the same valid time must see the same truth field. This pins the
    reftime+step -> valid-window mapping -- the classic off-by-one in
    verification wiring -- using nothing but the run's own output, no blessed
    reference.
    """
    groups: dict[tuple[str, datetime], list[tuple[int, str, float]]] = {}
    for (param, leadtime), rows in rows_by_key.items():
        for row in rows:
            valid = datetime.strptime(row["reftime"], DATETIME_FMT) + timedelta(
                hours=leadtime
            )
            groups.setdefault((param, valid), []).append(
                (leadtime, row["reftime"], float(row["truth_mean"]))
            )

    shared = {key: entries for key, entries in groups.items() if len(entries) > 1}
    # Guard against the check silently becoming vacuous if the config's dates or
    # lead times stop overlapping.
    assert shared, (
        "no valid time is reached by more than one (reftime, leadtime) pair, so "
        "this cross-check would be vacuous: dates and sal.leadtimes in "
        f"{CONFIG.name} must overlap (reftimes={EXPECTED_REFTIMES}, "
        f"leadtimes={LEADTIMES})"
    )

    for (param, valid), entries in sorted(shared.items()):
        ref_leadtime, ref_reftime, ref_value = entries[0]
        for leadtime, reftime, value in entries[1:]:
            assert math.isclose(value, ref_value, rel_tol=TRUTH_RTOL), (
                f"{param}: truth_mean for valid time {valid:%Y-%m-%d %H}Z differs "
                f"depending on how it was reached: {value} via (init {reftime}, "
                f"+{leadtime}h) vs {ref_value} via (init {ref_reftime}, "
                f"+{ref_leadtime}h). The reftime+step -> accumulation-window "
                f"mapping is inconsistent."
            )


@pytest.mark.longtest
def test_sal_small():
    """Run the experiment workflow on the minimal SAL config and check that the
    scores it produces are *consistent*, not merely present.

    Baseline-only (ICON-CH2-CTRL), no inference -- so no GPU, MLflow or DWH is
    needed, only access to /store_new (the ICON-CH2-EPS baseline archive and the
    KENDA-CH1 truth zarr). Marked ``longtest`` so it is skipped by default and on
    GitHub Actions, and runs on the CSCS balfrin runner via the ``longtest``
    pipeline (``pytest tests/integration -m longtest``).

    Beyond "the command exits 0 and a CSV exists", this asserts:
      * the CSV carries the documented metadata header, column set, scoring
        raster size and one row per configured init time;
      * domain-mean precipitation is finite and non-negative;
      * the amplitude component equals the identity recomputed from the row's
        own mean columns, which ties A to the fields the means were taken from;
      * S/A/L respect their Wernli et al. (2008) bounds;
      * S/L are not NaN for every init (the all-NaN guard); and
      * the truth column depends only on valid time, cross-checked across two
        independent output files.

    Scope: this covers ``verification_sal_baseline``. The run path
    (``verification_sal``) depends on ``inference_execute``, so exercising it
    needs either a GPU or the inference-replay fixture; the only untested delta
    is the ``run_root/<reftime>/[grib]`` resolution in verification_sal.py.
    """
    # Skip cleanly when the baseline archive / gridded truth are not reachable
    # (keyed off the paths in the config), instead of failing inside snakemake.
    missing = [p for p in (BASELINE_ROOT, TRUTH_ROOT) if not p.exists()]
    if missing:
        pytest.skip(
            "SAL inputs not accessible (need /store_new access, e.g. on "
            "balfrin): " + ", ".join(str(p) for p in missing)
        )

    result = subprocess.run(
        ["evalml", "experiment", str(CONFIG)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"evalml experiment failed (exit {result.returncode}).\n"
        f"stdout tail:\n{result.stdout[-2000:]}\n"
        f"stderr tail:\n{result.stderr[-2000:]}"
    )

    rows_by_key: dict[tuple[str, int], list[dict]] = {}
    for param in PARAMS:
        assert re.fullmatch(r"TOT_PREC\d*", param), (
            f"{CONFIG.name}: sal.params entry {param!r} is not a precipitation param"
        )
        for leadtime in LEADTIMES:
            csvs = _find_sal_csvs(param, leadtime)
            assert len(csvs) == 1, (
                f"expected exactly one SAL CSV for {param} at +{leadtime}h under "
                f"{OUT_ROOT}/data/baselines/*/sal/, got {csvs}"
            )
            path = csvs[0]
            header, rows = _read_sal_csv(Path(path))

            assert rows, f"{path}: no data rows"
            assert list(rows[0]) == EXPECTED_COLUMNS, (
                f"{path}: unexpected columns {list(rows[0])} "
                f"(expected {EXPECTED_COLUMNS})"
            )
            _check_header(header, param, leadtime, path)
            _check_rows(rows, param, leadtime, path)
            rows_by_key[(param, leadtime)] = rows

    _check_truth_depends_only_on_valid_time(rows_by_key)
