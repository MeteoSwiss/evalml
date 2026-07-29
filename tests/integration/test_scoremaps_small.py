import glob
import os
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent / "configs" / "scoremaps_small.yaml"

# This test reads the baseline archive and gridded truth live from /store_new
# (no inference / fixture: the ICON-CH2-CTRL baseline is static archive data).
# Read the input paths from the config so the test and config never disagree,
# and skip cleanly when they are absent (e.g. off balfrin) rather than failing
# deep inside snakemake — the analog of the meteogram test's fixture skip.
_CFG = yaml.safe_load(CONFIG.read_text())
BASELINE_ROOT = Path(next(r["baseline"] for r in _CFG["runs"] if "baseline" in r)["root"])
TRUTH_ROOT = Path(_CFG["truth"]["root"])

# The scoremaps block plots one map per (param, score, region, season, init_hour,
# leadtime). The fixture requests both an instantaneous param (T_2M) and a
# de-accumulated one (TOT_PREC6) at BIAS / switzerland / all / all / 6 h, so one
# PNG per parameter is expected.
EXPECTED_PARAMS = ["T_2M", "TOT_PREC6"]
LEADTIME = 6
STATS = ("BIAS", "RMSE", "MAE", "STDE")

# --- PNG "is this a real map?" discriminator -------------------------------
# When the plotted field is all-NaN, plot_scoremaps.mo.py still writes a valid,
# non-empty PNG: a grey (#cccccc) "No data" placeholder with black coastlines.
# So "a non-empty PNG exists" does NOT prove the map is meaningful. A real score
# map is filled with colormap shading (plus a colourbar); the placeholder is not.
# We therefore require a minimum fraction of *saturated* (colourful) pixels:
# grey/black/white all have HSV saturation ~0, colormap colours do not.
# The thresholds are deliberately loose — they only separate "real map" from
# "grey placeholder", not fine plot details. Confirmed against the first blessed
# CSCS run: the grey placeholder measures ~0.0, while the real maps measured
# 0.319 (T_2M) and 0.030 (TOT_PREC6, whose near-zero-bias field is mostly the
# unsaturated white centre of the diverging colormap). 0.01 sits ~3x below the
# tighter of the two and far above the placeholder.
SAT_THRESHOLD = 0.2  # HSV saturation above which a pixel counts as "coloured"
MIN_COLOURED_FRACTION = 0.01  # a real map has well above this; a placeholder ~0

# --- Golden reference (tolerance comparison, NOT bit-reproducibility) -------
# Summary of the unstratified (season="all", init_hour=-999) bucket — the same
# bucket that is plotted — captured from a blessed CSCS run. To (re-)bless: run
# the test on balfrin with EVALML_UPDATE_REFERENCE=1, eyeball the printed block
# (and the PNGs) once, paste it here, and commit. Compared with rtol=1e-3 so
# library/platform float noise does not cause false failures; re-bless only when
# the fixture inputs or the science deliberately change. Until a param is
# blessed (value is None) the reference comparison is skipped with a warning —
# the N_total>0 guard and the PNG not-grey check still run.
# Blessed 2026-07-24 from a balfrin CSCS run (EVALML_UPDATE_REFERENCE=1).
REFERENCE: dict[str, dict | None] = {
    "T_2M": {
        "N_total": 2295960,
        "BIAS": {"mean": -0.162372, "min": -9.21579, "max": 6.83401},
        "RMSE": {"mean": 0.902101, "min": 0.0011017, "max": 14.4667},
        "MAE": {"mean": 0.798512, "min": 0.000946045, "max": 11.878},
        "STDE": {"mean": 0.584206, "min": 7.62939e-05, "max": 11.878},
    },
    "TOT_PREC6": {
        "N_total": 2295960,
        "BIAS": {"mean": -0.0399991, "min": -50.6713, "max": 10.1969},
        "RMSE": {"mean": 0.112297, "min": 0.0, "max": 51.1054},
        "MAE": {"mean": 0.0966864, "min": 0.0, "max": 50.6713},
        "STDE": {"mean": 0.0617382, "min": 0.0, "max": 11.3508},
    },
}
RTOL = 1e-3
ATOL = 1e-4  # absorbs near-zero stats (e.g. a BIAS mean close to 0) where rtol bites


def _coloured_fraction(png_path: Path) -> float:
    """Fraction of pixels with HSV saturation above ``SAT_THRESHOLD``.

    ~0 for the grey "No data" placeholder; clearly positive for a real,
    colormap-shaded score map.
    """
    with Image.open(png_path) as img:
        sat = np.asarray(img.convert("HSV"), dtype=np.float32)[..., 1] / 255.0
    return float((sat > SAT_THRESHOLD).mean())


def _check_field_invariants(ds: xr.Dataset, param: str) -> None:
    """Reference-free sanity checks on the full unstratified-bucket fields.

    These must run on the per-grid-cell fields, NOT on spatial averages:
    ``RMSE**2 == STDE**2 + BIAS**2`` is a pointwise identity (by construction
    from the running accumulators) that does not survive spatial averaging, and
    ``RMSE >= MAE`` likewise holds cell-by-cell. They catch a metric-wiring bug
    that would still render as a plausible, colourful — but wrong — map.
    """
    agg = ds.sel(season="all", init_hour=-999)
    n = agg[f"{param}.N"].values
    rmse = agg[f"{param}.RMSE"].values.astype(np.float64)
    mae = agg[f"{param}.MAE"].values.astype(np.float64)
    stde = agg[f"{param}.STDE"].values.astype(np.float64)
    bias = agg[f"{param}.BIAS"].values.astype(np.float64)

    valid = np.isfinite(rmse) & (n > 0)
    assert valid.any(), f"{param}: no valid grid cells in the unstratified bucket"
    r, m, s, b = rmse[valid], mae[valid], stde[valid], bias[valid]

    assert (r >= 0).all() and (m >= 0).all() and (s >= 0).all(), (
        f"{param}: negative RMSE/MAE/STDE at some grid cell"
    )
    assert (r >= m - ATOL).all(), f"{param}: RMSE < MAE at some grid cell"
    # Pointwise identity. atol absorbs float32 storage noise and the STDE
    # variance clamp (STDE is max(var, 0)**0.5) at cells where the error is
    # effectively constant, i.e. RMSE**2 == BIAS**2.
    np.testing.assert_allclose(
        r**2,
        s**2 + b**2,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"{param}: RMSE**2 != STDE**2 + BIAS**2 (pointwise identity broken)",
    )


def _summarise_bucket(ds: xr.Dataset, param: str) -> dict:
    """Spatial mean/min/max of each stat plus total N in the unstratified
    (season="all", init_hour=-999) bucket."""
    agg = ds.sel(season="all", init_hour=-999)
    n = agg[f"{param}.N"].values
    summary: dict = {"N_total": int(np.nansum(n))}
    for stat in STATS:
        vals = agg[f"{param}.{stat}"].values
        finite = vals[np.isfinite(vals)]
        summary[stat] = {
            "mean": float(finite.mean()),
            "min": float(finite.min()),
            "max": float(finite.max()),
        }
    return summary


def _check_against_reference(param: str, summary: dict) -> None:
    ref = REFERENCE.get(param)
    if ref is None:
        warnings.warn(
            f"No blessed reference for {param}; skipping the summary comparison. "
            "Run with EVALML_UPDATE_REFERENCE=1 to record one.",
            stacklevel=2,
        )
        return
    assert summary["N_total"] == ref["N_total"], (
        f"{param}: accumulated sample count changed "
        f"({summary['N_total']} vs reference {ref['N_total']})"
    )
    for stat in STATS:
        for key in ("mean", "min", "max"):
            np.testing.assert_allclose(
                summary[stat][key],
                ref[stat][key],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"{param}.{stat}.{key} drifted from the blessed reference",
            )


def _round(value: float) -> float:
    """Round to 6 significant figures for a readable pasteable reference block."""
    return float(f"{value:.6g}")


def _print_reference_block(recorded: dict[str, dict]) -> None:
    print("\n# ---- paste into REFERENCE (thresholds/tolerance already set) ----")
    for param, summary in recorded.items():
        parts = [f'"N_total": {summary["N_total"]}']
        for stat in STATS:
            s = summary[stat]
            parts.append(
                f'"{stat}": {{"mean": {_round(s["mean"])}, '
                f'"min": {_round(s["min"])}, "max": {_round(s["max"])}}}'
            )
        print(f'    "{param}": {{{", ".join(parts)}}},')
    print("# ---- end reference block ----")


@pytest.mark.longtest
def test_scoremaps():
    """Run the experiment workflow on the minimal scoremaps config and check that
    the score maps it produces are *meaningful*, not merely present.

    Baseline-only (ICON-CH2-CTRL), no inference — so no GPU, MLflow, or DWH is
    needed, only access to /store_new (the ICON-CH2-EPS baseline archive and the
    KENDA-CH1 truth zarr). Marked ``longtest`` so it is skipped by default (and on
    GitHub Actions, which runs ``pytest tests/unit`` only) and runs on the CSCS
    balfrin runner, which invokes ``pytest tests/integration -m longtest``
    (see ci/cscs.yml).

    Beyond "the command exits 0 and a PNG exists", this asserts, per param:
      * the PNG is a real colormap-shaded map, not the grey "No data" placeholder;
      * the score-map NetCDF actually accumulated samples (N_total > 0), which is
        the check that catches a silent all-NaN run;
      * the per-grid-cell fields obey the metric invariants RMSE >= MAE and the
        RMSE**2 == STDE**2 + BIAS**2 identity (reference-free); and
      * a summary of the plotted (unstratified) bucket matches a blessed golden
        reference within tolerance (once recorded — see REFERENCE / record mode).

    Exact per-pixel numerics are already unit-tested (tests/unit/test_data_input.py
    for de-accumulation / step-0 IC, tests/unit/test_verification.py for the metric
    maths), so this test checks end-to-end wiring + sane, stable output rather than
    re-deriving values.
    """
    # Skip cleanly when the baseline archive / gridded truth are not reachable
    # (keyed off the paths in the config), instead of failing inside snakemake.
    missing = [p for p in (BASELINE_ROOT, TRUTH_ROOT) if not p.exists()]
    if missing:
        pytest.skip(
            "scoremaps inputs not accessible (need /store_new access, e.g. on "
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

    record = bool(os.environ.get("EVALML_UPDATE_REFERENCE"))
    recorded: dict[str, dict] = {}

    for param in EXPECTED_PARAMS:
        # --- PNG: exists, non-empty, and a real map (not the grey placeholder) ---
        pngs = glob.glob(
            str(PROJECT_ROOT / f"output/results/**/scoremaps/**/{param}_*.png"),
            recursive=True,
        )
        assert pngs, (
            f"expected score-map PNG for {param} was not produced under "
            "output/results/**/scoremaps/"
        )
        assert all(Path(p).stat().st_size > 0 for p in pngs), (
            f"score-map PNG for {param} is empty"
        )
        for p in pngs:
            frac = _coloured_fraction(Path(p))
            assert frac > MIN_COLOURED_FRACTION, (
                f"{param} score-map PNG {Path(p).name} looks like the grey "
                f"'No data' placeholder (coloured fraction {frac:.3f} <= "
                f"{MIN_COLOURED_FRACTION}): the plotted field is empty/all-NaN."
            )

        # --- NetCDF: has all stats, accumulated samples, matches reference ---
        ncs = glob.glob(
            str(
                PROJECT_ROOT
                / f"output/data/baselines/**/scoremaps/{param}_{LEADTIME}_*.nc"
            ),
            recursive=True,
        )
        assert ncs, f"no score-map NetCDF produced for {param}"
        with xr.open_dataset(ncs[0]) as ds:
            for stat in STATS + ("N",):
                assert f"{param}.{stat}" in ds, f"{param}.{stat} missing from {ncs[0]}"
            assert {"season", "init_hour"}.issubset(set(ds.dims))
            _check_field_invariants(ds, param)
            summary = _summarise_bucket(ds, param)

        # all-NaN guard: samples were actually accumulated (reference-free).
        assert summary["N_total"] > 0, (
            f"{param}: no samples accumulated in the unstratified bucket — the "
            "run produced an all-NaN score map."
        )

        recorded[param] = summary
        if not record:
            _check_against_reference(param, summary)

    if record:
        _print_reference_block(recorded)
        pytest.skip(
            "EVALML_UPDATE_REFERENCE set: reference summary printed above (use "
            "`pytest -s` to see it); paste it into REFERENCE and commit."
        )
