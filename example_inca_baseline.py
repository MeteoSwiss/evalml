import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_input import load_INCA_baseline_from_netcdf

logging.basicConfig(level=logging.INFO, format="%(message)s")

ROOT = Path("/store_new/mch/msclim/INCA")

# --- Hourly output (steps 0–6 h) ---
# T_2M/TD_2M: step 0 from current reftime file, steps 1–6 from the file at
# reftime − 10 min (positional index). Valid times remain anchored to reftime.
ds_1h = load_INCA_baseline_from_netcdf(
    root=ROOT,
    reftime=datetime(2025, 1, 1, 0, 0),
    steps=list(range(7)),   # 0–6  (hours)
    params=["T_2M", "TD_2M", "TOT_PREC", "FF_10M", "DD_10M", "U_10M", "V_10M", "CLCT", "VMAX_10M"],
    freq="1h",
)
def print_ds(label, ds):
    print(f"\n=== {label} ===")
    print(ds)
    print("  Coordinates:")
    for coord in ["lat", "lon"]:
        c = ds[coord]
        print(f"    {coord}: shape={c.shape}, range=[{float(c.min()):.3f}, {float(c.max()):.3f}], attrs={c.attrs}")
    print("  Variable attributes:")
    for var in ds.data_vars:
        print(f"    {var}: {ds[var].attrs}")


print_ds("freq='1h'", ds_1h)

# --- 10-minute output (steps 0–36 × 10 min) ---
# Sources: FF_10min (since 2025-05), DD_10min / WG_10min (since 2025-10), RR for TOT_PREC.
# T_2M and TD_2M remain hourly → NaN at non-hourly steps.
ds_10min = load_INCA_baseline_from_netcdf(
    root=ROOT,
    reftime=datetime(2025, 10, 1, 0, 0),
    steps=list(range(37)),  # 0–36  (× 10 min = 0 … 6 h)
    params=["T_2M", "TD_2M", "TOT_PREC", "FF_10M", "DD_10M", "U_10M", "V_10M", "CLCT", "VMAX_10M"],
    freq="10min",
)
print_ds("freq='10min'", ds_10min)

# --- 5-minute output (steps 0–72 × 5 min) ---
# Only TOT_PREC is supported at 5min (source: RP, available since 2025-05).
ds_5min = load_INCA_baseline_from_netcdf(
    root=ROOT,
    reftime=datetime(2025, 10, 1, 0, 0),
    steps=list(range(73)),  # 0–72  (× 5 min = 0 … 6 h)
    params=["TOT_PREC"],
    freq="5min",
)
print_ds("freq='5min'", ds_5min)

# --- Missing reftime → all-NaN fallback ---
# If no INCA files exist for the given reftime, each missing variable is filled
# with NaN; lat/lon are still attached using the canonical INCA grid.
ds_missing = load_INCA_baseline_from_netcdf(
    root=ROOT,
    reftime=datetime(1990, 1, 1, 0, 0),  # no INCA files for this date
    steps=[0, 1, 2, 3],
    params=["T_2M", "TOT_PREC"],
    freq="1h",
)
print_ds("missing reftime (all NaN)", ds_missing)
