"""Verify the extracted point files, and delete any that are incomplete.

extract_points.py skips an init whose output file already exists. That is the
right behaviour for resuming an interrupted array, but it makes a truncated file
dangerous: it would be skipped on every rerun and then silently used in the
analysis. Files written while the CAPSTOR filesystem was being unmounted for the
2026-09-15 maintenance are exactly that risk.

So existence is not the test. Each file must open, carry the expected variables,
have the number of steps its run implies, and contain no NaNs.

    uv run python workflow/scripts/check_points.py            # report only
    uv run python workflow/scripts/check_points.py --delete   # remove bad files

After --delete, rebuild the missing-task list and resubmit the array; the
deleted inits are then redone.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

# Steps expected per run label, from the config: hourly models write 0..120,
# the Varda forecaster writes 0,6,...,120.
EXPECTED_STEPS = {
    "Multistep-stage-C": 121,
    "Varda-single-stage-C": 121,
    "Varda-forecaster-6h": 21,
}
# TOT_PREC1 is absent from the 6-hourly run by design: a 1 h accumulation cannot
# be formed from 6-hourly output.
BASE_VARS = {"T_2M", "TD_2M", "U_10M", "V_10M", "PS"}
N_STATIONS = 145


def check(path: Path) -> str | None:
    """Return a reason string if the file is bad, else None."""
    label = path.parent.name
    try:
        d = xr.open_dataset(path)
    except Exception as e:  # truncated or unreadable
        return f"unreadable ({type(e).__name__})"
    with d:
        want_steps = EXPECTED_STEPS.get(label)
        if want_steps and d.sizes.get("step") != want_steps:
            return f"{d.sizes.get('step')} steps, expected {want_steps}"
        if d.sizes.get("station") != N_STATIONS:
            return f"{d.sizes.get('station')} stations, expected {N_STATIONS}"
        missing = BASE_VARS - set(d.data_vars)
        if missing:
            return f"missing variables {sorted(missing)}"
        if want_steps == 121 and "TOT_PREC1" not in d.data_vars:
            return "missing TOT_PREC1 in an hourly run"
        for v in d.data_vars:
            a = d[v].values
            # Step 0 of TOT_PREC1 is legitimately undefined: an accumulation
            # over the hour before the initial time does not exist. Ignore NaNs
            # there, but treat them as corruption anywhere else. Drop step 0 when
            # analysing precipitation.
            if v == "TOT_PREC1":
                a = a[1:]
            if np.isnan(a).any():
                return f"{v} has {int(np.isnan(a).sum())} NaNs after step 0"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("output/points"))
    ap.add_argument("--delete", action="store_true",
                    help="remove bad files so the array redoes them")
    args = ap.parse_args()

    files = sorted(args.root.rglob("*.nc"))
    bad = []
    for f in files:
        reason = check(f)
        if reason:
            bad.append((f, reason))

    by_run: dict[str, int] = {}
    for f in files:
        key = f"{f.parent.parent.name}/{f.parent.name}"
        by_run[key] = by_run.get(key, 0) + 1
    print(f"=== {len(files)} files, {len(bad)} bad ===")
    for k in sorted(by_run):
        print(f"  {by_run[k]:3d}  {k}")
    for f, reason in bad:
        print(f"  BAD {f}: {reason}")
        if args.delete:
            f.unlink()
    if bad and args.delete:
        print(f"\ndeleted {len(bad)} files; rebuild the joblist and resubmit")
    elif bad:
        print("\nrerun with --delete to remove them")


if __name__ == "__main__":
    main()
