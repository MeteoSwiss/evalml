"""Regroup flat, multi-init GRIB files into evalml's per-init layout.

External forecasters (e.g. WeatherGenerator) can write one GRIB file per rank,
each holding messages for many initialisation times and lead times. evalml's
``grib_forecaster`` runs expect instead::

    {out_root}/{YYYYMMDDHHMM}/grib/{YYYYMMDDHHMM}_{step:03d}.grib

This script reads every input file, groups messages by (dataDate, dataTime,
step) and appends the raw, unmodified message bytes to the matching output
file. Messages that appear in more than one input file (overlapping ranks) are
written once.

Usage:
    python workflow/tools/split_grib_by_init.py OUT_ROOT FILE [FILE ...]

Example (WeatherGenerator predictions, surface + pressure levels):
    python workflow/tools/split_grib_by_init.py \\
        /scratch/.../x44nsdo5_cosmo_all \\
        /scratch/.../x44nsdo5_cosmo_all/prediction_*.grib
"""

import argparse
from collections import Counter
from pathlib import Path

import eccodes

DEDUP_KEYS = ("shortName", "typeOfLevel", "level", "number", "stepType")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_root", type=Path)
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()

    seen: set[tuple] = set()
    handles: dict[Path, object] = {}
    written: Counter[Path] = Counter()
    skipped = 0

    try:
        for path in args.files:
            with path.open("rb") as f:
                while (h := eccodes.codes_grib_new_from_file(f)) is not None:
                    try:
                        date = eccodes.codes_get(h, "dataDate")
                        time = eccodes.codes_get(h, "dataTime")
                        step = int(eccodes.codes_get(h, "endStep"))
                        ident = tuple(
                            eccodes.codes_get(h, k) if eccodes.codes_is_defined(h, k) else None
                            for k in DEDUP_KEYS
                        )
                        key = (date, time, step, ident)
                        if key in seen:
                            skipped += 1
                            continue
                        seen.add(key)

                        init = f"{date:08d}{time:04d}"
                        out = args.out_root / init / "grib" / f"{init}_{step:03d}.grib"
                        if out not in handles:
                            out.parent.mkdir(parents=True, exist_ok=True)
                            # Truncate on first open so reruns don't duplicate.
                            handles[out] = out.open("wb")
                        handles[out].write(eccodes.codes_get_message(h))
                        written[out] += 1
                    finally:
                        eccodes.codes_release(h)
    finally:
        for fh in handles.values():
            fh.close()

    inits = {p.parent.parent.name for p in written}
    print(
        f"Wrote {sum(written.values())} messages to {len(written)} files "
        f"across {len(inits)} inits under {args.out_root} "
        f"({skipped} duplicate messages skipped)."
    )


if __name__ == "__main__":
    main()
