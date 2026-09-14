"""Build the (grib_dir, output) task list for the extract_points slurm array.

Run labels are resolved from the checkpoint hash embedded in the run directory
name, so the mapping to Varda / multistep is derived rather than hardcoded by
position:

    forecaster-0eb1-*                       -> Multistep-stage-C      (hourly)
    forecaster-a69e-*                       -> Varda-forecaster-6h    (6-hourly)
    temporal_downscaler-*-on-forecaster-a69e-* -> Varda-single-stage-C (hourly)

Emits one line per task: "<grib_dir> <output_nc>". Tasks whose output already
exists are still listed, because extract_points.py skips them cheaply and that
keeps array indices stable across resubmissions.
"""

import argparse
from pathlib import Path

LABELS = {
    "temporal_downscaler": "Varda-single-stage-C",
    "0eb1": "Multistep-stage-C",
    "a69e": "Varda-forecaster-6h",
}


def label_for(run_dir: Path) -> str:
    name = run_dir.name
    if name.startswith("temporal_downscaler"):
        return LABELS["temporal_downscaler"]
    for h, lbl in LABELS.items():
        if name.startswith(f"forecaster-{h}"):
            return lbl
    raise SystemExit(f"cannot label run directory {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs="+", default=["winter2024", "summer2024"])
    ap.add_argument("--root", type=Path, default=Path("output"))
    ap.add_argument("--out-root", type=Path, default=Path("output/points"))
    args = ap.parse_args()

    n = 0
    for season in args.seasons:
        runs_dir = args.root / season / "data" / "runs"
        if not runs_dir.is_dir():
            raise SystemExit(f"missing {runs_dir}")
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            label = label_for(run_dir)
            for grib in sorted(run_dir.glob("*/*/grib")):
                init = grib.parent.name
                if not (len(init) == 12 and init.isdigit()):
                    continue
                out = args.out_root / season / label / f"{init}.nc"
                print(f"{grib} {out}")
                n += 1
    if n == 0:
        raise SystemExit("no tasks found")


if __name__ == "__main__":
    main()
