"""Overwrite prognostic variables at the overlap steps of a temporal-downscaler
run with the parent forecaster's values.

A multi-step temporal downscaler is given the forecaster states at the window
boundaries (e.g. ``explicit_times.input: [0, 6]``) and predicts the intermediate
hourly steps plus the right boundary (``target: [1, 2, 3, 4, 5, 6]``). At the
overlap step (here 6, and every multiple of the window stride) the downscaler
therefore *re-predicts* a prognostic state it was handed as input. For those
prognostic variables the forecaster's value is the consistent one, so
forecaster-alone and forecaster+downscaler metrics should agree there; they only
differ because of this redundant re-prediction. Diagnostic variables (e.g.
hourly-accumulated ``tp``) are genuinely downscaler-only at those steps and are
left untouched.

This script, enabled per-run via ``copy_prognostic_from_forecaster``, copies the
forecaster's prognostic fields onto the downscaler GRIB at the overlap steps, in
place, for both output streams (the LAM ``{dt}_{step}.grib`` and the global
``ifs-{dt}_{step}.grib``).

--------------------------------------------------------------------------------
1:1 GRID ASSUMPTION
--------------------------------------------------------------------------------
The copy replaces only the GRIB data section (via ``field.clone(values=...)``),
keeping the downscaler message headers. This assumes the downscaler output grid
is pointwise identical to the forecaster output grid — true for a *temporal*
downscaler (it refines time, not space). The assumption is enforced per field by
an equality check on ``numberOfValues``; if it is ever violated (e.g. a spatial
downscaler on a finer grid) the script aborts loudly rather than silently
misaligning fields.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import earthkit.data as ekd
import yaml

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def _load_metadata(metadata_path: Path) -> dict:
    with open(metadata_path, "r") as f:
        return json.load(f)


def prognostic_variables(metadata: dict) -> set[str]:
    """Return the prognostic output variable names, read from the checkpoint
    metadata.

    ``output.prognostic`` are the variables that are both input and output
    (copied from the forecaster); ``output.diagnostic`` (e.g. ``tp``) are
    output-only and left untouched.
    """
    out = metadata["data_indices"]["data"]["data"]["output"]
    idx_to_name = {v: k for k, v in out["name_to_index"].items()}
    return {idx_to_name[i] for i in out["prognostic"]}


def lead_time_hours(config: dict) -> int:
    """Forecast horizon in hours, read from the rendered inference config's
    ``lead_time`` (e.g. ``120h``). This is the authoritative run horizon and,
    unlike the GRIB filenames, is a stable structured field."""
    lead_time = str(config["lead_time"]).strip()
    if lead_time.endswith("h"):
        return int(lead_time[:-1])
    if lead_time.endswith("d"):
        return int(lead_time[:-1]) * 24
    return int(lead_time)  # bare number of hours


def overlap_steps(metadata: dict, max_step: int) -> list[int]:
    """Global lead times at which the downscaler re-predicts a forecaster boundary.

    Derived from ``config.training.explicit_times``: the within-window overlap is
    ``set(input) & set(target)``; the window advances by the input span (stride).
    Tiling those offsets across windows up to the forecast horizon gives the
    global overlap steps (e.g. {6, 12, 18, ...}).
    """
    explicit_times = metadata["config"]["training"]["explicit_times"]
    downscale_input, downscale_target = explicit_times["input"], explicit_times["target"]
    offsets = sorted(set(downscale_input) & set(downscale_target)) # offsets = inputs that are repredicted.
    if not offsets:
        return []
    stride = max(downscale_input) - min(downscale_input) # stride = window span
    if stride <= 0:
        return []
    windows = range(0, max_step + 1, stride)
    return sorted({w + off for off in offsets for w in windows if w + off <= max_step})


def _namer_map(config: dict) -> dict[str, str]:
    """Build the LAM ``shortName -> anemoi name`` map from the rendered inference
    config's ``namer.rules``. This is the exact mapping the downscaler used to
    read the forecaster input, so it is the correct inverse for its LAM output."""

    def _find(obj):
        if isinstance(obj, dict):
            namer = obj.get("namer")
            if isinstance(namer, dict) and "rules" in namer:
                return namer["rules"]
            for v in obj.values():
                r = _find(v)
                if r:
                    return r
        elif isinstance(obj, list):
            for v in obj:
                r = _find(v)
                if r:
                    return r
        return None

    rules = _find(config) or []
    return {
        rule[0]["shortName"]: rule[1]
        for rule in rules
        if isinstance(rule, list) and len(rule) == 2 and "shortName" in rule[0]
    }


def _anemoi_name(short_name, level, type_of_level, ifs: bool, namer: dict) -> str | None:
    """Map a GRIB field to its anemoi variable name.

    Global/IFS stream: surface shortNames already equal the anemoi names
    (``2t``, ``sp``, ``tp``, ...); pressure levels become ``{shortName}_{level}``
    (``t_500``). LAM stream: ICON shortNames go through the config ``namer``
    (``T``->``t_{level}``, ``T_2M``->``2t``); unknown names return None.
    """
    if ifs:
        return f"{short_name}_{level}" if type_of_level == "isobaricInhPa" else short_name
    template = namer.get(short_name)
    if template is None:
        return None
    return template.format(level=level) if "{level}" in template else template


def _field_key(field) -> tuple:
    return (
        field.metadata("shortName"),
        field.metadata("level"),
        field.metadata("typeOfLevel"),
    )


def _find_step_file(directory: Path, step: int, ifs: bool) -> Path | None:
    """Locate the GRIB file for a given step in a stream.

    anemoi-inference has written step suffixes both zero-padded (``_006.grib``)
    and bare (``_6.grib``); accept either. LAM files start with the date, global
    files with the ``ifs-`` prefix.
    """
    patterns = (
        [f"ifs-*_{step:03d}.grib", f"ifs-*_{step}.grib"]
        if ifs
        else [f"[0-9]*_{step:03d}.grib", f"[0-9]*_{step}.grib"]
    )
    for pat in patterns:
        matches = sorted(directory.glob(pat))
        if matches:
            if len(matches) > 1:
                LOG.warning(
                    "Multiple files match %s in %s; using %s", pat, directory, matches[0]
                )
            return matches[0]
    return None


def _patch_stream_file(
    downscaler_file: Path,
    forecaster_file: Path,
    prognostic: set[str],
    ifs: bool,
    namer: dict,
) -> int:
    """Overwrite prognostic fields in ``downscaler_file`` in place with the
    matching forecaster fields. Returns the number of fields replaced."""
    ds_data = ekd.from_source("file", str(downscaler_file)).to_fieldlist()
    fc_data = ekd.from_source("file", str(forecaster_file)).to_fieldlist()

    # Index forecaster fields by (shortName, level, typeOfLevel) for lookup.
    fc_by_key = {_field_key(f): f for f in fc_data}

    patched = 0
    out_fields = []
    for field in ds_data:
        short_name, level, type_of_level = _field_key(field)
        name = _anemoi_name(short_name, level, type_of_level, ifs, namer)
        if name is None or name not in prognostic:
            out_fields.append(field)  # diagnostic / forcing / unknown -> keep as-is
            continue
        match = fc_by_key.get((short_name, level, type_of_level))
        if match is None:
            LOG.warning(
                "No forecaster field for prognostic %s (%s/%s) in %s; keeping "
                "downscaler value.",
                name,
                short_name,
                level,
                forecaster_file.name,
            )
            out_fields.append(field)
            continue
        # 1:1 grid assumption (see module docstring): a pure value copy is only
        # valid when both fields live on the same grid. Enforce it per field.
        n_ds = field.metadata("numberOfValues")
        n_fc = match.metadata("numberOfValues")
        if n_ds != n_fc:
            raise ValueError(
                f"Grid mismatch for {name} ({short_name}/{level}) in "
                f"{downscaler_file.name}: downscaler has {n_ds} values but "
                f"forecaster has {n_fc}. The copy_prognostic_from_forecaster "
                "value-copy requires the downscaler and forecaster to share the "
                "same output grid (true for a temporal downscaler)."
            )
        # `set(values=...)` returns a new field with the data swapped and all
        # metadata kept. Equivalent alternative: `type(field).from_field(field,
        # data=match.to_numpy())`.
        out_fields.append(field.set(values=match.to_numpy()))
        patched += 1

    tmp = downscaler_file.with_suffix(downscaler_file.suffix + ".patched")
    ekd.FieldList.from_fields(out_fields).to_target("file", str(tmp))
    os.replace(tmp, downscaler_file)  # atomic in-place replace
    return patched


def main(args: argparse.Namespace) -> int:
    metadata = _load_metadata(args.metadata)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    prognostic = prognostic_variables(metadata)
    overlaps = overlap_steps(metadata, lead_time_hours(config))

    if not overlaps:
        LOG.info(
            "No overlap between explicit_times input/target (or empty stride); "
            "nothing to patch. Touching okfile and exiting."
        )
        Path(args.okfile).touch()
        return 0

    LOG.info("Patching downscaler prognostic variables (%d): %s", len(prognostic), sorted(prognostic))
    LOG.info("Overlap steps to patch: %s", overlaps)

    namer = _namer_map(config)  # LAM shortName -> anemoi name

    total = 0
    for step in overlaps:
        for ifs in (False, True): # prefix "ifs" is for global models.
            ds_file = _find_step_file(args.grib_dir, step, ifs=ifs)
            fc_file = _find_step_file(args.forecaster_dir, step, ifs=ifs)
            stream = "ifs" if ifs else "lam"
            if ds_file is None:
                LOG.info("No downscaler %s file for step %s; skipping.", stream, step)
                continue
            if fc_file is None:
                LOG.warning(
                    "No forecaster %s file for step %s; cannot patch %s.",
                    stream,
                    step,
                    ds_file.name,
                )
                continue
            n = _patch_stream_file(ds_file, fc_file, prognostic, ifs, namer)
            assert len(prognostic) == n
            LOG.info(
                "Patched %d prognostic field(s) in %s (step %s, %s stream) from %s",
                n,
                ds_file.name,
                step,
                stream,
                fc_file.name,
            )
            total += n

    LOG.info("Done. Replaced %d prognostic field(s) in total.", total)
    Path(args.okfile).touch()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Overwrite prognostic variables at the overlap steps of a temporal "
            "downscaler run with the parent forecaster's values (in place)."
        )
    )
    parser.add_argument(
        "--grib-dir", type=Path, required=True, help="Downscaler GRIB output dir."
    )
    parser.add_argument(
        "--forecaster-dir",
        type=Path,
        required=True,
        help="Forecaster GRIB dir (the run's forecaster/ symlink).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Downscaler checkpoint anemoi.json.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Rendered downscaler inference config.yaml (for lead_time and the "
        "LAM namer).",
    )
    parser.add_argument("--okfile", type=Path, required=True, help="Okfile to touch.")
    args = parser.parse_args()

    raise SystemExit(main(args))
