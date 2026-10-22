"""Script to prepare configuration and working directory for inference runs."""

import logging
import yaml
import shutil
from pathlib import Path

from evalml.helpers import setup_logger
from data_input import _FDB_MODEL, _FDB_GLOBAL_MODEL, _FDB_PARAM_IDS, _FDB_GLOBAL_PARAM_IDS, _FDB_IDENTITY, _FDB_SCHEMA


def prepare_config(
    default_config_path: str,
    output_config_path: str,
    params: dict,
    output_backend: str = "grib",
    forecaster_fdb_root: str | None = None,
):
    """Prepare the configuration file for the inference run.

    Overrides default configuration parameters with those provided in params
    and writes the updated configuration to output_config_path.

    Parameters
    ----------
    default_config_path : str
        Path to the default configuration file.
    output_config_path : str
        Path where the updated configuration file will be written.
    params : dict
        Dictionary of parameters to override in the default configuration.
    output_backend : str
        "grib" (default) leaves output.tee untouched -- the inference config is
        itself the single source of truth, always written in grib: form. "fdb"
        rewrites every `output.tee` entry from `grib:` to `fdb:` (dropping the
        grib-only `path` key) -- see `_convert_tee_to_fdb`.
    forecaster_fdb_root : str | None
        If this run's forecaster declares `output: fdb`, the FDB root its output
        was archived into -- rewrites this downscaler's own input.cutout entries
        from `grib:` to `fdb:` so it reads the forecaster's output directly from
        FDB instead of a local grib symlink. None (default) leaves input.cutout
        untouched (the forecaster writes local grib, read via the symlink
        prepare_temporal_downscaler already creates).
    """

    with open(default_config_path, "r") as f:
        config = yaml.safe_load(f)

    if output_backend == "fdb":
        config = _convert_tee_to_fdb(config)

    if forecaster_fdb_root is not None:
        config = _convert_cutout_to_fdb(config, forecaster_fdb_root)

    config = _override_recursive(config, params)

    with open(output_config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def prepare_workdir(workdir: Path, resources_root: Path):
    """Prepare the working directory for the inference run.

    Creates necessary subdirectories and copies resource files.

    Parameters
    ----------
    workdir : Path
        Path to the working directory.
    resources_root : Path
        Path to the root directory containing resource files.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "resources").mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        resources_root / "templates", workdir / "resources", dirs_exist_ok=True
    )
    shutil.copytree(
        resources_root / "metadata", workdir / "resources", dirs_exist_ok=True
    )


def stage_holdout_stations(smk, workdir: Path, LOG: logging.Logger):
    """Copy the holdout station file, if the rule has one as input, into the
    working directory under the fixed name the inference config refers to
    (e.g. the nudge_toward_observation filter's holdout_station_file)."""
    holdout = smk.input.get("holdout")
    if not holdout:
        return
    target = workdir / smk.params.holdout_workdir_name
    shutil.copyfile(str(holdout), target)
    with open(target) as f:
        n_stations = sum(1 for _ in f) - 1  # minus header
    LOG.info("Staged %d holdout stations at %s", n_stations, target)


def prepare_temporal_downscaler(smk):
    """Prepare the temporal downscaler for the inference run.

    Required steps:
    - prepare working directory
    - prepare forecaster directory
    - prepare config
    """
    LOG = _setup_logger(smk)

    # prepare working directory
    workdir = _get_workdir(smk)
    prepare_workdir(workdir, smk.params.resources_root)
    LOG.info("Prepared working directory at %s", workdir)
    res_list = "\n".join([str(fn) for fn in Path(workdir / "resources").rglob("*")])
    LOG.info("Resources: \n%s", res_list)
    stage_holdout_stations(smk, workdir, LOG)

    # prepare forecaster directory
    fct_run_id = smk.params.forecaster_run_id
    forecaster_fdb_root = smk.params.forecaster_fdb_root
    if fct_run_id != "null":
        if forecaster_fdb_root is None:
            fct_workdir = (
                smk.params.output_root / "runs" / fct_run_id / smk.wildcards.init_time
            )
            (workdir / "forecaster").symlink_to(fct_workdir / "grib")
            LOG.info(
                "Created symlink to forecaster grib directory at %s", workdir / "forecaster"
            )
        else:
            # Still an output directory of this rule (see inference.smk's
            # inference_prepare_temporal_downscaler), so it must exist, but
            # unused -- input.cutout reads the forecaster's output straight
            # from FDB instead.
            (workdir / "forecaster").mkdir(parents=True, exist_ok=True)
            LOG.info(
                "Forecaster output is in FDB (%s) -- downscaler input.cutout will be "
                "rewritten to read it directly, no local grib symlink needed.",
                forecaster_fdb_root,
            )
    else:
        (workdir / "forecaster").mkdir(parents=True, exist_ok=True)
        (workdir / "forecaster/.dataset").touch()
        LOG.info(
            "No forecaster run ID provided; using dataset placeholder at %s",
            workdir / "forecaster/.dataset",
        )

    # prepare config
    overrides = _overrides_from_params(smk)
    prepare_config(
        smk.input.config,
        smk.output.config,
        overrides,
        output_backend=smk.params.output_backend,
        forecaster_fdb_root=forecaster_fdb_root,
    )

    LOG.info("Wrote config file at %s", smk.output.config)
    with open(smk.output.config, "r") as f:
        config_content = f.read()
    LOG.info("Config: \n%s", config_content)

    okfile = Path(smk.output.okfile)
    okfile.parent.mkdir(parents=True, exist_ok=True)
    okfile.touch()
    LOG.info("Interpolator preparation complete.")


def prepare_forecaster(smk):
    """Prepare the forecaster for the inference run.

    Required steps:
    - prepare working directory
    - prepare config
    """
    LOG = _setup_logger(smk)

    workdir = _get_workdir(smk)
    prepare_workdir(workdir, smk.params.resources_root)
    (workdir / "grib").mkdir(parents=True, exist_ok=True)
    LOG.info("Prepared working directory at %s", workdir)
    res_list = "\n".join([str(fn) for fn in Path(workdir / "resources").rglob("*")])
    LOG.info("Resources: \n%s", res_list)
    stage_holdout_stations(smk, workdir, LOG)

    overrides = _overrides_from_params(smk)
    prepare_config(
        smk.input.config,
        smk.output.config,
        overrides,
        output_backend=smk.params.output_backend,
    )
    LOG.info("Wrote config file at %s", smk.output.config)
    with open(smk.output.config, "r") as f:
        config_content = f.read()
    LOG.info("Config: \n%s", config_content)

    okfile = Path(smk.output.okfile)
    okfile.parent.mkdir(parents=True, exist_ok=True)
    okfile.touch()
    LOG.info("Forecaster preparation complete.")


# TODO: just pass a dictionary of config overrides to the rule's params
def _overrides_from_params(smk) -> dict:
    return {
        "checkpoint": str(Path(smk.input.checkpoint).resolve()),
        "date": smk.params.reftime_to_iso,
        "lead_time": smk.params.lead_time,
    }


def _get_workdir(smk) -> Path:
    run_id = smk.wildcards.run_id
    init_time = smk.wildcards.init_time
    return smk.params.output_root / "runs" / run_id / init_time


def _setup_logger(smk) -> logging.Logger:
    run_id = smk.wildcards.run_id
    init_time = smk.wildcards.init_time
    logger_name = f"{smk.rule}_{run_id}_{init_time}"
    LOG = setup_logger(logger_name, log_file=smk.log[0])
    return LOG


def _convert_tee_to_fdb(config: dict) -> dict:
    """Rewrite every `output.tee` entry from `{grib: {...}}` to `{fdb: {...}}`.

    The inference config is maintained once, in grib: form, as the single
    source of truth for both backends -- see
    resources/inference/configs/sgm-temporal-downscaler-global_trimedge_multi.yaml,
    whose output.tee encoding/templates/post_processors/modifiers are already
    FDB-compatible (generatingProcessIdentifier, marsStream/marsType, etc.) and
    were kept in parity with the (now-deleted) *-fdb-parity.yaml by hand before
    this. Only the grib-specific `path` key doesn't apply to fdb: and is
    dropped; everything else under the tee entry carries over unchanged.
    """
    tee = config.get("output", {}).get("tee")
    if not tee:
        raise ValueError(
            "output_backend='fdb' but this inference config has no output.tee "
            "entries to convert -- check the referenced config file."
        )
    new_tee = []
    for entry in tee:
        if set(entry.keys()) != {"grib"}:
            raise ValueError(
                f"output_backend='fdb' expects every output.tee entry to be a "
                f"single 'grib:' block, got keys {sorted(entry.keys())} -- "
                "this config isn't written in the expected grib: source-of-truth form."
            )
        inner = dict(entry["grib"])
        inner.pop("path", None)
        new_tee.append({"fdb": inner})
    config["output"]["tee"] = new_tee
    return config


# Which FDB model/param-id table a downscaler input.cutout entry reads from,
# keyed by its grid name -- "lam_0" is the regional/ICON grid (VARDA's own local
# GPI=180 table), "global" is the IFS/global grid (standard WMO/ECMWF param ids).
# See _FDB_IDENTITY (data_input/__init__.py) for the class/stream/type/expver
# each model resolves to.
_CUTOUT_FDB_PARAM_IDS = {
    "lam_0": (_FDB_MODEL, _FDB_PARAM_IDS),
    "global": (_FDB_GLOBAL_MODEL, _FDB_GLOBAL_PARAM_IDS),
}


def _convert_cutout_to_fdb(config: dict, fdb_root: str) -> dict:
    """Rewrite the downscaler's input.cutout entries from `{grib: {...}}` to
    `{fdb: {...}}`, so it reads its forecaster's output directly from FDB
    instead of a local grib symlink.

    `pre_processors`/`namer` under the grib: block carry over unchanged -- they
    operate on the already-decoded field regardless of how the raw bytes were
    fetched. The grib-only `path` key is dropped. In its place we add
    `fdb_config` (pointing at the forecaster's FDB root), `param_id_map`
    (translating the checkpoint's own param names -- already rewritten to ICON
    shortnames by this config's `patch_metadata`, see anemoi-inference's
    Metadata.patch -- to this archive's real param ids), and the static
    class/stream/type/expver/model identity, via anemoi-inference's FDBInput
    (registered as `fdb`, inputs/fdb.py in the anemoi-inference source):
    these keys all pass straight through as constructor kwargs.
    """
    cutout = config.get("input", {}).get("cutout")
    if not cutout:
        raise ValueError(
            "forecaster declares output: fdb but this downscaler's input has no "
            "input.cutout entries to convert -- check the referenced inference config."
        )
    new_cutout = []
    for entry in cutout:
        if len(entry) != 1:
            raise ValueError(f"Expected a single-key cutout entry, got {sorted(entry.keys())}")
        (grid_name, block), = entry.items()
        if grid_name not in _CUTOUT_FDB_PARAM_IDS:
            raise ValueError(
                f"Unknown cutout grid '{grid_name}' -- add its FDB identity to "
                "_CUTOUT_FDB_PARAM_IDS before converting it to an fdb: input."
            )
        if set(block.keys()) != {"grib"}:
            raise ValueError(
                f"Expected cutout entry '{grid_name}' to be a single 'grib:' block, "
                f"got keys {sorted(block.keys())} -- this config isn't written in "
                "the expected grib: source-of-truth form."
            )
        inner = dict(block["grib"])
        inner.pop("path", None)
        model, param_id_map = _CUTOUT_FDB_PARAM_IDS[grid_name]
        inner["fdb_config"] = {
            "type": "local",
            "engine": "toc",
            "schema": str(_FDB_SCHEMA.resolve()),
            "spaces": [{"handler": "Default", "roots": [{"path": fdb_root}]}],
        }
        inner["param_id_map"] = dict(param_id_map)
        inner.update(_FDB_IDENTITY[model])
        new_cutout.append({grid_name: {"fdb": inner}})
    config["input"]["cutout"] = new_cutout
    return config


def _override_recursive(original: dict, updates: dict) -> dict:
    """Recursively override values in the original dictionary with those from the updates dictionary."""
    for key, value in updates.items():
        if (
            isinstance(value, dict)
            and key in original
            and isinstance(original[key], dict)
        ):
            original[key] = _override_recursive(original[key], value)
        else:
            original[key] = value
    return original


def main(smk):
    """Main function to run the Snakemake workflow."""
    if smk.rule == "inference_prepare_forecaster":
        prepare_forecaster(smk)
    elif smk.rule == "inference_prepare_temporal_downscaler":
        prepare_temporal_downscaler(smk)
    else:
        raise ValueError(f"Unknown rule: {smk.rule}")


if __name__ == "__main__":
    snakemake = snakemake  # type: ignore # noqa: F821
    raise SystemExit(main(snakemake))
