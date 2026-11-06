"""Script to prepare configuration and working directory for inference runs."""

import copy
import logging
import yaml
import shutil
from pathlib import Path, PurePosixPath

from evalml.helpers import setup_logger


def prepare_config(
    default_config_path: str,
    output_config_path: str,
    params: dict,
    output_variables: list[str] | None = None,
    logger: logging.Logger | None = None,
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
    output_variables : list of str, optional
        Variables to write to every GRIB output (see `set_output_variables`).
        If None, the outputs of the default configuration are left as they are.
    logger : logging.Logger, optional
        Logger for warnings; defaults to a logger named after this module.
    """

    with open(default_config_path, "r") as f:
        config = yaml.safe_load(f)

    config = _override_recursive(config, params)
    if output_variables:
        logger = logger or logging.getLogger(__name__)
        config = set_output_variables(config, output_variables, logger)

    with open(output_config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def set_output_variables(
    config: dict, variables: list[str], logger: logging.Logger
) -> dict:
    """Set the variables written by every GRIB output of an anemoi-inference config.

    Sets `variables` on each `grib` output under `config["output"]`, either a bare
    `grib` output or any depth of `tee` outputs, overriding a `variables` already
    set in the template. If the config has no GRIB output, a warning is logged and
    the config is left unchanged.

    Parameters
    ----------
    config : dict
        The anemoi-inference configuration, modified in place.
    variables : list of str
        Variable names as the GRIB output sees them; shell-style globs are allowed.
    logger : logging.Logger
        Logger for the warning.

    Returns
    -------
    dict
        The updated configuration.
    """
    n_grib = _set_grib_variables(config.get("output"), variables)
    if n_grib == 0:
        logger.warning(
            "output_variables is set, but the inference config has no GRIB output; "
            "the config is left unchanged."
        )
    return config


def _set_grib_variables(output, variables: list[str]) -> int:
    """Set `variables` on every grib output in an output block; return how many."""
    n = 0
    for spec in _grib_outputs(output):
        # one list per output, so that yaml does not write anchors and aliases
        spec["variables"] = list(variables)
        n += 1
    return n


def _grib_outputs(output):
    """Yield the spec of every grib output in an output block, bare or in any
    depth of `tee` outputs. A short-form spec (the path only) is replaced in
    place by `{"path": ...}`, so that the yielded dicts can be modified."""
    if not isinstance(output, dict):
        return
    for kind, spec in output.items():
        if kind == "grib":
            if isinstance(spec, str):  # short form: the output path only
                spec = output[kind] = {"path": spec}
            yield spec
        elif kind == "tee":
            outputs = spec.get("outputs", []) if isinstance(spec, dict) else spec
            for sub in outputs or []:
                yield from _grib_outputs(sub)


def ensemble_encoding(member: int, members: int) -> dict:
    """GRIB keys that mark the output of one member of an ensemble of `members`.

    `dataType` (not `type`, which GRIB2 has no key for) marks a perturbed
    forecast. `eps` is not among them: see `member_config`.
    """
    return {
        "productDefinitionTemplateNumber": 1,
        "number": member,
        "numberOfForecastsInEnsemble": members,
        "dataType": "pf",
        "typeOfGeneratingProcess": 4,
    }


def member_config(config: dict, member: int, members: int) -> dict:
    """Return the anemoi-inference config of one ensemble member.

    In a copy of `config`, every grib output (bare or under `tee`) writes to a
    member directory `{NNN}` (NNN = zero-padded member number) inserted before
    the file name, e.g. `grib/{step:03}.grib` becomes `grib/001/{step:03}.grib`,
    and its `encoding` is merged with `ensemble_encoding` (the ensemble keys
    win over those of the template). `eps: 1` goes in `grib2_keys`: anemoi-inference
    looks for it there too, to encode accumulated fields with an ensemble template
    (PDTN 11), but only passes `grib2_keys` entries to ecCodes per parameter. In
    `encoding` it would reach ecCodes, and the COSMO definitions have no `eps` key.

    Raises
    ------
    ValueError
        If the config has no grib output: the members would overwrite each
        other's output.
    """
    config = copy.deepcopy(config)
    specs = list(_grib_outputs(config.get("output")))
    if not specs:
        raise ValueError(
            "An ensemble run needs a GRIB output in its inference config, "
            "but the config has none."
        )
    for spec in specs:
        path = PurePosixPath(spec["path"])
        spec["path"] = str(path.parent / f"{member:03d}" / path.name)
        spec["encoding"] = {
            **(spec.get("encoding") or {}),
            **ensemble_encoding(member, members),
        }
        spec["grib2_keys"] = {**(spec.get("grib2_keys") or {}), "eps": 1}
    return config


def write_member_configs(config_path: Path, workdir: Path, members: int) -> list[Path]:
    """Write `config_{NNN}.yaml` next to `config_path` for every member 1..members.

    Each member config is `member_config` of the config at `config_path`. The
    directories its grib outputs write to are created, relative to `workdir`.

    Returns
    -------
    list of Path
        The member config files, in member order.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    paths = []
    for member in range(1, members + 1):
        mconfig = member_config(config, member, members)
        for spec in _grib_outputs(mconfig["output"]):
            (workdir / PurePosixPath(spec["path"]).parent).mkdir(
                parents=True, exist_ok=True
            )
        path = config_path.with_name(f"config_{member:03d}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(mconfig, f, sort_keys=False)
        paths.append(path)
    return paths


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
    (workdir / "grib").mkdir(parents=True, exist_ok=True)
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
    if fct_run_id != "null":
        fct_workdir = (
            smk.params.output_root / "runs" / fct_run_id / smk.wildcards.init_time
        )
        (workdir / "forecaster").symlink_to(fct_workdir / "grib")
        LOG.info(
            "Created symlink to forecaster grib directory at %s", workdir / "forecaster"
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
        output_variables=smk.params.output_variables,
        logger=LOG,
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
    LOG.info("Prepared working directory at %s", workdir)
    res_list = "\n".join([str(fn) for fn in Path(workdir / "resources").rglob("*")])
    LOG.info("Resources: \n%s", res_list)
    stage_holdout_stations(smk, workdir, LOG)

    overrides = _overrides_from_params(smk)
    prepare_config(
        smk.input.config,
        smk.output.config,
        overrides,
        output_variables=smk.params.output_variables,
        logger=LOG,
    )
    LOG.info("Wrote config file at %s", smk.output.config)
    with open(smk.output.config, "r") as f:
        config_content = f.read()
    LOG.info("Config: \n%s", config_content)

    ensemble = smk.params.get("ensemble")
    if ensemble:
        member_paths = write_member_configs(
            Path(smk.output.config), workdir, ensemble["members"]
        )
        LOG.info(
            "Wrote %d ensemble member configs: %s",
            len(member_paths),
            ", ".join(p.name for p in member_paths),
        )
        with open(member_paths[0], "r") as f:
            LOG.info("Config of member 001: \n%s", f.read())

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
