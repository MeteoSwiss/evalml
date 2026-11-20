"""Checks of a run's `output_variables` against its checkpoint and the verified params.

anemoi-inference writes a field to a GRIB output only if its name is in the output's
`variables` list (exact match, shell-style globs from anemoi-inference 0.13.0), and
silently ignores names that match nothing. The names are those of the fields that reach
the output: the checkpoint's variable names, unless a post-processor renames them (from
anemoi-inference 0.11.2, transform filters rename fields to their GRIB param names) or
adds new fields (declared in the inference config, see `inference_config_variable_names`). These checks run before inference so that a typo, a GRIB short name
or a forgotten component of a verified parameter fails early instead of costing an
inference run.
"""

import json
import logging
import zipfile
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Iterable

import yaml

from data_input import _ICON_TO_IFS, _IFS_TO_ICON, DERIVED_PARAMS
from data_input import parse_aggregated_param

_GLOB_CHARS = set("*?[")


def is_glob(name: str) -> bool:
    return bool(_GLOB_CHARS & set(name))


def checkpoint_variable_names(checkpoint: str | Path) -> tuple[set[str], set[str]]:
    """Return the variable names of a checkpoint and the GRIB param names stored with them.

    The variable names are read from the `variable_types` of each dataset in the
    inference metadata (forcings, prognostic and diagnostic variables), or else from the
    `variables` lists of the dataset metadata. The GRIB param names are those of the
    `mars` request of each variable (`param`, with `_{levelist}` for a level), as
    anemoi-inference names the fields a transform filter passes back; names equal to a
    variable name are left out.

    Returns two empty sets if the checkpoint has no anemoi metadata.
    """
    with zipfile.ZipFile(checkpoint) as z:
        name = next(
            (n for n in z.namelist() if n.endswith("anemoi-metadata/anemoi.json")), None
        )
        if name is None:
            return set(), set()
        metadata = json.loads(z.read(name))

    variables: set[str] = set()
    for dataset in (metadata.get("metadata_inference") or {}).values():
        if isinstance(dataset, dict):
            for names in (dataset.get("variable_types") or {}).values():
                variables.update(names)
    datasets = _datasets(metadata)
    if not variables:
        for dataset in datasets:
            variables.update(dataset.get("variables") or [])
    return variables, _grib_names(datasets) - variables


def inference_config_variable_names(
    config: dict, workdir: Path
) -> tuple[set[str], set[str]]:
    """Return the field names an inference config declares, and their GRIB param names.

    Fields a post-processor adds (e.g. derive-diagnostic) are not in the checkpoint, but
    the config declares them in `typed_variables` or in the `variables_metadata` of its
    `patch_metadata` (inline or a file, relative to the working directory). The patch
    can also change the GRIB param names of checkpoint variables.
    """
    variables = set(config.get("typed_variables") or {})
    patch = config.get("patch_metadata") or {}
    if isinstance(patch, str):
        with open(Path(workdir) / patch) as f:
            patch = yaml.safe_load(f) or {}
    datasets = _datasets(patch)
    for dataset in datasets:
        variables.update(dataset.get("variables") or [])
        variables.update(dataset.get("variables_metadata") or {})
    return variables, _grib_names(datasets)


def _datasets(metadata: dict) -> list[dict]:
    """The per-dataset metadata entries ("data" for single-dataset checkpoints).

    Older checkpoints have the dataset metadata at the top level. Only the top-level
    `variables` of a dataset are the model's: the nested `specific` entries list the
    source datasets before the variables were selected.
    """
    dataset_md = metadata.get("dataset") or {}
    if "variables" in dataset_md or "variables_metadata" in dataset_md:
        dataset_md = {"data": dataset_md}
    return [d for d in dataset_md.values() if isinstance(d, dict)]


def _grib_names(datasets: list[dict]) -> set[str]:
    names = set()
    for dataset in datasets:
        for entry in (dataset.get("variables_metadata") or {}).values():
            mars = entry.get("mars") if isinstance(entry, dict) else None
            if isinstance(mars, dict) and mars.get("param"):
                level = mars.get("levelist")
                names.add(f"{mars['param']}_{level}" if level else mars["param"])
    return names


def _matches(entry: str, names: Iterable[str]) -> set[str]:
    if is_glob(entry):
        return {n for n in names if fnmatchcase(n, entry)}
    return {entry} & set(names)


def _icon_names(names: Iterable[str]) -> set[str]:
    """The evalml (ICON) parameter names a set of model field names is loaded as."""
    return {_IFS_TO_ICON.get(n, n) for n in names}


def _provided_by(param: str, icon_names: set[str]) -> bool:
    """Whether a verified param (e.g. TOT_PREC6, SP_10M) can be loaded from these fields."""
    base, _ = parse_aggregated_param(param)
    if base in icon_names:
        return True
    components = DERIVED_PARAMS.get(base)
    return components is not None and set(components) <= icon_names


def _requirement(param: str) -> str:
    """Model names a user would write to provide a param, for error messages."""
    base, _ = parse_aggregated_param(param)
    names = DERIVED_PARAMS.get(base, (base,))
    return ", ".join(_ICON_TO_IFS.get(n, n) for n in names)


def check_output_variables(
    output_variables: list[str],
    variables: set[str],
    grib_names: set[str],
    verified_params: Iterable[str],
    logger: logging.Logger,
) -> None:
    """Raise ValueError if `output_variables` would silently write less than intended.

    Parameters
    ----------
    output_variables : list of str
        The run's `output_variables` (names and shell-style globs).
    variables : set of str
        The checkpoint's variable names (see `checkpoint_variable_names`).
    grib_names : set of str
        GRIB param names stored with the checkpoint's variables that differ from the
        variable names. An entry matching only one of these is accepted with a warning:
        it is written only if a post-processor renames the field.
    verified_params : iterable of str
        Parameters verified from this run's output (e.g. T_2M, SP_10M, TOT_PREC6). A
        parameter the checkpoint can provide must be written, or its components for a
        derived parameter. Parameters the checkpoint cannot provide are not checked.
    logger : logging.Logger
        Logger for the warnings.

    Raises
    ------
    ValueError
        If an entry matches no checkpoint name, or a verified parameter the checkpoint
        can provide is not written.
    """
    unknown, written = [], set()
    for entry in output_variables:
        matched = _matches(entry, variables)
        if not matched and (matched := _matches(entry, grib_names)):
            logger.warning(
                "output_variables entry %r matches the GRIB param name of a checkpoint "
                "variable, not a variable name: it is written only if a post-processor "
                "renames the field to its GRIB param name (e.g. a transform filter with "
                "anemoi-inference >= 0.11.2).",
                entry,
            )
        if not matched:
            unknown.append(entry)
        written |= matched

    errors = []
    if unknown:
        errors.append(
            f"output_variables entries {unknown} match no variable of the checkpoint "
            "and would write nothing. Use the checkpoint's variable names: "
            f"{', '.join(sorted(variables))}."
        )

    provided = _icon_names(variables | grib_names)
    written_icon = _icon_names(written)
    missing = [
        p
        for p in dict.fromkeys(verified_params)
        if _provided_by(p, provided) and not _provided_by(p, written_icon)
    ]
    if missing:
        needs = "; ".join(f"{p} needs {_requirement(p)}" for p in missing)
        errors.append(
            f"Verified parameters {missing} are not in output_variables ({needs})."
        )

    if errors:
        raise ValueError(" ".join(errors))


def verified_params(config: dict) -> list[str]:
    """Parameters the experiment verifies from each candidate run's GRIB output.

    experiment.params (with its default) and the params of the enabled score maps and
    SAL verification. Empty without an experiment section. The showcase params are not
    included: a showcase section always exists (with defaults), so they cannot tell
    whether a showcase is produced.
    """
    from evalml.config import ExperimentConfig, SalConfig, ScoreMapsConfig

    experiment = config.get("experiment")
    if not experiment:
        return []
    params = list(
        experiment.get("params") or ExperimentConfig.model_fields["params"].default
    )
    for key, model in (("scoremaps", ScoreMapsConfig), ("sal", SalConfig)):
        section = experiment.get(key) or {}
        if section.get("enabled", False):
            params += section.get("params") or model.model_fields["params"].default
    return list(dict.fromkeys(params))
