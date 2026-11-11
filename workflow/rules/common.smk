import logging
import copy
from datetime import datetime, timedelta
import yaml
import hashlib
import json
from urllib.parse import urlparse

from evalml.config import MODEL_REGISTRY_ROOT
from data_input import inca, truth_type
from evalml.leadtimes import resolve_leadtimes

CONFIG_ROOT = Path("config").resolve()
OUT_ROOT = Path(config["locations"]["output_root"])
# the experiment block is optional (e.g. for showcase/sandbox configs)
EXPERIMENT_CONFIG = config.get("experiment") or {}
# "anemoi_dataset", "jretrieve" or "inca"
TRUTH_TYPE = truth_type(config["truth"]["root"])

DATETIME_FORMAT = "%Y-%m-%dT%H:%M"
HASH_LENGTH = 4

# Fields that determine the inference ENVIRONMENT. Changing these requires a new venv/squashfs.
ENV_HASH_FIELDS = {
    "checkpoint",
    "extra_requirements",
    "disable_local_eccodes_definitions",
}
# Fields excluded from ALL hashing (display/resource metadata only).
RUN_HASH_EXCLUDE = {"label", "inference_resources", "_is_candidate", "model_type"}
# Fields excluded from baseline hashing (display metadata only).
BASELINE_HASH_EXCLUDE = {"label"}
TRUTH_HASH_EXCLUDE = {"label"}


# ============================================================================
# Utility Functions
# ============================================================================


def parse_timedelta(td):
    """Parse a string representing a time delta (e.g., '1d', '2h') into a timedelta object."""
    if not isinstance(td, str):
        raise ValueError("Expected a string in the format 'Xd' or 'Xh'")
    magnitude, unit = int(td[:-1]), td[-1]
    match unit:
        case "d":
            return timedelta(days=magnitude)
        case "h":
            return timedelta(hours=magnitude)
        case _:
            raise ValueError(
                f"Unsupported time unit: {unit}. Only 'd' and 'h' are supported."
            )


def generate_json_hash(obj: object) -> str:
    """Generate a short hash of a JSON-serializable object."""
    json_str = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(json_str.encode()).hexdigest()[:HASH_LENGTH]


# ============================================================================
# Configuration Parsing
# ============================================================================


def parse_reference_times():
    """Parse reference times from the configuration."""
    cfg = config["dates"]
    if isinstance(cfg, list):
        return [datetime.strptime(t, DATETIME_FORMAT) for t in cfg]
    start = datetime.strptime(cfg["start"], DATETIME_FORMAT)
    end = datetime.strptime(cfg["end"], DATETIME_FORMAT)
    freq = parse_timedelta(cfg["frequency"])
    blacklist = {
        datetime.strptime(t, DATETIME_FORMAT) for t in cfg.get("blacklist", [])
    }
    times = []
    t = start
    while t <= end:
        if t not in blacklist:
            times.append(t)
        t += freq
    return times


def parse_regions():
    """Return a JSON list of region specs in config order.

    Each entry is either ``{"type": "bbox", "name": ..., "bbox": [...]}``
    or ``{"type": "shp", "name": ..., "path": ...}``, preserving the original
    order so the NetCDF region coordinate matches the config.
    """
    from evalml.config import PREDEFINED_REGIONS

    cfg = EXPERIMENT_CONFIG.get("stratification", {})
    root = cfg.get("root", "")
    result = []
    for entry in cfg.get("regions", []):
        if isinstance(entry, str):
            if entry in PREDEFINED_REGIONS:
                logging.getLogger("snakemake").info(
                    "Region %s found among the predefined regions (rectangles in lon/lat).  The evaluation will happen on the bounding box %s",
                    entry,
                    PREDEFINED_REGIONS[entry],
                )
                result.append(
                    {"type": "bbox", "name": entry, "bbox": PREDEFINED_REGIONS[entry]}
                )
            else:
                path = f"{root}/{entry}.shp"
                logging.getLogger("snakemake").info(
                    "Region %s not found in the predefined regions. Will try to locate file %s.",
                    entry,
                    path,
                )
                result.append({"type": "shp", "name": entry, "path": path})
        elif isinstance(entry, dict):
            name, bbox = next(iter(entry.items()))
            result.append({"type": "bbox", "name": name, "bbox": bbox})
    return json.dumps(result)


def parse_showcase_regions(domains=None):
    """Parse showcase domains from config.

    ``domains`` defaults to ``showcase.animations.domains``.

    Returns a dict mapping domain name -> {extent, projection, rotate, hours_per_revolution}.
    Named domains (strings) have extent=None and projection=None,
    meaning the plot script will fall back to the DOMAINS lookup.
    Custom domains carry their explicit extent and projection.
    rotate/hours_per_revolution only take effect when extent is None
    (full-globe domains).
    """
    if domains is None:
        domains = config["showcase"]["animations"]["domains"]
    result = {}
    for r in domains:
        if isinstance(r, str):
            result[r] = {
                "extent": None,
                "projection": None,
                "rotate": False,
                "hours_per_revolution": 96.0,
            }
        else:
            result[r["name"]] = {
                "extent": r["extent"],
                "projection": r["projection"],
                "rotate": r["rotate"],
                "hours_per_revolution": r["hours_per_revolution"],
            }
    return result


# ============================================================================
# Run entries configuration management
# ============================================================================


def _checkpoint_uri_type(checkpoint_uri: str):
    parsed_url = urlparse(checkpoint_uri)
    if parsed_url.netloc in [
        "mlflow.ecmwf.int",
        "service.meteoswiss.ch",
        "servicedepl.meteoswiss.ch",
    ]:
        return "mlflow"
    elif parsed_url.netloc == "huggingface.co":
        if not parsed_url.path.endswith(".ckpt"):
            raise ValueError(
                f"Expected a .ckpt file for HuggingFace checkpoint URI. Got: {checkpoint_uri}"
            )
        return "huggingface"
    elif parsed_url.netloc == "":
        if Path(checkpoint_uri).exists():
            return "local"
        else:
            raise ValueError(f"Local checkpoint path does not exist: {checkpoint_uri}")
    else:
        raise ValueError(f"Unknown checkpoint URI type: {checkpoint_uri}")


def model_id(checkpoint_uri: str) -> str:
    """Generate a model ID based on the checkpoint URI."""
    ckpt_type = _checkpoint_uri_type(checkpoint_uri)
    if ckpt_type == "mlflow":
        fragment = checkpoint_uri.split("#")[-1]
        if "/models/" in fragment:
            parts = fragment.strip("/").split("/")
            if len(parts) >= 4 and parts[2] == "versions":
                return f"{parts[1]}-v{parts[3]}"[:HASH_LENGTH]
            return f"{parts[1]}-latest"[:HASH_LENGTH]
        return checkpoint_uri.split("/")[-1][:HASH_LENGTH]
    elif ckpt_type == "huggingface":
        return checkpoint_uri.split("/")[-1].split(".")[0]
    elif ckpt_type == "local":
        # A checkpoint resolved from the model registry keeps its registered name as
        # the id, so output paths and dashboards say "amber-ridge", not "ambe".
        try:
            return Path(checkpoint_uri).relative_to(MODEL_REGISTRY_ROOT).parts[0]
        except ValueError:
            return checkpoint_uri.split("/")[-2][:HASH_LENGTH]


def register_run(model_type, run_config, as_candidate=True):
    """Parse a run configuration and assign a unique env_id and run_id.

    Assigns two identifiers:
    - env_id: Identifies the inference environment (venv, squashfs). Shared across
             runs with the same checkpoint and extra_requirements.
    - run_id: Extends env_id with a hash of inference parameters (config YAML, steps).
             Ensures each unique run configuration has its own output directory.
    """
    run_cfg = copy.deepcopy(run_config)
    mid = model_id(run_cfg["checkpoint"])

    out = {}
    if model_type == "temporal_downscaler":
        forecaster = run_cfg.get("forecaster")
        if forecaster is None:
            run_cfg["forecaster"] = None
            env_dep_suffix = "analysis"
            run_dep_suffix = "analysis"
        else:
            # Register the upstream forecaster recursively
            dep_entry = register_run("forecaster", forecaster, as_candidate=False)
            out |= dep_entry
            dep_run_id = next(iter(dep_entry))
            dep_env_id = dep_entry[dep_run_id]["env_id"]
            run_cfg["forecaster"]["run_id"] = dep_run_id
            run_cfg["forecaster"]["env_id"] = dep_env_id
            env_dep_suffix = dep_env_id  # env_id determines environment dependency
            run_dep_suffix = dep_run_id  # run_id determines output dependency

    # Compute env_id (determines which venv/squashfs to use)
    e_hash = env_entry_hash(run_cfg, model_type)
    env_id_base = f"{model_type}-{mid}-{e_hash}"
    if model_type == "temporal_downscaler":
        env_id = f"{env_id_base}-on-{env_dep_suffix}"
    else:
        env_id = env_id_base

    # Compute run_id (extends env_id with run-specific config hash)
    r_hash = run_specific_hash(run_cfg, model_type)
    run_id = f"{env_id}/{r_hash}"

    run_cfg["env_id"] = env_id
    run_cfg["_is_candidate"] = as_candidate
    run_cfg["model_type"] = model_type
    out[run_id] = run_cfg
    return out


def collect_all_runs() -> dict:
    """Collect all runs defined in the configuration, including secondary runs."""
    runs: dict[str, dict] = {}
    for run_entry in config["runs"]:
        model_type = next(iter(run_entry))
        if model_type == "baseline":
            continue
        run_config = run_entry[model_type]
        for run_id, run_cfg in register_run(model_type, run_config).items():
            if run_id in runs and runs[run_id].get("_is_candidate", False):
                # Preserve candidates by not letting a dependency registration
                # (as_candidate=False) demote a run that was already registered as
                # an explicit candidate. Order in config["runs"] must not matter.
                run_cfg["_is_candidate"] = True
                run_cfg["label"] = runs[run_id].get("label")
            runs[run_id] = run_cfg
    return runs


def collect_all_candidates():
    """Collect participating runs ('candidates') only."""
    runs = collect_all_runs()
    candidates = {}
    for run_id, run_config in runs.items():
        if run_config.get("_is_candidate", False):
            candidates[run_id] = run_config
    return candidates


def collect_all_envs() -> dict:
    """Collect unique inference environments from all registered runs.

    Returns a dict mapping env_id -> minimal environment config dict.
    """
    envs = {}
    for run_cfg in RUN_CONFIGS.values():
        env_id = run_cfg["env_id"]
        if env_id not in envs:
            envs[env_id] = {k: v for k, v in run_cfg.items() if k in ENV_HASH_FIELDS}
    return envs


def collect_all_baselines():
    """Collect all baselines defined in the configuration."""
    baselines = {}

    for run_entry in copy.deepcopy(config["runs"]):
        if "baseline" not in run_entry:
            continue
        baseline_config = run_entry["baseline"]
        baseline_id = f"baseline-{baseline_hash(baseline_config)}"
        baselines[baseline_id] = baseline_config

    return baselines


def resolve_reference_id(label: str) -> str:
    """Resolve a scorecard reference label to its participant ID.

    The reference can be a baseline (hash-based ID) or a candidate run
    ('env_id/config_hash'). Labels are unique across `runs` (config validation).
    """
    for baseline_id, cfg in BASELINE_CONFIGS.items():
        if cfg.get("label") == label:
            return baseline_id
    for run_id, cfg in RUN_CONFIGS.items():
        if cfg.get("_is_candidate", False) and cfg.get("label") == label:
            return run_id
    available = [cfg.get("label") for cfg in BASELINE_CONFIGS.values()] + [
        cfg.get("label")
        for cfg in RUN_CONFIGS.values()
        if cfg.get("_is_candidate", False)
    ]
    raise ValueError(
        f"No baseline or candidate run with label {label!r} found. "
        f"Available labels: {available}"
    )


def experiment_participant_ids() -> list[str]:
    """Baseline ids and candidate run ids: the forecasts compared in the experiment."""
    return list(BASELINE_CONFIGS) + [
        run_id
        for run_id, run_config in RUN_CONFIGS.items()
        if run_config.get("_is_candidate", False)
    ]


def participant_data_dir(participant_id: str) -> Path:
    if participant_id in BASELINE_CONFIGS:
        return OUT_ROOT / f"data/baselines/{participant_id}"
    return OUT_ROOT / f"data/runs/{participant_id}"


def collect_experiment_participants():
    return {
        pid: participant_data_dir(pid) / f"verif_aggregated_{VERIF_HASH}.nc"
        for pid in experiment_participant_ids()
    }


def participant_valid_mask_files(wildcards) -> list[Path]:
    """Valid masks of all participants for one reference time (rule
    verification_valid_mask*), or none when experiment.score_on_common_points is
    unset."""
    if not COMMON_POINTS_PARAMS:
        return []
    return [
        participant_data_dir(pid)
        / f"{wildcards.init_time}/valid_mask_{VALID_MASK_HASH}.nc"
        for pid in experiment_participant_ids()
    ]


# -----------------------------------------------
# Hashing functions
# -----------------------------------------------


def env_entry_hash(run_config: dict, model_type: str) -> str:
    """Hash of fields that determine the inference environment only.

    The environment (venv, squashfs) must be rebuilt if any of these change:
    - checkpoint (different model)
    - extra_requirements (different dependencies)
    - disable_local_eccodes_definitions (different ECCODES setup)
    - For temporal downscalers: the forecaster's env_id (different upstream model)
    """
    cfg = {k: v for k, v in run_config.items() if k in ENV_HASH_FIELDS}
    configs_to_hash = [cfg]
    if model_type == "temporal_downscaler" and run_config.get("forecaster"):
        # environment depends on which forecaster model (not which run config)
        configs_to_hash.append(run_config["forecaster"].get("env_id"))
    return generate_json_hash(configs_to_hash)


def run_specific_hash(run_config: dict, model_type: str) -> str:
    """Hash of fields that affect inference outputs but not the environment.

    Changes to these fields create a new run_id (new outputs) but reuse the environment:
    - steps (lead times)
    - config YAML file contents (inference parameters)
    - output_variables (variables written to the GRIB outputs), only when set, so
      that run_ids of configs without it are unchanged; sorted and deduplicated,
      so that the order of the list does not matter
    - ensemble (members and seed), only when set, so that run_ids of deterministic
      runs are unchanged
    - For temporal downscalers: the forecaster's run_id (which run's outputs to read)
    """
    run_fields = {"steps": run_config["steps"]}
    if run_config.get("output_variables") is not None:
        run_fields["output_variables"] = sorted(set(run_config["output_variables"]))
    if run_config.get("ensemble") is not None:
        run_fields["ensemble"] = run_config["ensemble"]
    configs_to_hash = [run_fields]
    with open(run_config["config"], "r") as f:
        configs_to_hash.append(yaml.safe_load(f))
    if model_type == "temporal_downscaler" and run_config.get("forecaster"):
        # run output depends on which forecaster RUN was used (not just the env)
        configs_to_hash.append(run_config["forecaster"].get("run_id"))
    return generate_json_hash(configs_to_hash)


def warn_ensembles_without_output_variables(run_configs: dict) -> None:
    """Warn about ensemble runs without output_variables.

    They write every field for every member, which multiplies storage by the
    number of members.
    """
    for run_id, run_config in run_configs.items():
        ensemble = run_config.get("ensemble")
        if ensemble is not None and run_config.get("output_variables") is None:
            logging.getLogger("snakemake").warning(
                "Run %s is an ensemble of %d members without output_variables: every "
                "field is written for every member. Set output_variables to limit "
                "storage.",
                run_id,
                ensemble["members"],
            )


def baseline_hash(baseline_config: dict) -> str:
    """Hash of fields that determine baseline identity (excludes display/legacy metadata)."""
    cfg = {k: v for k, v in baseline_config.items() if k not in BASELINE_HASH_EXCLUDE}
    return generate_json_hash(cfg)


def master_hash() -> str:
    """Generate a short hash of all the configurable components of the workflow."""
    configs_to_hash = [config]
    for run_id, run_config in RUN_CONFIGS.items():
        configs_to_hash.append(
            {
                "env": env_entry_hash(run_config, run_config["model_type"]),
                "run": run_specific_hash(run_config, run_config["model_type"]),
            }
        )
    return generate_json_hash(configs_to_hash)


def truth_hash(truth_config: dict) -> str:
    """Generate a short hash of the configs for the truth data."""
    cfg = {k: v for k, v in truth_config.items() if k not in TRUTH_HASH_EXCLUDE}
    return generate_json_hash(cfg)


def verif_hash(full_config: dict) -> str:
    """Hash of all settings that affect verification outputs.

    Combines the truth source with verification-method settings so that
    changing any of them (e.g. switching lapse_rate_correction on/off, or
    changing the station_holdout selection) produces new output
    paths and unconditionally triggers a rerun.
    """
    truth_cfg = {
        k: v for k, v in full_config["truth"].items() if k not in TRUTH_HASH_EXCLUDE
    }
    experiment_verif_cfg = {
        "lapse_rate_correction": full_config.get("lapse_rate_correction", True),
        "station_holdout": HOLDOUT_STATIONS,
    }
    # Scoring on common points makes the scores depend on the other participants.
    # Added only when set, so that existing outputs keep their paths.
    if COMMON_POINTS_PARAMS:
        experiment_verif_cfg["score_on_common_points"] = {
            "params": sorted(COMMON_POINTS_PARAMS),
            "participants": sorted(experiment_participant_ids()),
        }
    return generate_json_hash({"truth": truth_cfg, "verif": experiment_verif_cfg})


def valid_mask_hash(full_config: dict) -> str:
    """Hash of the settings that determine a participant's valid mask: the truth
    points it is mapped to and the masked parameters."""
    truth_cfg = {
        k: v for k, v in full_config["truth"].items() if k not in TRUTH_HASH_EXCLUDE
    }
    return generate_json_hash(
        {"truth": truth_cfg, "params": sorted(COMMON_POINTS_PARAMS)}
    )


def holdout_dep(_):
    """Holdout station file dependency: the CSV written by rule
    station_holdout, or no input when experiment.station_holdout is unset."""
    return [HOLDOUT_FILE] if HOLDOUT_STATIONS else []


def run_uses_holdout(run_id: str) -> bool:
    """Whether the run's inference config consumes the holdout station file.

    The contract is the file name: an inference config that references
    HOLDOUT_WORKDIR_NAME (e.g. as the nudge_toward_observation filter's
    holdout_station_file) gets the file staged in its working directory.
    """
    with open(RUN_CONFIGS[run_id]["config"], "r") as f:
        return HOLDOUT_WORKDIR_NAME in f.read()


def truth_file_dep(_):
    """Truth file dependency tracked by Snakemake.

    - anemoi dataset: the dataset itself
    - INCA: the static topography file; the archive directory is not tracked
      because its modification time changes whenever new runs are archived
    - jretrieve: none (live query)
    """
    root = config["truth"]["root"]
    if TRUTH_TYPE == "anemoi_dataset":
        return [root]
    if TRUTH_TYPE == "inca":
        return [str(inca.topography_file(root))]
    return []


# Fail fast: verify the prerequisites of the truth source at workflow-build time
# so a misconfigured environment is caught at launch, before any (expensive)
# inference job runs.
if TRUTH_TYPE == "jretrieve":
    from data_input.jretrieve import check_prerequisites, parse_selection

    _, _jretrieve_stage, _, _ = parse_selection(config["truth"]["root"])
    check_prerequisites(_jretrieve_stage)
elif TRUTH_TYPE == "inca":
    inca.check_prerequisites(config["truth"]["root"])


TRUTH_HASH = truth_hash(config["truth"])
HOLDOUT_STATIONS = sorted(EXPERIMENT_CONFIG.get("station_holdout") or [])
HOLDOUT_FILE = (
    OUT_ROOT / f"data/station_holdout/{generate_json_hash(HOLDOUT_STATIONS)}.csv"
)
HOLDOUT_WORKDIR_NAME = "holdout_stations.csv"
REGIONS = parse_regions()
_showcase = config["showcase"]
SHOWCASE_CONFIG = {
    "regions": parse_showcase_regions(),
    "params": _showcase["params"],
    "fps": _showcase["animations"]["frames_per_second"],
}
_comparisons = _showcase["comparisons"]
COMPARISONS_CONFIG = {
    "enabled": _comparisons["enabled"],
    "regions": parse_showcase_regions(_comparisons["domains"]),
    "baselines": _comparisons["baselines"],
    "lead_times": _comparisons["lead_times"],
    # station truth (jretrievedwh) cannot be drawn as a map
    "truth": _comparisons["truth"] and TRUTH_TYPE != "jretrieve",
}
EXPERIMENT_PARAMS = EXPERIMENT_CONFIG.get(
    "params", ["T_2M", "TD_2M", "SP_10M", "PS", "PMSL", "TOT_PREC6"]
)
REFTIMES = parse_reference_times()
RUN_CONFIGS = collect_all_runs()
ENV_CONFIGS = collect_all_envs()
BASELINE_CONFIGS = collect_all_baselines()
COMMON_POINTS_PARAMS = EXPERIMENT_CONFIG.get("score_on_common_points") or []
# After RUN_CONFIGS and BASELINE_CONFIGS: the hash includes the participants
# when experiment.score_on_common_points is set.
VERIF_HASH = verif_hash(config)
VALID_MASK_HASH = valid_mask_hash(config)
EXPERIMENT_PARTICIPANTS = collect_experiment_participants()
# Fail fast: an inference config that consumes the holdout station file needs
# experiment.station_holdout to be set.
if not HOLDOUT_STATIONS:
    for _run_id in RUN_CONFIGS:
        if run_uses_holdout(_run_id):
            raise ValueError(
                f"Inference config {RUN_CONFIGS[_run_id]['config']} of run {_run_id} "
                f"references {HOLDOUT_WORKDIR_NAME}, but experiment.station_holdout "
                "is not set."
            )
warn_ensembles_without_output_variables(RUN_CONFIGS)
_scorecard = EXPERIMENT_CONFIG.get("scorecards") or {}
SCORECARD_CONFIGS = (
    _scorecard.get("sections", {}) if _scorecard.get("enabled", True) else {}
)


def balfrin_cpus(mem_mb, actual_cpus=1, node_mem_mb=512_000, node_cpus=256):
    """Minimum cores to request on Balfrin based on memory usage.

    Balfrin CPU nodes have no per-job memory control: any job can consume the
    full node memory (512 GB / 256 cores).  To avoid starving other jobs,
    reserve cores proportional to expected memory so the scheduler implicitly
    limits total memory usage:

        cores = max(actual_cpus, ceil(mem_mb / node_mem_mb * node_cpus))
    """
    import math

    mem_cpus = math.ceil(mem_mb / node_mem_mb * node_cpus)
    return max(actual_cpus, mem_cpus)


def scoremap_runtime(wc):
    """SLURM runtime (minutes) for a scoremap job, which loops over all reference
    times. Accumulated parameters (TOT_PREC*) re-accumulate hourly analysis
    fields up to the lead time, so they scale with it. Per-reftime costs measured
    on Balfrin (MRB-980, REA-L-CH1 truth): ~4 s, plus ~0.3 s per lead hour for
    accumulated parameters; 1.5x headroom, at least 60 min."""
    import math

    seconds = 4.0
    if wc.param.startswith("TOT_PREC"):
        seconds += 0.3 * int(wc.leadtime)
    return max(60, math.ceil(1.5 * len(REFTIMES) * seconds / 60))


def run_grib_dir(run_id, init_time):
    """GRIB directory of a run's inference output, as read by downstream rules.

    Deterministic runs write ``grib/``; ensemble runs write ``grib/{NNN}/`` per member
    and, until ensemble verification lands, downstream rules use member 001 only.
    """
    base = Path(OUT_ROOT) / f"data/runs/{run_id}/{init_time}/grib"
    if RUN_CONFIGS[run_id].get("ensemble") is not None:
        return base / "001"
    return base


for _run_id, _rc in RUN_CONFIGS.items():
    if _rc.get("ensemble") is not None:
        logging.getLogger("snakemake").warning(
            "Run %s is an ensemble of %d members: verification, plots and MEC use "
            "member 001 only until ensemble verification is implemented.",
            _run_id,
            _rc["ensemble"]["members"],
        )
