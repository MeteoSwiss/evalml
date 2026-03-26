import logging
import copy
from datetime import datetime, timedelta
import yaml
import hashlib
import json
from urllib.parse import urlparse

CONFIG_ROOT = Path("config").resolve()
OUT_ROOT = Path(config["locations"]["output_root"])

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
    times = []
    t = start
    while t <= end:
        times.append(t)
        t += freq
    return times


def parse_regions():
    """Parse regions from the configuration."""
    cfg = config["stratification"]
    regions = [f"{cfg['root']}/{region}.shp" for region in cfg["regions"]]
    regions_txt = ",".join(regions)
    return regions_txt


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
        return checkpoint_uri.split("/")[-1][:HASH_LENGTH]
    elif ckpt_type == "huggingface":
        return checkpoint_uri.split("/")[-1].split(".")[0]
    elif ckpt_type == "local":
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
    if model_type == "interpolator":
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
    if model_type == "interpolator":
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
        run_config = run_entry[model_type]
        runs |= register_run(model_type, run_config)
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
    for baseline_entry in copy.deepcopy(config["baselines"]):
        baseline_type = next(iter(baseline_entry))
        baseline_config = baseline_entry[baseline_type]
        baseline_id = baseline_config.pop("baseline_id")
        baselines[baseline_id] = baseline_config
    return baselines


def collect_experiment_participants():
    participants = {}
    for base in BASELINE_CONFIGS.keys():
        participants[base] = OUT_ROOT / f"data/baselines/{base}/verif_aggregated.nc"
    for exp in RUN_CONFIGS.keys():
        if RUN_CONFIGS[exp].get("_is_candidate", False):
            participants[exp] = OUT_ROOT / f"data/runs/{exp}/verif_aggregated.nc"
    return participants


# -----------------------------------------------
# Hashing functions
# -----------------------------------------------


def env_entry_hash(run_config: dict, model_type: str) -> str:
    """Hash of fields that determine the inference environment only.

    The environment (venv, squashfs) must be rebuilt if any of these change:
    - checkpoint (different model)
    - extra_requirements (different dependencies)
    - disable_local_eccodes_definitions (different ECCODES setup)
    - For interpolators: the forecaster's env_id (different upstream model)
    """
    cfg = {k: v for k, v in run_config.items() if k in ENV_HASH_FIELDS}
    configs_to_hash = [cfg]
    if model_type == "interpolator" and run_config.get("forecaster"):
        # environment depends on which forecaster model (not which run config)
        configs_to_hash.append(run_config["forecaster"].get("env_id"))
    return generate_json_hash(configs_to_hash)


def run_specific_hash(run_config: dict, model_type: str) -> str:
    """Hash of fields that affect inference outputs but not the environment.

    Changes to these fields create a new run_id (new outputs) but reuse the environment:
    - steps (lead times)
    - config YAML file contents (inference parameters)
    - For interpolators: the forecaster's run_id (which run's outputs to read)
    """
    configs_to_hash = [{"steps": run_config["steps"]}]
    with open(run_config["config"], "r") as f:
        configs_to_hash.append(yaml.safe_load(f))
    if model_type == "interpolator" and run_config.get("forecaster"):
        # run output depends on which forecaster RUN was used (not just the env)
        configs_to_hash.append(run_config["forecaster"].get("run_id"))
    return generate_json_hash(configs_to_hash)


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


REGIONS = parse_regions()
REFTIMES = parse_reference_times()
RUN_CONFIGS = collect_all_runs()
ENV_CONFIGS = collect_all_envs()
BASELINE_CONFIGS = collect_all_baselines()
EXPERIMENT_PARTICIPANTS = collect_experiment_participants()
