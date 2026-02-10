import logging
import copy
from datetime import datetime, timedelta
import yaml
import hashlib
import json

CONFIG_ROOT = Path("config").resolve()
OUT_ROOT = Path(config["locations"]["output_root"])

DATETIME_FORMAT = "%Y-%m-%dT%H:%M"
HASH_LENGTH = 4


# ============================================================================
# Utility Functions
# ============================================================================


def parse_toml(toml_file, key):
    """Parse a key (e.g. 'project.requires-python') from a TOML file handle."""
    import toml

    content = toml.load(toml_file)
    # support dotted keys
    for part in key.split("."):
        content = content.get(part, {})
    if isinstance(content, str):
        return content.lstrip(">=< ").strip()
    raise ValueError(f"Expected a string for key '{key}', got: {content}")


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


def register_run(model_type, run_config, as_candidate=True):
    """Parse a run configuration and assign a unique run ID."""
    run_cfg = copy.deepcopy(run_config)
    mlflow_id_short = run_cfg["mlflow_id"][:HASH_LENGTH]
    run_id_prefix = f"{model_type}-{mlflow_id_short}"
    run_id_hash = run_entry_hash(run_cfg)
    run_cfg["_is_candidate"] = as_candidate
    run_cfg["model_type"] = model_type
    run_id = f"{run_id_prefix}{run_id_hash}"
    out = {}
    if model_type == "interpolator":
        forecaster = run_cfg["forecaster"]
        if forecaster is None:
            dependency = "analysis"
        else:
            dependency_entry = register_run(
                "forecaster", forecaster, as_candidate=False
            )
            dependency = next(iter(dependency_entry))
            out |= dependency_entry
            run_cfg["forecaster"]["run_id"] = dependency
        run_id += f"-on-{dependency}"
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


def master_hash() -> str:
    """Generate a short hash of all the configurable components of the workflow."""
    configs_to_hash = [config]
    for run_id, run_config in RUN_CONFIGS.items():
        configs_to_hash.append(run_entry_hash(run_config))
    return generate_json_hash(configs_to_hash)


RUN_ENTRY_HASH_EXCLUDE = ["label", "_is_candidate"]


def run_entry_hash(run_config: dict) -> str:
    """Generate a short hash of a run entry."""
    cfg = copy.deepcopy(run_config)
    for key in RUN_ENTRY_HASH_EXCLUDE:
        cfg.pop(key, None)
    configs_to_hash = [cfg]
    with open(cfg["config"], "r") as f:
        configs_to_hash.append(yaml.safe_load(f))
    if "forecaster" in cfg and cfg["forecaster"] is not None:
        configs_to_hash.append(run_entry_hash(cfg["forecaster"]))
    return generate_json_hash(configs_to_hash)


REGIONS = parse_regions()
REFTIMES = parse_reference_times()
RUN_CONFIGS = collect_all_runs()
BASELINE_CONFIGS = collect_all_baselines()
EXPERIMENT_PARTICIPANTS = collect_experiment_participants()
