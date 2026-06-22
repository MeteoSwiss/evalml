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
    """Parse regions from the configuration."""
    cfg = config["experiment"]["stratification"]
    regions = [f"{cfg['root']}/{region}.shp" for region in cfg["regions"]]
    regions_txt = ",".join(regions)
    return regions_txt


def parse_showcase_regions():
    """Parse showcase domains from config.

    Returns a dict mapping domain name -> {extent, projection}.
    Named domains (strings) have extent=None and projection=None,
    meaning the plot script will fall back to the DOMAINS lookup.
    Custom domains carry their explicit extent and projection.
    """
    result = {}
    for r in (
        config.get("showcase", {})
        .get("animations", {})
        .get("domains", ["globe", "europe", "switzerland"])
    ):
        if isinstance(r, str):
            result[r] = {"extent": None, "projection": None}
        else:
            result[r["name"]] = {
                "extent": r.get("extent"),
                "projection": r.get("projection", "orthographic"),
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


def resolve_baseline_id(label: str) -> str:
    """Resolve a baseline label to its hash-based ID.

    Scorecard configs reference baselines by human-readable label (e.g. 'IFS').
    This finds the matching baseline_id in BASELINE_CONFIGS.
    Raises ValueError if the label doesn't match any registered baseline.
    """
    for baseline_id, cfg in BASELINE_CONFIGS.items():
        if cfg.get("label") == label:
            return baseline_id
    available = [cfg.get("label") for cfg in BASELINE_CONFIGS.values()]
    raise ValueError(
        f"No baseline with label {label!r} found. "
        f"Available baseline labels: {available}"
    )


def collect_experiment_participants():
    participants = {}
    for base in BASELINE_CONFIGS.keys():
        participants[base] = (
            OUT_ROOT / f"data/baselines/{base}/verif_aggregated_{VERIF_HASH}.nc"
        )
    for exp in RUN_CONFIGS.keys():
        if RUN_CONFIGS[exp].get("_is_candidate", False):
            participants[exp] = (
                OUT_ROOT / f"data/runs/{exp}/verif_aggregated_{VERIF_HASH}.nc"
            )
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
    - For temporal downscalers: the forecaster's run_id (which run's outputs to read)
    """
    configs_to_hash = [{"steps": run_config["steps"]}]
    with open(run_config["config"], "r") as f:
        configs_to_hash.append(yaml.safe_load(f))
    if model_type == "temporal_downscaler" and run_config.get("forecaster"):
        # run output depends on which forecaster RUN was used (not just the env)
        configs_to_hash.append(run_config["forecaster"].get("run_id"))
    return generate_json_hash(configs_to_hash)


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
    changing either (e.g. switching lapse_rate_correction on/off) produces
    new output paths and unconditionally triggers a rerun.
    """
    truth_cfg = {
        k: v for k, v in full_config["truth"].items() if k not in TRUTH_HASH_EXCLUDE
    }
    experiment_verif_cfg = {
        "lapse_rate_correction": full_config.get("experiment", {}).get(
            "lapse_rate_correction", True
        ),
    }
    return generate_json_hash({"truth": truth_cfg, "verif": experiment_verif_cfg})


def truth_file_dep(_):
    """Truth file dependency: a real path for zarr, but a live-query
    marker (no input file) for jretrieve."""
    root = config["truth"]["root"]
    return [] if "jretrieve" in str(root) else [root]


# Fail fast: when the truth source is the live DWH (jretrievedwh), verify its
# prerequisites at workflow-build time so a misconfigured environment is caught
# at launch, before any (expensive) inference job runs.
if "jretrieve" in str(config["truth"]["root"]):
    from data_input.jretrieve import check_prerequisites, parse_selection

    _, _jretrieve_stage, _ = parse_selection(config["truth"]["root"])
    check_prerequisites(_jretrieve_stage)


TRUTH_HASH = truth_hash(config["truth"])
VERIF_HASH = verif_hash(config)
REGIONS = parse_regions()
SHOWCASE_REGIONS = parse_showcase_regions()
SHOWCASE_PARAMS = config.get("showcase", {}).get("params", ["T_2M", "SP_10M"])
EXPERIMENT_PARAMS = config.get("experiment", {}).get(
    "params", ["T_2M", "TD_2M", "U_10M", "V_10M", "PS", "PMSL", "TOT_PREC"]
)
REFTIMES = parse_reference_times()
RUN_CONFIGS = collect_all_runs()
ENV_CONFIGS = collect_all_envs()
BASELINE_CONFIGS = collect_all_baselines()
EXPERIMENT_PARTICIPANTS = collect_experiment_participants()
_scorecard = config.get("experiment", {}).get("scorecards") or {}
SCORECARD_CONFIGS = (
    _scorecard.get("sections", {}) if _scorecard.get("enabled", True) else {}
)


# Period-accumulated params verify a [lead - period, lead] window, so they have
# no value at lead times shorter than one step spacing (e.g. no 0h precip map).
# Short and canonical names both appear across the workflow (showcases vs maps).
ACCUMULATED_PARAMS = {"TOT_PREC", "tp"}


def resolve_leadtimes(steps_spec, requested="all", param=None):
    """Lead times to compute for a single participant.

    A run or baseline produces only the lead times in its own ``steps`` spec
    (``start/stop/step``, hours). This returns those of the ``requested``
    selection that the participant actually produces — the literal ``"all"``
    (every produced lead time) or an explicit list of ints — so a 36h lead is
    never requested of an ICON-CH1 baseline (steps ``0/33/6``), nor a >120h
    lead of ICON-CH2. Explicitly requested lead times the participant cannot
    produce are skipped with a warning. For accumulated ``param``s, lead times
    shorter than one step spacing are dropped (no accumulation window).
    """
    start, end, step = map(int, steps_spec.split("/"))
    supported = set(range(start, end + 1, step))
    wanted = supported if requested == "all" else set(requested)

    unsupported = sorted(wanted - supported)
    if unsupported:
        logging.getLogger("snakemake").warning(
            "Skipping lead time(s) %sh: not produced by forecast steps '%s'.",
            unsupported,
            steps_spec,
        )

    valid = wanted & supported
    if param in ACCUMULATED_PARAMS:
        valid = {lt for lt in valid if lt >= step}
    return sorted(valid)
