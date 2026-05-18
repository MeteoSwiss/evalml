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
        baseline_id = Path(baseline_config["root"]).stem
        baselines[baseline_id] = baseline_config

    # Backward compatibility with legacy top-level `baselines` block.
    for baseline_entry in copy.deepcopy(config.get("baselines", [])):
        baseline_type = next(iter(baseline_entry))
        baseline_config = baseline_entry[baseline_type]
        baseline_id = Path(baseline_config["root"]).stem
        baseline_config.pop("baseline_id", None)
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
SHOWCASE_REGIONS = parse_showcase_regions()
SHOWCASE_PARAMS = config.get("showcase", {}).get("params", ["T_2M", "SP_10M"])
REFTIMES = parse_reference_times()
RUN_CONFIGS = collect_all_runs()
ENV_CONFIGS = collect_all_envs()
BASELINE_CONFIGS = collect_all_baselines()
EXPERIMENT_PARTICIPANTS = collect_experiment_participants()


# ============================================================================
# Showcase animation helpers
# ============================================================================


def sanitize_label(label: str) -> str:
    """Sanitize a run label for use as a path component."""
    import re as _re
    return _re.sub(r"[^a-zA-Z0-9_-]", "_", label)


def collect_zarr_sources() -> dict:
    """Collect zarr-based sources (truth + baselines) keyed by their label.

    Returns a dict mapping label -> {root, step, total_hours, source_type}.
    """
    sources = {}

    # Truth (analysis)
    truth_cfg = config.get("truth", {})
    if truth_cfg and "root" in truth_cfg:
        label = truth_cfg.get("label", "truth")
        sources[label] = {
            "root": truth_cfg["root"],
            "step": 1,
            "total_hours": 120,
            "source_type": "analysis",
        }

    # Baselines
    for baseline_id, cfg in BASELINE_CONFIGS.items():
        label = cfg.get("label", baseline_id)
        _, total, step = map(int, cfg.get("steps", "0/120/1").split("/"))
        sources[label] = {
            "root": cfg["root"],
            "step": step,
            "total_hours": total,
            "source_type": "baseline",
        }

    return sources


def _resolve_label(label: str) -> dict:
    """Resolve a label to a source descriptor used in comparison entries.

    Returns a dict with:
      type       — ``'run'`` or ``'zarr'``
      run_id     — present when type == 'run'
      label      — present when type == 'zarr'
      step       — time step in hours
    """
    # ML runs (candidates and non-candidates such as nested forecasters)
    for run_id, cfg in RUN_CONFIGS.items():
        if cfg.get("label") == label:
            return {
                "type": "run",
                "run_id": run_id,
                "step": int(cfg["steps"].split("/")[2]),
            }
    # Zarr sources (truth / baselines)
    if label in ZARR_SOURCES:
        z = ZARR_SOURCES[label]
        return {"type": "zarr", "label": label, "step": z["step"]}

    available_runs = sorted({cfg.get("label") for cfg in RUN_CONFIGS.values() if cfg.get("label")})
    available_zarr = sorted(ZARR_SOURCES.keys())
    raise ValueError(
        f"No source found with label {label!r}. "
        f"ML run labels: {available_runs}. "
        f"Zarr source labels: {available_zarr}."
    )


def label_to_run_id(label: str) -> str:
    """Return the run_id for the given label (ML runs only).

    Searches both candidate and non-candidate runs (e.g. nested forecasters).
    Raises ValueError if not found.
    """
    for run_id, cfg in RUN_CONFIGS.items():
        if cfg.get("label") == label:
            return run_id
    available = sorted({cfg.get("label") for cfg in RUN_CONFIGS.values() if cfg.get("label")})
    raise ValueError(
        f"No run found with label {label!r}. Available ML run labels: {available}"
    )


def parse_showcase_animation_runs() -> list:
    """Return the run_ids to animate individually.

    If ``animations.runs`` is set in the showcase config, filter by those labels
    (ML runs only; zarr sources have their own animation pipeline).
    Otherwise return all candidate run_ids.
    """
    labels = config.get("showcase", {}).get("animations", {}).get("runs")
    if labels is None:
        return list(collect_all_candidates().keys())
    return [label_to_run_id(label) for label in labels]


def parse_showcase_comparisons() -> list:
    """Parse ``animations.comparisons`` from the showcase config.

    Each returned entry has:
      id     — sanitised ``{left_label}_vs_{right_label}`` path component
      left   — source descriptor (type, run_id/label, step)
      right  — source descriptor (type, run_id/label, step)
    """
    comparisons = config.get("showcase", {}).get("animations", {}).get("comparisons", [])
    result = []
    for c in comparisons:
        left_label = c["left"]
        right_label = c["right"]
        result.append({
            "id": f"{sanitize_label(left_label)}_vs_{sanitize_label(right_label)}",
            "left": _resolve_label(left_label),
            "right": _resolve_label(right_label),
        })
    return result


ZARR_SOURCES = collect_zarr_sources()
SHOWCASE_ANIMATION_RUN_IDS = parse_showcase_animation_runs()
SHOWCASE_COMPARISONS = parse_showcase_comparisons()
