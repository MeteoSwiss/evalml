import logging
import copy
from datetime import datetime, timedelta
import yaml
import hashlib
import json

CONFIG_ROOT = Path("config").resolve()
OUT_ROOT = Path(config["locations"]["output_root"])


def short_hash_config():
    """Generate a short hash of the configuration file."""
    configs_to_hash = []
    for run_id, run_config in RUN_CONFIGS.items():
        with open(run_config["config"], "r") as f:
            configs_to_hash.append(yaml.safe_load(f))
        if "forecaster" in run_config and run_config["forecaster"] is not None:
            with open(run_config["forecaster"]["config"], "r") as f:
                configs_to_hash.append(yaml.safe_load(f))
    cfg_str = json.dumps([config, *configs_to_hash], sort_keys=True)
    return hashlib.sha256(cfg_str.encode()).hexdigest()[:8]


def short_hash_runconfig(run_config):
    """Generate a short hash of the run block in the config file."""
    # 'label' has no functional impact on the results of a model run, so we exclude it
    cfg = copy.deepcopy(run_config)
    cfg.pop("label", None)
    cfg.pop("_is_candidate", None)

    cfg_str = json.dumps(
        cfg,
        sort_keys=True,
        separators=(",", ":"),  # consistent separators for hashing
        ensure_ascii=False,
    )
    return hashlib.sha256(cfg_str.encode()).hexdigest()[:8]


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


def _parse_timedelta(td):
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


def _reftimes():
    cfg = config["dates"]
    if isinstance(cfg, list):
        return [datetime.strptime(t, "%Y-%m-%dT%H:%M") for t in cfg]
    start = datetime.strptime(cfg["start"], "%Y-%m-%dT%H:%M")
    end = datetime.strptime(cfg["end"], "%Y-%m-%dT%H:%M")
    freq = _parse_timedelta(cfg["frequency"])
    times = []
    t = start
    while t <= end:
        times.append(t)
        t += freq
    return times


REFTIMES = _reftimes()


def _run_config(run_entry: dict) -> tuple[str, dict]:
    model_type = next(iter(run_entry))
    run_config = copy.deepcopy(run_entry[model_type])
    run_config["model_type"] = model_type
    return run_config


def _make_run_key(prefix: str, run_config: dict) -> tuple[str, str]:
    """ """
    hsh = short_hash_runconfig(run_config)
    return f"{prefix}-{hsh}"


def _register_forecaster_dependency(runs: dict, forecaster_cfg: dict) -> str:
    """
    Register a forecaster config as a non-candidate dependency.
    Returns the generated forecaster run_id (key in runs).
    """
    fcst_cfg = copy.deepcopy(forecaster_cfg)
    fcst_cfg["model_type"] = "forecaster"

    run_label = f"forecaster-{fcst_cfg["mlflow_id"][:4]}"
    fcst_key = _make_run_key(run_label, fcst_cfg)
    if fcst_key in runs:
        return fcst_key  # already registered

    fcst_cfg["_is_candidate"] = False  # exclude from outputs
    runs[fcst_key] = fcst_cfg
    return fcst_key


def collect_all_runs() -> dict:
    """Collect all runs defined in the configuration, including secondary runs."""
    runs: dict[str, dict] = {}
    for run_entry in config["runs"]:
        run_cfg = _run_config(run_entry)

        model_type = run_cfg["model_type"]
        mlflow_id4 = run_cfg["mlflow_id"][:4]
        prefix = f"{model_type}-{mlflow_id4}"

        if model_type == "interpolator":
            forecaster = run_cfg.get("forecaster")

            if not forecaster:
                # "analysis" dependency marker
                suffix = "analysis"

            else:
                fcst_key = _register_forecaster_dependency(runs, forecaster)
                run_cfg["forecaster"]["run_id"] = fcst_key
                suffix = f"forecaster-{forecaster['mlflow_id'][:4]}"

            run_label = f"{prefix}-on-{suffix}"
        else:
            run_label = prefix

        run_key = _make_run_key(run_label, run_cfg)  # unique run identifier

        run_cfg["_is_candidate"] = True
        runs[run_key] = run_cfg

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


def _inference_routing_fn(wc):

    run_config = RUN_CONFIGS[wc.run_id]

    if run_config["model_type"] == "forecaster":
        input_path = f"logs/prepare_inference_forecaster/{wc.run_id}-{wc.init_time}.ok"
    elif run_config["model_type"] == "interpolator":
        input_path = (
            f"logs/prepare_inference_interpolator/{wc.run_id}-{wc.init_time}.ok"
        )
    else:
        raise ValueError(f"Unsupported model type: {run_config['model_type']}")

    return OUT_ROOT / input_path


def _regions():
    cfg = config["stratification"]
    regions = [f"{cfg['root']}/{region}.shp" for region in cfg["regions"]]
    # convert list of strings in regions to comma-separated string
    regions_txt = ",".join(regions)
    return regions_txt


REGION_TXT = _regions()


RUN_CONFIGS = collect_all_runs()
BASELINE_CONFIGS = collect_all_baselines()
EXPERIMENT_PARTICIPANTS = collect_experiment_participants()
