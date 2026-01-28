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
    if "label" in run_config:
        run_config = copy.deepcopy(run_config)
        run_config.pop("label")
    cfg_str = json.dumps(run_config, sort_keys=True)
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


def collect_all_runs():
    """Collect all runs defined in the configuration, including secondary runs."""
    runs = {}
    for run_entry in copy.deepcopy(config["runs"]):
        model_type = next(iter(run_entry))
        run_config = run_entry[model_type]
        run_config["model_type"] = model_type
        run_id = run_config["mlflow_id"][0:4]

        if model_type == "interpolator":
            if "forecaster" not in run_config or run_config["forecaster"] is None:
                fcst_id = "ana"
            else:
                fcst_id = run_config["forecaster"]["mlflow_id"][0:4]
                # Ensure a proper 'forecaster' entry exists with model_type
                fore_cfg = copy.deepcopy(run_config["forecaster"])
                fore_cfg["model_type"] = "forecaster"
                # make sure we don't hash the is_candidate status
                fore_id = short_hash_runconfig(fore_cfg)
                fore_cfg["is_candidate"] = False  # exclude from outputs
                runs[f"{fcst_id}-{fore_id}"] = fore_cfg
                # add run_id of forecaster to interpolator config
                run_config["forecaster"]["run_id"] = f"{fcst_id}-{fore_id}"
            run_id = f"{run_id}-{fcst_id}"

        # add the hash of the config to the run id
        run_id = f"{run_id}-{short_hash_runconfig(run_config)}"

        # make sure we don't hash the is_candidate status
        run_config["is_candidate"] = True
        # Register this (possibly composite) run inside the loop
        runs[run_id] = run_config
    return runs


def collect_all_candidates():
    """Collect participating runs ('candidates') only."""
    runs = collect_all_runs()
    candidates = {}
    for run_id, run_config in runs.items():
        if run_config.get("is_candidate", False):
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
        if RUN_CONFIGS[exp].get("is_candidate", False):
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


def _enable_nudging(wc):
    run_config = RUN_CONFIGS[wc.run_id]
    if run_config["model_type"] == "interpolator":
        if RUN_CONFIGS[run_config["forecaster"]["run_id"]]["nudging"]:
            return rules.nudge_analysis.output.okfile
    return []

def _regions():
    cfg = config["stratification"]
    regions = [f"{cfg['root']}/{region}.shp" for region in cfg["regions"]]
    # convert list of strings in regions to comma-separated string
    regions_txt = ",".join(regions)
    return regions_txt


REGION_TXT = _regions()


RUN_CONFIGS = collect_all_runs()
BASELINE_CONFIGS = collect_all_baselines()
EXPERIMENT_HASH = short_hash_config()
EXPERIMENT_PARTICIPANTS = collect_experiment_participants()
