import copy
from datetime import datetime, timedelta
import yaml
import hashlib
import logging
import json

logging.basicConfig(level=logging.WARNING)

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
    """Collect all runs defined in the configuration."""
    runs = {}
    for run_entry in copy.deepcopy(config["runs"]):
        model_type = next(iter(run_entry))
        run_config = run_entry[model_type]
        run_config["model_type"] = model_type
        run_id = run_config.pop("run_id")
        runs[run_id] = run_config
        if model_type == "interpolator":
            if "forecaster" not in run_config or run_config["forecaster"] is None:
                logging.warning(
                    f"Interpolator '{run_id}' has not been set a 'forecaster' config. Please use the 'interpolator_from_files.yaml' config to run an interpolator without a forecaster."
                )
                return runs
            run_id = run_config["forecaster"]["run_id"]
            runs[run_id] = run_config["forecaster"]
    return runs


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
    for baseline_entry in config["baselines"]:
        baseline = next(iter(baseline_entry.values()))
        baseline_id = baseline["baseline_id"]
        label = baseline.get("label", baseline_id)
        participants[baseline_id] = (
            OUT_ROOT / f"data/baselines/{baseline_id}/verif_aggregated.nc"
        )
    for run_entry in config["runs"]:
        # every run entry is a single-key dict
        # where the key is the model type ("forecaster", "interpolator", etc.)
        run = next(iter(run_entry.values()))
        run_id = run["run_id"]
        label = run.get("label", run_id)
        participants[label] = OUT_ROOT / f"data/runs/{run_id}/verif_aggregated.nc"
    return participants


def _inference_routing_fn(wc):

    run_config = RUN_CONFIGS[wc.run_id]

    if run_config["model_type"] == "forecaster":
        input_path = f"logs/inference_forecaster/{wc.run_id}-{wc.init_time}.ok"
    elif run_config["model_type"] == "interpolator":
        input_path = f"logs/inference_interpolator/{wc.run_id}-{wc.init_time}.ok"
    else:
        raise ValueError(f"Unsupported model type: {run_config['model_type']}")

    return OUT_ROOT / input_path


RUN_CONFIGS = collect_all_runs()
BASELINE_CONFIGS = collect_all_baselines()
EXPERIMENT_PARTICIPANTS = collect_experiment_participants()
