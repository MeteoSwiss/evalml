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
        if "forecaster" in run_config:
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


def _reftimes_groups():
    cfg = config["dates"]
    group_size = config["execution"]["run_group_size"]
    groups = []
    for i in range(0, len(REFTIMES), group_size):
        group = REFTIMES[i : i + group_size]
        groups.append(group)
    return groups


REFTIMES = _reftimes()

REFTIMES_GROUPS = _reftimes_groups()
REFTIME_TO_GROUP = {
    reftime.strftime("%Y%m%d%H%M"): group_index
    for group_index, group in enumerate(REFTIMES_GROUPS)
    for reftime in group
}


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
            run_id = run_config["forecaster"]["run_id"]
            runs[run_id] = run_config["forecaster"]
    return runs


def collect_all_baselines():
    """Collect all baselines defined in the configuration."""
    baselines = config.get("baseline", {})
    if isinstance(baselines, list):
        return baselines
    elif isinstance(baselines, dict):
        return list(baselines.keys())
    elif isinstance(baselines, str):
        return [baselines]
    else:
        raise ValueError("Baselines should be a list, dict, or string.")


def collect_experiment_participants():
    participants = {}
    for baseline in collect_all_baselines():
        participants[baseline] = (
            OUT_ROOT / f"data/baselines/{baseline}/verif_aggregated.nc"
        )
    for run_entry in config["runs"]:
        # every run entry is a single-key dict
        # where the key is the model type ("forecaster", "interpolator", etc.)
        run = next(iter(run_entry.values()))
        run_id = run["run_id"]
        label = run.get("label", run_id)
        participants[label] = OUT_ROOT / f"data/runs/{run_id}/verif_aggregated.nc"
    return participants


RUN_CONFIGS = collect_all_runs()
EXPERIMENT_PARTICIPANTS = collect_experiment_participants()
