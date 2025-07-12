from datetime import datetime, timedelta
import yaml
import hashlib
import json

CONFIG_ROOT = Path("config").resolve()
OUT_ROOT = Path(config["locations"]["output_root"]).resolve()


def short_hash_config():
    """Generate a short hash of the configuration file."""
    with open(CONFIG_ROOT / "anemoi_inference.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg_str = json.dumps([config, cfg], sort_keys=True)
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
    return [cfg["run_id"] for cfg in config["runs"].values()]


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
            OUT_ROOT / f"data/baselines/{baseline}/verif_aggregated.csv"
        )
    for name, run in config["runs"].items():
        label = run.get("label", name)
        participants[label] = (
            OUT_ROOT / f"data/runs/{run['run_id']}/verif_aggregated.csv"
        )
    return participants


EXPERIMENT_PARTICIPANTS = collect_experiment_participants()
