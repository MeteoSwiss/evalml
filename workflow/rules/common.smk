from datetime import datetime, timedelta
import yaml


configfile: "config/anemoi_inference.yaml"

CONFIG_ROOT = Path("config").resolve()
OUT_ROOT = Path(config["locations"]["output_root"]).resolve()


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
    cfg = config["init_times"]
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
    cfg = config["init_times"]
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


def collect_study_participants(wc):

    with open(CONFIG_ROOT / f"studies/{wc.study}.yaml", "r") as f:
        study = yaml.safe_load(f)
    
    baselines = study.get("baselines", [])
    experiments = study.get("experiments", [])
    if not baselines and not experiments:
        raise ValueError(f"Study '{wc.study}' has no baselines or experiments defined.")
    participants = []
    for baseline in baselines:
        participants.append(OUT_ROOT / f"baselines/{baseline}/verif_aggregated.csv")
    for experiment in experiments:
        participants.append(OUT_ROOT / f"experiments/{experiment}/verif_aggregated.csv")
    return participants