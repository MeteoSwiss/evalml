from datetime import datetime, timedelta
import yaml


configfile: "config/anemoi_inference.yaml"


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


# def study_reftimes(study):
#     cfg = config["studies"][study]["init_times"]
#     start = datetime.strptime(cfg["start"], "%Y-%m-%dT%H:%M")
#     end = datetime.strptime(cfg["end"], "%Y-%m-%dT%H:%M")
#     freq = _parse_timedelta(cfg["frequency"])
#     times = []
#     t = start
#     while t <= end:
#         times.append(t)
#         t += freq
#     return times


# def study_reftimes_groups(study):
#     cfg = config["studies"][study]["init_times"]
#     reftimes = study_reftimes(study)
#     group_size = config["execution"]["run_group_size"]
#     groups = []
#     for i in range(0, len(reftimes), group_size):
#         group = reftimes[i : i + group_size]
#         groups.append(group)
#     return groups

# def reftimes_to_group(reftimes_groups):
#     """Convert a list of reference times to a list of groups."""
#     group_size = config["execution"]["run_group_size"]
#     return [
#         reftimes[i : i + group_size] for i in range(0, len(reftimes), group_size)
#     ]

