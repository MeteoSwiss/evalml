from datetime import datetime, timedelta
import yaml


configfile: "config/anemoi_inference.yaml"


def _parse_timedelta(td):
    if not isinstance(td, str):
        raise ValueError("Expected a string in the format 'Xd' or 'Xh'") 
    magnitude, unit = int(td[:-1]), td[-1]
    match unit:
        case 'd':
            return timedelta(days=magnitude)
        case 'h':
            return timedelta(hours=magnitude)
        case _:
            raise ValueError(f"Unsupported time unit: {unit}. Only 'd' and 'h' are supported.")

def _reftimes():
    cfg = config["init_times"]
    start = datetime.strptime(cfg["start"], "%Y-%m-%dT%H:%M")
    end = datetime.strptime(cfg["end"], "%Y-%m-%dT%H:%M")
    freq = _parse_timedelta(cfg["frequency"])
    times = []
    t = start
    while t <= end:
        times.append(t.strftime("%Y%m%d%H%M"))
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
    reftime: group_index
    for group_index, group in enumerate(REFTIMES_GROUPS)
    for reftime in group
}
