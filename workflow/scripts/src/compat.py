from datetime import datetime
from pathlib import Path

import numpy as np


def load_state_from_raw(file: Path, paramlist: list[str] | None = None) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    _state: dict[str, np.ndarray] = np.load(file, allow_pickle=True)
    reftime = datetime.strptime(file.parents[1].name, "%Y%m%d%H%M")
    validtime = datetime.strptime(file.stem, "%Y%m%d%H%M%S")
    state = {}
    state["longitudes"] = _state["longitudes"]
    state["latitudes"] = _state["latitudes"]
    state["valid_time"] = validtime
    state["lead_time"] = state["valid_time"] - reftime
    state["forecast_reference_time"] = reftime
    state["fields"] = {} 
    for key, value in _state.items():
        if key.startswith("field_"):
            state["fields"][key.removeprefix("field_")] = value
    return state
