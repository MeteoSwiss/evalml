"""Watch the realv2 inference output and build the GIF once the 120h run finishes.

Polls `realv2.nc` until its last step reaches the 120h lead time
(date + 120h = 2024-02-06T00:00) with real (non-NaN) data and a stable file
size, then renders the prediction-vs-truth animation. Run detached in the
background; it prints progress and exits after saving the GIF.
"""

import time
from pathlib import Path

import numpy as np
import xarray as xr

REALV2 = Path(
    "/users/rradev/evalml/output/data/runs/forecaster-dadd-20dc/"
    "f9d8/202401010000/realv2.nc"
)
OUTFN = Path("/users/rradev/evalml/figures/realv2_vmax10m.gif")
TARGET_END = np.datetime64("2024-02-06T00:00", "ns")  # date 2024-02-01 + 120h
POLL_SECONDS = 30


def last_state() -> tuple[np.datetime64 | None, bool]:
    """Return (last valid_time, last_step_has_data) or (None, False) if unreadable."""
    try:
        with xr.open_dataset(REALV2) as ds:
            times = ds["time"].values
            if len(times) == 0:
                return None, False
            last = ds["VMAX_10M"].isel(time=-1).values
            return np.datetime64(times[-1], "ns"), not bool(np.all(np.isnan(last)))
    except Exception:
        return None, False


def main() -> None:
    print(f"watching {REALV2} until last step == {TARGET_END}", flush=True)
    prev_reported = None
    prev_size = -1
    while True:
        last_time, has_data = last_state()
        if last_time is not None and last_time != prev_reported:
            print(f"  current last step: {str(last_time)[:16]} (data={has_data})", flush=True)
            prev_reported = last_time

        if last_time is not None and last_time >= TARGET_END and has_data:
            size = REALV2.stat().st_size
            if size == prev_size:  # unchanged over one poll -> write finished
                print("120h run complete and file stable; rendering GIF", flush=True)
                break
            prev_size = size

        time.sleep(POLL_SECONDS)

    from paper_plots_realv2_animation import main as animate

    animate(OUTFN)
    print("done", flush=True)


if __name__ == "__main__":
    main()
