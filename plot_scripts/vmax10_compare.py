"""Comparison paper plot: realv2 prediction vs the real (truth) dataset.

Renders, for each forecast step in the realv2 inference output, three panes on
the shared COSMO-CH rotated-pole grid:

  - Prediction: VMAX_10M from the anemoi-inference `realv2.nc`.
  - Truth: VMAX_10M from the dataset realv2 is trained against,
    `mch-realch1-fdb-1km-...-pl13-v2.0.zarr`, sampled at the same valid times.
  - Difference: prediction − truth (diverging colormap, centred on 0).

Prediction and truth share one continuous gust colorbar; the difference column
has its own symmetric one. Both datasets are on the same 1.1M-cell grid with the
same cell ordering, so a single rotated-pole triangulation (built in
`paper_plots_realv2`) is reused for every pane, and each field is drawn with the
same fast, continuous gouraud `tripcolor`.
"""

from pathlib import Path

import cartopy.crs as ccrs  # noqa: F401  (kept for clarity; ROTATED_POLE comes from below)
import earthkit.plots  # noqa: F401  applies the paper styling via matplotlib rcParams
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import Normalize

from paper_plots_realv2 import (
    CMAP,
    GUST_MAX,
    REALV2_NC,
    ROTATED_POLE,
    VARIABLE,
    add_gridlines,
    draw_field,
    format_time,
    projected_triangulation,
)

TRUTH_ZARR = Path(
    "/store_new/mch/msopr/ml/datasets/"
    "mch-realch1-fdb-1km-2005-2025-1h-pl13-v2.0.zarr"
)
DIFF_CMAP = "RdBu_r"  # red: prediction too high, blue: too low


def load_model_series(
    nc_path: Path, variable: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (lon, lat, fields[time, values], times) for all forecast steps."""
    ds = xr.open_dataset(nc_path)
    lons = ds["longitude"].values
    lats = ds["latitude"].values
    lons = np.where(lons > 180, lons - 360, lons)
    return lons, lats, ds[variable].values, ds["time"].values


def load_truth_series(
    zarr_path: Path, variable: str, times: np.ndarray
) -> np.ndarray:
    """Return truth fields[time, values] sampled at the model's valid times."""
    ds = xr.open_zarr(zarr_path, consolidated=False)
    var_idx = list(ds.attrs["variables"]).index(variable)
    dates = ds["dates"].values
    fields = []
    for t in times:
        match = np.where(dates == t)[0]
        if match.size == 0:
            raise ValueError(f"Truth dataset has no time {t}")
        fields.append(ds["data"][int(match[0]), var_idx, 0, :].values)
    return np.asarray(fields)


def main(outfn: Path) -> None:
    outfn.parent.mkdir(parents=True, exist_ok=True)
    lons, lats, pred, times = load_model_series(REALV2_NC, VARIABLE)

    # The realv2 stream has no input, so its written initial state is all-NaN.
    # Keep only the forecast steps that actually carry a prediction.
    has_data = ~np.all(np.isnan(pred), axis=1)
    pred, times = pred[has_data], times[has_data]
    truth = load_truth_series(TRUTH_ZARR, VARIABLE, times)

    tri, finite = projected_triangulation(lons, lats)
    pred, truth = pred[:, finite], truth[:, finite]
    diff = pred - truth

    n = len(times)
    gust_norm = Normalize(0, GUST_MAX)
    dmax = max(2.0, float(np.ceil(np.nanpercentile(np.abs(diff), 99))))
    diff_norm = Normalize(-dmax, dmax)

    fig, axes = plt.subplots(
        n, 3, figsize=(15, 3.1 * n + 0.6),
        subplot_kw={"projection": ROTATED_POLE},
    )
    axes = np.atleast_2d(axes)
    col_titles = ["Prediction (realv2)", "Truth (ICON-REA-L-CH1)", "Prediction − Truth"]

    m_gust = m_diff = None
    for r in range(n):
        m_gust = draw_field(axes[r, 0], tri, pred[r], norm=gust_norm, cmap=CMAP)
        draw_field(axes[r, 1], tri, truth[r], norm=gust_norm, cmap=CMAP)
        m_diff = draw_field(axes[r, 2], tri, diff[r], norm=diff_norm, cmap=DIFF_CMAP)
        for c in range(3):
            add_gridlines(axes[r, c], left=(c == 0), bottom=(r == n - 1))
        axes[r, 0].text(
            -0.22, 0.5, format_time(times[r]), rotation=90,
            va="center", ha="center", transform=axes[r, 0].transAxes, fontsize=11,
        )
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title)

    cb_gust = fig.colorbar(
        m_gust, ax=axes[:, :2], location="bottom",
        shrink=0.5, pad=0.04, aspect=40, extend="max",
    )
    cb_gust.set_label("10 m max wind gust [m s$^{-1}$]")
    cb_diff = fig.colorbar(
        m_diff, ax=axes[:, 2], location="bottom",
        shrink=0.85, pad=0.04, aspect=18, extend="both",
    )
    cb_diff.set_label("difference [m s$^{-1}$]")

    fig.suptitle(
        "realv2 10 m max wind gust — prediction vs truth", y=0.99, fontsize=14
    )
    fig.savefig(outfn, bbox_inches="tight", dpi=150)
    print(f"saved: {outfn}")


if __name__ == "__main__":
    main(Path("figures/realv2_vmax10m_compare.png"))
