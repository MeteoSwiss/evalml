import logging
import string
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from data_input import (
    load_forecast_data,
    load_truth_data,
    parse_steps,
    truth_type,
)
from verification import CIRCULAR_PARAMS, apply_lapse_rate_correction_inplace
from verification.spatial import map_forecast_to_truth

LOG = logging.getLogger(__name__)

# Colours as in the meteograms of the Varda-single paper (Figs. 8 and 9).
OBS_COLOR = "mediumseagreen"
# Optional gridded reference (e.g. a reanalysis), dashed.
REFERENCE_COLOR = "tab:purple"
# One colour per forecast run, in config order (first run red as Varda-single in the paper).
FORECAST_COLORS = ["tab:red", "black", "tab:orange", "tab:purple", "tab:brown"]
BASELINE_COLORS = {
    "ICON-CH1": "deepskyblue",
    "ICON-CH2": "navy",
}


KELVIN_PARAMS = {"T_2M", "TD_2M"}
ZERO_CELSIUS_IN_KELVIN = 273.15


def _lead_hours(times, init_time: datetime) -> np.ndarray:
    """Hours between each valid time and the forecast initialisation time."""
    return (np.asarray(times, dtype="datetime64[ns]") - np.datetime64(init_time)) / (
        np.timedelta64(1, "h")
    )


def _baseline_color(label: str, default: str) -> str:
    """Paper colour for ICON-CH1/CH2 baselines (matched on the label), else default."""
    for key, color in BASELINE_COLORS.items():
        if key in label.upper():
            return color
    return default


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def _ylabel(param: str, attrs: dict, units: str) -> str:
    label = attrs.get("shortName", "") or param
    return f"{label} ({units})" if units else label


def _draw_meteogram(
    ax,
    param: str,
    init_time: datetime,
    analysis: tuple[str, xr.Dataset],
    references: list[tuple[str, xr.Dataset]],
    baselines: list[tuple[str, xr.Dataset]],
    forecasts: list[tuple[str, int, list[xr.Dataset]]],
) -> None:
    """Draw the meteogram of one param at one station (all inputs mapped to it).

    Observations green, a reference (e.g. a reanalysis) dashed purple, baselines dashed (paper colours for ICON-CH1/CH2), runs in
    FORECAST_COLORS order; ensembles as min-max band plus dashed mean. Wind direction
    wraps at 0/360 deg, so it is drawn as dots (ensemble: every member).
    """
    line_style = (
        {"ls": "none", "marker": "o", "ms": 3} if param in CIRCULAR_PARAMS else {}
    )
    # Temperatures in degrees Celsius, as in the paper.
    offset = ZERO_CELSIUS_IN_KELVIN if param in KELVIN_PARAMS else 0.0

    analysis_label, analysis_ds = analysis
    ax.plot(
        _lead_hours(analysis_ds["time"].values, init_time),
        analysis_ds[param].values.ravel() - offset,
        color=OBS_COLOR,
        label=analysis_label,
        **line_style,
    )
    for label, ds in references:
        ax.plot(
            _lead_hours(ds["time"].values, init_time),
            ds[param].values.ravel() - offset,
            color=REFERENCE_COLOR,
            label=label,
            **(line_style or {"ls": "--"}),
        )
    for i, (label, ds) in enumerate(baselines, start=1):
        ax.plot(
            _lead_hours(ds["valid_time"].values, init_time),
            ds[param].values.ravel() - offset,
            color=_baseline_color(label, default=f"C{i}"),
            label=label,
            # Baselines dashed, as the ICON lines in the paper's Fig. 8.
            **(line_style or {"ls": "--"}),
        )
    for i, (label, members, member_ds) in enumerate(forecasts):
        color = FORECAST_COLORS[i % len(FORECAST_COLORS)]
        x = _lead_hours(member_ds[0]["valid_time"].values, init_time)
        curves = np.stack([ds[param].values.ravel() - offset for ds in member_ds])
        if not members:
            ax.plot(x, curves[0], color=color, label=label, **line_style)
        elif param in CIRCULAR_PARAMS:
            # A min-max band or plain mean wraps wrongly at 0/360 deg: dots.
            for k, y in enumerate(curves):
                ax.plot(
                    x,
                    y,
                    color=color,
                    alpha=0.5,
                    label=f"{label} ({members} members)" if k == 0 else None,
                    **line_style,
                )
        else:
            ax.fill_between(
                x,
                curves.min(axis=0),
                curves.max(axis=0),
                color=color,
                alpha=0.2,
                lw=0,
                label=f"{label} min-max ({members} members)",
            )
            ax.plot(x, curves.mean(axis=0), color=color, ls="--", label=f"{label} mean")
    if param in CIRCULAR_PARAMS:
        ax.set_ylim(0, 360)
        ax.set_yticks([0, 90, 180, 270, 360])
    ax.grid(alpha=0.3)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--forecast",
        action="append",
        type=str,
        default=[],
        help="Directory with forecast grib data (repeatable, one per run).",
    )
    parser.add_argument(
        "--forecast_steps",
        action="append",
        type=parse_steps,
        default=[],
        help=(
            "Forecast steps in the format 'start/stop/step' for each forecast "
            "(repeatable, must match --forecast count)."
        ),
    )
    parser.add_argument(
        "--forecast_label",
        action="append",
        type=str,
        default=[],
        help="Label for each forecast line in plot legend (repeatable).",
    )
    parser.add_argument(
        "--forecast_members",
        action="append",
        type=int,
        default=[],
        help=(
            "Number of ensemble members for each forecast (repeatable, 0 for a "
            "deterministic run). For an ensemble, --forecast is the grib/ directory "
            "holding one sub-directory per member (001, 002, ...); the plot shows the "
            "min-max range of the members and their mean. Defaults to 0 for all."
        ),
    )
    parser.add_argument(
        "--baseline",
        action="append",
        type=str,
        default=[],
        help="Path to baseline data root (repeatable).",
    )
    parser.add_argument(
        "--baseline_steps",
        action="append",
        type=parse_steps,
        default=[],
        help=(
            "Forecast steps in the format 'start/stop/step' for each baseline "
            "(repeatable, must match --baseline count)."
        ),
    )
    parser.add_argument(
        "--baseline_label",
        action="append",
        type=str,
        default=[],
        help="Label for each baseline line in plot legend (repeatable).",
    )
    parser.add_argument(
        "--analysis", type=str, default=None, help="Path to analysis data root"
    )
    parser.add_argument(
        "--analysis_label",
        type=str,
        default="truth",
        help="Label for analysis line in plot legend.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help=(
            "Optional gridded reference (anemoi .zarr, e.g. a reanalysis), drawn "
            "dashed at the station's grid point, at the observation times."
        ),
    )
    parser.add_argument(
        "--reference_label",
        type=str,
        default="reference",
        help="Label for the reference line in the plot legend.",
    )
    parser.add_argument("--date", type=str, default=None, help="reference datetime")
    parser.add_argument("--outdir", type=str, help="output directory")
    parser.add_argument(
        "--param",
        nargs="+",
        type=str,
        help="Parameter(s); one meteogram per param and station.",
    )
    parser.add_argument("--stations", nargs="+", type=str, help="station IDs")
    parser.add_argument(
        "--lapse_rate_correction",
        action="store_true",
        default=False,
        help="Apply lapse-rate correction to T_2M and PS.",
    )
    parser.add_argument(
        "--panel",
        type=str,
        default=None,
        help=(
            "Also write all meteograms assembled in one figure to this file "
            "(rows: params, columns: stations, one legend)."
        ),
    )

    args = parser.parse_args()

    forecast_grib_dirs = [Path(path) for path in args.forecast]
    forecast_steps_list = args.forecast_steps
    forecast_labels = args.forecast_label
    forecast_members = args.forecast_members or [0] * len(forecast_grib_dirs)
    if not forecast_grib_dirs:
        raise ValueError("At least one --forecast is required.")
    if not (
        len(forecast_grib_dirs)
        == len(forecast_steps_list)
        == len(forecast_labels)
        == len(forecast_members)
    ):
        raise ValueError(
            "Mismatched forecast arguments: --forecast, --forecast_steps, "
            "--forecast_label and --forecast_members (if given) must be provided "
            "the same number of times."
        )
    # Observations at every valid time any forecast or baseline covers (e.g. hourly
    # when ICON is hourly and the ML runs 6-hourly); station observations always
    # hourly over the full forecast range.
    obs_steps = sorted(set().union(*forecast_steps_list, *args.baseline_steps))
    if truth_type(args.analysis) == "jretrieve":
        obs_steps = list(range(max(obs_steps) + 1))
    analysis_root = Path(args.analysis)
    analysis_label = args.analysis_label
    baseline_roots = [Path(path) for path in args.baseline]
    baseline_steps = args.baseline_steps
    baseline_labels = args.baseline_label
    if len(baseline_roots) != len(baseline_steps):
        raise ValueError(
            "Mismatched baseline arguments: --baseline and --baseline_steps "
            "must be provided the same number of times."
        )
    if len(baseline_labels) != len(baseline_roots):
        raise ValueError(
            "Mismatched baseline arguments: --baseline and --baseline_label "
            "must be provided the same number of times."
        )
    init_time = datetime.strptime(args.date, "%Y%m%d%H%M")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    params = args.param
    stations = args.stations
    init_str = init_time.strftime("%Y%m%d%H%M")

    LOG.info(
        "Plotting meteograms: params=%s, stations=%s, init_time=%s",
        params,
        stations,
        init_time,
    )

    LOG.info("Loading analysis data from %s", analysis_root)
    analysis_ds = load_truth_data(analysis_root, init_time, obs_steps, params).squeeze()

    # Build station coordinate lookup from the loaded analysis dataset so that
    # any station with data for the plotted parameters is found (a fixed
    # parameter like rre150h0 would exclude stations like JUN).
    # A gridded truth on a (y, x) grid (e.g. INCA) has no station axis.
    if "values" in analysis_ds.dims:
        catalog_lookup = {
            str(sta): (float(lat), float(lon), float(elev))
            for sta, lat, lon, elev in zip(
                analysis_ds["values"].values,
                analysis_ds["latitude"].values,
                analysis_ds["longitude"].values,
                analysis_ds["elevation"].values,
            )
        }
    else:
        LOG.warning(
            "Truth %s has no station axis; meteograms require station observations.",
            analysis_root,
        )
        catalog_lookup = {}

    reference_ds = None
    if args.reference:
        LOG.info("Loading reference '%s' from %s", args.reference_label, args.reference)
        reference_ds = load_truth_data(
            Path(args.reference), init_time, obs_steps, params
        ).squeeze()

    # Load gridded data once — shared across all station plots. Each forecast is a
    # list of member datasets: one for a deterministic run, N for an ensemble.
    forecast_ds_list = []
    for grib_dir, steps, label, members in zip(
        forecast_grib_dirs, forecast_steps_list, forecast_labels, forecast_members
    ):
        member_dirs = (
            [grib_dir / f"{m:03d}" for m in range(1, members + 1)]
            if members
            else [grib_dir]
        )
        LOG.info(
            "Loading forecast '%s' from %s (%s)",
            label,
            grib_dir,
            f"{members} members" if members else "deterministic",
        )
        forecast_ds_list.append(
            [
                load_forecast_data(d, init_time, steps, params).squeeze()
                for d in member_dirs
            ]
        )
    forecast_ds = forecast_ds_list[0][0]

    baseline_ds_list = []
    for root, step, label in zip(baseline_roots, baseline_steps, baseline_labels):
        LOG.info("Loading baseline '%s' from %s", label, root)
        baseline_ds_list.append(
            load_forecast_data(root, init_time, step, params).squeeze()
        )

    panel_fig = panel_axes = None
    if args.panel:
        panel_fig, panel_axes = plt.subplots(
            len(params),
            len(stations),
            figsize=(5 * len(stations), 2.8 * len(params)),
            sharex=True,
            squeeze=False,
        )

    # Loop over stations — data is loaded once, mapping is per station
    for col, station in enumerate(stations):
        LOG.info("Plotting station %s (%d/%d)", station, col + 1, len(stations))
        if station not in catalog_lookup:
            LOG.warning(
                "Station %r has no observations for %s — writing placeholders.",
                station,
                params,
            )
            for param in params:
                outfn = outdir / f"{init_str}_{param}_{station}.png"
                fig, ax = plt.subplots()
                ax.text(
                    0.5,
                    0.5,
                    f"No observations for {param}\nat station {station}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
                plt.savefig(outfn)
                plt.close(fig)
                LOG.info("saved placeholder: %s", outfn)
            if panel_axes is not None:
                panel_axes[0, col].set_title(f"{station} (no observations)")
            continue
        lat, lon, elev = catalog_lookup[station]
        station_ds = xr.Dataset(
            coords={
                "values": [station],
                "latitude": ("values", [lat]),
                "longitude": ("values", [lon]),
                "elevation": ("values", [elev]),
            }
        )

        forecast_station_ds_list = [
            [map_forecast_to_truth(ds, station_ds) for ds in member_ds]
            for member_ds in forecast_ds_list
        ]
        analysis_station_ds = map_forecast_to_truth(analysis_ds, station_ds)
        baseline_station_ds_list = [
            map_forecast_to_truth(ds, station_ds) for ds in baseline_ds_list
        ]
        reference_station_ds_list = (
            [map_forecast_to_truth(reference_ds, station_ds)]
            if reference_ds is not None
            else []
        )

        if args.lapse_rate_correction:
            member_station_ds = [ds for m in forecast_station_ds_list for ds in m]
            for ds in (
                member_station_ds + baseline_station_ds_list + reference_station_ds_list
            ):
                apply_lapse_rate_correction_inplace(ds, station_ds, params)

        draw_args = dict(
            init_time=init_time,
            analysis=(analysis_label, analysis_station_ds),
            references=[(args.reference_label, ds) for ds in reference_station_ds_list],
            baselines=list(zip(baseline_labels, baseline_station_ds_list)),
            forecasts=list(
                zip(forecast_labels, forecast_members, forecast_station_ds_list)
            ),
        )
        for row, param in enumerate(params):
            attrs = forecast_ds[param].attrs.get("parameter", {})
            units = "°C" if param in KELVIN_PARAMS else attrs.get("units", "")
            name = attrs.get("name", "") or param

            fig, ax = plt.subplots()
            _draw_meteogram(ax, param, **draw_args)
            ax.legend()
            ax.set_ylabel(_ylabel(param, attrs, units))
            ax.set_xlabel(f"Lead time from {init_time:%Y-%m-%d %H:%M} (h)")
            ax.set_title(f"{name} at {station}")
            outfn = outdir / f"{init_str}_{param}_{station}.png"
            plt.savefig(outfn)
            plt.close(fig)
            LOG.info("saved: %s", outfn)

            if panel_axes is not None:
                ax = panel_axes[row, col]
                _draw_meteogram(ax, param, **draw_args)
                letter = string.ascii_lowercase[(row * len(stations) + col) % 26]
                ax.text(
                    0.01,
                    0.97,
                    f"({letter})",
                    transform=ax.transAxes,
                    va="top",
                    fontweight="bold",
                )
                if row == 0:
                    ax.set_title(station)
                if col == 0:
                    ax.set_ylabel(f"{name}\n({units})" if units else name)

    if panel_fig is not None:
        for ax in panel_axes[-1]:
            ax.set_xlabel(f"Lead time from {init_time:%Y-%m-%d %H:%M} (h)")
            ax.set_xticks(range(0, int(max(obs_steps)) + 1, 24))
        # One legend for all panels, below the figure (two rows if long).
        handles, labels = {}, []
        for ax in panel_axes.flat:
            for h, lbl in zip(*ax.get_legend_handles_labels()):
                if lbl not in handles:
                    handles[lbl] = h
                    labels.append(lbl)
        ncol = len(labels) if len(labels) <= 4 else (len(labels) + 1) // 2
        legend_rows = -(-len(labels) // max(ncol, 1))
        panel_fig.legend(
            [handles[lbl] for lbl in labels],
            labels,
            loc="lower center",
            ncol=max(ncol, 1),
            frameon=False,
        )
        panel_fig.tight_layout(rect=(0, 0.03 + 0.04 * legend_rows, 1, 1))
        panel_fig.savefig(args.panel, dpi=150)
        plt.close(panel_fig)
        LOG.info("saved panel: %s", args.panel)


if __name__ == "__main__":
    main()
