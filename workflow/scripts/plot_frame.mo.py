import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import logging
    from argparse import ArgumentParser
    from datetime import datetime
    from pathlib import Path

    import cartopy.crs as ccrs
    import earthkit.plots as ekp
    import geopandas as gpd
    import numpy as np
    import xarray as xr
    from shapely.geometry import MultiPoint

    from data_input import load_baseline_from_zarr, load_truth_data
    from plotting import DOMAINS, StatePlotter
    from plotting.colormap_defaults import CMAP_DEFAULTS
    from plotting.compat import load_state_from_grib
    from verification.spatial import map_forecast_to_truth

    return (
        ArgumentParser,
        CMAP_DEFAULTS,
        DOMAINS,
        MultiPoint,
        Path,
        StatePlotter,
        ccrs,
        datetime,
        ekp,
        gpd,
        load_baseline_from_zarr,
        load_state_from_grib,
        load_truth_data,
        logging,
        map_forecast_to_truth,
        np,
        xr,
    )


@app.cell
def _(logging):
    LOG = logging.getLogger(__name__)
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)
    return (LOG,)


@app.cell
def _(ArgumentParser, Path, datetime):
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["forecast", "truth", "error"],
        help="Which field to plot.",
    )
    parser.add_argument(
        "--forecast",
        type=str,
        default=None,
        help="Forecast root (GRIB directory or baseline zarr). Required for forecast/error mode.",
    )
    parser.add_argument(
        "--truth",
        type=str,
        default=None,
        help="Truth zarr root. Required for truth/error mode.",
    )
    parser.add_argument("--date", type=str, required=True, help="reference datetime")
    parser.add_argument("--outfn", type=str, required=True, help="output filename")
    parser.add_argument("--leadtime", type=str, required=True, help="leadtime (hours)")
    parser.add_argument("--param", type=str, required=True, help="parameter")
    parser.add_argument("--region", type=str, required=True, help="name of region")

    args = parser.parse_args()
    mode = args.mode
    reftime = datetime.strptime(args.date, "%Y%m%d%H%M")
    outfn = Path(args.outfn)
    lead_time_h = int(args.leadtime)
    param = args.param
    region = args.region
    forecast_root = Path(args.forecast) if args.forecast else None
    truth_root = Path(args.truth) if args.truth else None

    if mode in ("forecast", "error") and forecast_root is None:
        raise ValueError(f"--forecast is required for mode={mode!r}")
    if mode in ("truth", "error") and truth_root is None:
        raise ValueError(f"--truth is required for mode={mode!r}")

    return (
        forecast_root,
        lead_time_h,
        mode,
        outfn,
        param,
        reftime,
        region,
        truth_root,
    )


@app.cell
def _(param):
    # Parameters loaded from the underlying dataset to compute the plotted field
    if param == "SP_10M":
        load_params = ["U_10M", "V_10M"]
    elif param == "SP":
        load_params = ["U", "V"]
    else:
        load_params = [param]
    return (load_params,)


@app.cell
def _(
    LOG,
    MultiPoint,
    gpd,
    load_baseline_from_zarr,
    load_state_from_grib,
    np,
):
    def load_forecast_xr(root, reftime, lead_time_h, params):
        """Load forecast data for a single lead time as an xarray.Dataset.

        Routes to `load_state_from_grib` for GRIB directories (keeps IFS global
        concatenation) and to `load_baseline_from_zarr` for zarr roots.
        The returned dataset carries a `values` dim, `lat`/`lon` coords, and a
        `lam_envelope` GeoSeries stashed in `ds.attrs["lam_envelope"]`.
        """
        if any(root.glob("*.grib")):
            grib_file = root / f"{reftime:%Y%m%d%H%M}_{lead_time_h:03d}.grib"
            LOG.info("Loading forecast GRIB: %s", grib_file)
            return load_state_from_grib(
                grib_file, paramlist=params, output_type="xarray"
            )
        LOG.info("Loading baseline zarr: %s", root)
        ds = load_baseline_from_zarr(
            root=root, reftime=reftime, steps=[lead_time_h], params=params
        )
        ds = ds.sel(lead_time=np.timedelta64(lead_time_h, "h"))
        if "y" in ds.dims and "x" in ds.dims:
            ds = ds.stack(values=("y", "x"))
        lam_hull = MultiPoint(
            list(zip(ds["lon"].values.tolist(), ds["lat"].values.tolist()))
        ).convex_hull
        ds.attrs["lam_envelope"] = gpd.GeoSeries([lam_hull], crs="EPSG:4326")
        return ds

    return (load_forecast_xr,)


@app.cell
def _(
    LOG,
    MultiPoint,
    forecast_root,
    gpd,
    lead_time_h,
    load_forecast_xr,
    load_params,
    load_truth_data,
    map_forecast_to_truth,
    mode,
    np,
    reftime,
    truth_root,
):
    fcst_ds = None
    truth_ds = None
    lam_envelope = None

    if mode in ("forecast", "error"):
        fcst_ds = load_forecast_xr(forecast_root, reftime, lead_time_h, load_params)
        lam_envelope = fcst_ds.attrs.get("lam_envelope")

    if mode in ("truth", "error"):
        LOG.info("Loading truth data from %s", truth_root)
        truth_ds = load_truth_data(
            root=truth_root,
            reftime=reftime,
            steps=[lead_time_h],
            params=load_params,
        )
        _valid_time = np.datetime64(reftime) + np.timedelta64(lead_time_h, "h")
        truth_ds = truth_ds.sel(time=_valid_time)
        if "y" in truth_ds.dims and "x" in truth_ds.dims:
            truth_ds = truth_ds.stack(values=("y", "x"))
        if lam_envelope is None:
            lam_hull = MultiPoint(
                list(
                    zip(
                        truth_ds["lon"].values.tolist(),
                        truth_ds["lat"].values.tolist(),
                    )
                )
            ).convex_hull
            lam_envelope = gpd.GeoSeries([lam_hull], crs="EPSG:4326")

    if mode == "error":
        # Align forecast to truth grid (fast-path when grids already match).
        fcst_ds = map_forecast_to_truth(fcst_ds, truth_ds)

    return fcst_ds, lam_envelope, truth_ds


@app.cell
def _(np):
    try:
        import pint  # type: ignore

        _ureg = pint.UnitRegistry()

        def _k_to_c(arr):
            try:
                return (_ureg.Quantity(arr, _ureg.kelvin).to(_ureg.degC)).magnitude
            except Exception:
                return arr - 273.15

        def _m_to_mm(arr):
            try:
                return (
                    _ureg.Quantity(arr, _ureg.meter).to(_ureg.millimeter)
                ).magnitude
            except Exception:
                return arr * 1000

    except Exception:

        def _k_to_c(arr):
            return arr - 273.15

        def _m_to_mm(arr):
            return arr * 1000

    def extract_field(param: str, ds):
        """Return (1D numpy array, units) for the plotted quantity of `param`."""
        if param in ("T_2M", "TD_2M", "T", "TD"):
            return _k_to_c(np.asarray(ds[param].values).ravel()), "degC"
        if param == "SP_10M":
            u = np.asarray(ds["U_10M"].values).ravel()
            v = np.asarray(ds["V_10M"].values).ravel()
            return np.sqrt(u**2 + v**2), "m/s"
        if param == "SP":
            u = np.asarray(ds["U"].values).ravel()
            v = np.asarray(ds["V"].values).ravel()
            return np.sqrt(u**2 + v**2), "m/s"
        if param == "TOT_PREC":
            return _m_to_mm(np.asarray(ds[param].values).ravel()), "mm"
        return np.asarray(ds[param].values).ravel(), None

    return (extract_field,)


@app.cell
def _(extract_field, fcst_ds, mode, param, truth_ds):
    if mode == "forecast":
        field, units_override = extract_field(param, fcst_ds)
        cmap_key = param
        source_ds = fcst_ds
    elif mode == "truth":
        field, units_override = extract_field(param, truth_ds)
        cmap_key = param
        source_ds = truth_ds
    else:  # error
        fcst_field, units_override = extract_field(param, fcst_ds)
        truth_field, _ = extract_field(param, truth_ds)
        field = fcst_field - truth_field
        cmap_key = f"{param}_error"
        source_ds = fcst_ds

    lons = np.asarray(source_ds["lon"].values).ravel()
    lats = np.asarray(source_ds["lat"].values).ravel()
    return cmap_key, field, lats, lons, units_override


@app.cell
def _(CMAP_DEFAULTS, ekp):
    def get_style(param_key, units_override=None):
        """Get style and colormap settings for the plot."""
        cfg = CMAP_DEFAULTS[param_key]
        units = units_override if units_override is not None else cfg.get("units", "")
        return {
            "style": ekp.styles.Style(
                levels=cfg.get("bounds", cfg.get("levels", None)),
                extend="both",
                units=units,
                colors=cfg.get("colors", None),
            ),
            "norm": cfg.get("norm", None),
            "cmap": cfg.get("cmap", None),
            "levels": cfg.get("levels", None),
            "vmin": cfg.get("vmin", None),
            "vmax": cfg.get("vmax", None),
            "colors": cfg.get("colors", None),
        }

    return (get_style,)


@app.cell
def _(
    DOMAINS,
    LOG,
    StatePlotter,
    ccrs,
    cmap_key,
    field,
    get_style,
    lam_envelope,
    lats,
    lead_time_h,
    lons,
    mode,
    np,
    outfn,
    param,
    reftime,
    region,
    units_override,
):
    _plotter = StatePlotter(lons, lats, outfn.parent)
    _fig = _plotter.init_geoaxes(
        nrows=1,
        ncols=1,
        projection=DOMAINS[region]["projection"],
        bbox=DOMAINS[region]["extent"],
        name=region,
        size=(6, 6),
    )
    _subplot = _fig.add_map(row=0, column=0)

    _plotter.plot_field(_subplot, field, **get_style(cmap_key, units_override))

    if lam_envelope is not None:
        _subplot.ax.add_geometries(
            lam_envelope,
            edgecolor="black",
            facecolor="none",
            crs=ccrs.PlateCarree(),
        )

    _valid_time = np.datetime64(reftime) + np.timedelta64(lead_time_h, "h")
    _validtime_str = (
        str(_valid_time).replace("-", "").replace(":", "").replace("T", "")[:12]
    )
    if mode == "error":
        _fig.title(f"{param} error (fcst - truth), time: {_validtime_str}")
    else:
        _fig.title(f"{param} ({mode}), time: {_validtime_str}")

    _fig.save(outfn, bbox_inches="tight", dpi=200)
    LOG.info("saved: %s", outfn)
    return


if __name__ == "__main__":
    app.run()
