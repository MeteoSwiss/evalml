from contextlib import contextmanager
from pathlib import Path

import cartopy.crs as ccrs
import earthkit.plots as ekp
import numpy as np
from matplotlib.tri import Triangulation

State = dict[str, np.ndarray | dict[str, np.ndarray]]

_PROJECTIONS: dict[str, ccrs.Projection] = {
    "platecarree": ccrs.PlateCarree(),
    "orthographic": ccrs.Orthographic(central_longitude=5.0, central_latitude=45.0),
    # added some pojections to test the behaviour, can be deleted later
    "rotatedlatlon": ccrs.RotatedPole(pole_longitude=-170.0, pole_latitude=43.0),
    "azimuthalequidist": ccrs.AzimuthalEquidistant(),
}


# Mapping of region names to their geographic extent and projection
# extent [lon_min, lon_max, lat_min, lat_max] in PlateCarree coordinates
def get_projection(name: str) -> "ccrs.Projection":
    """Look up a projection by name."""
    if name not in _PROJECTIONS:
        raise ValueError(
            f"Unknown projection {name!r}. Available: {list(_PROJECTIONS)}"
        )
    return _PROJECTIONS[name]


DOMAINS = {
    "globe": {
        "extent": None,  # full globe view
        "projection": _PROJECTIONS["orthographic"],
    },
    "europe": {
        "extent": [-16.0, 25.0, 30.0, 65.0],
        "projection": _PROJECTIONS["orthographic"],
    },
    "centraleurope": {
        "extent": [-1.5, 18, 41.5, 51],
        "projection": _PROJECTIONS["orthographic"],
    },
    "icon-ch": {
        "extent": [0, 17.5, 40.5, 53.0],
        "projection": _PROJECTIONS["orthographic"],
    },
    "alps": {
        "extent": [4.5, 16.0, 43.5, 48.5],
        "projection": _PROJECTIONS["orthographic"],
    },
    "radar": {
        "extent": [3.2, 12.5, 43.6, 49.4],
        "projection": _PROJECTIONS["orthographic"],
    },
    "switzerland": {
        "extent": [5.6, 10.8, 45.0, 48.6],
        "projection": _PROJECTIONS["orthographic"],
    },
}


class StatePlotter:
    """A class to plot state fields on various DOMAINS."""

    def __init__(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        out_dir: Path,
    ):
        """Initialize the StatePlotter object.

        Latitudes and longitudes are passed during initialization so that
        the triangulation can be computed once and reused for all plots.
        The reference coordinate system of lon/lat is assumed to be PlateCarree()
        currently.

        Parameters
        ----------
        lon : np.ndarray
            The longitudes of the grid.
        lat : np.ndarray
            The latitudes of the grid.
        out_dir : Path
            The output directory to save the plots.
        """

        self.lon = lon
        self.lat = lat
        out_dir.mkdir(exist_ok=True, parents=True)
        self.out_dir = out_dir

        self.tri = Triangulation(self.lon, self.lat)

    def init_geoaxes(
        self,
        projection: ccrs.Projection,
        bbox: list[float] | None,
        nrows: int = 1,
        ncols: int = 1,
        size: tuple[float] | None = None,
        name: str | None = None,
    ) -> ekp.Figure:
        """Initialize a figure with GeoAxes for plotting fields.

        Parameters
        ----------
        projection : cartopy.crs.Projection
            The projection used for the region.
        bbox : list[float] or None
            The bounding box [lon_min, lon_max, lat_min, lat_max] in PlateCarree coordinates.
            If None, the full projection extent is used.
        name : str, optional
            The name of the region.
        nrows : int, optional
            The number of rows in the figure, by default 1.
        ncols : int, optional
            The number of columns in the figure, by default 1.
        size : tuple
            size of the figure in inches

        Returns
        -------
        earthkit.plots.Figure
            The figure object.
        """
        if bbox is not None and len(bbox) != 4:
            raise ValueError(
                "bbox must be a list of four floats [lon_min, lon_max, lat_min, lat_max]"
            )

        ekp_fig = ekp.Figure(
            crs=projection,
            domain=bbox,
            rows=nrows,
            columns=ncols,
            size=size,
        )
        self.fig = ekp_fig
        return ekp_fig

    def plot_field(
        self,
        subplot: ekp.Map,
        field: np.ndarray,
        style: ekp.styles.Style | None = None,
        colorbar: bool = True,
        title: str | None = None,
        gridline_labels: bool = True,
        **kwargs,
    ):
        """Plot a field on a Map object.

        Parameters
        ----------
        subplot : earthkit.plots.Map
            The Map subplot object to plot on.
        field : np.ndarray
            The field to plot.
        style : ekp.styles.Style, optional
            Earthkit.plots style for the map plot.
        colorbar : bool
            Whether to plot a colorbar, by default True.
        title: str, optional
            Map subplot title.
        gridline_labels : bool
            Whether to draw lat/lon degree labels on the gridlines, by
            default True. Set to False for views whose center rotates
            between frames (e.g. a rotating globe animation), where the
            labels' varying width would otherwise make the map's position
            shift from frame to frame.
        kwargs : dict
            Additional keyword arguments to pass to ax.tripcolor, including cmap,
            vmin, vmax, etc.
        """

        proj = subplot._crs
        # transform data coordinates to map coordinate reference system outside
        # of the plotting function is a lot faster than letting tricontourf or
        # tripcolor handle it in general, but not sure if using earthkit
        # removed for now to simplify the workflow
        if isinstance(proj, ccrs.Orthographic):
            triang, mask = self._orthographic_tri(proj)
        else:
            triang, mask = self.tri, slice(None, None)
        x, y = triang.x, triang.y
        # TODO: this is hardcoded for when the initial state has two timesteps
        # need to ditch this later
        field = field[mask]
        field = field[-1] if field.ndim == 2 else field.squeeze()
        finite = np.isfinite(field)
        # TODO: clip data to domain would make plotting faster (especially tripcolor)
        # tried using Map.domain.extract() but too memory heavy (probably uses
        # meshgrid in the background), implement clipping with e.g.
        #   points = np.column_stack((lon,lat));  mask = domain_polygon.contains_points(points)
        # would work but needs handling domain crossing dateline etc.

        # TODO: tricontourf/tripcolor can handle a Triangulation when used directly,
        # for some reason this doesn not work when using it with earthkit-plot,
        # guess is that the earthkit-plots check throwing:
        # "ValueError: x and y arrays must have the same length" is incorrect,
        # therefore using x,y here
        # subplot.tripcolor(  # also works but is slower
        # have to overwrite _plot_kwargs to avoid earthkit-plots trying to pass transform
        # PlateCarree based on NumpySource

        # Normalize style and color-related kwargs
        style_to_use, plot_kwargs = self._prepare_plot_kwargs(style, kwargs)

        # Temporarily suppress earthkit-plots internal source-based kwargs
        with self._temporary_plot_kwargs_override(subplot):
            subplot.tricontourf(
                x=x[finite],
                y=y[finite],
                z=field[finite],
                style=style_to_use,
                transform=proj,
                **plot_kwargs,
            )  # for earthkit.plots to work properly cmap and norm are needed here
        # TODO: gridlines etc would be nicer to have in the init, but I didn't get
        # them to overlay the plot layer

        if gridline_labels:
            subplot.standard_layers()
        else:
            subplot.land()
            subplot.coastlines()
            subplot.borders()
            subplot.gridlines(draw_labels=False)

        if colorbar:
            subplot.legend()
        if title:
            subplot.title(title)

    def _prepare_plot_kwargs(
        self,
        style: ekp.styles.Style | None,
        kwargs: dict,
    ) -> tuple[ekp.styles.Style | None, dict]:
        """Return a cleaned style and plot kwargs without mutating the input."""
        plot_kwargs = dict(kwargs)

        # Discrete colors mode: if explicit 'colors' provided, drop cmap
        colors = plot_kwargs.get("colors", None)
        if colors is not None:
            plot_kwargs.pop("cmap", None)
            plot_kwargs.setdefault(
                "no_style", True
            )  # avoid interpolation being performed by earthkit-plots resulting in an error
            return style, plot_kwargs

        # Continuous mode: remove None entries so configure_style doesn't call
        # style.with_overrides(colors=None, vmin=None, ...) and wipe the style's
        # pre-configured values.
        for key in ("colors", "levels", "cmap", "norm", "vmin", "vmax"):
            if plot_kwargs.get(key) is None:
                plot_kwargs.pop(key, None)

        return style, plot_kwargs

    @contextmanager
    def _temporary_plot_kwargs_override(self, subplot: ekp.Map):
        """Temporarily override internal _plot_kwargs to avoid transform issues."""
        has_attr = hasattr(subplot, "_plot_kwargs")
        old = getattr(subplot, "_plot_kwargs", None)
        subplot._plot_kwargs = lambda source: {}
        try:
            yield
        finally:
            if has_attr:
                subplot._plot_kwargs = old
            else:
                try:
                    delattr(subplot, "_plot_kwargs")
                except Exception:
                    pass

    def _orthographic_tri(
        self, proj: ccrs.Projection
    ) -> tuple[Triangulation, np.ndarray]:
        """Compute the triangulation for an orthographic-family projection."""
        x, y, _ = proj.transform_points(ccrs.PlateCarree(), self.lon, self.lat).T
        mask = ~(np.isnan(x) | np.isnan(y))
        return Triangulation(x[mask], y[mask]), mask
