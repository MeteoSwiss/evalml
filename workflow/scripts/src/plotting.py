from functools import cached_property
from pathlib import Path

import cartopy.crs as ccrs
import earthkit.plots as ekp
import numpy as np
from matplotlib.tri import Triangulation

State = dict[str, np.ndarray | dict[str, np.ndarray]]

PROJECTIONS: dict[str, ccrs.Projection] = {
    "platecarree": ccrs.PlateCarree(),
    "orthographic": ccrs.Orthographic(central_longitude=5.0, central_latitude=45.0),
    # added some pojections to test the behaviour, can be deleted later
    "rotatedlatlon": ccrs.RotatedPole(pole_longitude=-170.0, pole_latitude=43.0),
    "azimuthalequidist": ccrs.AzimuthalEquidistant(),
}
"""Mapping of projection names to their cartopy projection objects."""

REGION_EXTENTS = {  # coordinate reference system: PlateCarree()
    "europe": [-16.0, 25.0, 30.0, 65.0],
    "centraleurope": [-2.6, 19.5, 40.2, 52.3],
    "switzerland": [0, 17.5, 40.5, 53.0],
}
"""Mapping of region names to their extents."""


class StatePlotter:
    """A class to plot state fields on various map projections and regions."""

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
        if lon.shape != lat.shape:
            Z = np.meshgrid(lon, lat)
            self.lon = Z[0]
            self.lat = Z[1]
        out_dir.mkdir(exist_ok=True, parents=True)
        self.out_dir = out_dir

        self.tri = Triangulation(self.lon, self.lat)

    def init_geoaxes(
        self,
        nrows: int = 1,
        ncols: int = 1,
        projection: str = "orthographic",
        size: tuple[float] | None = None,
        region: str | None = None,
    ) -> ekp.Figure:
        """Initialize a figure with GeoAxes for plotting fields.

        Parameters
        ----------
        nrows : int, optional
            The number of rows in the figure, by default 1.
        ncols : int, optional
            The number of columns in the figure, by default 1.
        projection : str, optional
            The projection of the map, by default "orthographic".
        size : tuple
            size of the figure in inches
        region : str, optional
            The region to plot, by default None.

        Returns
        -------
        earthkit.plots.Figure
            The figure object.
        """

        proj = PROJECTIONS.get(projection, PROJECTIONS["orthographic"])

        domain = None
        if region:
            domain = ekp.geo.domains.Domain(
                bbox=REGION_EXTENTS[region],
                crs=ccrs.PlateCarree(),  # coordinate reference system of the region coords
                name=region,
            )

        ekp_fig = ekp.Figure(
            crs=proj,  # coordinate reference system of the map
            domain=domain,
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
        kwargs : dict
            Additional keyword arguments to pass to ax.tripcolor, including cmap,
            vmin, vmax, etc.
        """

        proj = subplot._crs
        # transform data coordinates to map coordinate reference system outside
        # of the plotting function is a lot faster than letting tricontourf or
        # tripcolor handle it in general, but not sure if using earthkit
        # removed for now to simplify the workflow
        if proj == PROJECTIONS["orthographic"]:
            triang, mask = self._orthographic_tri
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
        subplot._plot_kwargs = lambda source: {}
        subplot.tricontourf(
            x=x[finite],
            y=y[finite],
            z=field[finite],
            style=style,
            transform=proj,
            **kwargs,  # for earthkit.plots to work properly cmap and norm are needed here
        )
        # TODO: gridlines etc would be nicer to have in the init, but I didn't get
        # them to overlay the plot layer
        subplot.standard_layers()

        if colorbar:
            subplot.legend()
        if title:
            subplot.title(title)

    @cached_property
    def _orthographic_tri(self) -> Triangulation:
        """Compute the triangulation for the orthographic projection."""
        x, y, _ = PROJECTIONS["orthographic"].transform_points(ccrs.PlateCarree(), self.lon, self.lat).T
        mask = ~(np.isnan(x) | np.isnan(y))
        return Triangulation(x[mask], y[mask]), mask

