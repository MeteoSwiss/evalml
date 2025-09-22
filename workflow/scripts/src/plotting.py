import typing as tp
from pathlib import Path
from functools import cached_property

import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

State = dict[str, np.ndarray | dict[str, np.ndarray]]

PROJECTIONS: dict[str, ccrs.Projection] = {
    "plate_carree": ccrs.PlateCarree(),
    "orthographic": ccrs.Orthographic(central_longitude=5.0, central_latitude=45.0),
}
"""Mapping of projection names to their cartopy projection objects."""

REGION_EXTENTS = {
    "europe": [-16.0, 25.0, 30.0, 65.0],
    "central_europe": [-2.6, 19.5, 40.2, 52.3],
    "switzerland": [-1.5, 17.5, 40.5, 53.0],
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
        nrows: int = 1,
        ncols: int = 1,
        projection: str = "orthographic",
        region: str | None = None,
        coastlines: bool = True,
        **kwargs,
    ) -> tuple[plt.Figure, tp.Sequence[GeoAxes]]:
        """Initialize a figure with GeoAxes for plotting fields.

        Parameters
        ----------
        nrows : int, optional
            The number of rows in the figure, by default 1.
        ncols : int, optional
            The number of columns in the figure, by default 1.
        projection : str, optional
            The projection of the map, by default "orthographic".
        region : str, optional
            The region to plot, by default None.
        coastlines : bool, optional
            Whether to plot coastlines, by default True.

        Returns
        -------
        tuple[plt.Figure, tp.Sequence[GeoAxes]]
            The figure and the GeoAxes objects.
        """

        proj = PROJECTIONS.get(projection, PROJECTIONS["orthographic"])
        fig, ax = plt.subplots(nrows, ncols, subplot_kw={"projection": proj})
        ax: GeoAxes | tp.Sequence[GeoAxes] = (
            [ax] if nrows == 1 and ncols == 1 else ax.ravel()
        )

        for i in range(nrows * ncols):
            ax[i].set_global()
            if coastlines:
                ax[i].coastlines()

            if region != "globe":
                ax[i].set_extent(REGION_EXTENTS[region], crs=ccrs.PlateCarree())

        return fig, ax

    def plot_field(
        self,
        ax: GeoAxes,
        field: np.ndarray,
        region: str | None = None,
        validtime: str = "",
        colorbar: dict | bool = True,
        **kwargs,
    ):
        """Plot a field on a GeoAxes object.

        Parameters
        ----------
        ax : GeoAxes
            The GeoAxes object to plot on.
        field : np.ndarray
            The field to plot.
        region : str, optional
            The region to plot, by default None.
        colorbar : dict | bool, optional
            Whether to plot a colorbar, by default True.
            If a dictionary, it is passed as keyword arguments to plt.colorbar.
        kwargs : dict
            Additional keyword arguments to pass to ax.tripcolor, including cmap,
            vmin, vmax, etc.
        """

        proj = ax.projection

        if proj == PROJECTIONS["orthographic"]:
            triang, mask = self._ortographic_tri
        else:
            triang, mask = self.tri, slice(None, None)

        # TODO: this is hardcoded for when the initial state has two timesteps
        # need to ditch this later
        field = field[-1] if field.ndim == 2 else field.squeeze()

        im = ax.tripcolor(triang, field[mask], **kwargs)
        ax.text(
            0.05,
            0.95,
            f"Time: {validtime}",
            transform=ax.transAxes,
            fontsize=12,
            color="white",
            verticalalignment="top",
            bbox=dict(facecolor="black", alpha=0.5),
        )

        if region and region != "globe":
            ax.set_extent(REGION_EXTENTS[region], crs=ccrs.PlateCarree())

        if colorbar:
            colorbar = {
                "orientation": "horizontal",
                "pad": 0.04,
                "aspect": 45,
                "extend": "both",
                "shrink": 0.75,
            } | (colorbar if isinstance(colorbar, dict) else {})
            plt.colorbar(im, **colorbar)

    @cached_property
    def _ortographic_tri(self) -> Triangulation:
        """Compute the triangulation for the orthographic projection."""
        x, y, _ = (
            PROJECTIONS["orthographic"]
            .transform_points(ccrs.PlateCarree(), self.lon, self.lat)
            .T
        )
        mask = ~(np.isnan(x) | np.isnan(y))
        return Triangulation(x[mask], y[mask]), mask
