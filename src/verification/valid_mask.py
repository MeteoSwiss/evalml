"""Score all participants on the same truth points.

Some forecast fields are undefined at part of the domain, e.g. ICON leaves
HZEROCL and SNOWLMT undefined where the level lies below the ground, and CIN
where there is no unstable parcel.

For the parameters listed in ``experiment.score_on_common_points``, the workflow
scores every participant only where *all* participants have a value:

1. ``forecast_valid_mask`` records, per participant and reference time, where
   its forecast (mapped to the truth points) has a value.
2. ``combine_valid_masks`` intersects the masks of all participants.
3. ``apply_valid_mask`` sets forecast and truth to NaN outside the intersection,
   before ``verify``.

The mask is decided per time stamp and lead time. Where every participant has a
value, nothing is removed and the scores equal those without the mask.
"""

import logging
from functools import reduce

import xarray as xr

LOG = logging.getLogger(__name__)


def forecast_valid_mask(fcst: xr.Dataset, params: list[str]) -> xr.Dataset:
    """Where the forecast has a value, for each of ``params`` it provides.

    ``fcst`` must already be mapped to the truth points (``map_forecast_to_truth``)
    and hold a single reference time. Parameters the forecast does not provide are
    left out, so they do not restrict the other participants. A lead time where the
    field is missing everywhere (no data for that lead time) is marked valid
    everywhere for the same reason.
    """
    present = [p for p in params if p in fcst.data_vars]
    mask = fcst[present].notnull()
    if "forecast_reference_time" in mask.dims:
        mask = mask.squeeze("forecast_reference_time", drop=True)
    spatial_dims = [d for d in mask.dims if d != "step"]
    no_data = ~mask.any(dim=spatial_dims)
    return (mask | no_data).reset_coords(drop=True)


def combine_valid_masks(masks: list[xr.Dataset]) -> xr.Dataset:
    """Intersection of the participants' masks, per parameter.

    A participant restricts only the parameters and lead times its mask contains.
    The truth points must be identical in all masks.
    """
    params = sorted({p for m in masks for p in m.data_vars})
    combined = {}
    for param in params:
        per_participant = [m[param] for m in masks if param in m.data_vars]
        steps = reduce(
            lambda a, b: a.union(b), (m.indexes["step"] for m in per_participant)
        )
        aligned = [m.reindex(step=steps, fill_value=True) for m in per_participant]
        combined[param] = reduce(lambda a, b: a & b, xr.align(*aligned, join="exact"))
    return xr.Dataset(combined)


def apply_valid_mask(
    fcst: xr.Dataset, truth: xr.Dataset, mask: xr.Dataset
) -> tuple[xr.Dataset, xr.Dataset]:
    """Set forecast and truth to NaN outside the mask, for the parameters it holds.

    ``truth`` must be aligned to the forecast lead times (``truth.sel(time=fcst
    ["valid_time"])``). The truth is masked even when this forecast lacks the
    parameter, so the truth statistics are the same for every participant. Lead
    times that are not in the mask are left unchanged.
    """
    fcst, truth = fcst.copy(), truth.copy()
    for param, valid in mask.data_vars.items():
        LOG.info(
            "Common points: %s keeps %.1f%% of the points",
            param,
            100 * float(valid.mean()),
        )
        if param in fcst.data_vars:
            fcst[param] = _mask_outside(fcst[param], valid)
        if param in truth.data_vars:
            truth[param] = _mask_outside(truth[param], valid)
    return fcst, truth


def _mask_outside(data: xr.DataArray, valid: xr.DataArray) -> xr.DataArray:
    valid = valid.reindex(step=data["step"], fill_value=True)
    # where() silently drops points that do not match: require identical points
    xr.align(data, valid, join="exact")
    return data.where(valid)
