"""verify_ensemble(): ensemble scores on synthetic data."""

import itertools

import numpy as np
import pytest
import xarray as xr

from verification import verify, verify_ensemble

_FRT = np.datetime64("2024-01-01T00", "ns")
_ALL = {"type": "bbox", "name": "all", "bbox": [1.5, 16, 43, 49.5]}
_WEST = {"type": "bbox", "name": "west", "bbox": [1.5, 7.5, 43, 49.5]}


def _stations(n):
    """Coordinates of n stations in the 'all' region; those with lon < 7.5 are 'west'."""
    return {
        "values": [f"S{k:05d}" for k in range(n)],
        "longitude": ("values", np.linspace(5.0, 10.0, n)),
        "latitude": ("values", np.linspace(46.0, 47.0, n)),
        "forecast_reference_time": _FRT,
    }


def _datasets(members, truth, param="T_2M", coords=None, dims=("values",)):
    """members: (N, *dims), truth: dims."""
    n_members = members.shape[0]
    coords = coords if coords is not None else _stations(truth.shape[-1])
    fcst = xr.Dataset(
        {param: (("member", *dims), members)},
        coords={**coords, "member": np.arange(1, n_members + 1)},
    )
    obs = xr.Dataset({param: (dims, truth)}, coords=coords)
    return fcst, obs


def _run(fcst, obs, **kwargs):
    kwargs.setdefault("regions", [_ALL])
    return verify_ensemble(fcst, obs, "ens", "truth", num_workers=1, **kwargs)


def _sel(result, var, **sel):
    sel = {"source": "ens", "region": "all", "station_group": "all", **sel}
    return result[var].sel(sel).squeeze("forecast_reference_time", drop=True)


def _fair_crps_brute(members, truth):
    """Fair CRPS per point by the double sum (members on axis 0)."""
    n = members.shape[0]
    skill = np.abs(members - truth).mean(axis=0)
    spread = np.abs(members[:, None] - members[None, :]).sum(axis=(0, 1))
    return skill - spread / (2 * n * (n - 1))


# ---------------------------------------------------------------------------
# Scores against brute force
# ---------------------------------------------------------------------------


def test_scores_match_brute_force_per_region_and_station_group():
    rng = np.random.default_rng(1)
    n, n_members = 40, 5
    members = rng.normal(size=(n_members, n))
    truth = rng.normal(size=n)
    fcst, obs = _datasets(members, truth)
    holdout = ["S00001", "S00004", "S00010", "S00033"]

    result = _run(fcst, obs, regions=[_ALL, _WEST], holdout_stations=holdout)

    lon = fcst["longitude"].values
    is_holdout = np.isin(fcst["values"].values, holdout)
    groups = {"all": np.ones(n, bool), "holdout": is_holdout, "holdin": ~is_holdout}
    regions = {"all": np.ones(n, bool), "west": lon < 7.5}
    crps = _fair_crps_brute(members, truth)
    for (region, rmask), (group, gmask) in itertools.product(
        regions.items(), groups.items()
    ):
        sel = rmask & gmask
        at = dict(region=region, station_group=group)
        assert _sel(result, "T_2M.CRPS", **at).item() == pytest.approx(crps[sel].mean())
        assert _sel(result, "T_2M.ENS_VAR", **at).item() == pytest.approx(
            members[:, sel].var(axis=0, ddof=1).mean()
        )
        assert _sel(result, "T_2M.MEMBER_MSE", **at).item() == pytest.approx(
            ((members[:, sel] - truth[sel]) ** 2).mean()
        )
        ranks = (truth[sel] > members[:, sel]).sum(axis=0)  # no ties
        np.testing.assert_allclose(
            _sel(result, "T_2M.rank_hist", **at).values,
            np.bincount(ranks, minlength=n_members + 1),
        )


def test_crps_matches_scores_library():
    import scores

    rng = np.random.default_rng(2)
    members = rng.gamma(2.0, size=(4, 30))
    truth = rng.gamma(2.0, size=30)
    fcst, obs = _datasets(members, truth)

    result = _run(fcst, obs)

    expected = scores.probability.crps_for_ensemble(
        fcst["T_2M"], obs["T_2M"], "member", method="fair"
    )
    assert _sel(result, "T_2M.CRPS").item() == pytest.approx(expected.item())


def test_identical_members_crps_equals_mae():
    """With identical members the fair CRPS is the MAE and the spread is 0."""
    rng = np.random.default_rng(3)
    single = rng.normal(size=25)
    truth = rng.normal(size=25)
    fcst, obs = _datasets(np.stack([single] * 3), truth)

    result = _run(fcst, obs)
    det = verify(fcst.mean("member"), obs, "ens", "truth", [_ALL], num_workers=1)

    assert _sel(result, "T_2M.CRPS").item() == pytest.approx(
        _sel(det, "T_2M.MAE").item()
    )
    assert _sel(result, "T_2M.ENS_VAR").item() == pytest.approx(0.0, abs=1e-12)
    assert _sel(result, "T_2M.MEMBER_MSE").item() == pytest.approx(
        _sel(det, "T_2M.MSE").item()
    )


# ---------------------------------------------------------------------------
# Spread-skill ratio and rank histogram shape
# ---------------------------------------------------------------------------


def _calibration_case(spread_scale, n=20000, n_members=10, seed=4):
    """Truth ~ N(mu, 1); members ~ N(mu, spread_scale**2), mu varies per point."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(scale=3.0, size=n)
    truth = mu + rng.normal(size=n)
    members = mu + spread_scale * rng.normal(size=(n_members, n))
    fcst, obs = _datasets(members, truth)
    ens = _run(fcst, obs)
    det = verify(fcst.mean("member"), obs, "ens", "truth", [_ALL], num_workers=1)
    # The ensemble scores merge into verify()'s output of the ensemble mean
    merged = xr.merge([det, ens], compat="no_conflicts", join="outer")
    n_members = ens.attrs["n_members"]
    ssr = np.sqrt(
        _sel(merged, "T_2M.ENS_VAR").item() / _sel(merged, "T_2M.MSE").item()
    ) * np.sqrt((n_members + 1) / n_members)
    hist = _sel(merged, "T_2M.rank_hist").values
    return ssr, hist / hist.sum()


def test_perfect_ensemble_ssr_one_flat_rank_histogram():
    ssr, hist = _calibration_case(1.0)
    assert ssr == pytest.approx(1.0, abs=0.03)
    np.testing.assert_allclose(hist, 1 / 11, atol=0.01)


def test_underdispersive_ensemble_ssr_below_one_u_shaped_histogram():
    ssr, hist = _calibration_case(0.5)
    assert ssr < 0.7
    assert hist[0] > 2 / 11 and hist[-1] > 2 / 11
    assert hist[5] < 1 / 11


def test_overdispersive_ensemble_ssr_above_one_dome_histogram():
    ssr, hist = _calibration_case(2.0)
    assert ssr > 1.5
    assert hist[0] < 0.5 / 11 and hist[-1] < 0.5 / 11
    assert hist[5] > 1 / 11


# ---------------------------------------------------------------------------
# Ties, missing values
# ---------------------------------------------------------------------------


def test_ties_are_split_evenly_between_ranks():
    """Truth 0 mm with members (0, 0, 1, 2): ranks 0, 1 and 2 share each point."""
    n = 9
    members = np.tile(np.array([0.0, 0.0, 1.0, 2.0])[:, None], (1, n))
    fcst, obs = _datasets(members, np.zeros(n), param="TOT_PREC")
    hist = _sel(_run(fcst, obs), "TOT_PREC.rank_hist").values
    np.testing.assert_allclose(hist, [3, 3, 3, 0, 0], atol=1e-12)


def test_all_tied_spreads_over_all_ranks():
    """Dry everywhere: the truth ties every member, so all N+1 ranks get 1/(N+1)."""
    n = 10
    fcst, obs = _datasets(np.zeros((4, n)), np.zeros(n), param="TOT_PREC")
    result = _run(fcst, obs)
    np.testing.assert_allclose(
        _sel(result, "TOT_PREC.rank_hist").values, np.full(5, n / 5), atol=1e-12
    )
    assert _sel(result, "TOT_PREC.CRPS").item() == 0.0


def test_nan_member_makes_point_missing():
    rng = np.random.default_rng(5)
    n = 20
    members = rng.normal(size=(4, n))
    truth = rng.normal(size=n)
    members[2, 7] = np.nan  # one member missing at one point
    truth[3] = np.nan  # truth missing elsewhere: not a forecast gap
    fcst, obs = _datasets(members, truth)
    thresholds = {"T_2M": {"gt": [0.0]}}

    strict = _run(fcst, obs, threshold_dict=thresholds)
    for var in ("CRPS", "ENS_VAR", "MEMBER_MSE", "rank_hist", "prob_table"):
        assert np.isnan(_sel(strict, f"T_2M.{var}").values).all(), var

    tolerant = _run(fcst, obs, threshold_dict=thresholds, max_missing_fraction=0.1)
    keep = np.ones(n, bool)
    keep[[3, 7]] = False
    assert _sel(tolerant, "T_2M.CRPS").item() == pytest.approx(
        _fair_crps_brute(members[:, keep], truth[keep]).mean()
    )
    assert _sel(tolerant, "T_2M.rank_hist").sum().item() == keep.sum()
    assert _sel(tolerant, "T_2M.prob_table").sum().item() == keep.sum()


def test_truth_gaps_do_not_mask_scores():
    rng = np.random.default_rng(6)
    truth = rng.normal(size=10)
    truth[:5] = np.nan
    fcst, obs = _datasets(rng.normal(size=(3, 10)), truth)
    result = _run(fcst, obs)
    assert not np.isnan(_sel(result, "T_2M.CRPS").item())
    assert _sel(result, "T_2M.rank_hist").sum().item() == 5


# ---------------------------------------------------------------------------
# Probability tables
# ---------------------------------------------------------------------------


def test_prob_table_counts_and_threshold_dim_match_verify():
    rng = np.random.default_rng(7)
    n, n_members = 50, 4
    members = rng.gamma(0.8, size=(n_members, n))
    truth = rng.gamma(0.8, size=n)
    fcst, obs = _datasets(members, truth, param="TOT_PREC")
    thresholds = {"TOT_PREC": {"gt": [0.1, 1.0], "lt": [0.5]}}
    holdout = ["S00002", "S00020", "S00041"]

    result = _run(
        fcst,
        obs,
        regions=[_ALL, _WEST],
        threshold_dict=thresholds,
        holdout_stations=holdout,
    )
    det = verify(
        fcst.mean("member"),
        obs,
        "ens",
        "truth",
        [_ALL, _WEST],
        threshold_dict=thresholds,
        holdout_stations=holdout,
        num_workers=1,
    )

    table = result["TOT_PREC.prob_table"]
    assert table.dims == (
        "forecast_reference_time",
        "source",
        "region",
        "station_group",
        "TOT_PREC.threshold",
        "n_members_yes",
        "obs",
    )
    assert list(table["TOT_PREC.threshold"].values) == list(
        det["TOT_PREC.threshold"].values
    )
    assert list(table["n_members_yes"].values) == list(range(n_members + 1))
    assert list(table["obs"].values) == [0, 1]
    # Every valid point is counted once per threshold
    totals = table.sum(["n_members_yes", "obs"])
    n_valid = det["TOT_PREC.contingency_table"].sel(contingency="total_count")
    np.testing.assert_array_equal(
        totals.squeeze("forecast_reference_time").sel(source="ens").values,
        n_valid.squeeze("forecast_reference_time")
        .sel(source="ens")
        .transpose(*totals.dims[2:])
        .values,
    )
    # Brute force for one threshold, region "west", all stations
    west = fcst["longitude"].values < 7.5
    k = (members[:, west] > 1.0).sum(axis=0)
    o = (truth[west] > 1.0).astype(int)
    expected = np.zeros((n_members + 1, 2))
    np.add.at(expected, (k, o), 1)
    np.testing.assert_array_equal(
        _sel(
            result,
            "TOT_PREC.prob_table",
            region="west",
            **{"TOT_PREC.threshold": "gt_1p0"},
        ).values,
        expected,
    )


# ---------------------------------------------------------------------------
# Grids, special parameters, dims
# ---------------------------------------------------------------------------


def test_gridded_truth():
    """INCA-like (y, x) truth: works and has the single 'all' station group."""
    ny, nx, n_members = 4, 5, 3
    rng = np.random.default_rng(8)
    lat, lon = np.meshgrid(
        np.linspace(46.0, 47.0, ny), np.linspace(5.0, 10.0, nx), indexing="ij"
    )
    coords = {
        "y": np.arange(ny),
        "x": np.arange(nx),
        "latitude": (("y", "x"), lat),
        "longitude": (("y", "x"), lon),
        "forecast_reference_time": _FRT,
        "step": [0, 6],
    }
    members = rng.normal(size=(n_members, 2, ny, nx))
    truth = rng.normal(size=(2, ny, nx))
    fcst, obs = _datasets(members, truth, coords=coords, dims=("step", "y", "x"))

    result = _run(
        fcst.chunk({"step": 1, "member": 1}),
        obs,
        regions=[_ALL, _WEST],
        threshold_dict={"T_2M": {"gt": [0.0]}},
    )

    assert list(result["station_group"].values) == ["all"]
    west = lon < 7.5
    for step in range(2):
        crps = _fair_crps_brute(members[:, step], truth[step])
        assert _sel(result, "T_2M.CRPS", step=step * 6).item() == pytest.approx(
            crps.mean()
        )
        assert _sel(
            result, "T_2M.CRPS", step=step * 6, region="west"
        ).item() == pytest.approx(crps[west].mean())
    assert (_sel(result, "T_2M.rank_hist").sum("rank") == ny * nx).all()
    assert (
        _sel(result, "T_2M.prob_table", region="west").sum(["n_members_yes", "obs"])
        == west.sum()
    ).all()
    with pytest.raises(ValueError, match="Holdout stations"):
        _run(fcst, obs, holdout_stations=["CHM"])


def test_wind_direction_and_precipitation_type():
    """No CRPS/spread for angles and class codes; PTYPE still gets prob tables."""
    n = 12
    rng = np.random.default_rng(9)
    codes = np.array([0, 1, 5, 7])
    pt_members = codes[rng.integers(0, 4, size=(4, n))].astype(float)
    pt_truth = codes[rng.integers(0, 4, size=n)].astype(float)
    dd_members = rng.uniform(0, 360, size=(4, n))
    coords = _stations(n)
    fcst = xr.Dataset(
        {
            "PTYPE": (("member", "values"), pt_members),
            "DD_10M": (("member", "values"), dd_members),
        },
        coords={**coords, "member": np.arange(1, 5)},
    )
    obs = xr.Dataset(
        {
            "PTYPE": ("values", pt_truth),
            "DD_10M": ("values", rng.uniform(0, 360, size=n)),
        },
        coords=coords,
    )

    result = _run(fcst, obs, threshold_dict={"PTYPE": {"eq": [0, 1, 5, 7]}})

    for param in ("PTYPE", "DD_10M"):
        for var in ("CRPS", "ENS_VAR", "MEMBER_MSE", "rank_hist"):
            assert np.isnan(_sel(result, f"{param}.{var}").values).all()
    assert "DD_10M.prob_table" not in result
    rain = _sel(result, "PTYPE.prob_table", **{"PTYPE.threshold": "eq_1"})
    expected = np.zeros((5, 2))
    np.add.at(expected, ((pt_members == 1).sum(0), (pt_truth == 1).astype(int)), 1)
    np.testing.assert_array_equal(rain.values, expected)


def test_output_dims_match_verify_continuous_scores():
    rng = np.random.default_rng(10)
    coords = {**_stations(8), "step": [0, 6, 12]}
    fcst, obs = _datasets(
        rng.normal(size=(3, 3, 8)),
        rng.normal(size=(3, 8)),
        coords=coords,
        dims=("step", "values"),
    )
    result = _run(fcst.chunk({"step": 1}), obs, holdout_stations=["S00001"])
    det = verify(
        fcst.mean("member"),
        obs,
        "ens",
        "truth",
        [_ALL],
        holdout_stations=["S00001"],
        num_workers=1,
    )

    assert result.attrs["n_members"] == 3
    for var in ("CRPS", "ENS_VAR", "MEMBER_MSE"):
        assert result[f"T_2M.{var}"].dims == det["T_2M.MAE"].dims
    assert result["T_2M.rank_hist"].dims == (*det["T_2M.MAE"].dims, "rank")
    assert list(result["rank"].values) == [0, 1, 2, 3]
    assert "member" not in result.dims
    merged = xr.merge([det, result], compat="no_conflicts", join="outer")
    assert merged.sizes["source"] == 2  # ens and truth, no new source


def test_needs_a_member_dim_with_two_members():
    fcst, obs = _datasets(np.zeros((1, 5)), np.zeros(5))
    with pytest.raises(ValueError, match="2 members"):
        _run(fcst, obs)
    with pytest.raises(ValueError, match="member"):
        _run(fcst.isel(member=0), obs)
