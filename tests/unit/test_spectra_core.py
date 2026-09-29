import numpy as np
import pytest
import xarray as xr

from spectra import compute, core


def _cosine_field(n=128, wavelength_px=16):
    x = np.arange(n)
    return np.cos(2 * np.pi * x[None, :] / wavelength_px) * np.ones((n, 1))


def test_radial_spectrum_returns_fixed_length_with_nan_padding():
    # A coeff array with energy only in a couple of modes -> most bins empty (NaN).
    coeff = np.zeros((8, 8))
    coeff[0, 3] = 5.0
    kx = np.arange(8) / (2.0 * 8) / 1.0
    ky = np.arange(8) / (2.0 * 8) / 1.0
    wl, power = core.radial_spectrum(coeff, kx, ky, nbins=4, norm=1.0 / 64)
    assert wl.shape == (4,) and power.shape == (4,)
    assert np.isnan(power).any()  # empty bins are NaN, not dropped
    assert np.nanmax(power) > 0
    assert np.all(np.diff(wl) > 0)  # wavelengths strictly ascending


def test_dct_spectrum_peaks_at_input_wavelength():
    dx = 2.0  # km per pixel
    wavelength_px = 16
    field = _cosine_field(n=128, wavelength_px=wavelength_px)
    wl, power = core.dct_power_spectrum(field, dx)
    peak_wl = wl[np.nanargmax(power)]
    expected = wavelength_px * dx  # 32 km
    assert peak_wl == pytest.approx(expected, rel=0.25)


def test_constant_field_has_no_power():
    field = np.full((64, 64), 7.3)
    wl, power = core.dct_power_spectrum(field, 1.0)
    assert np.nansum(power) == pytest.approx(0.0, abs=1e-9)


def test_combined_spectrum_scales_and_sums():
    field = _cosine_field(64, 8)
    wl1, p1 = core.dct_power_spectrum(field, 1.0)
    # two identical fields with factor 0.5 -> 0.5*(p+p) == p
    wl2, p2 = core.combined_spectrum(
        core.dct_power_spectrum, [field, field], 1.0, factor=0.5
    )
    np.testing.assert_allclose(np.nan_to_num(p2), np.nan_to_num(p1), rtol=1e-6)


def test_aggregate_spectra_averages_over_init(tmp_path):
    wl = np.array([100.0, 50.0, 25.0])

    def mk(power, init):
        ds = xr.Dataset(
            {"power": (("variable", "leadtime", "wavenumber"), np.array([[power]]))},
            coords={
                "variable": ["T_2M"],
                "leadtime": [6],
                "wavenumber": np.arange(3),
                "wavelength": ("wavenumber", wl),
            },
            attrs={"dx_km": 1.1, "npoints": 10, "label": "m"},
        )
        p = tmp_path / f"s_{init}.nc"
        ds.to_netcdf(p)
        return p

    a = mk([1.0, 2.0, 3.0], 0)
    b = mk([3.0, 4.0, 5.0], 1)
    agg = compute.aggregate_spectra([a, b])
    np.testing.assert_allclose(
        agg["power"].sel(variable="T_2M", leadtime=6).values, [2.0, 3.0, 4.0]
    )
    assert agg.attrs["label"] == "m"
    np.testing.assert_array_equal(agg["wavelength"].values, wl)


def test_aggregate_spectra_nanmean_ignores_missing(tmp_path):
    wl = np.array([100.0, 50.0])

    def mk(power, init):
        ds = xr.Dataset(
            {"power": (("variable", "leadtime", "wavenumber"), np.array([[power]]))},
            coords={
                "variable": ["T_2M"],
                "leadtime": [6],
                "wavenumber": np.arange(2),
                "wavelength": ("wavenumber", wl),
            },
            attrs={"dx_km": 1.1, "npoints": 10, "label": "m"},
        )
        p = tmp_path / f"s_{init}.nc"
        ds.to_netcdf(p)
        return p

    a = mk([np.nan, 2.0], 0)
    b = mk([4.0, 4.0], 1)
    agg = compute.aggregate_spectra([a, b])
    np.testing.assert_allclose(
        agg["power"].sel(variable="T_2M", leadtime=6).values, [4.0, 3.0]
    )
