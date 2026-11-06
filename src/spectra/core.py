"""Numpy-only 2D variance (power) spectra of regular-grid fields.

Ported from the `icon-power-spectra` skill. The only behavioural change vs. the
skill is that `radial_spectrum` returns FIXED-LENGTH arrays (length `nbins`,
`NaN` for empty bins) on a deterministic, edge-centred wavenumber grid, so
spectra from different init times share an identical wavelength coordinate and
can be stacked/averaged. Variance-conserving normalization is unchanged.
"""

from __future__ import annotations

import numpy as np
import scipy.fft


def _hann2d(ny: int, nx: int) -> np.ndarray:
    win = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    return win / np.sqrt((win**2).mean())


def radial_spectrum(coeff, kx, ky, nbins, norm, oro=None):
    """Radially bin |coeff|^2 over isotropic |k| [cycles/km] into a density.

    Returns (wavelength_km, power) each of length `nbins`, sorted by ascending
    wavelength, with `NaN` in bins that contain no samples. `norm` scales the
    summed coefficients to a variance-conserving spectrum so spectra from grids
    of different resolution overlap where both resolve. `oro` (transform of the
    orography on the same grid), if given, removes the terrain-coherent
    component per band (unused in v1).
    """
    kk = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2).ravel()
    c = coeff.ravel()
    h = oro.ravel() if oro is not None else None

    k_max = min(kx.max(), ky.max())
    edges = np.linspace(0.0, k_max, nbins + 1)
    dk = edges[1] - edges[0]
    which = np.digitize(kk, edges)

    power = np.full(nbins, np.nan)
    for b in range(1, nbins + 1):
        sel = which == b
        if not np.any(sel):
            continue
        cb = c[sel]
        s_ff = float(np.vdot(cb, cb).real)
        if h is not None:
            hb = h[sel]
            s_hh = float(np.vdot(hb, hb).real)
            if s_hh > 0:
                s_fh = np.vdot(hb, cb)
                s_ff -= abs(s_fh) ** 2 / s_hh
        val = max(s_ff, 0.0) * norm / dk
        power[b - 1] = val if val > 0.0 else np.nan

    centre_k = 0.5 * (edges[:-1] + edges[1:])
    wavelength = 1.0 / centre_k
    order = np.argsort(wavelength)
    return wavelength[order], power[order]


def dct_power_spectrum(field, dx_km, nbins=None, oro_coeff=None):
    """2D DCT (Denis et al. 2002) variance spectrum of a regular-grid field."""
    field = field - field.mean()
    ny, nx = field.shape
    nbins = nbins or min(ny, nx) // 2
    coeff = scipy.fft.dctn(field, type=2, norm="ortho")
    kx = np.arange(nx) / (2.0 * nx) / dx_km
    ky = np.arange(ny) / (2.0 * ny) / dx_km
    return radial_spectrum(coeff, kx, ky, nbins, 1.0 / (nx * ny), oro_coeff)


def fft_power_spectrum(field, dx_km, nbins=None, oro_coeff=None):
    """2D windowed-FFT (Hann) spectrum, normalized to match the DCT."""
    # NOTE: ported verbatim from the icon-power-spectra skill. FFT is the
    # optional (non-default) comparison method; the magnitude binning follows
    # the reference implementation exactly. Prefer DCT for quantitative use.
    field = field - field.mean()
    ny, nx = field.shape
    nbins = nbins or min(ny, nx) // 2
    coeff = scipy.fft.fft2(field * _hann2d(ny, nx))
    kx = scipy.fft.fftfreq(nx, d=dx_km)
    ky = scipy.fft.fftfreq(ny, d=dx_km)
    return radial_spectrum(coeff, kx, ky, nbins, 1.0 / (nx * ny) ** 2, oro_coeff)


SPECTRUM_FUNCS = {"dct": dct_power_spectrum, "fft": fft_power_spectrum}


def combined_spectrum(
    spectrum_fn, fields_2d, dx_km, nbins=None, factor=1.0, oro_coeff=None
):
    """Sum the spectra of one or more component fields, scaled by `factor`.

    For wind kinetic energy pass [u, v] and factor=0.5 -> 0.5*(|u_hat|^2+|v_hat|^2).
    Empty and zero-power bins count as zero, so a constant (e.g. dry) field gives a
    valid all-zero spectrum. If any input field contains NaN (missing data), the
    whole spectrum is NaN, so the gap carries through to the aggregation.
    """
    if not fields_2d:
        raise ValueError("combined_spectrum requires at least one field")
    missing = any(np.isnan(f).any() for f in fields_2d)
    wl = total = None
    for f in fields_2d:
        w, p = spectrum_fn(f, dx_km, nbins, oro_coeff=oro_coeff)
        wl = w
        total = np.nan_to_num(p) if total is None else total + np.nan_to_num(p)
    if missing:
        return wl, np.full_like(total, np.nan)
    return wl, factor * total
