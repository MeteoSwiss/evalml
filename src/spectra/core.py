"""Numpy-only 2D variance (power) spectra of regular-grid fields.

Ported from the `icon-power-spectra` skill. The only behavioural change vs. the
skill is that `radial_spectrum` returns FIXED-LENGTH arrays (length `nbins`,
`NaN` for empty bins) on a deterministic, edge-centred wavenumber grid, so
spectra from different init times share an identical wavelength coordinate and
can be stacked/averaged. Variance-conserving normalization is unchanged.
"""

from __future__ import annotations

from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft

KM_PER_DEG = 111.2

REFERENCE_GRIDS = {
    "O96 (~104 km)": 90.0 / 96.0 * KM_PER_DEG,
    "N320 (~31 km)": 0.28125 * KM_PER_DEG,
    "CERRA (~9 km)": 0.10 * KM_PER_DEG,
}

EFF_RES_FACTOR = 6.0
TRANSITION_KM = 400.0


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
    NaN bins are treated as zero in the sum so the common coordinate is preserved.
    """
    if not fields_2d:
        raise ValueError("combined_spectrum requires at least one field")
    wl = total = None
    for f in fields_2d:
        w, p = spectrum_fn(f, dx_km, nbins, oro_coeff=oro_coeff)
        wl = w
        total = np.nan_to_num(p) if total is None else total + np.nan_to_num(p)
    return wl, factor * total


def _add_slope_guide(ax, n, label, wl_range, anchor):
    wl = np.array(wl_range, dtype=float)
    wl0, p0 = anchor
    power = p0 * (wl / wl0) ** n
    ax.plot(wl, power, ls=":", color="0.35", lw=1.3)
    ax.annotate(
        label,
        xy=(wl.min(), power[np.argmin(wl)]),
        xytext=(3, 3),
        textcoords="offset points",
        color="0.35",
        fontsize=11,
        style="italic",
    )


def plot_power_spectra(
    spectra,
    out_path,
    title,
    grid_dx_km=None,
    model_short="model",
    model_color="#1f3b73",
    extra_lines=None,
):
    """Plot one or more spectra vs wavelength with the standard annotations.

    spectra : {label: (wavelength_km, power, color, linestyle, alpha, lw)}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, (wl, power, color, ls, alpha, lw) in spectra.items():
        ax.loglog(wl, power, lw=lw, label=label, color=color, ls=ls, alpha=alpha)

    lines = [(lbl, wl, "0.55") for lbl, wl in REFERENCE_GRIDS.items()]
    if grid_dx_km is not None:
        lines += [
            (f"{model_short} grid", grid_dx_km, model_color),
            (f"{model_short} eff.", EFF_RES_FACTOR * grid_dx_km, model_color),
            (f"{model_short} Nyq.", 2.0 * grid_dx_km, model_color),
        ]
    lines += list(extra_lines or [])
    _, ymax = ax.get_ylim()
    label_y = cycle([1.0, 0.6, 0.35])
    for label, wl, color in sorted(lines, key=lambda t: t[1]):
        ax.axvline(wl, color=color, ls="--", lw=1.0, alpha=0.8)
        ax.text(
            wl,
            ymax * next(label_y),
            " " + label,
            rotation=90,
            va="top",
            ha="right",
            color=color,
            fontsize=8.5,
        )

    ref_wl, ref_p = next(iter(spectra.values()))[:2]
    ref_wl = np.asarray(ref_wl)
    ref_p = np.asarray(ref_p)
    finite = np.isfinite(ref_wl) & np.isfinite(ref_p)
    ref_wl, ref_p = ref_wl[finite], ref_p[finite]
    meso = (ref_wl >= 5) & (ref_wl <= TRANSITION_KM)
    if meso.any():
        _add_slope_guide(
            ax,
            5.0 / 3.0,
            r"$k^{-5/3}$",
            (ref_wl[meso].min(), ref_wl[meso].max()),
            (np.median(ref_wl[meso]), 5 * np.median(ref_p[meso])),
        )

    ax.set_xlabel("Wavelength [km]")
    ax.set_ylabel("Power (variance density)")
    ax.set_title(title)
    ax.invert_xaxis()
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
