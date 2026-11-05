"""Plotting of power spectra: single-figure helper with reference annotations,
and the experiment overlay (truth + all participants)."""

from __future__ import annotations

from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from spectra.core import KM_PER_DEG

REFERENCE_GRIDS = {
    "O96 (~104 km)": 90.0 / 96.0 * KM_PER_DEG,
    "N320 (~31 km)": 0.28125 * KM_PER_DEG,
    "CERRA (~9 km)": 0.10 * KM_PER_DEG,
}

EFF_RES_FACTOR = 6.0
TRANSITION_KM = 400.0

_PALETTE = ["#1f3b73", "#c1272d", "#1b7837", "#6a3d9a", "#e08214", "#4d4d4d"]


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


def plot_experiment_spectra(
    truth_file, participant_files, out_dir, variables, lead_times
):
    """One overlay figure per (variable, lead time): truth + all participants."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    participants = [xr.open_dataset(f) for f in participant_files]
    try:
        with xr.open_dataset(truth_file) as truth:
            for var in variables:
                for step in lead_times:
                    spectra = {}
                    t_wl = truth["wavelength"].values
                    t_p = truth["power"].sel(variable=var, leadtime=step).values
                    spectra[f"{truth.attrs['label']} (truth)"] = (
                        t_wl,
                        t_p,
                        "k",
                        "-",
                        1.0,
                        2.0,
                    )
                    color = cycle(_PALETTE)
                    for ds in participants:
                        wl = ds["wavelength"].values
                        p = ds["power"].sel(variable=var, leadtime=step).values
                        spectra[ds.attrs["label"]] = (wl, p, next(color), "-", 0.9, 1.6)
                    out = out_dir / f"spectrum_{var}_{step:03d}.png"
                    # Reference grid/eff-res/Nyquist lines use the TRUTH grid's dx (labelled
                    # "truth"); each participant curve keeps its own wavelength axis.
                    plot_power_spectra(
                        spectra,
                        out,
                        f"Power spectrum — {var} @ +{step}h",
                        grid_dx_km=float(truth.attrs["dx_km"]),
                        model_short="truth",
                        model_color="k",
                    )
    finally:
        for ds in participants:
            ds.close()
