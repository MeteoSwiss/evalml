"""
Diagnostic script to investigate 6-hourly variability in Varda-Single bias.
Stratifies by init_hour and season to find the source of oscillation.
"""
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

VARDA_FILE = Path(
    "output/data/runs/temporal_downscaler-f927-1ee3-on-forecaster-c304-23e7"
    "/495c/verif_aggregated_2b83.nc"
)
BASELINE_FILES = {
    "ICON-CH1-CTRL": Path("output/data/baselines/baseline-7e02/verif_aggregated_2b83.nc"),
    "ICON-CH2-CTRL": Path("output/data/baselines/baseline-ce47/verif_aggregated_2b83.nc"),
}
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/figures/diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

PARAMS = ["TOT_PREC", "U_10M"]
SEASONS = ["DJF", "MAM", "JJA", "SON"]
INIT_HOURS = [0, 6, 12, 18]
STEPS_MAX = 54  # only Varda goes to 120; zoom in on 0-54h to see oscillation clearly

COLORS_INIT = {0: "#1b7837", 6: "#762a83", 12: "#e08214", 18: "#2166ac"}
COLORS_SEASON = {"DJF": "#2166ac", "MAM": "#4dac26", "JJA": "#d01c8b", "SON": "#f1a340"}

xscale_kw = dict(
    functions=(
        lambda x: np.sign(x) * np.abs(x) ** 0.7,
        lambda x: np.sign(x) * np.abs(x) ** (1 / 0.7),
    )
)
xticks = mticker.FixedLocator([0, 6, 12, 18, 24, 36, 48])


def load_bias(path, param, source=None):
    """Return DataArray of BIAS for param from given file."""
    ds = xr.open_dataset(ROOT / path)
    if source is not None:
        ds = ds.sel(source=source)
    da = ds[f"{param}.BIAS"]
    # convert step to hours
    steps = da.step.values.astype("timedelta64[h]").astype(int)
    da = da.assign_coords(step=steps)
    return da


def plot_stratified(param, fig_suffix):
    """Two-panel figure: left=by init_hour, right=by season."""
    varda = load_bias(VARDA_FILE, param, source="Varda-Single")
    # load one baseline for reference (all init_hours combined)
    try:
        ch1 = load_bias(BASELINE_FILES["ICON-CH1-CTRL"], param, source="ICON-CH1-CTRL")
        ch1_all = ch1.sel(season="all", init_hour=-999, region="all")
    except Exception:
        ch1_all = None
    try:
        ch2 = load_bias(BASELINE_FILES["ICON-CH2-CTRL"], param, source="ICON-CH2-CTRL")
        ch2_all = ch2.sel(season="all", init_hour=-999, region="all")
    except Exception:
        ch2_all = None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    # ---- Panel A: stratify by init_hour ----
    ax = axes[0]
    # "all" aggregation (what the paper plot shows)
    v_all = varda.sel(season="all", init_hour=-999, region="all")
    mask = v_all.step <= STEPS_MAX
    ax.plot(v_all.step[mask], v_all.values[mask], color="black", lw=1.5,
            label="Varda-Single (all)", zorder=5)
    for ih in INIT_HOURS:
        v_ih = varda.sel(season="all", init_hour=ih, region="all")
        ax.plot(v_ih.step[mask], v_ih.values[mask], color=COLORS_INIT[ih],
                lw=0.9, alpha=0.85, label=f"init {ih:02d}Z")
    if ch1_all is not None:
        m = ch1_all.step <= STEPS_MAX
        ax.plot(ch1_all.step[m], ch1_all.values[m], color="#008dff",
                lw=1.0, ls="--", label="ICON-CH1-CTRL", alpha=0.7)
    if ch2_all is not None:
        m = ch2_all.step <= STEPS_MAX
        ax.plot(ch2_all.step[m], ch2_all.values[m], color="#003a7d",
                lw=1.0, ls="--", label="ICON-CH2-CTRL", alpha=0.7)
    ax.axhline(0, color="k", ls=":", lw=0.7)
    ax.set_xscale("function", **xscale_kw)
    ax.xaxis.set_major_locator(xticks)
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("BIAS")
    ax.set_title(f"{param} BIAS — stratified by init_hour")
    ax.legend(fontsize=7)

    # ---- Panel B: stratify by season ----
    ax = axes[1]
    ax.plot(v_all.step[mask], v_all.values[mask], color="black", lw=1.5,
            label="Varda-Single (all)", zorder=5)
    for s in SEASONS:
        v_s = varda.sel(season=s, init_hour=-999, region="all")
        ax.plot(v_s.step[mask], v_s.values[mask], color=COLORS_SEASON[s],
                lw=0.9, alpha=0.85, label=s)
    ax.axhline(0, color="k", ls=":", lw=0.7)
    ax.set_xscale("function", **xscale_kw)
    ax.xaxis.set_major_locator(xticks)
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("BIAS")
    ax.set_title(f"{param} BIAS — stratified by season")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fname = OUT / f"diagnose_6h_{param}_{fig_suffix}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")
    return fname


def plot_amplitude_analysis(param):
    """Compute 6-hourly oscillation amplitude per (init_hour, season) stratum."""
    varda = load_bias(VARDA_FILE, param, source="Varda-Single")
    mask = (varda.step >= 6) & (varda.step <= STEPS_MAX)

    fig, axes = plt.subplots(len(SEASONS), len(INIT_HOURS),
                             figsize=(14, 10), sharey=True, sharex=True)

    for si, s in enumerate(SEASONS):
        for ii, ih in enumerate(INIT_HOURS):
            ax = axes[si, ii]
            v = varda.sel(season=s, init_hour=ih, region="all")
            m = v.step <= STEPS_MAX
            ax.plot(v.step[m], v.values[m], color="black", lw=0.8)
            ax.axhline(0, color="k", ls=":", lw=0.5)
            # highlight 6h-multiple steps
            v6 = v.sel(step=[s for s in v.step.values if s % 6 == 0 and s <= STEPS_MAX])
            ax.plot(v6.step, v6.values, "ro", ms=3, zorder=5)
            if si == 0:
                ax.set_title(f"init {ih:02d}Z", fontsize=9)
            if ii == 0:
                ax.set_ylabel(s, fontsize=9)
            if si == len(SEASONS) - 1:
                ax.set_xlabel("Lead (h)", fontsize=8)
            ax.set_xscale("function", **xscale_kw)
            ax.xaxis.set_major_locator(mticker.FixedLocator([0, 6, 12, 24, 36, 48]))
            ax.tick_params(labelsize=7)

    fig.suptitle(f"{param} BIAS by season × init_hour (red dots = 6h-multiple steps)", fontsize=11)
    fig.tight_layout()
    fname = OUT / f"diagnose_6h_{param}_grid.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")
    return fname


def print_oscillation_stats(param):
    """Print the amplitude of 6h oscillation at each step for each stratum."""
    varda = load_bias(VARDA_FILE, param, source="Varda-Single")

    print(f"\n{'='*60}")
    print(f"  {param}: 6h-oscillation amplitude analysis")
    print(f"{'='*60}")

    # For each init_hour, compute std of bias at 6h-anchor steps vs non-anchor steps
    print(f"\n{'init_hour':>12} {'season':>8} {'std_anchor':>12} {'std_inter':>12} {'ratio':>8}")
    print("-" * 58)

    for s in ["all"] + SEASONS:
        for ih in [-999] + INIT_HOURS:
            v = varda.sel(season=s, init_hour=ih, region="all")
            steps = v.step.values
            anchor = v.sel(step=[x for x in steps if x % 6 == 0 and 6 <= x <= STEPS_MAX])
            inter = v.sel(step=[x for x in steps if x % 6 != 0 and x <= STEPS_MAX])
            # oscillation as std of difference between consecutive anchor and inter steps
            if len(anchor) > 2 and len(inter) > 2:
                # simple proxy: std across steps (how much it wiggles)
                all_vals = v.sel(step=[x for x in steps if x <= STEPS_MAX]).values
                anchor_vals = anchor.values
                inter_vals = inter.values
                osc = np.std(all_vals)
                ih_label = "all" if ih == -999 else f"{ih:02d}Z"
                print(f"{ih_label:>12} {s:>8} {np.std(anchor_vals):12.4f} {np.std(inter_vals):12.4f} {np.std(anchor_vals)/max(np.std(inter_vals), 1e-9):8.2f}")


if __name__ == "__main__":
    for param in PARAMS:
        plot_stratified(param, "stratified")
        plot_amplitude_analysis(param)
        print_oscillation_stats(param)
    print(f"\nAll figures saved to {OUT}")
