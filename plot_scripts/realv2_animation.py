"""Animation: realv2 prediction vs truth (VMAX_10M) side by side over the forecast.

Builds a GIF with two panes — prediction (the realv2 anemoi-inference output) and
truth (the ICON-REA-L-CH1 dataset) — sharing one continuous gust colorbar, with
the valid datetime shown on top. One frame per forecast step (the all-NaN initial
state of the realv2 stream is skipped). Reuses the fast rotated-pole gouraud
`tripcolor` machinery from `paper_plots_realv2`, building each triangle mesh once
and only swapping the field values per frame, so even a 120 h run animates quickly.
"""

from pathlib import Path

import earthkit.plots  # noqa: F401  applies the paper styling via matplotlib rcParams
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image

from paper_plots_realv2 import (
    CMAP,
    REALV2_NC,
    ROTATED_POLE,
    VARIABLE,
    draw_field,
    format_time,
    projected_triangulation,
)
from paper_plots_realv2_compare import (
    TRUTH_ZARR,
    load_model_series,
    load_truth_series,
)

FPS = 2  # frames per second in the GIF


def main(outfn: Path, fps: int = FPS) -> None:
    outfn.parent.mkdir(parents=True, exist_ok=True)
    lons, lats, pred, times = load_model_series(REALV2_NC, VARIABLE)

    # The realv2 stream has no input, so its written initial state is all-NaN.
    has_data = ~np.all(np.isnan(pred), axis=1)
    pred, times = pred[has_data], times[has_data]
    truth = load_truth_series(TRUTH_ZARR, VARIABLE, times)

    tri, finite = projected_triangulation(lons, lats)
    pred, truth = pred[:, finite], truth[:, finite]

    # One fixed colour scale for the whole animation. Use a high percentile
    # rather than the raw max so a few extreme mountain-peak gusts in the truth
    # (max ~56 m/s vs a 99.9th percentile ~30) don't wash out the bulk of the
    # field; those extremes still read via the colorbar's "max" extension.
    hi = max(np.nanpercentile(pred, 99.9), np.nanpercentile(truth, 99.9))
    vmax = float(np.ceil(hi / 5.0) * 5.0)
    norm = Normalize(0, vmax)

    fig, (ax_pred, ax_truth) = plt.subplots(
        1, 2, figsize=(13, 5.2), subplot_kw={"projection": ROTATED_POLE}
    )
    # Fixed layout (no tight bbox) so positions don't jitter between frames.
    # Generous bottom margin so the colorbar label is never clipped.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.20, wspace=0.04)
    mesh_pred = draw_field(ax_pred, tri, pred[0], norm=norm, cmap=CMAP)
    mesh_truth = draw_field(ax_truth, tri, truth[0], norm=norm, cmap=CMAP)
    ax_pred.set_title("Predicted")
    ax_truth.set_title("REAL-L-CH1")

    # Dedicated colorbar axes, centred under the two panes.
    cax = fig.add_axes([0.32, 0.13, 0.36, 0.03])
    cbar = fig.colorbar(
        mesh_truth, cax=cax, orientation="horizontal", extend="max",
        ticks=np.arange(0, vmax + 1, 5),
    )
    cbar.set_label("10 m wind gust [m s$^{-1}$]")
    suptitle = fig.suptitle(format_time(times[0]), y=0.97, fontsize=15)

    # Render every frame to RGB. The colour scale (norm) is created once and
    # never changes, so only the field data and the date differ between frames.
    frames = []
    for i in range(len(times)):
        mesh_pred.set_array(pred[i])
        mesh_truth.set_array(truth[i])
        suptitle.set_text(format_time(times[i]))
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(Image.fromarray(rgb))
    plt.close(fig)

    # Encode the GIF with ONE fixed palette (a dense viridis ramp + neutrals for
    # the white background / grey coastlines / black text). A shared palette
    # keeps the colours and the colorbar identical in every frame and renders
    # the gradient smoothly instead of in GIF banding.
    palette_img = _fixed_palette(CMAP)
    qframes = [f.quantize(palette=palette_img, dither=Image.NONE) for f in frames]
    qframes[0].save(
        outfn, save_all=True, append_images=qframes[1:],
        duration=int(1000 / fps), loop=0, disposal=2, optimize=False,
    )
    print(f"saved: {outfn} ({len(times)} frames @ {fps} fps, scale 0-{vmax:g} m/s)")


def _fixed_palette(cmap_name: str, n_colors: int = 232) -> Image.Image:
    """A 256-entry palette: a fine `cmap_name` ramp plus a grey/black/white ramp."""
    ramp = (plt.get_cmap(cmap_name)(np.linspace(0, 1, n_colors))[:, :3] * 255)
    greys = np.linspace(0, 255, 256 - n_colors)[:, None].repeat(3, axis=1)
    pal = np.vstack([ramp, greys]).round().astype(np.uint8)
    palette_img = Image.new("P", (1, 1))
    palette_img.putpalette(pal.flatten().tolist())
    return palette_img


if __name__ == "__main__":
    main(Path("figures/realv2_vmax10m.gif"))
