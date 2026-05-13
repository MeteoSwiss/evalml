"""Stable source -> color mapping shared with the dashboard.

The dashboard uses Vega-Lite's ``tableau10`` categorical scheme and pins its
``color.scale.domain`` to the alphabetically-sorted full source list so the
mapping stays bijective regardless of dashboard filters. The matplotlib plots
use the same palette and ordering so a given source has the same color in
every figure produced from a verification run.

Both the dashboard and the matplotlib side wrap around when there are more
than ``len(TABLEAU10)`` sources, at which point two sources will share a
color. Switch palettes (e.g. to ``tableau20`` or a deterministic HSV ramp)
if that becomes a problem.
"""

# Vega-Lite "tableau10" scheme:
# https://vega.github.io/vega/docs/schemes/#tableau10
TABLEAU10: list[str] = [
    "#4c78a8",
    "#f58518",
    "#e45756",
    "#72b7b2",
    "#54a24b",
    "#eeca3b",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    "#bab0ac",
]


def source_color_map(sources) -> dict[str, str]:
    """Return ``{source: color}`` over unique sources, ordered alphabetically.

    Wraps around for more than ``len(TABLEAU10)`` sources, matching Vega-Lite's
    behaviour for a categorical scale whose domain exceeds the scheme.
    """
    ordered = sorted(set(sources))
    return {s: TABLEAU10[i % len(TABLEAU10)] for i, s in enumerate(ordered)}
