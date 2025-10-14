import pathlib
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

# Base directory for colormap files
BASE_DIR = (
    pathlib.Path(__file__).resolve().parents[2] / "resources" / "report" / "plotting"
)


def load_ncl_colormap(filename):
    """Load colormap file into a matplotlib ListedColormap and BoundaryNorm.

    Returns
    -------
    dict
        Dictionary containing the colormap and normalisation generated from the
        colormap file
        {cmap : matplotlib.colors.ListedColormap,
         norm : matplotlib.colors.BoundaryNorm  }
    """
    cmap_path = BASE_DIR / filename
    if not cmap_path.exists():
        raise FileNotFoundError(f"Colormap file not found: {cmap_path}")
    with open(cmap_path, "r") as f:
        lines = f.readlines()

    # Remove header
    lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith(";")]

    # Number of levels on first line
    try:
        n_levs = int(lines[0])
    except ValueError:
        raise ValueError(
            f"Expected number of levels in first non-header line of {cmap_path}"
        )

    # Colormap bounds on second line
    bounds = [float(x) for x in lines[1].split()]
    if len(bounds) != n_levs:
        raise ValueError(f"Bounds must have {n_levs} values, got {len(bounds)}")

    # RGB values
    rgb_lines = lines[2:]
    rgb = np.array([[int(x) for x in line.split()] for line in rgb_lines], dtype=float)
    rgb /= 255.0  # scale to [0,1] for matplotlib
    if len(rgb) != n_levs + 1:
        raise ValueError(f"Expected {n_levs} RGB rows, got {len(rgb)}.")

    # Create colormap and norm
    cmap = ListedColormap(colors=rgb[1:-1], name=pathlib.Path(filename).stem)
    cmap.set_under(rgb[0])
    cmap.set_over(rgb[-1])
    norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)

    return {"cmap": cmap, "norm": norm, "bounds": bounds}
