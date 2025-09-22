import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# WInd RGB values
rgb_list = [
    (255, 255, 255),
    (239, 244, 209),
    (232, 244, 158),
    (170, 206, 99),
    (226, 237, 22),
    (255, 237, 0),
    (255, 237, 130),
    (244, 209, 127),
    (237, 165, 73),
    (229, 140, 61),
    (219, 124, 61),
    (239, 7, 61),
    (232, 86, 163),
    (155, 112, 168),
    (99, 112, 247),
    (127, 150, 255),
    (142, 178, 255),
    (181, 201, 255),
]


levels = [
    4.0,
    6.0,
    10.0,
    14.0,
    18.0,
    22.0,
    26.0,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
    60.0,
    70.0,
    80.0,
    90.0,
    100.0,
]

# levels = [(lv / 100.0) * 75.0 for lv in levels]

# Normalize your levels between 0 and 1 for colormap mapping
min_level = min(levels)
max_level = max(levels)
normalized_positions = [(lv - min_level) / (max_level - min_level) for lv in levels]

# Normalize RGB values
normalized_colors = [(r / 255, g / 255, b / 255) for r, g, b in rgb_list]

# Create the colormap using LinearSegmentedColormap
wind_cmap = LinearSegmentedColormap.from_list(
    "custom_map", list(zip(normalized_positions, normalized_colors))
)

levels = [
    -18.0,
    -16.0,
    -14.0,
    -12.0,
    -10.0,
    -8.0,
    -6.0,
    -4.0,
    -2.0,
    0.0,
    2.0,
    4.0,
    6.0,
    8.0,
    10.0,
    12.0,
    14.0,
    16.0,
    18.0,
    20.0,
    22.0,
    24.0,
    26.0,
    28.0,
    30.0,
    32.0,
    34.0,
    36.0,
    38.0,
]

rgb_list = [
    (109, 227, 255),
    (175, 240, 255),
    (255, 196, 226),
    (255, 153, 204),
    (255, 0, 255),
    (128, 0, 128),
    (0, 0, 128),
    (70, 70, 255),
    (51, 102, 255),
    (133, 162, 255),
    (255, 255, 255),
    (204, 204, 204),
    (179, 179, 179),
    (153, 153, 153),
    (96, 96, 96),
    (128, 128, 0),
    (0, 92, 0),
    (0, 128, 0),
    (51, 153, 102),
    (157, 213, 0),
    (212, 255, 91),
    (255, 255, 0),
    (255, 184, 112),
    (255, 153, 0),
    (255, 102, 0),
    (255, 0, 0),
    (188, 75, 0),
    (171, 0, 56),
    (128, 0, 0),
    (163, 112, 255),
]

# Normalize your levels between 0 and 1 for colormap mapping
min_level = min(levels)
max_level = max(levels)
normalized_positions = [(lv - min_level) / (max_level - min_level) for lv in levels]

# Normalize RGB values
normalized_colors = [(r / 255, g / 255, b / 255) for r, g, b in rgb_list]

# Create the colormap using LinearSegmentedColormap
t2m_cmap = LinearSegmentedColormap.from_list(
    "custom_map", list(zip(normalized_positions, normalized_colors))
)

levels = [30, 45, 60, 75, 90, 95]
rgb_list = [
    (251, 155, 52),
    (253, 206, 102),
    (254, 255, 153),
    (206, 254, 154),
    (120, 240, 116),
    (55, 202, 51),
    (54, 177, 52),
]

# Normalize your levels between 0 and 1 for colormap mapping
min_level = min(levels)
max_level = max(levels)
normalized_positions = [(lv - min_level) / (max_level - min_level) for lv in levels]

# Normalize RGB values
normalized_colors = [(r / 255, g / 255, b / 255) for r, g, b in rgb_list]

# Create the colormap using LinearSegmentedColormap
qv_cmap = LinearSegmentedColormap.from_list(
    "custom_map", list(zip(normalized_positions, normalized_colors))
)

FIELD_DEFAULTS = {
    "sp": {"cmap": plt.get_cmap("coolwarm", 11), "vmin": 800 * 100, "vmax": 1100 * 100},
    "2d": {"cmap": plt.get_cmap("inferno", 11), "vmin": 240, "vmax": 300},
    "2t": {"cmap": t2m_cmap, "vmin": -18, "vmax": 38},
    "10v": {"cmap": plt.get_cmap("GnBu", 11), "vmin": 0, "vmax": 40},
    "10u": {"cmap": plt.get_cmap("GnBu", 11), "vmin": 0, "vmax": 25},
    "uv": {"cmap": wind_cmap, "vmin": 0, "vmax": 40},
    "10si": {"cmap": plt.get_cmap("GnBu", 11), "vmin": 0, "vmax": 25},
    "t_850": {"cmap": plt.get_cmap("inferno", 11), "vmin": 220, "vmax": 310},
    "z_850": {"cmap": plt.get_cmap("coolwarm", 11), "vmin": 8000, "vmax": 17000},
    "q_925": {"cmap": qv_cmap, "vmin": 0, "vmax": 0.0125},
}
"""Mapping of field names to good default plotting parameters."""
