"""Sphinx configuration for the EvalML documentation."""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

# Project layout: docs/source/conf.py -> repo root is two parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

# -- Project information -----------------------------------------------------

project = "EvalML"
author = "MeteoSwiss"
copyright = f"%Y, {author}"

try:
    release = metadata.version("evalml")
except metadata.PackageNotFoundError:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_click",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

autosummary_generate = False
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Napoleon picks up both Google- and NumPy-style docstrings present in src/.
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Imports that may not be available on the docs builder (heavy scientific stack).
# We don't currently mock anything because RTD installs the project, but if a
# build env can't pull cartopy/earthkit, add modules here.
autodoc_mock_imports: list[str] = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "click": ("https://click.palletsprojects.com/en/stable/", None),
    "snakemake": ("https://snakemake.readthedocs.io/en/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# Keep the first build green; flip to True once docstring coverage improves.
nitpicky = False

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = f"EvalML {version}"

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "titles_only": False,
}
