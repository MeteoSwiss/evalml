"""Lightweight package root. Deliberately avoids importing anything heavy
(cartopy, earthkit.plots, matplotlib) at module level: Python always runs a
package's __init__.py before any of its submodules, so an eager import here
would force that whole visualization stack to load even for a submodule like
plotting.compat, which only loads data and never renders anything.

DOMAINS/get_projection/StatePlotter (plotting.viz's actual content) are
re-exported lazily via module __getattr__ (PEP 562), so `from plotting import
DOMAINS` still works exactly as before for callers that do plot -- they just
pay the cartopy/earthkit.plots import cost at that point instead of always.
"""

import numpy as np

State = dict[str, np.ndarray | dict[str, np.ndarray]]

_LAZY_VIZ_NAMES = ("DOMAINS", "get_projection", "StatePlotter")


def __getattr__(name):
    if name in _LAZY_VIZ_NAMES:
        from . import viz

        return getattr(viz, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
