import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "workflow" / "scripts"))
from plot_meteogram import resolve_plot_variable  # noqa: E402


def test_tot_prec_plotted_as_step_totals():
    assert resolve_plot_variable("TOT_PREC", list(range(0, 121, 6))) == "TOT_PREC6"
    assert resolve_plot_variable("TOT_PREC", list(range(0, 34, 1))) == "TOT_PREC1"


def test_other_params_unchanged():
    assert resolve_plot_variable("SP_10M", list(range(0, 121, 6))) == "SP_10M"
    assert resolve_plot_variable("TOT_PREC6", list(range(0, 121, 6))) == "TOT_PREC6"


def test_single_step_keeps_param():
    assert resolve_plot_variable("TOT_PREC", [6]) == "TOT_PREC"
