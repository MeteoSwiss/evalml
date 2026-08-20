"""Tests for the promoted publication style module."""

from evalml.publication import style


def test_param_label_known_and_fallback():
    assert style.param_label("T_2M") == "2m Temperature"
    # Unknown codes fall back to the code itself.
    assert style.param_label("ZZZ") == "ZZZ"


def test_line_style_source_selection():
    assert style.line_style("ICON-CH1-CTRL")["color"] == style.COLOR_CH1
    assert style.line_style("ICON-CH2-CTRL")["color"] == style.COLOR_CH2
    assert style.line_style("Varda-Single")["color"] == style.COLOR_VARDA
    # EPS mean sources are dashed.
    assert style.line_style("ICON-CH1-EPS mean")["linestyle"] == "--"
    # The observations source is markers-only (no line).
    assert style.line_style(style.OBS_LABEL)["linestyle"] == "none"


def test_mplstyle_path_exists():
    p = style.mplstyle_path()
    assert p.name == "publication.mplstyle"
    assert p.is_file()
