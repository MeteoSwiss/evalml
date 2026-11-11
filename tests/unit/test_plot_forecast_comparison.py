import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "workflow" / "scripts"))
import plot_forecast_comparison  # noqa: E402
from plot_forecast_comparison import parse_source  # noqa: E402


def test_parse_source():
    assert parse_source("ICON-CH1=/store/x=y") == ("ICON-CH1", "/store/x=y")


@pytest.mark.parametrize("text", ["no-separator", "=root", "label="])
def test_parse_source_rejects(text):
    with pytest.raises(ValueError):
        parse_source(text)


@pytest.fixture
def run_main(monkeypatch, tmp_path):
    """Run main() without plotting; loaders raise for roots listed in `errors`."""
    loaded = []

    def run(args, errors):
        def loader(root, *_):
            if root in errors:
                raise errors[root]
            loaded.append(root)
            return np.zeros(1), np.zeros(1), np.zeros(1)

        monkeypatch.setattr(plot_forecast_comparison, "load_source", loader)
        monkeypatch.setattr(plot_forecast_comparison, "load_truth", loader)
        monkeypatch.setattr(plot_forecast_comparison, "StatePlotter", lambda *_: None)
        monkeypatch.setattr(
            sys,
            "argv",
            ["plot_forecast_comparison.py", *args]
            + ["--date", "202401010000", "--leadtime", "6", "--param", "T_2M"]
            + ["--regions_json", "{}", "--outdir", str(tmp_path)],
        )
        plot_forecast_comparison.main()
        return loaded

    return run


def test_missing_baseline_and_truth_are_skipped(run_main):
    args = ["--source", "ML=run", "--baseline", "ICON=icon", "--truth", "truth"]
    errors = {"icon": FileNotFoundError("icon"), "truth": LookupError("truth")}
    assert run_main(args, errors) == ["run"]


def test_candidate_failure_raises(run_main):
    with pytest.raises(FileNotFoundError):
        run_main(["--source", "ML=run"], {"run": FileNotFoundError("run")})


def test_unexpected_baseline_failure_raises(run_main):
    with pytest.raises(ValueError):
        run_main(["--baseline", "ICON=icon"], {"icon": ValueError("corrupt")})


def test_nothing_loaded_raises(run_main):
    with pytest.raises(RuntimeError):
        run_main(["--baseline", "ICON=icon"], {"icon": FileNotFoundError("icon")})
