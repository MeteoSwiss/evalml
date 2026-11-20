import importlib.util
import sys
from pathlib import Path

import xarray as xr

SCRIPT = Path(__file__).resolve().parents[2] / "workflow/scripts/spectra_compute.py"
# spectra_compute.py loaded by path (workflow/scripts is not a package).
_spec = importlib.util.spec_from_file_location("spectra_compute", SCRIPT)
_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_script)


def _run(monkeypatch, tmp_path, extra_args):
    calls = {}

    def fake_load_forecast_data(root, reftime, steps, params, member="000"):
        calls["steps"] = steps
        calls["member"] = member
        return xr.Dataset()

    monkeypatch.setattr(_script, "load_forecast_data", fake_load_forecast_data)
    monkeypatch.setattr(
        _script, "compute_source_spectra", lambda *args, **kwargs: xr.Dataset()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "spectra_compute.py",
            "--forecast",
            str(tmp_path),
            "--reftime",
            "202406150000",
            "--steps",
            "0/24/6",
            "--lead_times",
            "6,24",
            "--variables",
            "T_2M",
            "--label",
            "baseline",
            "--output",
            str(tmp_path / "spectra.nc"),
            *extra_args,
        ],
    )
    _script.main()
    return calls


def test_member_is_passed_to_loader(monkeypatch, tmp_path):
    calls = _run(monkeypatch, tmp_path, ["--member", "median"])
    assert calls["member"] == "median"


def test_member_defaults_to_control(monkeypatch, tmp_path):
    calls = _run(monkeypatch, tmp_path, [])
    assert calls["member"] == "000"


def test_loads_only_requested_lead_times(monkeypatch, tmp_path):
    calls = _run(monkeypatch, tmp_path, [])
    assert calls["steps"] == [6, 24]
