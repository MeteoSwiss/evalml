from pathlib import Path

import pytest
import yaml

CONFIGS = sorted(
    (Path(__file__).resolve().parents[2] / "resources/inference/configs").glob("*.yaml")
)


def _rescales(node, out):
    """Collect (filter key, param, scale, offset) of every rescale transform filter."""
    if isinstance(node, list):
        for item in node:
            _rescales(item, out)
    elif isinstance(node, dict):
        for key, value in node.items():
            if key.endswith("_transform_filter") and isinstance(value, dict):
                f = (
                    value.get("rescale", value)
                    if value.get("filter", "rescale") == "rescale"
                    else {}
                )
                if "param" in f:
                    out.append((key, f["param"], f["scale"], f["offset"]))
            _rescales(value, out)
    return out


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_tp_rescale_has_grib_name_twin(path):
    """anemoi-inference <= 0.11.1 exposes the model name (tp) to output filters, >= 0.11.2
    and GRIB inputs the GRIB name (TOT_PREC); a filter on only one name silently does nothing."""
    rescales = _rescales(yaml.safe_load(path.read_text()), [])
    for key, param, scale, offset in rescales:
        if param == "tp":
            assert (key, "TOT_PREC", scale, offset) in rescales, path.name
