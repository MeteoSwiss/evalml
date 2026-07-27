from pathlib import Path
from evalml.fixtures import fixture_grib_dir


def test_fixture_grib_dir_mirrors_output_layout():
    got = fixture_grib_dir("/fx", "forecaster-abcd/6640", "202503010000")
    assert got == Path("/fx/data/runs/forecaster-abcd/6640/202503010000/grib")


def test_fixture_grib_dir_accepts_path_and_int_init():
    got = fixture_grib_dir(Path("/fx"), "temporal-x-on-forecaster-abcd/1a2b", 202503010000)
    assert got == Path("/fx/data/runs/temporal-x-on-forecaster-abcd/1a2b/202503010000/grib")
