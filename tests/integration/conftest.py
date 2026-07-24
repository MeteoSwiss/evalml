import os
from pathlib import Path

import pytest

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "jretrieve"
_BIN_DIR = _FIXTURE_DIR / "bin"


@pytest.fixture
def mock_jretrieve(monkeypatch):
    """Intercept jretrieve by placing the mock binary first on PATH.

    fixtures/jretrieve/bin/jretrievedwh.py lives on the shared /scratch
    filesystem, so it is accessible from both the pytest node and any SLURM
    compute nodes that Snakemake dispatches plot_meteogram jobs to.

    monkeypatch restores os.environ to its original state after the test, so
    real credentials and PATH are unaffected in subsequent runs.
    """
    monkeypatch.setenv("PATH", f"{_BIN_DIR}:{os.environ.get('PATH', '')}")
    monkeypatch.setenv("JRETRIEVE_FIXTURE_DIR", str(_FIXTURE_DIR))
    monkeypatch.setenv("JRETRIEVE_CLIENT_ID", "mock")
    monkeypatch.setenv("JRETRIEVE_CLIENT_SECRET", "mock")
