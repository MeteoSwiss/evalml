import os
import shutil
from pathlib import Path

import pytest

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "jretrieve"


@pytest.fixture
def mock_jretrieve(monkeypatch, tmp_path):
    """Intercept jretrieve by placing mock_jretrievedwh.py first on PATH.

    The subprocess spawned by ``evalml showcase`` inherits the patched PATH and
    JRETRIEVE_FIXTURE_DIR, so the mock binary is resolved by shutil.which and
    serves pre-stored fixture CSVs instead of contacting the real DWH.

    monkeypatch restores os.environ to its original state after the test, so
    real credentials and PATH are unaffected in subsequent runs.
    """
    fake_bin = tmp_path / "jretrievedwh.py"
    shutil.copy(_FIXTURE_DIR / "mock_jretrievedwh.py", fake_bin)
    fake_bin.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH', '')}")
    monkeypatch.setenv("JRETRIEVE_FIXTURE_DIR", str(_FIXTURE_DIR))
    monkeypatch.setenv("JRETRIEVE_CLIENT_ID", "mock")
    monkeypatch.setenv("JRETRIEVE_CLIENT_SECRET", "mock")
