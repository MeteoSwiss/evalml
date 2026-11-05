from evalml.cli import workflow_env


def test_uv_cache_on_scratch(monkeypatch):
    monkeypatch.setenv("SCRATCH", "/scratch/me")
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    assert workflow_env()["UV_CACHE_DIR"] == "/scratch/me/.cache/uv"


def test_uv_cache_user_choice_kept(monkeypatch):
    monkeypatch.setenv("SCRATCH", "/scratch/me")
    monkeypatch.setenv("UV_CACHE_DIR", "/custom")
    assert workflow_env()["UV_CACHE_DIR"] == "/custom"


def test_uv_cache_untouched_without_scratch(monkeypatch):
    monkeypatch.delenv("SCRATCH", raising=False)
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    assert "UV_CACHE_DIR" not in workflow_env()
