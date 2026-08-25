"""The experiment store, exercised against temp directories — never the real paths."""

import json
import os

import pytest
import yaml

from evalml import registration, store


@pytest.fixture
def models_store(tmp_path):
    """A model store the devml way: <name>/model.json + a sticky experiments/ dir."""
    root = tmp_path / "models"
    for name in ("amber-ridge", "calm-breeze"):
        (root / name / "experiments").mkdir(parents=True)
        (root / name / "model.json").write_text(json.dumps({"name": name}))
    return root


def make_results(
    base, models_store, identity="20260824_myexp_ab12", checkpoint="amber-ridge"
):
    results = base / identity
    (results / "dashboard").mkdir(parents=True)
    (results / "dashboard" / "index.html").write_text("<html/>")
    runs = [
        # Both checkpoint spellings that count as a registered model.
        {"forecaster": {"checkpoint": checkpoint, "steps": "0/120/6"}},
        {
            "forecaster": {
                "checkpoint": str(models_store / "calm-breeze" / "inference-last.ckpt"),
                "steps": "0/120/6",
            }
        },
        {"baseline": {"label": "ICON-CH1", "root": "/x", "steps": "0/120/6"}},
    ]
    (results / "config.yaml").write_text(
        yaml.safe_dump({"description": "A test experiment.", "runs": runs})
    )
    return results


def register(results, exp_store, models_store, **kwargs):
    kwargs.setdefault("no_publish", True)
    return registration.register_results(
        results, store_dir=exp_store, models_store=models_store, **kwargs
    )


def test_lifecycle(tmp_path, models_store):
    exp_store = tmp_path / "experiments"
    results = make_results(tmp_path, models_store)

    # A dry run copies nothing; the real thing is immutable, cross-linked, indexed.
    name = register(results, exp_store, models_store, dry_run=True)
    assert name == "myexp-20260824" and not (exp_store / name).exists()
    register(results, exp_store, models_store)
    target = exp_store / name
    manifest = json.loads((target / "experiment.json").read_text())
    assert manifest["models"] == ["amber-ridge", "calm-breeze"]
    assert manifest["baselines"] == ["ICON-CH1"]
    assert manifest["identity"] == "20260824_myexp_ab12"
    assert (target / "experiment.json").stat().st_mode & 0o777 == 0o444
    link = models_store / "amber-ridge" / "experiments" / name
    assert os.readlink(str(link)) == "../../../experiments/%s" % name
    assert [r["name"] for r in store.load(exp_store)["experiments"]] == [name]

    # The publish section owns its markers, is located with or without them, and an
    # unchanged table still matches once Confluence stamps macro-ids onto the anchors.
    body = store.section(store.scan(exp_store)[0], exp_store, retired=[])
    assert "pageId=%s#amber-ridge" % store.MODELS_PAGE in body
    page = "<p>i</p>" + body + "<h2>o</h2>"
    start, end, how = store.locate(page)
    assert how == "markers" and page[start:end] == body
    eaten = page.replace(store.START, "").replace(store.END, "")
    assert store.locate(eaten)[2] == "heading"
    stamped = body.replace('ac:name="anchor"', 'ac:name="anchor" ac:macro-id="1"')
    assert store.same_table(stamped, body)
    with pytest.raises(SystemExit, match="not the default store"):
        store.cmd_publish(store=exp_store)

    # Derived state is repairable...
    store.index_path(exp_store).unlink()
    link.unlink()
    store.cmd_list(store=exp_store, models_store=models_store, rebuild_index=True)
    assert store.index_path(exp_store).is_file() and link.is_symlink()

    # ...and unregistering tombstones the name, unlinks, and deletes the payload last.
    store.cmd_unregister(
        name, store=exp_store, models_store=models_store, yes=True, no_publish=True
    )
    tomb = store.orphan(exp_store, name)
    assert not (exp_store / name).exists() and (tomb / "NOTE.txt").is_file()
    assert not (tomb / "results").exists() and not link.is_symlink()
    loaded = store.load(exp_store)
    assert loaded["experiments"] == []
    assert [r["name"] for r in loaded["retired"]] == [name]


def test_refusals(tmp_path, models_store, monkeypatch):
    exp_store = tmp_path / "experiments"
    name = register(make_results(tmp_path, models_store), exp_store, models_store)
    fresh = make_results(tmp_path / "b", models_store, identity="20260825_myexp_cd34")

    with pytest.raises(SystemExit, match="already a registration"):  # same identity
        register(
            make_results(tmp_path / "a", models_store),
            exp_store,
            models_store,
            name="other",
        )
    store.orphan(exp_store, "spent").mkdir()
    with pytest.raises(SystemExit, match="never reused"):  # tombstones retire names
        register(fresh, exp_store, models_store, name="spent")
    with pytest.raises(SystemExit, match="not a registered model"):
        register(
            make_results(
                tmp_path / "c", models_store, identity="x", checkpoint="misty-tarn"
            ),
            exp_store,
            models_store,
        )
    monkeypatch.setattr(os, "getuid", lambda uid=os.getuid(): uid + 1)
    with pytest.raises(SystemExit, match="not you"):  # only the owner unregisters
        store.cmd_unregister(
            name, store=exp_store, models_store=models_store, yes=True, no_publish=True
        )
