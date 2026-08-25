"""Registering an experiment: the config-reading half of the experiment store.

`evalml.store` owns the store layout, the index, the symlinks and the Confluence page.
This module reads the results directory's `config.yaml`, extracts what the store
records — description, registered models, baselines — adds evalml's own provenance
(version, commit, dirty state), and hands a store-shaped payload to
`evalml.store.register`.

What identifies an evaluation: the results directory's basename,
`{date}_{config-label}_{confighash}` — the name the workflow itself derives for one
evaluation run (see EXPERIMENT_NAME in workflow/Snakefile). Re-running the same config
produces the same basename and is refused as already registered; re-running it on
another day is a new evaluation and registers cleanly.

Which models an experiment used: every `checkpoint` in the config that is either a bare
registered-model name (`amber-ridge`) or a path under the model store resolves to a
model-store name; anything else (MLflow URLs, Hugging Face URLs, plain local paths) has
no stable identity and is not cross-referenced.
"""

import re
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from evalml import store
from evalml.config import REGISTERED_MODEL_PATTERN

PROJECT_ROOT = Path(__file__).parents[2]


def evalml_provenance():
    """Version, commit and clean/dirty state of the evalml that registers. Recorded, not
    gated on — an experiment run from a dirty tree is still worth archiving, but a reader
    must be able to tell. Best effort: an evalml installed without its repo has no commit."""
    try:
        from importlib.metadata import version

        release = version("evalml")
    except Exception:  # noqa: BLE001 — provenance is recorded, never fatal
        release = ""

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    commit, dirty = "", None
    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "-uno"))
    except (subprocess.CalledProcessError, OSError):
        pass
    return {"version": release, "commit": commit, "dirty": dirty}


def run_blocks(config):
    """Every forecaster / temporal-downscaler block in the config, including a temporal
    downscaler's nested upstream forecaster. Baselines are not run blocks."""
    blocks = []
    for item in config.get("runs") or []:
        if not isinstance(item, dict):
            continue
        for key in ("forecaster", "temporal_downscaler"):
            block = item.get(key)
            if isinstance(block, dict):
                blocks.append(block)
                nested = block.get("forecaster")
                if isinstance(nested, dict):
                    blocks.append(nested)
    return blocks


def model_names(config, models_store):
    """The registered model-store names an experiment's checkpoints refer to, in config
    order, deduplicated. Both spellings count: the bare name the config schema resolves
    at run time, and an explicit path under the model store."""
    prefix = str(models_store).rstrip("/") + "/"
    names = []
    for block in run_blocks(config):
        checkpoint = str(block.get("checkpoint") or "")
        name = ""
        if REGISTERED_MODEL_PATTERN.match(checkpoint):
            name = checkpoint
        elif checkpoint.startswith(prefix):
            name = checkpoint[len(prefix) :].split("/", 1)[0]
        if name and name not in names:
            names.append(name)
    return names


def baseline_labels(config):
    labels = []
    for item in config.get("runs") or []:
        block = item.get("baseline") if isinstance(item, dict) else None
        if isinstance(block, dict):
            label = str(block.get("label") or "").strip()
            if label and label not in labels:
                labels.append(label)
    return labels


def default_name(identity):
    """A slug from the identity, `{date}_{label}_{hash}` -> `{label-slug}-{date}` — e.g.
    `20260824_forecasters-ich1-oper_ab12` -> `forecasters-ich1-oper-20260824`.
    Descriptive and self-dating; a collision (same config registered twice) is refused by
    the store with a hint to pass an explicit name."""
    parts = identity.split("_")
    if len(parts) >= 3 and re.fullmatch(r"\d{8}", parts[0]):
        date, label = parts[0], "_".join(parts[1:-1])
    else:
        date, label = datetime.now().strftime("%Y%m%d"), identity
    slug = re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", label.lower())).strip("-")
    return "%s-%s" % (slug or "experiment", date)


def load_config(results_dir, config_path):
    """The config that produced the run: `config.yaml` inside the results directory (the
    workflow copies it there), or an explicit --config for results that predate that."""
    path = Path(config_path) if config_path else Path(results_dir) / "config.yaml"
    if not path.is_file():
        store.fail(
            "%s does not contain config.yaml — re-run the experiment with a current "
            "evalml (the workflow now copies the config into the results), or pass "
            "--config pointing at the YAML that produced it" % results_dir
        )
    try:
        return yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        store.fail("%s is not readable YAML: %s" % (path, exc))


def register_results(
    results_dir,
    name=None,
    config_path=None,
    store_dir=store.STORE,
    models_store=store.MODELS_STORE,
    dry_run=False,
    no_publish=False,
):
    """Extract the payload from the results directory and register it. Returns the name."""
    results_dir = Path(results_dir)
    config = load_config(results_dir, config_path)
    identity = results_dir.resolve().name
    meta = {
        "description": str(config.get("description") or "").strip(),
        "models": model_names(config, models_store),
        "baselines": baseline_labels(config),
        "evalml": {**evalml_provenance(), "config": "%s/config.yaml" % store.RESULTS},
        "identity": identity,
    }
    return store.register(
        results_dir,
        name or default_name(identity),
        meta,
        store=store_dir,
        models_store=models_store,
        config_src=config_path,
        dry_run=dry_run,
        no_publish=no_publish,
    )
