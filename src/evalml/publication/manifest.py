"""Build, write and load the publication manifest.

The manifest persists everything the publication figures need to locate their
input data without recomputing any hash:

* which runs/baselines participate (id, label, role, model_type, steps, member),
* the truth source (label, hash, gridded vs station/obs),
* the enumerated initialisation times,
* concrete data paths plus raw path templates for any uncommon combination.

``build_manifest`` is a pure dict-in/dict-out transform (no I/O, no Snakemake
globals) so it is unit-testable and can be called from the Snakemake ``run:``
block that has the in-memory globals to hand.
"""

import json
import os
from pathlib import Path

SCHEMA_VERSION = 1

DEFAULT_MANIFEST_RELPATH = "publication/manifest.json"

# Path templates, relative to ``output_root``. run_id / baseline_id are opaque
# (run_id contains '/'), so these are only ever str.format-joined, never split.
PATH_TEMPLATES = {
    "run_verif": "data/runs/{run_id}/verif_aggregated_{truth_hash}.nc",
    "run_grib": "data/runs/{run_id}/{init_time}/grib",
    "run_scoremap": "data/runs/{run_id}/scoremaps/{param}_{leadtime}_{truth_hash}.nc",
    "baseline_verif": "data/baselines/{baseline_id}/verif_aggregated_{truth_hash}.nc",
    "baseline_scoremap": "data/baselines/{baseline_id}/scoremaps/{param}_{leadtime}_{truth_hash}.nc",
}


def _join(output_root: str, relative: str) -> str:
    """Join a relative template onto output_root keeping forward slashes."""
    return f"{output_root.rstrip('/')}/{relative}"


def _truth_type(root: str) -> str:
    """Mirror common.smk ``truth_file_dep``: jretrieve markers vs a zarr path."""
    return "jretrieve" if "jretrieve" in str(root) else "zarr"


def build_manifest(
    *,
    run_configs: dict,
    baseline_configs: dict,
    truth_cfg: dict | None,
    truth_hash: str,
    reftimes,
    output_root: str,
    publication_cfg: dict | None,
    master_hash: str,
    generated_at: str | None = None,
) -> dict:
    """Assemble the manifest dict from the in-memory workflow globals.

    Parameters mirror the globals defined in ``workflow/rules/common.smk``:
    ``RUN_CONFIGS``, ``BASELINE_CONFIGS``, ``config["truth"]``, ``TRUTH_HASH``,
    ``REFTIMES`` (list of datetimes), ``str(OUT_ROOT)``, ``config["publication"]``
    and ``master_hash()``.
    """
    output_root = str(output_root)
    truth_root = (truth_cfg or {}).get("root", "")
    ttype = _truth_type(truth_root)

    init_times = sorted(t.strftime("%Y%m%d%H%M") for t in reftimes)

    participants = []
    # Baselines first, then candidates, matching collect_experiment_participants()
    # ordering so publication_figures source ordering is unchanged.
    for baseline_id, cfg in baseline_configs.items():
        participants.append(
            {
                "id": baseline_id,
                "label": cfg.get("label", baseline_id),
                "role": "baseline",
                "model_type": "baseline",
                "steps": cfg.get("steps"),
                "member": cfg.get("member"),
                "source_root": cfg.get("root"),
                "is_candidate": False,
                "paths": {
                    "verif_aggregated": _join(
                        output_root,
                        PATH_TEMPLATES["baseline_verif"].format(
                            baseline_id=baseline_id, truth_hash=truth_hash
                        ),
                    ),
                    "scoremap_template": _join(
                        output_root,
                        PATH_TEMPLATES["baseline_scoremap"].format(
                            baseline_id=baseline_id,
                            truth_hash=truth_hash,
                            param="{param}",
                            leadtime="{leadtime}",
                        ),
                    ),
                },
            }
        )

    for run_id, cfg in run_configs.items():
        if not cfg.get("_is_candidate", False):
            continue
        participants.append(
            {
                "id": run_id,
                "label": cfg.get("label") or run_id,
                "role": "candidate",
                "model_type": cfg.get("model_type"),
                "steps": cfg.get("steps"),
                "member": None,
                "source_root": None,
                "is_candidate": True,
                "paths": {
                    "verif_aggregated": _join(
                        output_root,
                        PATH_TEMPLATES["run_verif"].format(
                            run_id=run_id, truth_hash=truth_hash
                        ),
                    ),
                    "grib_dir_template": _join(
                        output_root,
                        PATH_TEMPLATES["run_grib"].format(
                            run_id=run_id, init_time="{init_time}"
                        ),
                    ),
                    "scoremap_template": _join(
                        output_root,
                        PATH_TEMPLATES["run_scoremap"].format(
                            run_id=run_id,
                            truth_hash=truth_hash,
                            param="{param}",
                            leadtime="{leadtime}",
                        ),
                    ),
                },
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "master_hash": master_hash,
        "output_root": output_root,
        "truth": {
            "label": (truth_cfg or {}).get("label"),
            "hash": truth_hash,
            "type": ttype,
            "root": truth_root,
            "gridded": ttype == "zarr",
        },
        "dates": {
            "start": init_times[0] if init_times else None,
            "end": init_times[-1] if init_times else None,
            "init_times": init_times,
        },
        "participants": participants,
        "path_templates": PATH_TEMPLATES,
        "publication": publication_cfg or {},
    }
    if generated_at is not None:
        manifest["generated_at"] = generated_at
    return manifest


def write_manifest(path, manifest: dict) -> None:
    """Serialise a manifest dict to ``path`` as pretty JSON (creates parents)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def default_manifest_path(output_root: str | None = None) -> Path:
    """Resolve the default manifest location.

    Precedence: ``$EVALML_MANIFEST`` env var > ``<output_root>/publication/
    manifest.json`` > ``output/publication/manifest.json``.
    """
    env = os.environ.get("EVALML_MANIFEST")
    if env:
        return Path(env)
    root = output_root or "output"
    return Path(root) / DEFAULT_MANIFEST_RELPATH


def load_manifest_dict(path=None) -> dict:
    """Load and return the raw manifest dict, applying default-path precedence."""
    p = Path(path) if path else default_manifest_path()
    if not p.exists():
        raise FileNotFoundError(
            f"Publication manifest not found at {p}. Generate it with "
            f"`evalml make <config> output/publication/manifest.json` "
            f"(or set $EVALML_MANIFEST)."
        )
    with p.open() as f:
        return json.load(f)


def load_manifest(path=None):
    """Load the manifest and wrap it in a :class:`evalml.publication.resolver.Manifest`."""
    from evalml.publication.resolver import Manifest

    return Manifest(load_manifest_dict(path))
