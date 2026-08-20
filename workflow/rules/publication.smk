# ----------------------------------------------------- #
# Publication-grade figures workflow                    #
# ----------------------------------------------------- #
# The publication target produces *only* the manifest (a run/baseline -> hash ->
# data-path mapping). The result files it points at (verification NC files,
# scoremaps, GRIB) are produced by the experiment/verification workflow
# (`evalml experiment`, driven by the `experiment:` config blocks), not here.
# All publication-ready plotting happens in the standalone notebooks under
# `notebooks/publication/`, driven only by the manifest — see
# docs/publication_figures.md.

from pathlib import Path as _Path
from evalml.publication.manifest import config_slug as _config_slug

# The manifest is namespaced by a config slug (config file stem + short master
# hash) so each config file produces a distinct manifest even when two configs
# share the same truth label.
_CONFIGFILE_STEM = _Path(workflow.configfiles[0]).stem if workflow.configfiles else "config"
CONFIG_SLUG = _config_slug(_CONFIGFILE_STEM)


rule publication_manifest:
    # Persist the run/baseline -> hash -> data-path mapping for the CLI/notebooks.
    #
    # Cheap localrule: dumps the in-memory workflow globals to JSON, so paths can be
    # resolved (interactively or by the figure rules) without recomputing any hash.
    # The master hash (a digest of the whole config) is a rule param, so Snakemake's
    # `params` rerun-trigger regenerates the manifest whenever the config content
    # changes — without re-running on a no-op file touch.
    input:
        script="src/evalml/publication/manifest.py",
    output:
        OUT_ROOT / f"manifests/manifest_{CONFIG_SLUG}.json",
    localrule: True
    params:
        master_hash=master_hash(),
    run:
        from evalml.publication.manifest import build_manifest, write_manifest

        manifest = build_manifest(
            run_configs=RUN_CONFIGS,
            baseline_configs=BASELINE_CONFIGS,
            truth_cfg=config.get("truth"),
            truth_hash=TRUTH_HASH,
            verif_hash=VERIF_HASH,
            reftimes=REFTIMES,
            output_root=str(OUT_ROOT),
            publication_cfg=config.get("publication", {}),
            master_hash=params.master_hash,
            config_slug=CONFIG_SLUG,
        )
        write_manifest(output[0], manifest)
