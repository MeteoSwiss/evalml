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

from evalml.publication.manifest import truth_slug

# The manifest is namespaced by the truth label so station-based and
# analysis-based runs don't overwrite each other's manifest.
TRUTH_SLUG = truth_slug((config.get("truth") or {}).get("label", ""))


rule publication_manifest:
    """Persist the run/baseline -> hash -> data-path mapping for the notebooks.

    Cheap localrule: dumps the in-memory workflow globals to JSON, so paths can be
    resolved (interactively or by the notebooks) without recomputing any hash. It
    does not depend on the heavy data — the notebooks open whatever result files
    exist at the recorded paths. The master hash (a digest of the whole config) is
    a rule param, so Snakemake's `params` rerun-trigger regenerates the manifest
    whenever the config content changes — without re-running on a no-op file touch.
    """
    input:
        script="src/evalml/publication/manifest.py",
    output:
        OUT_ROOT / f"publication/{TRUTH_SLUG}/manifest.json",
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
        )
        write_manifest(output[0], manifest)
