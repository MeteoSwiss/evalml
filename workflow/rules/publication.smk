# ----------------------------------------------------- #
# Publication-grade figures workflow                    #
# ----------------------------------------------------- #
# Snakemake produces the *results* and a *manifest* (run/baseline -> hash ->
# data-path mapping) and stops. All publication-ready plotting happens in the
# standalone notebooks under `notebooks/publication/`, driven only by the
# manifest — see docs/publication_figures.md.

from evalml.publication.manifest import truth_slug

# Publication outputs are namespaced by the truth label so station-based and
# analysis-based runs don't overwrite each other's manifest.
TRUTH_SLUG = truth_slug((config.get("truth") or {}).get("label", ""))


rule publication_manifest:
    input:
        script="src/evalml/publication/manifest.py",
    """Persist the run/baseline -> hash -> data-path mapping for the notebooks.

Cheap localrule: dumps the in-memory workflow globals to JSON, so paths can be
resolved (interactively or by the notebooks) without recomputing any hash.
The master hash (a digest of the whole config) is a rule param, so Snakemake's
`params` rerun-trigger regenerates the manifest whenever the config content
changes — without re-running on a no-op file touch.
"""
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


# --- Result dependencies for `publication_all` -------------------------------
# The scoremap NC files must be listed from in-memory globals (the manifest file
# a sibling rule produces cannot be read at DAG-build time). In the paper configs
# `experiment.scoremaps.enabled` is false while `publication.scoremaps.enabled`
# is true, so the publication target is the only thing pulling scoremap
# production — these helpers keep that dependency alive.


def _pub_candidate_run_id():
    """The single publication candidate run_id (raises if not exactly one)."""
    candidates = [rid for rid, cfg in RUN_CONFIGS.items() if cfg.get("_is_candidate")]
    if len(candidates) != 1:
        raise ValueError(
            f"The publication workflow expects exactly one candidate run; "
            f"found {len(candidates)}. Pick a single candidate in the config."
        )
    return candidates[0]


def _pub_scoremap_cfg():
    return (config.get("publication", {}) or {}).get("scoremaps") or {}


def _pub_scoremap_leadtimes(cfg):
    """Lead times (hours) to plot: publication.scoremaps.steps when set,
    otherwise falls back to experiment.scoremaps.leadtimes.
    """
    steps = cfg.get("steps")
    if steps is not None:
        return [int(s) for s in steps]
    return list(
        config.get("experiment", {}).get("scoremaps", {}).get("leadtimes", [6, 24])
    )


def _pub_scoremap_files():
    """Scoremap NC files (candidate + baseline) required by publication_all.

    Ordered leadtime-major (all params for leadtimes[0], then leadtimes[1], …),
    using the same path template the manifest/notebook resolves, so the declared
    dependency always matches what the scoremaps notebook reads.
    """
    cfg = _pub_scoremap_cfg()
    params = cfg.get("params", ["T_2M", "SP_10M"])
    leadtimes = _pub_scoremap_leadtimes(cfg)
    cand_id = _pub_candidate_run_id()
    base_id = resolve_baseline_id(cfg.get("baseline_label", "ICON-CH1-CTRL"))
    cand = [
        str(OUT_ROOT / f"data/runs/{cand_id}/scoremaps/{p}_{lt}_{TRUTH_HASH}.nc")
        for lt in leadtimes
        for p in params
    ]
    base = [
        str(OUT_ROOT / f"data/baselines/{base_id}/scoremaps/{p}_{lt}_{TRUTH_HASH}.nc")
        for lt in leadtimes
        for p in params
    ]
    return cand + base
