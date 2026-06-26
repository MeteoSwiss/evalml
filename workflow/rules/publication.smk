# ----------------------------------------------------- #
# Publication-grade figures workflow                    #
# ----------------------------------------------------- #
# The figures are rendered by the standalone `evalml.publication` CLI, which
# reads a manifest (run/baseline -> hash -> data-path mapping) instead of
# hand-assembled identifiers. The same CLI drives interactive use, so these
# rules are thin wrappers: produce the manifest + the data, then call the CLI.


rule publication_manifest:
    """Persist the run/baseline -> hash -> data-path mapping for the CLI/notebooks.

Cheap localrule: dumps the in-memory workflow globals to JSON, so paths can be
resolved (interactively or by the figure rules) without recomputing any hash.
The master hash (a digest of the whole config) is a rule param, so Snakemake's
`params` rerun-trigger regenerates the manifest whenever the config content
changes — without re-running on a no-op file touch.
"""
    output:
        OUT_ROOT / "publication/manifest.json",
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
            reftimes=REFTIMES,
            output_root=str(OUT_ROOT),
            publication_cfg=config.get("publication", {}),
            master_hash=params.master_hash,
        )
        write_manifest(output[0], manifest)


rule publication_figures:
    input:
        "workflow/scripts/publication_style.py",
        "workflow/scripts/publication.mplstyle",
        "workflow/scripts/publication_figures.py",
        manifest=rules.publication_manifest.output,
        verif=EXPERIMENT_PARTICIPANTS.values(),
    output:
        report(
            directory(OUT_ROOT / "figures/leadtime"),
            htmlindex="publication_figures.html",
        ),
    log:
        OUT_ROOT / "logs/figures/publication_figures.log",
    localrule: True
    shell:
        """
        python -m evalml.publication figures \
            --manifest {input.manifest} \
            --output {output} >{log} 2>&1
        """


def _pub_candidate_run_id():
    """The single publication candidate run_id (raises if not exactly one)."""
    candidates = [rid for rid, cfg in RUN_CONFIGS.items() if cfg.get("_is_candidate")]
    if len(candidates) != 1:
        raise ValueError(
            f"The publication workflow expects exactly one candidate run; "
            f"found {len(candidates)}. Pick a single candidate in the config."
        )
    return candidates[0]


def _meteogram_data_dep(wc):
    """Aggregated verification file of the candidate run (DAG dependency).

    Depending on this regular file (rather than the candidate's GRIB *directory*,
    which two inference_prepare rules ambiguously declare) guarantees inference has
    run for every reftime — including the meteogram's init_time — so the GRIB the
    CLI resolves from the manifest is present, with correct ordering.
    """
    run_id = _pub_candidate_run_id()
    return str(OUT_ROOT / f"data/runs/{run_id}/verif_aggregated_{TRUTH_HASH}.nc")


rule publication_meteogram:
    input:
        "workflow/scripts/publication_style.py",
        "workflow/scripts/publication.mplstyle",
        "workflow/scripts/publication_meteogram.py",
        manifest=rules.publication_manifest.output,
        verif=_meteogram_data_dep,
        eckit_grids=rules.data_download_eckit_geo_grids.output,
    output:
        report(
            directory(OUT_ROOT / "figures/meteogram"),
            htmlindex="publication_meteogram.html",
        ),
    log:
        OUT_ROOT / "logs/figures/publication_meteogram.log",
    resources:
        slurm_partition="postproc",
        cpus_per_task=8,
        mem_mb=32000,
        runtime="60m",
    shell:
        """
        set -euo pipefail
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        python -m evalml.publication meteogram \
            --manifest {input.manifest} \
            --output {output} >{log} 2>&1
        """


def _pub_scoremap_cfg():
    return (config.get("publication", {}) or {}).get("scoremaps") or {}


def _pub_scoremap_leadtimes(cfg):
    """Lead times to plot: the `leadtimes` list, or the singular `leadtime`."""
    if cfg.get("leadtimes"):
        return list(cfg["leadtimes"])
    if cfg.get("leadtime") is not None:
        return [cfg["leadtime"]]
    return [24]


def _pub_scoremap_inputs(wc):
    """Scoremap NC files for the candidate and baseline (DAG dependency).

    Computed from the in-memory globals (not the manifest file, which a sibling
    rule produces) using the same path template the CLI resolves from the manifest,
    so the declared inputs always match what the CLI plots. Files are ordered
    leadtime-major (all params for leadtimes[0], then leadtimes[1], …) to match
    how the scoremaps script slices them.
    """
    cfg = _pub_scoremap_cfg()
    params = cfg.get("params", ["T_2M", "SP_10M"])
    leadtimes = _pub_scoremap_leadtimes(cfg)
    cand_id = _pub_candidate_run_id()
    base_id = resolve_baseline_id(cfg.get("baseline_label", "ICON-CH1-CTRL"))
    return {
        "cand_files": [
            str(OUT_ROOT / f"data/runs/{cand_id}/scoremaps/{p}_{lt}_{TRUTH_HASH}.nc")
            for lt in leadtimes
            for p in params
        ],
        "base_files": [
            str(
                OUT_ROOT
                / f"data/baselines/{base_id}/scoremaps/{p}_{lt}_{TRUTH_HASH}.nc"
            )
            for lt in leadtimes
            for p in params
        ],
    }


rule publication_scoremaps:
    input:
        unpack(_pub_scoremap_inputs),
        "workflow/scripts/publication_scoremaps.py",
        "workflow/scripts/publication_style.py",
        "workflow/scripts/publication.mplstyle",
        manifest=rules.publication_manifest.output,
    output:
        report(
            directory(OUT_ROOT / "figures/scoremaps"),
            htmlindex="publication_scoremaps.html",
        ),
    log:
        OUT_ROOT / "logs/figures/publication_scoremaps.log",
    localrule: True
    shell:
        """
        python -m evalml.publication scoremaps \
            --manifest {input.manifest} \
            --output {output} >{log} 2>&1
        """
