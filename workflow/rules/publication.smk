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
    The active config file is an explicit input so the manifest is regenerated
    whenever the config changes.
    """
    input:
        workflow.configfiles[0] if workflow.configfiles else [],
    output:
        OUT_ROOT / "publication/manifest.json",
    localrule: True
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
            master_hash=master_hash(),
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


def _meteogram_candidate_grib(wc):
    """GRIB dir of the candidate run for the configured init_time (DAG dependency)."""
    run_id = _pub_candidate_run_id()
    init_time = config["publication"]["meteogram"]["init_time"]
    return str((Path(OUT_ROOT) / f"data/runs/{run_id}/{init_time}/grib").resolve())


rule publication_meteogram:
    input:
        "workflow/scripts/publication_style.py",
        "workflow/scripts/publication.mplstyle",
        "workflow/scripts/publication_meteogram.py",
        manifest=rules.publication_manifest.output,
        grib=_meteogram_candidate_grib,
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


def _pub_scoremap_inputs(wc):
    """Scoremap NC files for the candidate and baseline (DAG dependency).

    Computed from the in-memory globals (not the manifest file, which a sibling
    rule produces) using the same path template the CLI resolves from the manifest,
    so the declared inputs always match what the CLI plots.
    """
    cfg = _pub_scoremap_cfg()
    params = cfg.get("params", ["T_2M", "SP_10M"])
    leadtime = cfg.get("leadtime", 24)
    baseline_id = resolve_baseline_id(cfg.get("baseline_label", "ICON-CH1-CTRL"))
    return {
        "cand_files": expand(
            str(
                OUT_ROOT
                / f"data/runs/{{run_id}}/scoremaps/{{param}}_{{leadtime}}_{TRUTH_HASH}.nc"
            ),
            run_id=_pub_candidate_run_id(),
            param=params,
            leadtime=leadtime,
        ),
        "base_files": expand(
            str(
                OUT_ROOT
                / f"data/baselines/{{baseline_id}}/scoremaps/{{param}}_{{leadtime}}_{TRUTH_HASH}.nc"
            ),
            baseline_id=baseline_id,
            param=params,
            leadtime=leadtime,
        ),
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
