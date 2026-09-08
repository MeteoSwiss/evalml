# ----------------------------------------------------- #
# GRIB STAGING                                          #
# ----------------------------------------------------- #
# Symlinks a run's pre-generated GRIB into the standard per-run workdir layout
# for the case where evalml itself doesn't run inference to produce it:
# GRIB-model runs (e.g. spatial_downscaler, which supply GRIB from a workflow
# outside evalml) here, and fixture replay in inference.smk. Also owns the
# "is this run's GRIB ready" signal (`grib_okfile`) that plot.smk /
# verification.smk / verif_obs.smk consume — the same signal regardless of
# whether the GRIB came from anemoi-inference or an external staging symlink.

from pathlib import Path


# Shared by any rule that stages a run's GRIB into the standard per-run workdir
# layout via symlink (fixture replay, GRIB-model runs): mkdir the workdir,
# replace a stale staged-GRIB symlink but refuse to touch a real grib
# directory (that would be Snakemake-owned inference output, and a bare
# `ln -sfn` would otherwise nest the link inside it), then symlink and mark done.
_STAGE_GRIB_SYMLINK_SHELL = """
(
    set -euo pipefail
    mkdir -p {params.workdir}
    if [ -L {params.workdir}/grib ]; then
        rm -f {params.workdir}/grib
    elif [ -e {params.workdir}/grib ]; then
        echo "ERROR: {params.workdir}/grib is a real directory (Snakemake-owned inference output), not a previously-staged GRIB symlink. Refusing to delete it; move it aside and retry." >&2
        exit 1
    fi
    ln -sfn {params.source} {params.workdir}/grib
) >{log} 2>&1
touch {output.okfile}
"""


def _grib_model_source(wc):
    rc = RUN_CONFIGS[wc.run_id]
    return str((Path(rc["root"]) / wc.init_time / "grib").resolve())


rule grib_model_stage:
    """Symlink a GRIB-model run's (e.g. spatial_downscaler) pre-generated GRIB,
    produced entirely outside evalml, into the standard per-run workdir layout."""
    output:
        okfile=OUT_ROOT / "logs/grib_model_stage/{run_id}-{init_time}.ok",
    log:
        OUT_ROOT / "logs/grib_model_stage/{run_id}-{init_time}.log",
    localrule: True
    params:
        source=_grib_model_source,
        workdir=lambda wc: (OUT_ROOT / f"data/runs/{wc.run_id}/{wc.init_time}").resolve(),
    shell:
        _STAGE_GRIB_SYMLINK_SHELL


def _rule_for_model_type(model_type: str):
    """The rule that produces a run's okfile: grib_model_stage for GRIB-model
    run types, inference_execute for everything else."""
    return rules.grib_model_stage if model_type in GRIB_MODEL_TYPES else rules.inference_execute


def _grib_okfile_template(wc):
    """Unexpanded rule-output object for wc.run_id's okfile rule."""
    model_type = RUN_CONFIGS[wc.run_id]["model_type"]
    return _rule_for_model_type(model_type).output.okfile


def _grib_okfile_for(run_id: str, init_time: str) -> str:
    model_type = RUN_CONFIGS[run_id]["model_type"]
    template = _rule_for_model_type(model_type).output.okfile
    return template.format(run_id=run_id, init_time=init_time)


def grib_okfile(wc):
    """Path to wc.run_id's "GRIB is ready" marker, regardless of whether it was
    produced by evalml running anemoi-inference or staged from an external
    GRIB-model run. Input-function for `input: grib_okfile=grib_okfile`."""
    return _grib_okfile_for(wc.run_id, wc.init_time)
