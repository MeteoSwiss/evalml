"""Lead times each participant (run or baseline) produces, per parameter."""

import logging
import re

PARAMS_WITHOUT_STEP_ZERO_VALUE = {
    "TOT_PREC",
    "tp",
    "CLCT",
    "tcc",
    "CLCL",
    "lcc",
    "CLCM",
    "mcc",
    "CLCH",
    "hcc",
    "SSRD",
    "ssrd",
}


def resolve_leadtimes(steps_spec, requested="all", param=None):
    """Lead times to compute for a single participant.

    A run or baseline produces only the lead times in its own ``steps`` spec
    (``start/stop/step``, hours). This returns those of the ``requested``
    selection that the participant actually produces — the literal ``"all"``
    (every produced lead time) or an explicit list of ints — so a 36h lead is
    never requested of an ICON-CH1 baseline (steps ``0/33/6``), nor a >120h
    lead of ICON-CH2. Explicitly requested lead times the participant cannot
    produce are skipped with a warning. For accumulated ``param``s, lead times
    shorter than one step spacing are dropped (no accumulation window).
    """
    start, end, step = map(int, steps_spec.split("/"))
    supported = set(range(start, end + 1, step))
    wanted = supported if requested == "all" else set(requested)

    unsupported = sorted(wanted - supported)
    if unsupported:
        logging.getLogger("snakemake").warning(
            "Skipping lead time(s) %sh: not produced by forecast steps '%s'.",
            unsupported,
            steps_spec,
        )

    valid = wanted & supported
    # Showcase-facing names encode an accumulation window as a numeric suffix
    # (e.g. "TOT_PREC1", "TOT_PREC6"); strip it before checking membership so
    # those aggregated variants are recognized the same as their base param.
    base_param = re.sub(r"\d+$", "", param) if param else param
    if (
        param in PARAMS_WITHOUT_STEP_ZERO_VALUE
        or base_param in PARAMS_WITHOUT_STEP_ZERO_VALUE
    ):
        # For aggregated params (e.g. TOT_PREC6), the accumulation window encoded
        # in the suffix is the minimum valid lead time — not the run's step spacing.
        # At shorter leads the de-accumulation window doesn't fit (prev_lt < 0),
        # so the frame would fall back to showing a raw cumulative total.
        agg_match = re.search(r"\d+$", param) if param and base_param != param else None
        min_lt = int(agg_match.group()) if agg_match else step
        valid = {lt for lt in valid if lt >= min_lt}
    return sorted(valid)
