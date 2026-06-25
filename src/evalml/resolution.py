"""Pure, importable resolution helpers shared by the Snakemake workflow and the
standalone publication tooling.

These functions used to live in ``workflow/rules/common.smk`` where they were only
reachable from inside the Snakemake process. They are pure (no Snakemake globals,
no I/O) so that the publication resolver/CLI can reuse the *exact same* logic for
lead-time producibility and baseline label resolution instead of duplicating it.
``common.smk`` now imports thin wrappers around them.
"""

import logging
from typing import Iterable

# Period-accumulated params verify a [lead - period, lead] window, so they have
# no value at lead times shorter than one step spacing (e.g. no 0h precip map).
# Short and canonical names both appear across the workflow (showcases vs maps).
ACCUMULATED_PARAMS = {"TOT_PREC", "tp"}


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
    if param in ACCUMULATED_PARAMS:
        valid = {lt for lt in valid if lt >= step}
    return sorted(valid)


def resolve_baseline_id(label: str, baseline_configs: dict) -> str:
    """Resolve a baseline label to its hash-based ID.

    Scorecard / publication configs reference baselines by human-readable label
    (e.g. 'IFS'). This finds the matching baseline_id in ``baseline_configs``
    (a mapping of baseline_id -> config dict). Raises ValueError with the list
    of available labels if the label doesn't match any registered baseline.
    """
    for baseline_id, cfg in baseline_configs.items():
        if cfg.get("label") == label:
            return baseline_id
    available = [cfg.get("label") for cfg in baseline_configs.values()]
    raise ValueError(
        f"No baseline with label {label!r} found. "
        f"Available baseline labels: {available}"
    )


def leadtime_producible(steps_spec: str, leadtime: int, param: str | None = None) -> bool:
    """Whether a single ``leadtime`` (hours) is produced by ``steps_spec``.

    Thin convenience wrapper over :func:`resolve_leadtimes` for the common
    single-lead-time coherence check (publication scoremaps).
    """
    return leadtime in resolve_leadtimes(steps_spec, [leadtime], param=param)


def steps_to_leadtimes(steps_spec: str) -> list[int]:
    """All lead times (hours) produced by a ``start/stop/step`` spec."""
    start, end, step = map(int, steps_spec.split("/"))
    return list(range(start, end + 1, step))


__all__ = [
    "ACCUMULATED_PARAMS",
    "resolve_leadtimes",
    "resolve_baseline_id",
    "leadtime_producible",
    "steps_to_leadtimes",
]


def _coerce_int_list(values: Iterable) -> list[int]:
    """Best-effort coercion of an iterable of lead-time values to ints."""
    return [int(v) for v in values]
