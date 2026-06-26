"""Standalone publication-figure tooling for evalml.

This package lets the publication figures be produced both through the Snakemake
workflow (thin wrapper rules) and interactively / outside Snakemake, sharing one
code path. The keystone is a *manifest* (``output/publication/manifest.json``)
that persists the otherwise in-memory mapping of runs/baselines to their
deterministic hashes and on-disk data paths, so no consumer has to recompute a
hash or hand-assemble a cryptic ``run_id``.
"""

from evalml.publication.manifest import (
    SCHEMA_VERSION,
    build_manifest,
    default_manifest_path,
    load_manifest,
    write_manifest,
)
from evalml.publication.resolver import Manifest, Participant, ResolutionError

__all__ = [
    "SCHEMA_VERSION",
    "build_manifest",
    "write_manifest",
    "load_manifest",
    "default_manifest_path",
    "Manifest",
    "Participant",
    "ResolutionError",
]
