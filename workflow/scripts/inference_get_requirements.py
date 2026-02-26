"""
Generates a pip-compatible ``requirements.txt`` file from an Anemoi checkpoint's
embedded provenance metadata (anemoi.json).

Usage
-----
    python inference_get_requirements.py path/to/metadata.json [--overrides pkg==1.2,git+https://...]
"""

import argparse
import json
import sys
import warnings
from packaging.version import Version, InvalidVersion


CORE_SUBMODULES = {"models", "training", "graphs"}


BASE_DEPENDENCIES = [
    "anemoi-inference",
    "eccodes==2.39.1",
    "eccodes-cosmo-resources-python",
]

# Packages emitted in the output even when only found in provenance (not overrides).
# pytorch-lightning and torch-geometric are included here because the PyPI filter
# (extract_pypi_requirements) matches them via the "torch" substring.
PIN_PACKAGES = [
    "anemoi-datasets",
    "anemoi-models",
    "anemoi-graphs",
    "torch",
    "pytorch-lightning",
    "torch-geometric",
]

# Canonical names of BASE_DEPENDENCIES for membership tests (strips version pins).
_BASE_DEPENDENCY_NAMES: set[str] = set()
for _dep in BASE_DEPENDENCIES:
    _base_name = _dep.split("==")[0].strip() if "==" in _dep else _dep.strip()
    _BASE_DEPENDENCY_NAMES.add(_base_name)


def load_provenance(metadata_path: str) -> dict:
    """Load and return ``provenance_training`` from a checkpoint metadata file.

    Exits with an error message if the key is missing rather than returning an
    empty dict and producing a silently incomplete requirements file.
    """
    with open(metadata_path, "r") as f:
        data = json.load(f)

    provenance = data.get("provenance_training")
    if provenance is None:
        print(
            f"ERROR: 'provenance_training' key not found in {metadata_path}. "
            "Check that the correct metadata file was supplied.",
            file=sys.stderr,
        )
        sys.exit(1)

    return provenance


def default_torch_index(torch_version: Version) -> str | None:
    """Return the PyTorch wheel index URL for a given torch version, or ``None``.

    Returns ``None`` (and emits a warning) for unrecognised versions instead of
    raising, so callers can degrade gracefully.

    Note: version tuples use ``(major, minor)`` so torch 2.10 is correctly
    distinct from 2.1 — ``Version("2.10").minor == 10``, not ``1``.
    """

    CUDA_BY_TORCH: dict[tuple[int, int], str] = {
        (2, 10): "cu126",
        (2, 9): "cu126",
        (2, 8): "cu126",
        (2, 7): "cu126",
        (2, 6): "cu124",
        (2, 5): "cu124",
        (2, 4): "cu121",
        (2, 3): "cu121",
    }

    key = (torch_version.major, torch_version.minor)
    cuda_tag = CUDA_BY_TORCH.get(key)

    if cuda_tag is None:
        warnings.warn(
            f"No CUDA version mapping found for torch {torch_version}; "
            "--index-url will be omitted from the output.",
            stacklevel=2,
        )
        return None

    return f"https://download.pytorch.org/whl/{cuda_tag}"


def extract_pypi_requirements(
    module_versions: dict[str, str],
    distribution_names: dict[str, str],
) -> dict[str, str]:
    """Extract pinned PyPI requirements from ``module_versions`` provenance.

    Version strings that cannot be parsed, or that do
    not start with a digit (e.g. editable installs), are skipped.

    ``Version.base_version`` is used deliberately to strip pre/post/dev/local
    segments, keeping pins clean for reproducibility.
    """
    requirements: dict[str, str] = {}

    for module, version_str in module_versions.items():
        if module.startswith("_"):
            continue
        if not version_str or not version_str[0].isdigit():
            continue

        try:
            version = Version(version_str)
        except InvalidVersion:
            continue

        name = module.replace(".", "-")
        name = distribution_names.get(name, name)
        name = name.replace("_", "-")

        requirements[name] = version.base_version

    return requirements


def extract_git_requirements(
    git_versions: dict[str, dict],
) -> dict[str, str]:
    """Extract VCS install URLs from ``git_versions`` provenance.

    Only ``anemoi.*`` modules are processed; all other entries are
    intentionally ignored.  Packages belonging to the ``anemoi-core``
    monorepo (``models``, ``training``, ``graphs``) are emitted as
    subdirectory installs; all others get a direct repo URL.
    """
    requirements: dict[str, str] = {}

    for module, info in git_versions.items():
        if not module.startswith("anemoi."):
            continue

        sha1 = info.get("git", {}).get("sha1")
        if not sha1:
            continue

        submodule = module.split(".")[-1]
        name = module.replace(".", "-")

        if submodule in CORE_SUBMODULES:
            url = (
                "git+https://github.com/ecmwf/anemoi-core"
                f"@{sha1}#subdirectory={submodule}"
            )
        else:
            url = f"git+https://github.com/ecmwf/anemoi-{submodule}@{sha1}"

        requirements[name] = url

    return requirements


def _parse_url_package_name(url: str) -> str:
    """Derive a package name from a VCS/HTTP URL.

    Handles the common patterns::

        git+https://github.com/org/anemoi-inference@sha1
        git+https://github.com/org/anemoi-core@sha1#subdirectory=models
        https://example.com/some-package.tar.gz

    For ``anemoi-core`` subdirectory URLs the returned name is
    ``anemoi-<subdir>`` (e.g. ``anemoi-models``).
    """
    # Handle anemoi-core monorepo subdirectory installs first.
    if "anemoi-core" in url and "#subdirectory=" in url:
        subdir = url.split("#subdirectory=")[-1].strip()
        return f"anemoi-{subdir}"

    # Take the last path segment, strip query/fragment and ref (@...).
    segment = url.rstrip("/").split("/")[-1]
    segment = segment.split("?")[0].split("#")[0]  # remove query / fragment
    segment = segment.split("@")[0]  # remove git ref
    if segment.endswith(".git"):
        segment = segment[:-4]

    return segment


def parse_overrides(overrides: list[str]) -> dict[str, str | None]:
    """Parse override tokens into a ``{name: value}`` mapping.

    ``BASE_DEPENDENCIES`` are merged in first so that explicit ``--overrides``
    entries always win.  Each token may be:

    - ``name==version``  →  pinned PyPI package
    - a URL (``git+``, ``http://``, ``https://``)  →  VCS / direct install
    - bare ``name``  →  unpinned PyPI package (value is ``None``)
    """
    result: dict[str, str | None] = {}

    for item in [*BASE_DEPENDENCIES, *(overrides or [])]:
        item = item.strip()
        if not item:
            continue

        if "==" in item:
            name, version = item.split("==", 1)
            result[name.strip()] = version.strip()
        elif any(item.startswith(prefix) for prefix in ("git+", "http://", "https://")):
            name = _parse_url_package_name(item)
            result[name] = item
        else:
            result[item] = None

    return result


def format_requirements(
    python_version: str | None,
    pypi_requirements: dict[str, str],
    git_requirements: dict[str, str],
    overrides: dict[str, str | None],
) -> str:
    """Render the final requirements file as a string.

    Works on *copies* of the input dicts so the caller's data is not mutated.
    Packages are only emitted if their name appears in ``PIN_PACKAGES``,
    ``_BASE_DEPENDENCY_NAMES``, or the explicit overrides set.
    """
    # Work on copies to avoid mutating the caller's dicts.
    pypi_requirements = dict(pypi_requirements)
    git_requirements = dict(git_requirements)

    lines: list[str] = []

    lines.append("# This file is automatically generated from a checkpoint.")
    if python_version:
        lines.append(f"# Python: {python_version}")

    # Detect torch version (if any) for the index URL.
    torch_version: Version | None = None
    if "torch" in pypi_requirements:
        try:
            torch_version = Version(pypi_requirements["torch"])
        except InvalidVersion:
            pass

    default_index = default_torch_index(torch_version) if torch_version else None

    lines.append("")
    lines.append("# Default index (derived from torch version)")
    if default_index:
        lines.append(f"--index-url {default_index}")
    else:
        lines.append("# (torch version unknown or unrecognised — index-url omitted)")

    lines.append("")
    lines.append("# Fallback index (PyPI)")
    lines.append("--extra-index-url https://pypi.org/simple")

    # Apply overrides: remove the name from whichever bucket holds it, then
    # re-insert into the appropriate bucket.
    for name, value in overrides.items():
        pypi_requirements.pop(name, None)
        git_requirements.pop(name, None)

        if isinstance(value, str) and value.startswith(("http://", "https://", "git+")):
            git_requirements[name] = value
        else:
            pypi_requirements[name] = value  # type: ignore[assignment]  # may be None

    # The allow-list for output: PIN_PACKAGES + base dependency names + explicit overrides.
    allowed: set[str] = {*PIN_PACKAGES, *_BASE_DEPENDENCY_NAMES, *overrides.keys()}

    # Git requirements
    if git_requirements:
        lines.append("")
        lines.append("# Git requirements:")
        lines.append("")

        for name, url in sorted(git_requirements.items()):
            if name not in allowed:
                continue
            # If provenance also recorded a PyPI version for this package, note it.
            version = pypi_requirements.pop(name, None)
            if version:
                lines.append(f"# {name}=={version}")
            extra = "  # Extra (not from checkpoint)" if name in overrides else ""
            lines.append(f"{url}{extra}")

    # Release (PyPI) requirements
    if pypi_requirements:
        lines.append("")
        lines.append("# Releases requirements:")
        lines.append("")

        for name, version in sorted(pypi_requirements.items()):
            if name not in allowed:
                continue
            line = f"{name}=={version}" if version else f"{name}"
            line += "  # Extra (not from checkpoint)" if name in overrides else ""
            lines.append(line)

    return "\n".join(lines)


def main(args: argparse.Namespace) -> None:
    md = load_provenance(args.metadata)

    distribution_names: dict[str, str] = md.get("distribution_names", {})

    pypi_requirements = extract_pypi_requirements(
        md.get("module_versions", {}),
        distribution_names,
    )

    git_requirements = extract_git_requirements(
        md.get("git_versions", {}),
    )

    overrides = parse_overrides(args.overrides or [])

    output = format_requirements(
        python_version=md.get("python"),
        pypi_requirements=pypi_requirements,
        git_requirements=git_requirements,
        overrides=overrides,
    )

    print(output)


def _parse_overrides_arg(overrides: str) -> list[str]:
    """Split a comma-separated overrides string into a list of tokens."""
    if not overrides:
        return []
    return [item.strip() for item in overrides.split(",") if item.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a pip requirements file from an Anemoi checkpoint."
    )
    parser.add_argument("metadata", help="Path to the metadata JSON file")
    parser.add_argument(
        "--overrides",
        type=_parse_overrides_arg,
        default=None,
        help=(
            "Comma-separated list of requirement overrides.  Each item may be "
            "'name==version', a VCS URL (git+https://...), or a bare package name."
        ),
    )
    args = parser.parse_args()
    main(args)
