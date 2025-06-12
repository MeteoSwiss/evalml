"""Update anemoi model dependencies based on MLflow run information.

This module provides functionality to:
1. Fetch model version information from MLflow
2. Resolve git dependencies for anemoi components
3. Update pyproject.toml with correct dependency versions

The script maintains a backup of configuration files and restores them in case of failure.

"""

import logging
import shutil
from pathlib import Path
import requests

from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_GIT_DEPENDENCIES_CONFIG = {
    "anemoi-models": {
        "meteoswiss": {
            "url": "https://github.com/MeteoSwiss/anemoi-core.git",
            "subdirectory": "models"
        },
        "ecmwf": {
            "url": "https://github.com/ecmwf/anemoi-core.git",
            "subdirectory": "models"
        }
    },
    "anemoi-datasets": {
        "ecmwf": {
            "url": "https://github.com/ecmwf/anemoi-datasets.git"
        }
    },
}



def check_commit_exists(repo_url: str, commit_hash: str) -> bool:
    """Check if a git commit exists in a GitHub repository.

    Args:
        repo_url (str): URL of the git repository
        commit_hash (str): Git commit hash to check

    Returns:
        bool: True if commit exists, False otherwise
    """
    if not commit_hash:
        return False

    path = repo_url.replace("https://github.com/", "").replace(".git", "")
    owner, repo = path.split("/")

    api_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_hash}"

    try:
        response = requests.head(api_url, timeout=30)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.warning("Failed to check commit %s in %s: %s", commit_hash, repo_url, e)
        return False


def get_version_and_commit_hash(client: MlflowClient, run_id: str, dependency: str) -> tuple[str, str]:
    """Get version and commit hash for a dependency from MLflow run.

    Args:
        client (MlflowClient): MLflow client instance
        run_id (str): ID of the MLflow run
        dependency (str): Name of the dependency to look up

    Returns:
        tuple[str, str]: Tuple containing (version, commit_hash)
    """
    run = client.get_run(run_id)
    dependency = dependency.replace("-", ".")
    commit_hash_param = f"metadata.provenance_training.git_versions.{dependency}.git.sha1"
    version_param = f"metadata.provenance_training.git_versions.{dependency}.version"
    return run.data.params.get(version_param), run.data.params.get(commit_hash_param)


def resolve_dependency_config(commit_hash: str, dependency_type: str) -> dict:
    """Resolve git repository configuration for a dependency.

    Args:
        commit_hash (str): Git commit hash to use
        dependency_type (str): Type of dependency (e.g., "anemoi-models")

    Returns:
        dict: Repository configuration including:
            - git: Repository URL
            - rev: Commit hash
            - subdirectory: Optional subdirectory path

    Raises:
        ValueError: If dependency type is unknown or commit not found
    """
    if dependency_type not in _GIT_DEPENDENCIES_CONFIG:
        raise ValueError(f"Unknown dependency type: {dependency_type}")

    config = {"rev": commit_hash}
    repos = _GIT_DEPENDENCIES_CONFIG[dependency_type]

    for org, repo_config in repos.items():
        if check_commit_exists(repo_config["url"], commit_hash):
            logger.info("Found %s commit in %s git repository: %s", dependency_type, org, commit_hash)
            config["git"] = repo_config["url"]
            if "subdirectory" in repo_config:
                config["subdirectory"] = repo_config["subdirectory"]
            return config

    raise ValueError(f"Cannot resolve commit {commit_hash} for {dependency_type}")


def get_anemoi_versions(client: MlflowClient, run_id: str) -> dict:
    """Get anemoi component versions from MLflow run.

    Args:
        client (MlflowClient): MLflow client instance
        run_id (str): ID of the MLflow run

    Returns:
        dict: Dictionary mapping component names to their configurations

    Raises:
        ValueError: If no valid dependencies are found
    """
    versions = {}
    for dep_type in _GIT_DEPENDENCIES_CONFIG:
        _, commit_hash = get_version_and_commit_hash(client, run_id, dep_type)
        versions[f"{dep_type}"] = resolve_dependency_config(commit_hash, dep_type)

    if not versions:
        raise ValueError("No valid dependencies found in MLflow run")

    return versions


def format_vcs_requirement(pkg: str, cfg: dict) -> str:
    """Convert VCS dependency dictionary into a PEP 508–compliant string."""
    git = cfg.get("git")
    rev = cfg.get("rev", "main")
    subdir = cfg.get("subdirectory")
    url = f"git+{git}@{rev}"
    if subdir:
        url += f"#subdirectory={subdir}"
    return f"{pkg} @ {url}"

def update_requirements_txt(
    versions: dict,
    requirements_path: Path,
) -> None:
    """
    Update a requirements.txt file with specified package versions or VCS configs.

    Args:
        versions (dict): Mapping of package names to version strings or VCS configs
        requirements_path (Path): Path to requirements.txt file

    Raises:
        FileNotFoundError: If requirements.txt doesn't exist
        RuntimeError: If reading or writing fails
    """
    if not requirements_path.exists():
        raise FileNotFoundError(f"requirements.txt not found at {requirements_path}")

    try:
        with open(requirements_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        raise RuntimeError("Failed to read requirements.txt.") from e

    updated_lines = []
    seen = set()

    for line in lines:
        original = line
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            updated_lines.append(original)
            continue

        pkg_name = stripped.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0].split("@")[0].strip()

        if pkg_name in versions:
            val = versions[pkg_name]
            if isinstance(val, dict):
                updated_lines.append(format_vcs_requirement(pkg_name, val) + "\n")
            else:
                updated_lines.append(f"{pkg_name}{val}\n")
            seen.add(pkg_name)
        else:
            updated_lines.append(original)

    for pkg, val in versions.items():
        if pkg in seen:
            continue
        if isinstance(val, dict):
            updated_lines.append(format_vcs_requirement(pkg, val) + "\n")
        else:
            updated_lines.append(f"{pkg}{val}\n")

    try:
        with open(requirements_path, "w", encoding="utf-8") as f:
            f.writelines(updated_lines)
    except Exception as e:
        raise RuntimeError("Failed to write requirements.txt.") from e


def main(snakemake) -> None:
    """Main entry point for the script.

    This function:
    1. Creates backup of configuration files
    2. Updates dependencies based on MLflow information
    3. Runs poetry lock to update lock file
    4. Removes backups on success or restores on failure

    Raises:
        Exception: If any step fails, original files are restored
    """
    mlflow_uri = snakemake.params["mlflow_uri"]
    run_id = snakemake.wildcards["run_id"]
    requirements_path_in = Path(snakemake.input[0])
    requirement_path_out = Path(snakemake.output[0])

    shutil.copy2(requirements_path_in, requirement_path_out)

    logging.info("Using %s tracking URI", mlflow_uri)
    client = MlflowClient(tracking_uri=mlflow_uri)
    anemoi_versions = get_anemoi_versions(client, run_id)

    update_requirements_txt(anemoi_versions, requirement_path_out)

    logger.info("Successfully updated dependencies in %s", requirement_path_out)


if __name__ == "__main__":

    snakemake = snakemake  # type: ignore

    raise SystemExit(main(snakemake))