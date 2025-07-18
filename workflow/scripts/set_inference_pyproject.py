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
import toml

from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

from anemoi.utils.mlflow.auth import TokenAuth
from anemoi.utils.mlflow.client import AnemoiMlflowClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_GIT_DEPENDENCIES_CONFIG = {
    "anemoi-models": {
        "meteoswiss": {
            "url": "https://github.com/MeteoSwiss/anemoi-core.git",
            "subdirectory": "models",
        },
        "ecmwf": {
            "url": "https://github.com/ecmwf/anemoi-core.git",
            "subdirectory": "models",
        },
    },
    "anemoi-datasets": {
        "ecmwf": {"url": "https://github.com/ecmwf/anemoi-datasets.git"}
    },
}


def get_mlflow_client_given_runid(
    mlflow_uri: str | list[str], run_id: str
) -> MlflowClient:
    """Get an MLflow client for a given run ID.

    Parameters:
        mlflow_uri (str | list[str]): One or more MLflow tracking URIs to search.
        run_id (str): The MLflow run ID to look up.

    Returns:
        MlflowClient: A client configured for the server where the run was found.

    Raises:
        ValueError: If the run ID is not found in any of the provided URIs.
    """
    uris = [mlflow_uri] if isinstance(mlflow_uri, str) else mlflow_uri
    for uri in uris:
        if "ecmwf.int" in uri:
            TokenAuth(url=uri).login()
            client = AnemoiMlflowClient(uri, authentication=True)
        else:
            client = MlflowClient(tracking_uri=uri)
        try:
            client.get_run(run_id)
            return client
        except RestException:
            continue
    raise ValueError(
        f"Run ID {run_id} not found in any of the provided MLflow URIs: {uris}"
    )


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


def get_version_and_commit_hash(
    client: MlflowClient, run_id: str, dependency: str
) -> tuple[str, str]:
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
    commit_hash_param = (
        f"metadata.provenance_training.git_versions.{dependency}.git.sha1"
    )
    version_param = f"metadata.provenance_training.module_versions.{dependency}.version"
    version_param_alt = f"metadata.provenance_training.module_versions.{dependency}"
    version = run.data.params.get(version_param) or run.data.params.get(
        version_param_alt
    )
    return version, run.data.params.get(commit_hash_param)


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
            logger.info(
                "Found %s commit in %s git repository: %s",
                dependency_type,
                org,
                commit_hash,
            )
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
        version, commit_hash = get_version_and_commit_hash(client, run_id, dep_type)
        versions[f"{dep_type}"] = (
            resolve_dependency_config(commit_hash, dep_type) if commit_hash else version
        )

    if not versions:
        raise ValueError("No valid dependencies found in MLflow run")

    return versions


def get_python_version(client: MlflowClient, run_id: str) -> str | None:
    """Extract the Python version used in the MLflow run.

    Args:
        client (MlflowClient): MLflow client instance
        run_id (str): ID of the MLflow run

    Returns:
        str: Python version string, e.g. "3.10.14"

    Raises:
        ValueError: If no valid python version is found
    """
    run = client.get_run(run_id)
    python_version = run.data.params.get("metadata.provenance_training.python")

    if not python_version:
        raise ValueError("No valid python version found in MLflow run")

    return python_version


def get_path_to_checkpoints(client: MlflowClient, run_id: str) -> str:
    """Get the path to the checkpoints directory from MLflow run.

    Args:
        client (MlflowClient): MLflow client instance
        run_id (str): ID of the MLflow run

    Returns:
        str: Path to the checkpoints directory, e.g. "/scratch/mch/user/outpu/checkpoint"

    Raises:
        ValueError: If no valid checkpoints path is found
    """
    run = client.get_run(run_id)
    path = run.data.params.get("config.hardware.paths.checkpoints")

    if not path:
        raise ValueError("No valid checkpoints path found in MLflow run")

    return path


def format_vcs_pep508(pkg: str, cfg: dict) -> str:
    """Format a Git dependency as a valid PEP 508 string."""
    git = cfg["git"]
    rev = cfg.get("rev", "main")
    subdir = cfg.get("subdirectory")
    url = f"git+{git}@{rev}"
    if subdir:
        url += f"#subdirectory={subdir}"
    return f"{pkg} @ {url}"


def version_to_pep440_range(version: str) -> str:
    """Convert a Python version like '3.10.6' into a PEP 440 range like '>=3.10,<3.11'."""
    parts = version.split(".")
    if len(parts) < 2:
        raise ValueError("Version must have at least major and minor parts")

    major, minor = int(parts[0]), int(parts[1])
    return f">={major}.{minor},<{major}.{minor + 1}"


def update_pyproject_toml(
    versions: dict,
    toml_path: Path,
    python_version: str,
    checkpoints_path: str,
    run_mlflow_link: str,
) -> None:
    """
    Update pyproject.toml [project.dependencies] with versions or Git references.

    Args:
        versions (dict): Mapping of dependency names to version strings or VCS configs
        toml_path (Path): Path to pyproject.toml
        python_version (str): Python version string, e.g. "3.10.14"
        checkpoints_path (str): Path to the checkpoints directory
    """
    if not toml_path.exists():
        raise FileNotFoundError(f"{toml_path} not found")

    try:
        with open(toml_path, "r", encoding="utf-8") as f:
            config = toml.load(f)
    except Exception as e:
        raise RuntimeError("Failed to read pyproject.toml") from e

    deps = config.get("project", {}).get("dependencies", [])
    if not isinstance(deps, list):
        raise ValueError("[project.dependencies] must be a list")

    updated = []
    for dep in deps:
        pkg_name = dep.split("==")[0].split(">=")[0].split("@")[0].split()[0].strip()
        if pkg_name in versions:
            val = versions[pkg_name]
            new_dep = (
                format_vcs_pep508(pkg_name, val)
                if isinstance(val, dict)
                else f"{pkg_name}=={val}"
            )
            updated.append(new_dep)
        else:
            updated.append(dep)

    config["project"]["dependencies"] = updated
    config["project"]["requires-python"] = version_to_pep440_range(python_version)
    config.setdefault("tool", {})["anemoi"] = {"checkpoints_path": checkpoints_path}
    config["tool"]["anemoi"]["run_mlflow_link"] = run_mlflow_link

    try:
        with open(toml_path, "w", encoding="utf-8") as f:
            toml.dump(config, f)
    except Exception as e:
        raise RuntimeError("Failed to write pyproject.toml") from e


def main(snakemake) -> None:
    """Main entry point for the script.

    Raises:
        Exception: If any step fails, original files are restored
    """
    mlflow_uri = snakemake.config["locations"]["mlflow_uri"]
    run_id = snakemake.wildcards["run_id"]
    requirements_path_in = Path(snakemake.input[0])
    toml_path_out = Path(snakemake.output[0])

    shutil.copy2(requirements_path_in, toml_path_out)

    client = get_mlflow_client_given_runid(mlflow_uri, run_id)
    anemoi_versions = get_anemoi_versions(client, run_id)
    python_version = get_python_version(client, run_id)
    checkpoints_path = get_path_to_checkpoints(client, run_id)

    run_mlflow_link = client.tracking_uri + "/#/runs/" + run_id
    update_pyproject_toml(
        anemoi_versions,
        toml_path_out,
        python_version,
        checkpoints_path,
        run_mlflow_link,
    )

    logger.info("Successfully updated dependencies in %s", toml_path_out)


if __name__ == "__main__":
    snakemake = snakemake  # type: ignore # noqa: F821

    raise SystemExit(main(snakemake))
