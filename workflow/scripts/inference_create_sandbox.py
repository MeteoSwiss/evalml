import abc
import argparse
import logging
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List

import yaml
import jinja2


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
LOG = logging.getLogger(__name__)


class SandboxModule(abc.ABC):
    """Abstract base for sandbox components that prepare files for inclusion."""

    def __init__(self, work_dir: Path, strict: bool = False) -> None:
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.strict = strict

    @abc.abstractmethod
    def prepare(self) -> Dict[str, Path]:
        """
        Prepare and return a mapping of archive names to file paths.
        """
        ...


class ConfigModule(SandboxModule):
    """Loads anemoi-specific settings from pyproject.toml and writes a YAML config."""

    def __init__(self, checkpoint: Path, work_dir: Path, strict: bool) -> None:
        self.checkpoint = checkpoint
        super().__init__(work_dir / "config", strict)

    def prepare(self) -> Dict[str, Path]:
        config_data = {
            "checkpoint": (
                str(self.checkpoint) if self.strict else self.checkpoint.name
            ),
            "input": {"test": {"use_original_paths": True}},
            "allow_nans": False,
            "output": "printer",
        }
        out_file = self.work_dir / "config.yaml"
        with out_file.open("w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        # Include the checkpoint file if strict mode
        files = {str(Path("config/config.yaml")): out_file}
        if self.strict:
            files[self.checkpoint.name] = self.checkpoint
        return files


class LibsModule(SandboxModule):
    """Clones anemoi repositories into the work directory."""

    def __init__(
        self,
        repos: Dict[str, str],
        work_dir: Path,
        strict: bool = False,
    ) -> None:
        self.repos = repos
        super().__init__(work_dir / "libs", strict)

    def prepare(self) -> Dict[str, Path]:
        files: Dict[str, Path] = {}
        for name, url in self.repos.items():
            target = self.work_dir / name
            LOG.info(f"Cloning {url} into {target}")
            subprocess.run(
                ["git", "clone", "--depth=1", url, str(target)],
                check=True,
            )
            for file_path in target.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.work_dir.parent)
                    files[str(arcname)] = file_path
        return files


class ReadmeModule(SandboxModule):
    """Renders a README.md using a Jinja2 template."""

    def __init__(
        self,
        template_path: Path,
        context: Dict,
        work_dir: Path,
        strict: bool = False,
    ) -> None:
        self.template_path = template_path
        self.context = context
        super().__init__(work_dir / "readme", strict)

    def prepare(self) -> Dict[str, Path]:
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_path.parent)
        )
        template = env.get_template(self.template_path.name)
        output = template.render(**self.context)

        readme_file = self.work_dir / "README.md"
        with readme_file.open("w+") as f:
            f.write(output)

        return {"README.md": readme_file}


class ArchiveBuilder:
    """Combines files into a ZIP archive and verifies required entries."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._zip = zipfile.ZipFile(
            output_path, mode="w", compression=zipfile.ZIP_DEFLATED
        )

    def add_files(self, files: Dict[str, Path]) -> None:
        for arcname, fpath in files.items():
            if not fpath.is_file():
                raise FileNotFoundError(f"{fpath} not found for {arcname}")
            self._zip.write(fpath, arcname)

    def close(self) -> None:
        self._zip.close()

    def verify(self, required: List[str]) -> None:
        with zipfile.ZipFile(self.output_path, "r") as zf:
            contents = set(zf.namelist())
            missing = set(required) - contents
            if missing:
                raise RuntimeError(
                    f"Missing files in archive: {', '.join(sorted(missing))}"
                )
            LOG.info(f"Archive verified. Contents: {contents}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--requirements", type=Path, required=True, help="Path to requirements.txt"
    )
    parser.add_argument(
        "--readme-template", type=Path, required=True, help="Path to README.j2 template"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output ZIP archive path"
    )
    parser.add_argument(
        "--squashfs", type=Path, help="Optional SquashFS image to include"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict mode: no references to external paths",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOG.info("Starting sandbox archive creation")

    repos = {
        "anemoi-inference": "https://github.com/ecmwf/anemoi-inference.git",
        "anemoi-utils": "https://github.com/ecmwf/anemoi-utils.git",
        "anemoi-transform": "https://github.com/ecmwf/anemoi-transform.git",
    }

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        # Collect files from initial user inputs
        files: Dict[str, Path] = {
            args.requirements.name: args.requirements,
        }

        # Prepare modules
        cfg_module = ConfigModule(args.checkpoint, work_dir, args.strict)
        files.update(cfg_module.prepare())

        # Optionally include squashfs
        if args.squashfs:
            squash = args.squashfs.resolve()
            if args.strict:
                files[squash.name] = squash

        libs_module = LibsModule(repos, work_dir, strict=False)
        files.update(libs_module.prepare())

        # README generation
        context = {
            "image": args.squashfs.name if args.squashfs else None,
            "checkpoint_path": (
                cfg_module.checkpoint.name
                if args.strict
                else str(cfg_module.checkpoint)
            ),
            "strict": args.strict,
        }
        readme_module = ReadmeModule(args.readme_template, context, work_dir)
        files.update(readme_module.prepare())

        # Build archive
        arch = ArchiveBuilder(args.output)
        arch.add_files(files)
        arch.close()

    # Verify essential files
    arch.verify([args.requirements.name, "config/config.yaml"])
    LOG.info("Sandbox archive created at %s", args.output)


if __name__ == "__main__":
    main()


"""
Example usage (replace RUN_ID if needed):

export RUN_ID=2f962c89ff644ca7940072fa9cd088ec
python workflow/scripts/inference_create_sandbox.py \
    --pyproject output/data/runs/${RUN_ID}/pyproject.toml \
    --lockfile output/data/runs/${RUN_ID}/uv.lock \
    --squashfs output/data/runs/${RUN_ID}/venv.squashfs \
    --readme-template resources/inference/sandbox/readme.md.jinja2 \
    --output sandbox.zip && mkdir -p _sandbox && yes | unzip sandbox.zip -d _sandbox/


mkdir -p _sandbox && yes | unzip output/data/runs/2f962c89ff644ca7940072fa9cd088ec/sandbox.zip -d _sandbox/
"""
