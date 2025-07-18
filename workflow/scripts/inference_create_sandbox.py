import argparse
import logging
from pathlib import Path
import sys
import zipfile

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def program_summary_log(args):
    LOG.info("=" * 80)
    LOG.info("Creating sandbox zip file")
    LOG.info(f"Pyproject file: {args.pyproject}")
    LOG.info(f"Lockfile: {args.lockfile}")
    if args.squashfs:
        LOG.info(f"Squashfs image: {args.squashfs}")
    LOG.info(f"Output zip file: {args.output}")
    LOG.info("=" * 80)


def main(args: argparse.Namespace) -> None:
    program_summary_log(args)

    files_to_add = {
        args.pyproject.name: args.pyproject,
        args.lockfile.name: args.lockfile,
    }

    if args.squashfs:
        files_to_add[args.squashfs.name] = args.squashfs

    # Validate input files exist
    for label, path in files_to_add.items():
        if not path.is_file():
            sys.exit(f"Error: {label} does not exist or is not a file: {path}")

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write zip file
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zipf:
        for arcname, filepath in files_to_add.items():
            zipf.write(filepath, arcname=arcname)

    LOG.info("Created zip file: %s", args.output)

    with zipfile.ZipFile(args.output, "r") as zipf:
        zip_contents = zipf.namelist()
        for label, path in files_to_add.items():
            if label not in zip_contents:
                sys.exit(f"Error: {label} not found in the zip file: {args.output}")
        LOG.info("Contents of the created zip file: \n%s", "\n".join(zip_contents))

    LOG.info("Program completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Package files into a zipped directory."
    )

    parser.add_argument(
        "--pyproject", required=True, type=Path, help="Path to pyproject.toml"
    )
    parser.add_argument("--lockfile", required=True, type=Path, help="Path to lockfile")
    parser.add_argument("--squashfs", type=Path, help="Optional squashfs image")
    parser.add_argument(
        "--output", required=True, type=Path, help="Output zip file path"
    )

    args = parser.parse_args()

    main(args)


"""
Example usage:
python workflow/scripts/inference_create_sandbox.py \
    --pyproject output/data/runs/2f962c89ff644ca7940072fa9cd088ec/pyproject.toml \
    --lockfile output/data/runs/2f962c89ff644ca7940072fa9cd088ec/uv.lock \
    --output sandbox.zip
"""
# --squashfs /path/to/squashfs.img \
