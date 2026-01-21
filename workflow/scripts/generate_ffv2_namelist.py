import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import jinja2

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def _check_args(args):
	print('ids are ' + args.experiment_ids)
	ids = args.experiment_ids.split(',')
	print('fdbk dirs are ' + args.feedback_directories)
	fdbk_dirs = args.feedback_directories.split(',')
	# Allow list to end with comma
	if ids[-1] == '':
		ids.pop(-1)
	if fdbk_dirs[-1] == '':
		fdbk_dirs.pop(-1)
	if len(ids) != len(fdbk_dirs):
		raise ValueError(
            'lengths of experiment IDs and feedback directories differ:'
            f'{len(ids)} {args.experiment_ids} vs '
            f'{len(fdbk_dirs)} {args.feedback_directories}'
		)
	if not os.path.exists(args.output_directory) or not os.path.isdir(args.output_directory):
		raise FileNotFoundError(f'output directory {args.output_directory} '
						  'does not exist (or is not a directory)')
	if not os.path.exists(args.domain_table):
		raise FileNotFoundError(f'domain table location {args.domain_table} '
						  'does not exist. Check that it is mounted correctly '
						  ' in container.')
	if not os.path.exists(args.blacklists):
		raise FileNotFoundError(f'blacklists location {args.blacklists} '
						  'does not exist. Check that it is mounted correctly '
						  ' in container.')
	for fdbk_dir in fdbk_dirs:
		if not os.path.exists(fdbk_dir) or not os.path.isdir(fdbk_dir):
			raise FileNotFoundError(f'feedback directory {fdbk_dir} '
                        'does not exist. Check that MEC has run.')

# TODO: Make this a parameter as well.
def _make_veri_ens_member(experiment_ids: str) -> str:
	num_ids = len(experiment_ids.split(','))
	return ','.join(['-1'] * num_ids)

def main(args):
    # Render template with provided args
    context = {"experiment_ids": args.experiment_ids,
               "feedback_directories": args.feedback_directories,
               "veri_ens_member": _make_veri_ens_member(args.experiment_ids),
               "output_directory": args.output_directory,
               "experiment_description": args.experiment_description,
               "file_description": args.file_description,
               "domain_table": args.domain_table,
               "blacklists": args.blacklists
               }
    template_path = Path(args.template)
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(template_path.parent)))
    template = env.get_template(template_path.name)
    namelist = template.render(**context)
    LOG.info(f"FFV2 namelist created: {namelist}")

    out_path = Path(str(args.namelist))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(namelist)


if __name__ == "__main__":

	parser = ArgumentParser()
    
	parser.add_argument(
        "--template",
        type=str,
        help="path to jinja template",
	)

	parser.add_argument(
        "--namelist",
        type=str,
        help="full path to output namelist",
	)

	parser.add_argument(
        "--experiment_ids",
        type=str,
        help="namelist variable: comma-separated list of models to compare",
    )

	parser.add_argument(
        "--feedback_directories",
        type=str,
        help="namelist variable: comma-separated list of feedback directory locations",
    )

	parser.add_argument(
        "--output_directory",
        type=str,
        help="namelist variable: location to output score files (must already exist)",
    )
	
	parser.add_argument(
        "--experiment_description",
        type=str,
        help="namelist variable: string used to build the filenames of the "
          "intermediate scorefiles",
	)
     
	parser.add_argument(
        "--file_description",
        type=str,
        help="namelist variable: string used to build the filenames of the output",
    )

	parser.add_argument(
    	"--domain_table",
        type=str,
        help="namelist variable: path to domain table. This needs to be "
          "available from container through mounting."
	)

	parser.add_argument(
    	"--blacklists",
        type=str,
        help="namelist variable: path to blacklists. This needs to be "
          "available from container through mounting."
	)

	args = parser.parse_args()
	print(args)
	_check_args(args)

	main(args)
