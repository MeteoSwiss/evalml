import logging
import jinja2
# snakemake object inherited by default, but this enables code completion.
from snakemake.script import snakemake
from pathlib import Path

# Note: not currently in use; optional script in case we want to factor it out
# of the rules file
def main(args):
	#TODO: get wildcards working
	context = {}
	#context = {"init_time": snakemake.wildcards.init_time}
	template_path = Path(snakemake.input.template)
	logging.info('writing namelist to {template_filename}')
	env = jinja2.Environment(
		loader=jinja2.FileSystemLoader({template_path.parent})
	)
	template = env.get_template(template_path.name)
	namelist = template.render(**context)
	namelist_fn = Path(snakemake.output['namelist'])
	with namelist_fn.open("w+") as f:
		f.write(namelist)
	logging.info('finished writing namelist')


if __name__ == "__main__":
	main()