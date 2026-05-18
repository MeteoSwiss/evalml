# Style and conventions

This page collects the quick-reference conventions for code review. The
expanded version of the workflow conventions is in
[Workflow → Conventions](../workflow/conventions.md); this page focuses
on the Python source and the surrounding tooling.

## Python

- **Linting / formatting**: `ruff` and `ruff-format` (see
  `.pre-commit-config.yaml`). Line length follows the ruff default (88).
  Prefer fixing lint warnings rather than disabling them.
- **Type hints**: encouraged but not enforced. New public functions
  should have type hints; sphinx-autodoc renders them in the API
  reference via `autodoc_typehints = "description"`.
- **Docstrings**: Google or NumPy style; `napoleon` is enabled, so both
  render correctly. The first line should be a one-sentence summary —
  it ends up in `autosummary` tables.
- **Pydantic models**: live in `src/evalml/config.py`. New fields need a
  `Field(..., description="...")`; the description is what surfaces in
  the JSON Schema and in editor tooltips.
- **No top-level Snakemake-specific imports** in `src/`. The packages
  must remain importable outside a Snakemake run so the unit tests can
  exercise them.

## Snakemake

- Run `snakefmt workflow/` after edits. The `pre-commit` hook will catch
  it but local feedback is faster.
- Rule names follow `{module}_{operation}[_{sub_operation}]`. See
  [Workflow → Conventions](../workflow/conventions.md).
- Every rule declares a `log:` path under
  `OUT_ROOT/logs/{rule_name}/{wildcards}.log`.
- Outputs that are directories should be paired with a sentinel `.ok`
  file when downstream rules need a stable trigger.

## Pydantic schema

The JSON Schema at `workflow/tools/config.schema.json` is generated from
the Pydantic models. The pre-commit hook regenerates it; CI fails if the
committed schema has drifted.

To regenerate manually:

```bash
python src/evalml/config.py workflow/tools/config.schema.json
```

## Documentation

- New pages go under `docs/source/<section>/<name>.md` and need a toctree
  entry in `docs/source/index.md`.
- Cross-link with markdown links (`../user_guide/cli.md`) — Sphinx
  resolves them at build time and the `-W` flag in CI fails the build
  if a link is broken.
- Avoid embedding line numbers in `{literalinclude}` directives unless
  you really need them; prefer `:start-after:` / `:end-before:` markers
  if you need a slice.
- Don't add emojis to docs unless explicitly asked.

## Commits and PRs

- Commit hooks must be green before a push (`pre-commit install`
  ensures this).
- Keep PRs focused. A docs-only change that touches workflow rules
  should be split.
- Write commit messages in the imperative present tense ("Add
  ICON-CH1 patch metadata", not "Added…").
