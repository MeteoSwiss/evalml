# Config JSON Schema

The repository ships a JSON Schema generated from the Pydantic models in
`src/evalml/config.py`. It lives at `workflow/tools/config.schema.json`
and is regenerated automatically by a `pre-commit` hook so it never drifts
from the Python source of truth.

To regenerate it manually:

```bash
python src/evalml/config.py workflow/tools/config.schema.json
```

The same schema is also exposed programmatically:

```python
from evalml.config import generate_config_schema
schema = generate_config_schema()
```

## Using the schema in your editor

Add a YAML language-server hint at the top of any config file:

```yaml
# yaml-language-server: $schema=../workflow/tools/config.schema.json
```

VSCode (with the YAML extension), Neovim (with `coc-yaml` or
`yaml-language-server`), and most modern editors will then surface
hover-docs, autocompletion, and inline validation pulled directly from
the `Field(description=...)` strings.

## Full schema

```{literalinclude} ../../../workflow/tools/config.schema.json
:language: json
```
