# algutils

Algorithmic utilities: data tools, I/O, parameters, paths, caching, image transforms, and math (hist, geom, regress).

Extracted from the toolbox project as a standalone, installable package.

## Installation

```bash
# From this directory (editable)
pip install -e .

# With optional extras
pip install -e ".[io,units,dev]"
```

## Usage

```python
from algutils import logger, as_list
from algutils.param import TBox, YamlModel
from algutils.io.format import FileFormat
from algutils.io.imread import imread
```

## Development

- **pixi**: `pixi install` then `pixi run test` or `pixi run lint`
- **pip**: `pip install -e ".[dev]"` then `pytest src/algutils -v`
