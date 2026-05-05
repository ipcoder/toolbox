# ialdev-core

Core utilities for the `iad` toolbox, published as `ialdev-core` and imported as `iad.core`.

Use this package when you need the shared building blocks used by the other `ialdev-*` libraries: collection helpers, typed dictionaries, path transformations, file discovery, caching, logging, event timing, decorators, and NumPy/Pandas utility functions.

## Install

```bash
pip install ialdev-core
```

Requires Python `>=3.10`, NumPy `>=2.0`, and pandas `>=2.1.0`.

## Highlights

- `iad.core.short`: scalar/list normalization and keyword filtering (`as_list`, `as_iter`, `drop_undef`).
- `iad.core.tbox`: `TBox`, a convenience wrapper around `python-box`.
- `iad.core.fs`: path normalization, file discovery, locators, and transformable path templates.
- `iad.core.cache`: pickle-based caches and cached processing pipes.
- `iad.core.data`: NumPy/Pandas helpers, labels, units, binary packing, and table utilities.
- `iad.core.events`: timers, progress helpers, and joblib/tqdm integration.

## Examples

```python
from iad.core import as_list, drop_undef
from iad.core.tbox import TBox

names = as_list("sample")
config = TBox(drop_undef(root="/data", cache=None, batch=8))
```

```python
from iad.core.fs.filesproc import Locator
from iad.core.fs.paths import TransPath

locator = Locator("/data/project")
path_template = TransPath("{scene}/{frame}.png")
```

## Development

```bash
pip install -e .
pixi run test
pixi run lint
```
