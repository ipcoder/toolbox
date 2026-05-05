# ialdev-engines

Algorithm engine abstractions for the `iad` toolbox, published as `ialdev-engines` and imported as `iad.engines`.

Use this package when you want to wrap algorithms behind a standard interface with typed inputs/outputs, persistent configuration, registry lookup, and catalog generation.

## Install

```bash
pip install ialdev-engines
```

Requires Python `>=3.10`.

## Highlights

- `AlgoEngine`, the base class for algorithm wrappers.
- Engine registration through `engine_class`, `Registry`, and the global `engines` registry.
- `get_engine` convenience lookup by name, kind, or platform metadata.
- Labeled I/O types for data arrays with attached semantic labels.
- Catalog resource models for discovering engine metadata.

## Example

```python
from iad.engines import AlgoEngine, get_engine

class AddEngine(AlgoEngine, kind="demo"):
    class Inputs:
        x: int
        y: int

    class Outputs:
        value: int

    def _process(self, inputs):
        return {"value": inputs.x + inputs.y}, None

engine_cls = get_engine("AddEngine", kind="demo")
```

See the repository root `README.md` for workspace install and publishing notes.
