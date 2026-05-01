# Toolbox

A comprehensive Python toolbox for data management, processing, and visualization.

## Installation

### From source (with pixi workspace)

```bash
git clone https://github.com/yourusername/toolbox.git
cd toolbox
pixi install
```

### Editable pip install

```bash
pip install -e ./algutils -e ./datacast -e ./resman -e .
```

### Development installation

```bash
pip install -e ".[dev]"
```

## Package Architecture

The workspace contains three **standalone pip-installable packages** and a bridge module:

```
algutils    →  base utilities (no workspace deps)
datacast    →  folder scanning, data casting, collection  (depends on algutils)
resman      →  resource model framework with discovery    (depends on algutils + datacast)
toolbox.datasets  →  bridge wiring resman models to datacast core (depends on all above + inu)
```

## Usage

```python
# Standalone packages — import directly
from datacast import DataCaster, DataCollection, SinkRepo
from resman import resman, ModelsManager, ResourceModel

# Bridge — name-based convenience constructors
from toolbox.datasets import create_caster, create_collection, create_sink
caster = create_caster('ETH3D')
dc = create_collection('SmallQualityEval')

# Utilities
from algutils.param import YamlModel, TBox
```

## Other Subpackages

- `toolbox.vis` — Visualization tools
- `toolbox.engines` — Engine framework

## License

[Your License Here]
