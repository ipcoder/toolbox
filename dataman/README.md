# ialdev-dataman

Dataset management and data casting tools for the `iad` toolbox, published as `ialdev-dataman` and imported as `iad.dataman`.

Use this package to describe datasets with resource models, discover YAML-backed dataset definitions, build data collections, apply read/transformation pipelines, and create sink repositories for generated outputs.

## Install

```bash
pip install ialdev-dataman
```

Requires Python `>=3.10`, NumPy `>=2.0`, and pandas `>=2.0.0`.

## Highlights

- Dataset resource models: `DataSourceRM`, `SchemeRM`, `DatasetRM`, and `CollectionRM`.
- Factory helpers: `create_caster`, `create_collection`, and `create_sink`.
- Data collection APIs: `DataCollection`, `SinkRepo`, `CollectTable`, and `CollectSeries`.
- YAML/Pydantic helpers through `iad.dataman.pydantools`.
- Transform utilities for reading files into structured tables and applying image/data transforms.

## Examples

```python
from iad.dataman import create_caster, create_collection

caster = create_caster(
    source="/data/images",
    scheme="{scene}/{frame}.png",
    filters={"scene": "scene_001"},
)
collection = create_collection(datasets=[caster])
```

```python
from iad.dataman.models import DataSourceRM, DatasetRM, SchemeRM
from iad.dataman.resman import ModelsManager

manager = ModelsManager()
manager.register_models(DataSourceRM, SchemeRM, DatasetRM)
```

```python
from iad.dataman.pydantools import YamlModel

class DatasetConfig(YamlModel):
    name: str
    root: str
```
