# Quick Start Guide

## Installation

### Development Installation (Recommended)

```bash
cd /code/toolbox
pixi install          # installs everything via workspace
```

Or with pip (editable mode):

```bash
pip install -e ./algutils -e ./datacast -e ./resman -e .
```

## Verify Installation

```bash
python -c "import datacast; import resman; print('Success!')"
```

## Usage

```python
# Core data casting — standalone package
from datacast import DataCaster, DataCollection, CasterConfig
from datacast.scan import GuideScan

# Resource management — standalone package
from resman import resman, ModelsManager, ResourceModel

# Name-based factories — bridge module (requires all packages + inu)
from toolbox.datasets import create_caster, create_collection, create_sink
caster = create_caster('ETH3D')
dc = create_collection('KITTI', query=dict(subset='train'))

# Utilities — standalone package
from algutils.param import YamlModel, TBox
from algutils import logger, as_list
```

## Project Structure

```
/code/toolbox/              # Monorepo root (also the toolbox package)
├── pyproject.toml          # Root package config + pixi workspace
├── __init__.py             # toolbox package init
├── algutils/               # Standalone package: utilities
│   ├── pyproject.toml
│   ├── pixi.toml
│   └── src/algutils/
├── datacast/               # Standalone package: data casting
│   ├── pyproject.toml
│   ├── pixi.toml
│   └── src/datacast/
├── resman/                 # Standalone package: resource management
│   ├── pyproject.toml
│   ├── pixi.toml
│   └── src/resman/
├── datasets/               # Bridge module (toolbox.datasets)
│   ├── __init__.py
│   ├── models.py
│   └── factories.py
├── engines/                # Engine framework
└── vis/                    # Visualization tools
```

## Building Distributions

```bash
# Build individual packages
cd datacast && python -m build && cd ..
cd resman && python -m build && cd ..
python -m build   # root toolbox package
```
