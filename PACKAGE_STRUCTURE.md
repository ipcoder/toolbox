# Package Structure Guide

This workspace is a **monorepo** containing four standalone pip-installable packages
and two bridge modules that wire them together.

## Architecture

```
algutils    (standalone)   base utilities, no workspace deps
    ↑
datacast    (standalone)   folder scanning, data casting, collection
    ↑
resman      (standalone)   resource model framework with discovery
    ↑
algovis     (standalone)   image grids, interactive viewers, colormaps
    ↑
toolbox.datasets    (bridge)  domain-specific models + factory functions (requires inu)
toolbox.annotations (bridge)  issue annotation UI + AnnotatingKeyProcessor (requires inu, datacast, algovis)
```

Each standalone package can be installed and used independently.
The bridge modules live inside the root `toolbox` package and provide
domain-specific integrations that depend on multiple standalone packages plus `inu`.

## Directory Layout

```
/code/toolbox/                  # Monorepo root (also the "toolbox" package)
├── pyproject.toml              # Root package config + pixi workspace definition
├── __init__.py                 # toolbox package init
├── conftest.py                 # Root-level pytest fixtures
│
├── algutils/                   # ── Standalone package ──
│   ├── pyproject.toml
│   ├── pixi.toml
│   ├── src/algutils/           #   distributed code
│   │   ├── __init__.py
│   │   ├── cache.py
│   │   ├── param/
│   │   └── ...
│   └── tests/                  #   NOT distributed
│
├── datacast/                   # ── Standalone package ──
│   ├── pyproject.toml
│   ├── pixi.toml
│   ├── docs/                   #   project docs (not distributed)
│   ├── src/datacast/           #   distributed code
│   │   ├── __init__.py
│   │   ├── caster.py           #     DataCaster, CasterConfig, Labeler
│   │   ├── collect.py          #     DataCollection, SinkRepo
│   │   ├── scan.py             #     GuideScan
│   │   ├── transtools.py       #     Col, CollectTable, Fetchable
│   │   ├── transforms.py       #     image transform functions
│   │   └── labeled.py          #     LabelRules
│   └── tests/                  #   NOT distributed
│       ├── test_scheme.py
│       ├── test_collect.py
│       └── data/
│
├── resman/                     # ── Standalone package ──
│   ├── pyproject.toml
│   ├── pixi.toml
│   ├── docs/                   #   project docs
│   ├── src/resman/             #   distributed code
│   │   ├── __init__.py
│   │   └── resource.py         #     ResourceModel, ModelsManager, locatable
│   └── tests/                  #   NOT distributed
│       ├── test_resource.py
│       └── data/
│
├── vis/                        # ── Standalone package (algovis) ──
│   ├── pyproject.toml
│   ├── pixi.toml
│   ├── src/algovis/            #   distributed code
│   │   ├── __init__.py
│   │   ├── insight.py          #     imgrid, imhist, KeyProcessor
│   │   ├── interact.py         #     LinesDrawer, interactive widgets
│   │   ├── imageviewer.py      #     ImageViewer (Qt-based)
│   │   ├── mpl_utils.py        #     matplotlib helpers
│   │   ├── view3d.py           #     3D visualization
│   │   ├── colormaps/          #     custom colormaps
│   │   └── poly_to_mask/       #     polygon-to-mask utilities
│   └── tests/                  #   NOT distributed
│       ├── test_insight.py
│       └── test_interact.py
│
├── datasets/                   # ── Bridge module (toolbox.datasets) ──
│   ├── __init__.py             #   re-exports models + factories
│   ├── models.py               #   DataSourceRM, SchemeRM, DatasetRM, CollectionRM
│   ├── factories.py            #   create_caster, create_collection, create_sink
│   └── tests/
│       ├── test_models.py
│       ├── test_datasets.py
│       └── data/
│
├── annotations/                # ── Bridge module (toolbox.annotations) ──
│   ├── __init__.py             #   re-exports IssueCollection, VisIssue, AnnotatingKeyProcessor
│   ├── visual.py               #   VisIssue, AnnotatingKeyProcessor, show_issues_on_scenes
│   ├── collect.py              #   IssueCollection
│   └── tests/
│       └── test_issues.py
│
└── engines/                    # Other toolbox subpackages (not yet extracted)
```

### Key conventions

- **src layout**: each standalone package keeps source under `src/<name>/`
  so that `import <name>` only works after installation (not by accident from cwd).
- **Tests at project root**: `tests/` is a sibling of `src/`, not inside the
  distributed package.  Tests import the package the same way users do and are
  not shipped with `pip install`.
- **Docs at project root**: `docs/` contains project documentation, not
  distributed with the package.

## Dependency Graph (acyclic)

| Package               | Depends on                                      |
|-----------------------|-------------------------------------------------|
| `algutils`            | third-party only                                |
| `datacast`            | `algutils` + third-party                        |
| `resman`              | `algutils` + `datacast` + third-party           |
| `algovis`             | `algutils` + third-party (matplotlib, numpy ...) |
| `toolbox.datasets`    | `datacast` + `resman` + `inu`                   |
| `toolbox.annotations` | `algovis` + `datacast` + `inu`                  |
| `toolbox`             | all of the above                                |

## Installation

### Full workspace (pixi)

```bash
cd /code/toolbox
pixi install
```

### Editable pip install

```bash
pip install -e ./algutils -e ./datacast -e ./resman -e ./vis -e .
```

### Individual package

```bash
pip install -e ./datacast    # just datacast + algutils
pip install -e ./vis         # just algovis + algutils
```

## Usage

```python
# Standalone packages — import directly
from datacast import DataCaster, DataCollection, CasterConfig
from datacast.scan import GuideScan
from resman import resman, ModelsManager, ResourceModel
from algovis.insight import imgrid, imhist, KeyProcessor

# Bridge — name-based convenience constructors
from toolbox.datasets import create_caster, create_collection, create_sink
from toolbox.datasets import DatasetRM, SchemeRM, DataSourceRM

# Bridge — annotations with extensible KeyProcessor
from toolbox.annotations import AnnotatingKeyProcessor, VisIssue, IssueCollection
imgrid(im1, im2, key_processor_cls=AnnotatingKeyProcessor)

# Utilities
from algutils.param import YamlModel, TBox
from algutils import logger, as_list
```

## Running Tests

```bash
# All tests via pixi workspace
pixi run test

# Individual package
cd datacast && pytest tests -v
cd resman && pytest tests -v
cd vis && pytest tests -v

# Bridge tests (from repo root)
pytest datasets/tests -v
pytest annotations/tests -v
```

## Building Distributions

```bash
# Individual packages
cd datacast && python -m build
cd resman && python -m build
cd vis && python -m build

# Root toolbox package (includes bridges)
python -m build
```
