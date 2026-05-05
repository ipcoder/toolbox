# ialgdev

`ialgdev` is a dependency-only meta package for the `ialdev-*` Python packages in this repository. Install it when you want the full toolbox; install an individual `ialdev-*` package when you only need one namespace.

## Install

```bash
pip install ialgdev
```

From a checkout:

```bash
git clone https://github.com/yourusername/toolbox.git
cd toolbox
pixi install
pip install -e ./algutils -e ./fio -e ./imgtools -e ./maths -e ./vis -e ./dataman -e ./annotations -e ./engines -e .
```

Requires Python `>=3.10`.

## Packages

| PyPI package | Import namespace | Purpose |
|---|---|---|
| `ialdev-core` | `iad.core` | Shared utilities for collections, config, paths, caching, logging, events, and typed helpers. |
| `ialdev-io` | `iad.io` | Image, metadata, PFM, PLY, CIIF, TIFF, ZFP, bit-stream, and Inuitive/NU4 I/O helpers. |
| `ialdev-img` | `iad.img` | Image transforms, crops, regions, camera models, stereo camera metadata, and binning helpers. |
| `ialdev-maths` | `iad.maths` | Histograms, samplers, 2D geometry, planes, and regression utilities. |
| `ialdev-vis` | `iad.vis` | Matplotlib/Jupyter visualization helpers, image grids, histograms, colormaps, and optional 3D viewers. |
| `ialdev-dataman` | `iad.dataman` | Dataset resource models, data casting, collections, sinks, and YAML/Pydantic tooling. |
| `ialdev-annotations` | `iad.annotations` | Issue/ROI tables, CSV loading, and annotation visualization on scenes. |
| `ialdev-engines` | `iad.engines` | Algorithm engine base classes, labeled I/O types, registries, and catalogs. |

The source directory for `ialdev-io` is named `fio/` to avoid shadowing Python's standard library `io` module during development.

## Quick examples

```python
from iad.core import as_list, drop_undef
from iad.core.tbox import TBox

items = as_list("scene_001")
config = TBox(drop_undef(root="/data", cache=None))
```

```python
from iad.io import imread, imsave
from iad.img.tools import center_crop
from iad.vis import imgrid

image = imread("frame.tif")
crop = center_crop(image, width=256, height=256)
imgrid(image, crop, titles=["source", "crop"])
imsave("crop.tif", crop)
```

## Development and publishing

The individual `ialdev-*` packages use [Flit](https://flit.pypa.io/) and define one `[tool.flit.module]` each. The root `ialgdev` meta package uses `setuptools` and contains no `iad.*` modules.

```bash
pixi install
cd algutils
pixi run flit build --no-use-vcs
pixi run flit publish
```

Repeat the build/publish commands in each package directory as needed.
