# ialdev-io

Image and structured file I/O utilities for the `iad` toolbox, published as `ialdev-io` and imported as `iad.io`.

The source directory is named `fio/` instead of `io/` so local development does not shadow Python's standard library `io` module.

## Install

```bash
pip install ialdev-io
```

Requires Python `>=3.10`.

## Highlights

- General image loading/saving with `imread` and `imsave`.
- File format registry and metadata helpers in `iad.io.format`.
- PFM, PLY, TIFF tag, ZFP, bit-stream, and pickle helpers.
- CIIF conversion/loading utilities.
- Inuitive/NU4 helpers under `iad.io.inu`.

## Examples

```python
from iad.io import imread, imsave

image = imread("input.tif")
imsave("copy.tif", image)
```

```python
from iad.io.pfm import load_pfm, save_pfm

depth = load_pfm("depth.pfm")
save_pfm("depth_copy.pfm", depth)
```

```python
from iad.io.format import FileFormat

handler = FileFormat.find_handler("image.tif")
```
