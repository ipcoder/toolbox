# ialdev-vis

Visualization utilities for the `iad` toolbox, published as `ialdev-vis` and imported as `iad.vis`.

Use this package for quick image grids, histograms, Matplotlib helpers, custom colormaps, interactive annotation widgets, Jupyter logging, and optional Qt/3D visualization tools.

## Install

```bash
pip install ialdev-vis
```

Optional extras:

```bash
pip install "ialdev-vis[qt]"
pip install "ialdev-vis[jupyter]"
pip install "ialdev-vis[3d]"
pip install "ialdev-vis[all]"
```

Requires Python `>=3.10`.

## Highlights

- `imgrid` and `imhist` for fast inspection of images and distributions.
- Matplotlib helpers for figure capture and conversion to arrays/PIL images.
- Built-in custom colormaps.
- Polygon-to-mask helpers and interactive ROI tools.
- Optional Qt image viewer and optional `ipyvolume`/Open3D 3D views.

## Examples

```python
from iad.vis import imgrid, imhist

imgrid(left_image, right_image, titles=["left", "right"], clim="auto")
imhist(left_image, right_image, titles=["left", "right"])
```

```python
from iad.vis.mpl_utils import fig2img

pil_image = fig2img(figure)
```
