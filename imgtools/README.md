# ialdev-img

Image processing and camera utilities for the `iad` toolbox, published as `ialdev-img` and imported as `iad.img`.

Use this package for small image transforms, crop/window helpers, labeled regions, camera metadata, and stereo camera support.

## Install

```bash
pip install ialdev-img
```

Requires Python `>=3.10`.

## Highlights

- Image transforms: normalization, gamma correction, alpha blending, channel extraction, convolution, and shot noise.
- Crop/window helpers such as `center_crop`, `crop`, and sliding-window conversions.
- Camera models: `Camera`, `Sensor`, `Shot`, `StereoCam`, and `Resolution`.
- Region containers for label-coded masks and ROI-style operations.

## Examples

```python
from iad.img.tools import center_crop
from iad.img.transforms import gamma, norm

crop = center_crop(image, width=256, height=256)
display = gamma(norm(crop), g=2.2)
```

```python
from iad.img.camera import Camera, Resolution

camera = Camera(name="HD", pixels=Resolution.HD)
```
