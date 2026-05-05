# ialdev-maths

Numerical helpers for the `iad` toolbox, published as `ialdev-maths` and imported as `iad.maths`.

Use this package for histogram statistics, sampling, geometry primitives, plane fitting, and regression helpers used by image/data workflows.

## Install

```bash
pip install ialdev-maths
```

Requires Python `>=3.10`.

## Highlights

- Histogram tools: `Sampler`, `StatGather`, `StatGather2D`, `Hist2D`, and equal-bin statistics.
- Geometry primitives: `Vec2d`, `Rect`, `Pose`, ranges, and region checks.
- Plane utilities: plane fitting, axis IDs, and 3D plane representation.
- Regression helpers, including robust linear regression and an SVD plane estimator.

## Examples

```python
from iad.maths.geom.shapes import Rect, Vec2d

roi = Rect(Vec2d(10, 20), dim=Vec2d(100, 80))
inside = (25, 30) in roi
```

```python
from iad.maths.hist import Sampler, equal_bins_stats

sampler = Sampler(low=0, high=1, bins=16)
stats = equal_bins_stats(values, sampler, stats=True)
```

```python
from iad.maths.regress import robust_linear_regression

coef, intercept = robust_linear_regression(x, y)
```
