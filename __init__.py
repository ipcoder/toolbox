"""
Toolbox - A comprehensive Python toolbox for data management, processing, and visualization.

Standalone packages (pip-installable independently):
    - algutils: Utilities, image, and math (io, param, paths, transforms, etc.)
    - datacast: Folder scanning, data casting, and collection (resman-free core)
    - resman: Resource model framework with registration and discovery

Bridge modules (part of this package, wiring standalone packages together):
    - toolbox.datasets: Domain-specific resource models and factory functions
                        connecting resman resource models with datacast core

Other subpackages:
    - toolbox.vis: Visualization tools
    - toolbox.engines: Engine framework
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
]

