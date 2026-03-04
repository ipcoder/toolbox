"""
Toolbox - A comprehensive Python toolbox for data management, processing, and visualization.

Main subpackages:
    - datacast: Data casting and collection tools
    - resman: Resource management
    - vis: Visualization tools
    - engines: Engine framework

Utilities, image, and math (io, param, paths, transforms, hist, geom, etc.) are provided by ``algutils``.
"""

__version__ = "0.1.0"  # Update this to match pyproject.toml

# Optionally expose commonly used classes at package level
# Uncomment and add as needed:
# from toolbox.datacast import DataCaster, DataCollection
# from algutils.param import YamlModel, TBox
# from toolbox.resman import resman

__all__ = [
    "__version__",
    # Add commonly used exports here
]

