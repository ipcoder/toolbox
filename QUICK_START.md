# Quick Start Guide

## Installation

### Development Installation (Recommended)

```bash
cd /code/toolbox
pip install -e .
```

This installs the package in "editable" mode, so changes to source code are immediately available.

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

## Verify Installation

```bash
python -c "import toolbox; print('Success! Toolbox version:', toolbox.__version__)"
```

## Usage

Imports work exactly as before:

```python
from toolbox.datacast import DataCaster, DataCollection
from toolbox.param import YamlModel, TBox
from toolbox.resman import resman
from toolbox.utils import logger, as_list
from toolbox.io import imread, imwrite
```

## Building Distribution

```bash
# Install build tools
pip install build

# Build distributions
python -m build

# Distributions will be in dist/
```

## Project Structure

```
/code/toolbox/              # Project root (also the package)
├── pyproject.toml          # Package configuration
├── setup.py                # Legacy setup (optional)
├── README.md               # Package documentation
├── __init__.py             # Package initialization
├── datacast/               # Data casting subpackage
├── engines/                # Engine framework
├── image/                  # Image processing
├── io/                     # I/O operations
├── math/                   # Math utilities
├── param/                  # Parameter management
├── resman/                 # Resource management
├── utils/                  # Utility functions
└── vis/                    # Visualization tools
```

## Next Steps

1. Update `pyproject.toml`:
   - Version number
   - Author information
   - License
   - Repository URLs

2. Update `toolbox/__init__.py`:
   - Set version number
   - Optionally expose commonly used classes

3. Test your installation and imports!

