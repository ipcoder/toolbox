# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip

## Installation Methods

### 1. Development Installation (Recommended for contributors)

```bash
# Clone the repository
git clone https://github.com/yourusername/toolbox.git
cd toolbox

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### 2. Regular Installation from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/toolbox.git
cd toolbox

# Install
pip install .
```

### 3. Installation from PyPI (when published)

```bash
pip install toolbox
```

### 4. Installation with Optional Dependencies

```bash
# Install with all optional dependencies
pip install ".[all]"

# Install specific optional dependencies
pip install ".[dev]"
```

## Verifying Installation

After installation, verify it works:

```python
python -c "import toolbox; print(toolbox.__version__)"
```

## Building Distribution Packages

To build source and wheel distributions:

```bash
# Install build tools
pip install build

# Build distributions
python -m build

# Distributions will be in dist/
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure:
1. The package is properly installed: `pip list | grep toolbox`
2. You're using the correct Python environment
3. The package is in your Python path

### Missing Dependencies

If you get missing dependency errors:
```bash
pip install -e ".[all]"
```

