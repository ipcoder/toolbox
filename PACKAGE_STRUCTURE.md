# Package Structure Guide

This document explains the structure of the installable `toolbox` package.

## Current Structure

```
/code/
├── pyproject.toml          # Modern Python package configuration
├── setup.py                # Legacy setup script (for compatibility)
├── MANIFEST.in             # Files to include in distribution
├── README.md               # Package documentation
├── CHANGELOG.md            # Version history
├── INSTALLATION.md         # Installation instructions
├── .gitignore              # Git ignore patterns
└── toolbox/                # Main package directory
    ├── __init__.py         # Package initialization
    ├── conftest.py         # Pytest configuration
    ├── datacast/           # Data casting subpackage
    ├── engines/            # Engine framework subpackage
    ├── image/              # Image processing subpackage
    ├── io/                 # I/O operations subpackage
    ├── math/               # Math utilities subpackage
    ├── param/              # Parameter management subpackage
    ├── resman/             # Resource management subpackage
    ├── utils/              # Utility functions subpackage
    └── vis/                # Visualization subpackage
```

## Key Files Explained

### `pyproject.toml`
- Modern Python packaging configuration (PEP 517/518)
- Defines package metadata, dependencies, and build system
- Replaces `setup.py` for modern Python packaging
- **Action Required**: Update version, author, license, and URLs

### `setup.py`
- Kept for backward compatibility
- Minimal file that reads from `pyproject.toml`
- Can be removed if you only support modern pip/setuptools

### `MANIFEST.in`
- Specifies which files to include in source distribution
- Includes YAML files, documentation, and other data files
- Excludes test files and build artifacts

### `toolbox/__init__.py`
- Package initialization file
- Can expose commonly used classes at package level
- Currently minimal - expand as needed

## Installation Commands

### Development Mode (Editable Install)
```bash
pip install -e .
```
This installs the package in "editable" mode, so changes to source code are immediately available.

### Regular Install
```bash
pip install .
```

### With Development Dependencies
```bash
pip install -e ".[dev]"
```

### Build Distribution
```bash
python -m build
# Creates dist/toolbox-0.1.0.tar.gz and dist/toolbox-0.1.0-py3-none-any.whl
```

## Package Structure Decisions

### Why Flat Structure?
- Current structure uses a flat layout (package at root level)
- Simpler than src-layout for this use case
- No need to restructure existing code
- Works well with current import patterns (`from toolbox.xxx import yyy`)

### Alternative: src-layout
If you prefer src-layout:
```
/code/
├── pyproject.toml
├── src/
│   └── toolbox/
│       ├── __init__.py
│       └── ...
└── tests/  # Move tests here
```

To use src-layout, update `pyproject.toml`:
```toml
[tool.setuptools.package-dir]
"" = "src"
```

## What Gets Included

### Included in Distribution
- All Python modules (`.py` files)
- Package data files (`.yml`, `.yaml`, `.json`, `.md`, `.ipynb`, etc.)
- Type stubs (`py.typed`)

### Excluded from Distribution
- Test files (`test_*.py`, `conftest.py`)
- Test data directories
- Build artifacts (`__pycache__`, `.pyc`, `.pyo`)
- Development files (`.pytest_cache`, `.mypy_cache`)

## Next Steps

1. **Update `pyproject.toml`**:
   - Set correct version number
   - Add your name/email as author
   - Set correct license
   - Update repository URLs
   - Review and adjust dependencies

2. **Update `toolbox/__init__.py`**:
   - Add version number
   - Optionally expose commonly used classes

3. **Test Installation**:
   ```bash
   pip install -e .
   python -c "import toolbox; print('Success!')"
   ```

4. **Build and Test Distribution**:
   ```bash
   python -m build
   pip install dist/toolbox-*.whl
   ```

5. **Consider Adding**:
   - `LICENSE` file
   - `CONTRIBUTING.md` for contributors
   - `docs/` directory for documentation
   - CI/CD configuration (GitHub Actions, etc.)

## Publishing to PyPI (Future)

When ready to publish:

1. Create accounts on:
   - PyPI: https://pypi.org
   - Test PyPI: https://test.pypi.org

2. Build distributions:
   ```bash
   python -m build
   ```

3. Upload to Test PyPI first:
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```

4. Test installation from Test PyPI:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ toolbox
   ```

5. Upload to PyPI:
   ```bash
   python -m twine upload dist/*
   ```

## Dependencies Management

Current dependencies in `pyproject.toml`:
- Core: pydantic, pandas, numpy, pyyaml, etc.
- Dev: pytest, black, flake8, mypy

**Action Required**: Review and update dependency versions based on your actual requirements.

## Testing

Tests are currently in subpackage `tests/` directories. To run:

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest toolbox/
```

Consider consolidating tests in a top-level `tests/` directory if preferred.

