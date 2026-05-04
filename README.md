# ialgdev (monorepo meta-package)

The repository root installs **`ialgdev`**, a **dependency-only** meta-package: it has no `iad.*` code, only **`[project.dependencies]`** on the **`ialdev-*`** wheels. You do **not** need setuptools in particular—any **PEP 517** backend that can ship an empty wheel is fine. This repo uses **setuptools** for that; **Flit** is kept for the real libraries (one `[tool.flit.module]` each) and is a poor fit for a module-less meta wheel.

## Installation

### From source (with pixi workspace)

```bash
git clone https://github.com/yourusername/toolbox.git
cd toolbox
pixi install
```

### Editable pip install (all `ialdev-*` libraries)

```bash
pip install -e ./algutils -e ./fio -e ./imgtools -e ./maths -e ./vis -e ./dataman -e ./annotations -e ./engines -e .
```

The I/O project lives under **`fio/`** (not `io/`) so the directory name does not shadow the standard library `io` module.

### Development installation

```bash
pip install -e ".[dev]"
```

## Package Architecture

PyPI distribution names and import namespaces:

| Folder        | `pip install`     | `import`        |
|---------------|-------------------|-----------------|
| `algutils/`   | `ialdev-core`     | `iad.core`      |
| `fio/`        | `ialdev-io`       | `iad.io`        |
| `imgtools/`   | `ialdev-img`      | `iad.img`       |
| `maths/`      | `ialdev-maths`    | `iad.maths`     |
| `vis/`        | `ialdev-vis`      | `iad.vis`       |
| `dataman/`    | `ialdev-dataman`  | `iad.dataman`   |
| `annotations/`| `ialdev-annotations` | `iad.annotations` |
| `engines/`    | `ialdev-engines`  | `iad.engines`   |

## Publishing to PyPI (Flit)

The `ialdev-*` packages are built with **[Flit](https://flit.pypa.io/)** (`flit_core` backend in each `pyproject.toml`). From the repo root with Pixi:

```bash
pixi install   # provides ``flit`` and ``build`` CLI
cd algutils && pixi run flit build --no-use-vcs && pixi run flit publish
```

Repeat `build` / `publish` in `fio/`, `imgtools/`, `maths/`, `vis/`, `dataman/`, `annotations/`, `engines/` as needed. Use `--no-use-vcs` if the working tree is not clean (Flit otherwise checks version control for sdist contents).

Configure PyPI credentials with **keyring**, `~/.pypirc`, or `FLIT_USERNAME` / `FLIT_PASSWORD` / tokens per Flit docs.

The **`ialgdev`** meta-package at the repository root uses **setuptools**; publish it separately only if you distribute that name (e.g. `python -m build` at repo root).

## Usage

```python
from iad.core.param import YamlModel, TBox
from iad.dataman.datacast import DataCollection, SinkRepo
# See each package’s docs for full APIs.
```

## License

[Your License Here]
