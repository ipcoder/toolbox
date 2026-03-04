# Python 3.8 Compatibility Report

## Summary

The codebase has **incompatibilities** with Python 3.8 that need to be fixed.

## Issues Found

### 1. PEP 604 Union Syntax (`|` instead of `Union`) - **REQUIRES PYTHON 3.10+**

**Files affected:**
- `vis/insight.py` (lines 727, 729, 730, 981)
- `vis/annotations/visual.py` (line 419)

**Examples:**
```python
# ❌ Python 3.10+ only
labels: dict | list[dict] = None
cmap: str | list[str] = 'rain'
objects: object | list[object]

# ✅ Python 3.8 compatible
from typing import Union
labels: Union[dict, list[dict]] = None
cmap: Union[str, list[str]] = 'rain'
objects: Union[object, list[object]]
```

### 2. Match/Case Statements - **REQUIRES PYTHON 3.10+**

**Files affected:**
- `utils/pdtools.py` (lines 1156-1169, 2489-2496)
- `math/hist.py` (lines 655-664, 753-760, 1195-1208)

**Examples:**
```python
# ❌ Python 3.10+ only
match keep_levels:
    case True:
        ...
    case False:
        ...

# ✅ Python 3.8 compatible
if keep_levels is True:
    ...
elif keep_levels is False:
    ...
```

### 3. Generic Type Syntax (`list[str]` vs `List[str]`) - **OK with `from __future__ import annotations`**

**Status:** ✅ **COMPATIBLE** - 39 files use `from __future__ import annotations`, which makes this work in Python 3.8.

**Files using modern syntax:**
- Most files use `list[str]`, `dict[str, ...]` etc.
- All have `from __future__ import annotations` at the top
- This is **compatible** with Python 3.8

### 4. Walrus Operator (`:=`) - **COMPATIBLE**

**Status:** ✅ **COMPATIBLE** - Walrus operator is available in Python 3.8+

**Files using it:**
- `vis/insight.py` (lines 307, 394, 395, 1010, 1075)
- `vis/tests/test_insight.py` (line 229)

## Required Fixes

### Priority 1: Fix Union Syntax

**File: `vis/insight.py`**

Replace:
```python
labels: dict | list[dict] = None,
titles: list[str] = None,
cmap: str | list[str] = 'rain',
ticks: int | tuple[int, int] = 5,
hists: dict[str, np.ndarray] | np.ndarray,
```

With:
```python
from typing import Union
labels: Union[dict, list[dict]] = None,
titles: list[str] = None,  # OK with __future__ annotations
cmap: Union[str, list[str]] = 'rain',
ticks: Union[int, tuple[int, int]] = 5,
hists: Union[dict[str, np.ndarray], np.ndarray],
```

**File: `vis/annotations/visual.py`**

Replace:
```python
objects: object | list[object]
```

With:
```python
from typing import Union
objects: Union[object, list[object]]
```

### Priority 2: Replace Match/Case Statements

**File: `utils/pdtools.py` (around line 1156)**

Replace match/case with if/elif:
```python
# Current (3.10+)
match keep_levels:
    case True:
        levels = list(d.index.names)
    case False:
        levels = []
    case [*_levels]:
        levels = list(_levels)
    case str(x) | int(x):
        levels = [x]
    case _:
        levels = list(keep_levels)

# Python 3.8 compatible
if keep_levels is True:
    levels = list(d.index.names)
elif keep_levels is False:
    levels = []
elif isinstance(keep_levels, (list, tuple)):
    levels = list(keep_levels)
elif isinstance(keep_levels, (str, int)):
    levels = [keep_levels]
else:
    levels = list(keep_levels)
```

**File: `math/hist.py` (around line 655)**

Similar replacements needed for all match/case blocks.

## Testing Python 3.8 Compatibility

### Manual Test

```bash
# Test with Python 3.8
python3.8 -m py_compile toolbox/vis/insight.py
python3.8 -m py_compile toolbox/utils/pdtools.py
python3.8 -m py_compile toolbox/math/hist.py
```

### Automated Test

Add to CI/CD:
```yaml
- name: Test Python 3.8 compatibility
  run: |
    python3.8 -m compileall -q toolbox/
```

## Recommendations

1. **Fix all PEP 604 union syntax** - Replace `|` with `Union[]`
2. **Replace all match/case statements** - Use if/elif chains
3. **Add Python 3.8 to CI/CD** - Test against Python 3.8
4. **Consider raising minimum version** - If fixing is too difficult, consider Python 3.9 or 3.10 as minimum

## Files Requiring Changes

1. `vis/insight.py` - Union syntax + match/case
2. `vis/annotations/visual.py` - Union syntax
3. `utils/pdtools.py` - Match/case statements
4. `math/hist.py` - Match/case statements

## Estimated Effort

- **Union syntax fixes**: ~15 minutes (simple find/replace)
- **Match/case replacements**: ~1-2 hours (need to test logic)
- **Testing**: ~30 minutes

**Total: ~2-3 hours**


