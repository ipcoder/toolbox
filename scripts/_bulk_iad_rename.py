#!/usr/bin/env python3
"""Import-safe bulk rename for ialdev/iad refactor (.py only)."""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path("/code/toolbox")

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    ".pixi",
    "build",
    ".pytest_cache",
    "node_modules",
    "pydantools/src",
    "inu/src",
    ".cursor",
}

IMPORT_SUBST = [
    # (old_root, new_root) — apply longest roots first where needed
    ("pydantools", "iad.core.pydantools"),
    ("algutils", "iad.core"),
    ("iotools", "iad.io"),
    ("imgtools", "iad.img"),
    ("algovis", "iad.vis"),
    ("dataman", "iad.dataman"),
    ("maths", "iad.maths"),
    ("inu", "iad.io.inu"),
]


def skip_dir(path: Path) -> bool:
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        return True
    for p in rel.parts:
        if p in SKIP_DIRS or p == ".cursor":
            return True
    return False


def rewrite_py(text: str) -> str:
    """Rewrite import/module prefixes without touching unrelated ``word`` occurrences."""
    out = text
    for old, new in IMPORT_SUBST:
        out = re.sub(rf"\bfrom {re.escape(old)}\.", f"from {new}.", out)
        out = re.sub(rf"\bfrom {re.escape(old)}\s+import\b", f"from {new} import", out)
        out = re.sub(rf"\bimport {re.escape(old)}\.", f"import {new}.", out)
        # ``import foo`` as single module (last)
        out = re.sub(rf"(^|\s)import {re.escape(old)}\s*$", rf"\1import {new}", out, flags=re.M)
        out = re.sub(rf"(^|\s)import {re.escape(old)}\s*#", rf"\1import {new} #", out, flags=re.M)
    # import foo as bar (single segment)
    for old, new in IMPORT_SUBST:
        out = re.sub(
            rf"\bimport {re.escape(old)} as\b",
            f"import {new} as",
            out,
        )
    return out


def main() -> None:
    n = 0
    # Limit scope — whole-repo os.walk can be huge (snapshots, caches).
    roots = [
        ROOT / "algutils",
        ROOT / "dataman",
        ROOT / "maths",
        ROOT / "fio",
        ROOT / "imgtools",
        ROOT / "vis",
        ROOT / "annotations",
        ROOT / "engines",
    ]
    walk_roots = [p for p in roots if p.is_dir()]
    root_py = sorted(ROOT.glob("*.py"))
    for wr in walk_roots:
        for dirpath, dirnames, filenames in os.walk(wr):
            dp = Path(dirpath)
            if skip_dir(dp):
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and d != ".cursor"]
            for name in filenames:
                if not name.endswith(".py"):
                    continue
                path = dp / name
                try:
                    t = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                nt = rewrite_py(t)
                if nt != t:
                    path.write_text(nt, encoding="utf-8")
                    print(path.relative_to(ROOT))
                    n += 1
    for path in root_py:
        try:
            t = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        nt = rewrite_py(t)
        if nt != t:
            path.write_text(nt, encoding="utf-8")
            print(path.relative_to(ROOT))
            n += 1
    print(f"Rewrote {n} Python files.")


if __name__ == "__main__":
    main()
