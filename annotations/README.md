# ialdev-annotations

Annotation and issue-table utilities for the `iad` toolbox, published as `ialdev-annotations` and imported as `iad.annotations`.

Use this package to load issue/ROI collections from CSV, keep issue metadata in table form, query issues by labels, and visualize annotations on dataset scenes.

## Install

```bash
pip install ialdev-annotations
```

Requires Python `>=3.10`.

## Highlights

- `IssueCollection` for tabular issue/ROI collections.
- CSV loading with typed label rows and polygon fields.
- Query helpers for filtering issues by indexed labels.
- Visualization helpers: `VisIssue`, `show_issues_on_scenes`, and `AnnotatingKeyProcessor`.

## Examples

```python
from iad.annotations import IssueCollection

issues = IssueCollection.from_csv("issues.csv")
subset = issues.qix(dataset="FT3D", issue_type="S2")
```

```python
from iad.annotations import show_issues_on_scenes

show_issues_on_scenes(issues, scenes)
```

See the repository root `README.md` for workspace install and publishing notes.
