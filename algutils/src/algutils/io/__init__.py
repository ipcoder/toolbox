from .imread import imread
from .imwrite import imsave


def read_meta(path):
    """Read file at path and return dict (e.g. YAML/JSON). Used by extract_attr when meta is a path."""
    path = __import__('pathlib').Path(path)
    suffix = path.suffix.lower()
    if suffix in ('.yaml', '.yml'):
        import yaml
        return yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if suffix in ('.json', '.js', '.jso', '.jsn'):
        import json
        return json.loads(path.read_text(encoding='utf-8'))
    # fallback: try yaml
    import yaml
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}
