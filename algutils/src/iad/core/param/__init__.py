from frozendict import frozendict as fzdict

from . import paramaze as pa
from .tbox import TBox

__all__ = ['TBox', 'model_arguments', 'YamlModel', 'fzdict']


def __getattr__(name: str):
    """Lazy imports for YAML/Pydantic helpers shipped in ``iad.core.pydantools``."""
    if name == 'YamlModel':
        from iad.core.pydantools.models import YamlModel as _YamlModel
        return _YamlModel
    if name == 'model_arguments':
        from iad.core.pydantools.models import model_arguments as _model_arguments
        return _model_arguments
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
