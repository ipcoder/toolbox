from typing import Literal

from .core import AlgoEngine, Internals
from .label_io import OPTIONAL, LT
from .register import engine_class, Registry, engines

_UNDEF = object()
_OPT = str | Literal[_UNDEF]


def get_engine(name: str, *, kind: _OPT =_UNDEF, pfm: _OPT =_UNDEF):
    """
    Find *one* and return engine class as specified or fail!

    (Simplified wrapper around ``engines.find``. Use `engines` from `toolbox.engines` for more flexible interface.)

    :param name:
    :param kind:
    :param pfm:
    :return:
    """
    from ..utils import drop_undef
    return engines.find(name, out='class', **drop_undef(_undef=_UNDEF, ns=dict(kind=kind, pfm=pfm)))

