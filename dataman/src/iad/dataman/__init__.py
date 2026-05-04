"""iad.dataman — datacast, resman, and datasets bridge (``ialdev-dataman`` distribution)."""

from .models import DataSourceRM, SchemeRM, DatasetRM, CollectionRM, DSample, discover
from .factories import create_caster, create_collection, create_sink
from .datacast.collect import DataCollection, SinkRepo

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DataSourceRM",
    "SchemeRM",
    "DatasetRM",
    "CollectionRM",
    "DSample",
    "discover",
    "create_caster",
    "create_collection",
    "create_sink",
    "DataCollection",
    "SinkRepo",
]
