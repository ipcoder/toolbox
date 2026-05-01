"""
Bridge module connecting datacast core with resman resource models.

Provides domain-specific resource models (DataSourceRM, SchemeRM, DatasetRM,
CollectionRM) and factory functions for name-based construction of DataCaster,
DataCollection, and SinkRepo instances.
"""

from .models import DataSourceRM, SchemeRM, DatasetRM, CollectionRM, DSample, discover
from .factories import create_caster, create_collection, create_sink
