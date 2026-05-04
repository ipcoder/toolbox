"""
Factory functions bridging resman-based resource models with datacast core.

These provide the name-based convenience constructors that were previously
built into DataCaster and DataCollection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

from iad.core import as_list, as_iter


def create_caster(name_or_dataset=None, *, source=None, scheme=None,
                  filters=None, transforms=None, sample=None,
                  labels=None, temp_cache=False, cache=True,
                  progress=None):
    """
    Create a DataCaster by resolving names through resman resource models.

    Supports the same calling conventions as the old ``DataCaster`` constructor::

        create_caster('ETH3D')
        create_caster(source='/data/ETH3D', scheme='ETH3D')
        create_caster(DatasetRM(source='/', scheme='*'))

    :returns: a fully-constructed DataCaster
    """
    from .models import DatasetRM
    from .datacast.caster import DataCaster

    config_args = dict(
        name=name_or_dataset, source=source, scheme=scheme,
        filters=filters, transforms=transforms, sample=sample,
        labels=labels,
    )
    ds = DatasetRM.from_config(config_args, undefined=False, ignore=True)

    sample_dict = None
    if ds.sample is not None:
        sample_dict = ds.sample.dict() if hasattr(ds.sample, 'dict') else ds.sample

    return DataCaster(
        name=ds.name,
        root=ds.source.root,
        search=ds.scheme.search,
        labels=(ds.scheme.labels or {}) | (ds.labels or {}),
        mappings=ds.scheme.mappings or {},
        reverse=ds.scheme.reverse,
        bundle=as_list(ds.scheme.bundle),
        filters=ds.filters if isinstance(ds.filters, dict) else None,
        sample=sample_dict,
        temp_cache=temp_cache,
        cache=cache,
        progress=progress,
    )


def create_collection(name_or_datasets=None, *,
                      datasets=None, label_datasets=None,
                      query=None, bundle=None,
                      unique=False, data=None, drop=None,
                      cache=None, temp_cache=None,
                      calc_cache=False, calc_cache_rel=True,
                      progress=False, description=None):
    """
    Create a DataCollection by resolving names through resman resource models.

    Supports the old ``DataCollection`` calling conventions::

        create_collection('KITTI')
        create_collection(datasets=['MID14S', 'FT3D'])
        create_collection('SmallQualityEval')

    :returns: a fully-constructed DataCollection
    """
    from .models import DatasetRM, CollectionRM
    from .datacast.collect import DataCollection
    from iad.core import drop_undef

    if datasets is None and isinstance(name_or_datasets, str):
        cfg = dict(name=name_or_datasets, label_datasets=label_datasets,
                   query=query, bundle=bundle, description=description)
        col = CollectionRM.from_config(cfg, ignore=True, undefined=False)
        casters = [create_caster(d) for d in as_iter(col.datasets)]
        name = col.name
        query = query or col.query
        bundle = bundle or (as_list(col.bundle) if col.bundle else None)
    else:
        raw = datasets or ([name_or_datasets] if name_or_datasets else [])
        casters = [
            create_caster(d) if not hasattr(d, 'cached_pipe') else d
            for d in as_iter(raw)
        ]
        name = name_or_datasets if isinstance(name_or_datasets, str) else None

    dc_kws = drop_undef(
        data=data, drop=drop, cache=cache, temp_cache=temp_cache,
        calc_cache=calc_cache, calc_cache_rel=calc_cache_rel,
    )
    return DataCollection(
        name=name, datasets=casters, query=query, bundle=bundle,
        unique=unique, progress=progress, description=description,
        **dc_kws,
    )


def create_sink(dataset_or_scheme=None, root=None, *,
                data='data', select=None, labels=None, create_dir=True):
    """
    Create a SinkRepo by resolving names through resman resource models.

    Supports the old ``SinkRepo`` calling conventions::

        create_sink('DatasetName')
        create_sink(DatasetRM('DS'), root='/output')
        create_sink(SchemeRM('pattern'), root='/output')

    :returns: a fully-constructed SinkRepo
    """
    from .models import SchemeRM, DatasetRM
    from .datacast.collect import SinkRepo

    if root is None and isinstance(dataset_or_scheme, str):
        dataset_or_scheme = DatasetRM(dataset_or_scheme)

    if isinstance(dataset_or_scheme, DatasetRM):
        root = root or dataset_or_scheme.source.root
        dataset_or_scheme = dataset_or_scheme.scheme

    scheme = SchemeRM(dataset_or_scheme) if not isinstance(dataset_or_scheme, SchemeRM) else dataset_or_scheme

    return SinkRepo(
        root=root,
        search=scheme.search,
        labels=dict(scheme.labels) if scheme.labels else {},
        mappings=dict(scheme.mappings) if scheme.mappings else {},
        scheme_name=scheme.name,
        scheme_description=scheme.description,
        data=data, select=select,
        extra_labels=labels, create_dir=create_dir,
    )
