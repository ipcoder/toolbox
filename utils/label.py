from __future__ import annotations

__all__ = ['Array', 'Keys']

from collections import namedtuple
from typing import Dict, Sequence, Any, Collection, NamedTuple, Iterable

import pandas as pd

from toolbox.utils import as_list
from toolbox.utils.nptools import Array
from toolbox.utils.strings import compact_repr


class Keys(pd.core.indexes.frozen.FrozenList):  # TODO: Merge Keys into Labeled?
    def __init__(self, *keys):
        n = len(keys)
        if n == 1 and not isinstance(keys[0], str):
            assert hasattr(keys[0], '__len__')
            keys = keys[0]
        super().__init__(keys)

    @classmethod
    def from_index(cls, index: pd.MultiIndex):
        return cls(index.names)

    def label(self, *vals, strict=False, **kvs):
        """Construct Labels object by assigning values to the keys and verifying consistency.
        Support two forms, explicit and implicit:
        ::
            keys.label(k1=v1, k2=v2)
            keys.label(v1, v2)
            keys.label([v1, v2])   # same as previous

        They are equivalent if k1, k2 in keys, but explicit for is preferable
        for visibility, and flexibility, as ``strict`` argument may relax
        requirements for all the kw args being from keys.

        Second form is useful when operating arrays of values.

        :param strict: if strict is True and extra keys are used raise KeyError.
            otherwise just checking that all the expected keys are assigned.
        :param vals: sequence of values matching by the order keys in self,
                    as multiple positional arguments, or one of collection type
        :param kvs: key: value pairs of the labels.
        :return: Labels built from the keys
        """
        if vals:
            if len(vals) == 1 and isinstance(vals[0], (list, tuple)):
                vals = vals[0]
            if kvs:
                raise ValueError("Keyword and values only arguments can't be mixed!")
            if len(vals) != len(self):
                raise IndexError("Mismatch in length of values and keys")
            kvs = dict(zip(self, vals))

        if dif := (set(self).symmetric_difference(kvs) if strict else set(self).difference(kvs)):
            raise KeyError(f"Mismatched keys: {dif} (valid: {self})")
        return Labels(kvs)

    def order_values(self, *, strict=False, fill_missing=None, **kws) -> tuple:
        """Create tuple of values of all the keys in their order.

        :param strict: if True rise KeyError if kws contain unknown keys
        :param fill_missing: value to fill missing values or an exception class to raise
        :param kws: keys-values to extract values from
        :return: tuple of values of all the keys in their order
        """
        if issubclass(fill_missing, BaseException) and (dif := set(kws).difference(self)):
            fill_missing(f"Unknown keys: {dif}")
        if strict and (dif := set(self).difference(kws)):
            KeyError(f"Missing keys to initialize {dif}")
        return tuple(kws.get(k, fill_missing) for k in self)

    def __eq__(self, other: Sequence):
        return not set(self).symmetric_difference(other)


class Labels(dict):
    """
    Labels container class for labeled data with basic arithmetics
    """

    def __getattr__(self, item):
        if item in self:
            return self[item]
        return self.get(item, super().__getattribute__(item))

    def __setattr__(self, key, value):
        if hasattr(self, key):
            self[key] = value
        else:
            super().__setattr__(key, value)

    # noinspection PyMissingConstructor
    def __init__(self, *args: dict | NamedTuple | Iterable[tuple[str, Any]], **kws: Dict[str | Any]):
        """
        Construct labels in different forms:

         - From mixed collection of dicts, named tuples and kws:
         ::
            Labels({'k':1}, named(x=2, z=4), (), a=10, b=20)
         - From

        :param args:
        :param kws:
        """
        for d in args:  # merge dict representations of different kinds
            self.update(
                d if isinstance(d, dict)
                else d._asdict() if hasattr(d, '_asdict')  # named tuple
                else d  # Iterator over (key, value) tuples
            )
        self.update(kws)

    def __repr__(self):
        sep = '@#'
        items = sorted(self.items(), key=lambda kv: kv[0])
        s = sep.join(f'{k}: {compact_repr(v)}' for k, v in items)
        s = f"<{s.replace(sep, ', ')}>" if len(s) < 80 else s.replace(sep, '\n')
        return s

    __repr__.is_compact = True

    def drop(self, keys: str | Collection[str], strict=False):
        """Remove given key(s).

        :param keys: keys to remove
        :param strict: fail if requested key(s) not exist
        """
        keys = as_list(keys, collect=set)
        if strict and (missing := keys.difference(self)):
            raise ValueError(f"Some of the requested keys are {missing = }")
        return self.__class__((k, v) for k, v in self.items() if k not in keys)

    def select(self, *keys, strict=True):
        """
        Return subset of items in the given order.
        :param keys: keys to select the items by
        :param strict: raise KeyError if requested key does not exist
        :return:
        """
        return self.__class__(
            ((k, self[k]) for k in keys)
            if strict else
            ((k, self[k]) for k in keys if k in self)
        )

    def _from_set_operation(self, operation, other):
        if isinstance(other, str):
            other = {other}
        elif hasattr(other, '_fields'):
            other = other._fields
        elif not hasattr(other, '__iter__'):
            raise TypeError("Labels expect iterable or str or namedtuple")
        return self.__class__((k, self[k]) for k in operation(set(self), other))

    def __add__(self, other: dict | NamedTuple):
        return self.__class__(self, other)

    def __sub__(self, other: NamedTuple | Iterable | str):
        return self._from_set_operation(set.difference, other)

    def __and__(self, other: Iterable | NamedTuple | str):
        return self._from_set_operation(set.intersection, other)

    def __xor__(self, other: Iterable | NamedTuple | str):
        return self._from_set_operation(set.symmetric_difference, other)

    def __or__(self, other: dict):
        """lb2 = lb | dict"""
        return self.__class__(dict.__or__(self, other))

    def __ior__(self, other: dict):
        """lb |= dict"""
        return dict.__ior__(self, other)

    def to_keys(self):
        return Keys(*self)

    @property
    def tuple(self):
        return tuple(self.values())

    @property
    def namedtuple(self):
        return namedtuple('LabelsValues', self)(*self.values())

    def to_index(self):
        """Convert labels into a single row of ``MultiIndex``.

        If you need it for indexing, consider using ``tuple`` instead.
        """
        return pd.MultiIndex.from_tuples([tuple(self.values())], names=self)

    @classmethod
    def to_indices(cls, labels: list[Labels]):
        return pd.MultiIndex.from_tuples(map(lambda x: x.values(), labels), names=labels[0])

    @classmethod
    def from_frame(cls, df, *, index=True, data=True, squeeze=False) -> Labels | list[Labels]:
        """
        Convert DataFrame into list of Labels, with control over
        which data columns and levels of the index to use

        :param df: DataFrame or Series
        :param index: True to use all the levels of the Multi-index,
                 or specify required (False for none)
        :param data: True to use all the columns, or specify required
        :param squeeze: if list of labels contains only one Labels obj - return it
        :return: list of Labels or Labels correspondingly
        """
        if isinstance(df, pd.Series):
            df = df.to_frame()
            if isinstance(df.columns, pd.MultiIndex):
                df = df.T  # series was inverted

        data = [] if data is False else df.columns if data is True \
            else [data] if isinstance(data, str) else data
        df = df[data]
        if isinstance(index, bool):
            df = df.reset_index(drop=not index)
        else:
            df.reset_index(index).reset_index(drop=True)
        labels = [*map(cls, df.to_dict('index').values())]
        if squeeze and len(labels) == 1:
            return labels[0]
        return labels

    @classmethod
    def from_index(cls, index: pd.MultiIndex) -> list[Labels]:
        """
        Convert MultiIndex into Labels
        :param index: Multi-index to convert from
        :return: list of Labels
        """
        return Labels.from_frame(index.to_frame(index=False), index=False)

    def copy(self) -> Labels:
        return Labels(self)
