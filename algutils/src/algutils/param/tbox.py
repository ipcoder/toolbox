from __future__ import annotations

from typing import Iterable, Any

import pandas as pd
import pydantic
from box import Box

from ..datatools import complete_missing, UndefCond, UndefTypes
from ..strings import compact_repr, hash_str
from ..wrap import name_tuple

__all__ = ['TBox', 'UndefCond']

_intact_types = (tuple, list, pd.DataFrame, pd.Series)
UND_EMPTY = UndefCond(empty_dict=True)


class TBox(Box):
    _protected_keys = Box._protected_keys + [
        "_flatten",
        "_repr_json_",
        "diff",
        "issubset",
        "issuperset",
        "find_key",
        "remove",
        "discard",
        "hash_str"
    ]

    def __init__(self, *args: pydantic.BaseModel | dict | Iterable[tuple[str, Any]],
                 undef: UndefTypes = UND_EMPTY, **kw_box):
        """
        Initialize hierarchical dict-like structure.

        Note, that argument ``undef`` controlling behavior of ``setdefault``method
        may be provided in various forms, but is eventually converted into
        the most complete one: instance of ``UndefCond``:
         - Collection of undef objects
         - Callable -> bool
         - UndefCond (allows to control empty_dict)

        :param args: dicts to initialize with
        :param kw_box: either key=value pairs as a data items, or Box kw args
        :param undef: describes condition to consider a node undefined
        """
        kw_box.setdefault('box_intact_types', _intact_types)
        kw_box.setdefault('box_dots', True)  # dotted - default behaviour for TBox !

        if args:  # special conversions
            args = tuple(arg.dict() if isinstance(arg, pydantic.BaseModel) else arg for arg in args)

        if len(args) > 1:  # Box supports only 1 positional argument, so for more than that
            dict_arg = {}  # convert all the args into dicts and then merge them into one
            for arg in args:  # here we deal with dict and Iterable[tuple[k, v]]
                dict_arg.update(arg)
            args = (dict_arg,)

        super().__init__(*args, **kw_box)
        self._box_config['undef'] = undef if isinstance(undef, UndefCond) else UndefCond(undef=undef)

    def __setitem__(self, key, value):
        # ToDo risky - what about '1.32' or 'file.ext' or relative annotation ..
        if isinstance(key, str) and '.' in key:
            node, tail = key.split('.', 1)
            if node not in self:
                super().__setitem__(node, {})
            self[node].__setitem__(tail, value)
        else:
            super().__setitem__(key, value)

    def flatten(self):
        res = []
        for key, val in self.items():
            if isinstance(val, Box) and val:
                for subs, v in val.flatten():
                    res.append((f'{key}.{subs}', v))
            else:
                res.append((key, val))
        return res

    def _repr_json_(self):
        jbox = TBox(default_box=True)
        is_basic = lambda x: x is None or isinstance(x, (str, int, float))
        for k, v in self.items(True):
            jbox[k] = v if is_basic(v) or (
                    isinstance(v, (list, set, tuple))
                    and len(v) < 5 and all(map(is_basic, v))
            ) else compact_repr(v)
        return TBox(jbox)

    def __repr__(self):
        if not self: return f"{type(self).__name__}()"
        return '\n'.join(f'{k}: {compact_repr(v)}' for k, v in self.flatten())

    def hash_str(self, length: int = None):
        """Return hex hash string as unique id of the content.

        :param len: if provided limit string to this length (keep tail)
        """
        s = ''
        for k, v in sorted(self.items(True), key=lambda _: _[0]):
            if isinstance(v, list):
                v = [TBox(x).hash_str(length=1000)
                     if isinstance(x, dict) else x
                     for x in v]
            s += f"{k}:{str(v)}\n"

        return hash_str(s, length)

    def __copy__(self, **kws):
        old_kws = {k: v for k, v in self._box_config.items() if not k.startswith('_')}
        assert set(kws).issubset(old_kws), "copy only accepts Box standard arguments"
        old_kws.update(kws)
        return self.__class__(self.to_dict(), **old_kws)

    # def __deepcopy__(self, memodict={}):
    #     return self.__copy__()

    def copy(self, **kws):
        from copy import deepcopy
        return deepcopy(self)

    def diff(self, other: 'TBox'):
        """ Difference between two tree forms as tuple of
         - keys missing in the other,
         - keys missing in self,
         - keys of incomparable or not equal values
        :returns tuple(dif_keys, dif_values)

        """

        def not_same_value(k):
            try:
                v1, v2 = other[k], self[k]
                same = (v1 == v2)
                if type(same) is not bool:
                    import numpy as np
                    same = np.allclose(v1, v2)
            except Exception:
                same = False
            return not same

        other = other if isinstance(other, TBox) else TBox(other)
        my_keys, other_keys = set(self.keys(True)), set(other.keys(True))
        missing = my_keys.difference(other_keys)
        extra = other_keys.difference(my_keys)
        unequal = {*filter(not_same_value, my_keys.intersection(other_keys))}
        return name_tuple('TBoxDiffKeys', missing=missing, extra=extra, unequal=unequal)

    def to_yaml(self, filename=None, *, default_flow_style=False,
                encoding="utf-8", errors="strict", **yaml_kwargs):
        from ..filesproc import prepare_parent_folder, Path

        par = self.copy()  # type: TBox
        for k, v in par.items(True):
            if isinstance(v, Path):
                par[k] = str(v)
            if hasattr(v, '__qualname__'):
                par[k] = v.__qualname__

        if filename:
            prepare_parent_folder(filename)
            filename = str(filename)

        return Box.to_yaml(par, filename=filename, default_flow_style=default_flow_style,
                           encoding=encoding, errors=errors, **yaml_kwargs)

    def setdefault(self, key: str, default: dict | Any, *, undef: UndefTypes = None):
        """
        Include support for dotted (nested) keys for `key` arg,
        and dict-like (also nested) structure for default argument.

        If key is None - setting default for the entire tree, in which case
        default must be nested dict as well.

        May include check that mandatory fields (values set as TBox.REQUIRED
        in the default dict) are already defined in self, or raise ValueError.

        :param key: name (also nested) of the field or None for entire tree.
        :param default: a value to set to specified key
                        or dict (nested) if key points to a subtree
        :param undef: override TBox.undef
        :return: the value been set (or the (sub)tree if default dict is dict
        """
        if (sep := '.') in key:
            assert sep not in (key[0], key[-1]), f"Invalid dotted {key=}!"
            node = self  # Ensure dotted key node exists and not a leaf
            for k in key.split(sep):
                if not isinstance(node, dict):  # every next level must be a dict
                    raise KeyError(f'Position {k} in {key=} must be a node!')
                node = Box.setdefault(node, k, {})  # existed or just created

            if isinstance(node, dict) and not node:  # don't consider empty node as set!
                self[key] = default
                return self[key]

        default = TBox({key: default} if key else default)
        undef = getattr(self, '_box_config')['undef'] if undef is None else undef
        return complete_missing(self, default, undef=undef)

    def issubset(self, other: dict):
        if not isinstance(other, self.__class__):
            other = self.__class__(other)
        return set(self.keys(True)).issubset(other.keys(True))

    def issuperset(self, other: dict):
        if not isinstance(other, self.__class__):
            other = self.__class__(other)
        return set(other.keys(True)).issuperset(self.keys(True))

    def find_key(self, name, *, part='any', multi=False) -> str | list:
        """
        Find key(s) with deep (dotted) naming matching given name.

        Return full (dotted) key found or '' if not.

        If multiple matching keys are found return depending on *multi*:
         - list of them if ``True``
         - only the first if ``None``
         - raise ``KeyError`` if ``False``

        :param name: may include dots inside:  "x.y"
        :param part: key's part to match: ['any'] | 'start' | 'end'
        :param multi: allow multiple keys to return

        :return: a string or list of strings (keys found)
        """
        import re
        if not re.fullmatch(r'(\w+(\.\w+)?)+', name):
            return [] if multi else ''

        reg = re.compile({'any': rf'\b{name}\b',
                          'start': rf'^{name}\b',
                          'end': rf'\b{name}$'
                          }[part])

        found = [*filter(reg.search, self.keys(True))]
        if multi:
            return found
        elif len(found) > 1:
            raise KeyError(f"Multiple matches to {name=} been {found=}!")
        else:
            return found[0]

    def remove(self, keys: Iterable[str], *, strict=True):
        """Remove nodes addressed by the keys (dotted).
        :param keys: iterable over dotted keys to remove
        :param strict: if True - raise KeyError if any of keys is missing
                       otherwise - ignore and remove only found
        :return: a new TBox with keys removed
        """
        if isinstance(keys, str): keys = [keys]
        res = self.copy()
        for key in keys:
            try:
                del res[key]
            except KeyError as ex:
                if strict: raise ex
        return res

    def discard(self, keys: Iterable[str], *, strict=True):
        """Discard from this box nodes addressed by given keys (dotted).
        :param keys: iterable over dotted keys to remove
        :param strict: if True - raise KeyError if any of keys is missing
                       otherwise - ignore and remove only found
        :return: None
        """
        if isinstance(keys, str): keys = [keys]
        if not hasattr(keys, '__len__'):
            keys = [*keys]
        valid = [*filter(self.__contains__, keys)]
        if strict and len(valid) < len(keys):
            raise KeyError(f"Missing keys: {set(self.keys(True)).difference(valid)}")
        for key in valid:
            del self[key]
