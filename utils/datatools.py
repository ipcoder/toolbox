"""
Data structures manipulation tools
"""
from __future__ import annotations

from typing import Dict, Tuple, Collection, Union, Iterable, \
    Mapping, Any, Generator, Sequence, Callable, Literal, List, Type, TYPE_CHECKING

from .codetools import NamedObj

if TYPE_CHECKING:
    import re

DICTS = Union[Collection[dict], Dict[str, dict]]


def transpose(seq: Sequence[Any | Sequence[Any]], cols: int = None) -> list:
    """Rearrange sequence as in transposition of a matrix with cols
    composed of its elements, and return list of relocated elements.

    Supports 1d AND 2d cases:
        - 1D rearranges elements as if there were 2D, ``cols`` must divide ``len(seq)``
          ::
            seq[N] -> m[N/cols, cols].T -> seq[N]
        - 2D must be a sequence of sequences of equal length ``cols``

    Examples:

    >>> transpose([1,2,3,4,5,6], 3)
    [1, 4, 2, 5, 3, 6]

    >>> transpose([[1,2], [3,4], [5,6]])
    [[1, 3, 5], [2, 4, 6]]
    """

    if cols is None:
        cols = len(seq[0])
        return [list(r[i] for r in seq) for i in range(cols)]
    else:
        return [seq[i] for ofs in range(cols) for i in range(ofs, len(seq), cols)]


UndefTypes = Union['UndefCond', Collection, Callable[[Any], bool], Literal[None]]
UNDEF = NamedObj('UNDEF')


class UndefCond:
    """
    Contains condition of an item to be considered undefined.
    Instances are callable and implement this check.
    """

    def __init__(self, undef: UndefTypes = None, empty_dict=False):
        """
        May be initialized also by another UndefCond instance or
        a callable returning True if its argument should be considered undefined.

        In those cases ``empty_dict`` argument is ignored.

        If any element of provided collection IS a tested object, it is declared undefined.

        In addition, if empty_dict is True, then any empty object of dict subclass is undefined.

        An object which IS `UndefCond.UNDEF` is always considered undefined.
        """
        if isinstance(undef, UndefCond):
            self.undef = undef.undef
            self.empty_dict = undef.empty_dict
        elif undef is None:
            self.undef = ()
            self.empty_dict = empty_dict
        elif hasattr(undef, '__contains__'):
            self.undef = undef
            self.empty_dict = empty_dict
        elif isinstance(undef, Callable):
            self.undef = None
            self.empty_dict = None
            setattr(self, '__call__', undef)
        else:
            raise TypeError(f"Unsupported undef argument type f{type(undef)}")

    def __repr__(self):
        if self.empty_dict is None:  # constructed by callable
            rep = f"func<{self.__call__.__qualname__}>"
        else:
            rep = [f"{self.undef}"] if self.undef else []
            if self.empty_dict: rep.append("{}")
            rep = ', '.join(rep)
        return f"UndefCond({rep})"

    def __call__(self, x):
        return (x is UNDEF or x in self.undef or
                self.empty_dict and isinstance(x, dict) and not x)


def complete_missing(tree: dict, other: dict, *,
                     undef: Collection | Callable[[Any], bool] | UndefCond = None):
    """
    Complete missing elements in dict hierarchy from another tree,
    leaving values of existing leaves unchanged.

    :param tree: tree to complete with extra elements found in the other
    :param other: to take new elements from
    :param undef: definition of what constitutes an undefined object:
            None, a collection of such objects, or callable to performa


    :return: tree of all the new nodes added to the tree
    """
    undef = undef if isinstance(undef, UndefCond) else UndefCond(undef)

    new = type(tree)()  # new nodes added
    for k, ov in other.items():
        tv = dict.get(tree, k, UNDEF)
        # all the cases when tv is considered UNDEF
        if undef(tv):
            tree[k] = ov
        elif isinstance(tv, dict) and isinstance(ov, dict):
            ov = complete_missing(tv, ov, undef=undef)
        else:
            continue  # tv is defined and not a recursive 2 dicts case
        new[k] = ov
    return new


def common_dict(dicts: DICTS, unique=False
                ) -> Union[dict, Tuple[dict, DICTS]]:
    """
    Given a collection of dicts find a common sub-dictionary.
    Optionally return also list of remaining unique sub-dictionaries.

    :param dicts:  sequence or dictionary of dictionaries to select from
    :param unique: if True return also the unique sub-dictionaries
        as collection or dictionary, depending on the input
    :return: common | common, unique
    """
    from toolz import itemfilter, reduce, comp

    keys = None
    if isinstance(dicts, dict):
        keys, dicts = dicts.keys(), [*dicts.values()]

    for d in dicts:  # convert all lists to tuple making them hashable
        transform_node(d, lambda _: isinstance(_, list), 'value', lambda _: tuple(_), 'value')

    items_sets = map(lambda d: set(d.items()), dicts)
    common = reduce(set.intersection, items_sets)
    if unique:
        def remove_common(dct):
            return itemfilter(lambda it: it not in common, dct)

        uniques = map(comp(dict, remove_common), dicts)
        return dict(common), list(uniques) if keys is None else dict(zip(keys, uniques))
    return dict(common)


def zip_dict(*dicts, fillvalue=None, keys=None, skip=False, strict=False):
    """
    zip dictionaries with same keys into dict of tuples of the corresponding values.
    :param dicts: dicts to zip
    :param fillvalue: fill with this value if key is missing in some dictionary (unless strict is True!)
    :param keys: None | <iterable> - all the keys in the dicts OR use keys from specifically provided iterable
    :param skip: True|False
    :param strict: [False]|True - raise KeyError for any missing keys (otherwise use fillvalue)
    :return: Dict[Key: Tuple]
    """
    if keys is None:
        keys = {k for d in dicts for k in d}  # all the keys found in all the dicts

    def get(d, k):  # return key's value or fillvalue or raise Exception - depending on the settings
        return d[k] if strict else d.get(k, fillvalue)

    def available(k):  # check if the key is available in all the dicts
        return all((k in d) for d in dicts)

    return {k: tuple(get(d, k) for d in dicts) for k in keys if not skip or available(k)}


def split_sync_iter(itr: Iterable[Mapping[int, Any]], splitter=None, n=None) -> Tuple[Generator, ...]:
    """
    Given an iterable over items which are sequences of n elements, return
    n synchronized iterators over elements by their order (like transpose).

    Implementation note
    -------------------
    Since each iterator can be advanced independently, some of them could
    be consumed faster than the others. The source iterator is therefore
    advancing with the fastest, and internal buffers are used to keep
    unconsumed elements of slow iterators - thus the iterators are synced.

    :param itr: iterable over sequences of items of same length
    :param splitter: optional function to split the item into elements
                     default assumes item is an iterator over elements
    :param n: optional number of output channels,
              otherwise determined automatically by peeking
    :return: tuple with n iterators
    """
    from collections import deque
    from toolz import peek
    split = lambda x: splitter(x) if splitter else x

    if not n:
        sample, itr = peek(itr)
        n = len(split(sample))
    qs = [deque() for _ in range(n)]

    def _next(i):
        nonlocal qs
        if not qs[i]:
            items = split(next(itr))
            assert len(items) == n
            for q, item in zip(qs, items):
                q.appendleft(item)
        return qs[i].pop()

    def iterator(i):
        while True:
            yield _next(i)

    return tuple(map(iterator, range(len(qs))))


def unzip_dict(d):
    """
    Transpose dict of arrays into array of dict:

    Example:
        unzip_dict({'x': [1, 2, 3], 'y': [10, 20, 30]}) \
        ==
        [{'x': 1, 'y': 10}, {'x': 2, 'y': 20}, {'x': 3, 'y': 30}]

    :param d:  dictionary of arrays of same length
    :return: array of dicts of the same length
    """
    return [dict(zip(d.keys(), vals)) for vals in zip(*d.values())]


def merge_update(trg: dict, src: dict, *, copy=True,
                 conflict: Literal['error', 'replace', 'ignore'] = 'error', verb=False) -> dict:
    """
    Merge a src dict into the target by updating its values.
    ::
        merge_update(trg, src) is trg

    Make a copy to keep the original intact:
    ::
        new = merge_update(trg.copy(), src)

    Possible conflicts can be handles as instructed by control argument
    :param trg: dict to merge into
    :param src: dict to merge
    :param conflict: error | replace | ignore
                - error: raise KeyError
                - replace: use from src
                - ignore: - leave the old
    :param copy: copy nodes instead of referencing,
                 also may be a copy function
    :param verb: be verbose - warn if replace or ignore happens

    :return: updated version of the input dict.
    """
    from warnings import warn
    if not copy:
        copy = lambda x: x
    elif copy is True:
        from copy import copy

    for k, v in src.items():
        if k in trg and trg[k] is not None:
            if hasattr(v, 'keys'):
                merge_update(trg[k], v, conflict=conflict, copy=copy)
            else:
                if conflict == 'error':
                    raise KeyError(f'Conflict for key {k} ({trg[k]} vs  {v})')
                if conflict == 'replace':
                    trg[k] = copy(v)
                elif conflict != 'ignore':
                    raise ValueError(f'Unsupported conflict argument value: {conflict}')

                not verb or warn(f'Conflict resolved by "{conflict}" for key {k} ({trg[k]} vs  {v})')
        else:
            trg[k] = copy(v)
    return trg  # TODO: reconsider copy mechanism


def split_dict(d, cond: Callable):
    """
    Given dict and condition `function(key, val)` return two dicts:
    with items meeting conditions and the rest.

    :param d:
    :param cond:
    :return: cond_true_dict, cond_false_dict
    """
    pos, neg = {}, {}
    for k, v in d.items():
        (pos if cond(k, v) else neg)[k] = v
    return pos, neg


def rm_keys(d: dict, keys: str | re.Pattern | Iterable[str | re.Pattern],
            *, strict=False):
    """
    remove keys from dict and return it (Make copy before passing if needed!)
    :param d: dict
    :param keys: a scalar key or Iterable over keys
    :param strict: fails if a key required to be removed is not in ``d``
    :return: dict
    """

    regs, keys_to_rm = [], []
    for k in keys:  # separate regular expressions from the keys to remove
        (regs if not isinstance(k, str) or '*' in k or '?' in k else keys_to_rm).append(k)

    if regs:  # extend by the dict keys matching a regular expression
        from .regexp import filter_regex_matches
        keys_to_rm.extend(filter_regex_matches(regs, d))

    for k in keys_to_rm:
        d.pop(k) if strict else d.pop(k, None)
    return d


def drop_item(seq: Sequence, pos: int):
    """Drop item at given position in the sequence, and return list without it.

    :param seq: a list to drop item from
    :param pos: index of item to drop (could be negative to count from the end)

    Example:
    >>> drop_item([1,2,3], 1)
    [1, 3]
    >>> drop_item([1,2,3], -1)
    [1, 2]
    """
    n = len(seq)
    pos = n + pos if pos < 0 else pos
    return [a for i, a in enumerate(seq) if i != pos]


def unique(seq: Iterable, exclude: Iterable = ()) -> Generator:
    """From the given iterable over hashable items
    produce iterator with all the repeated occupancies filtered out.

    Additional items to exlcude may be optinally provided.

    Example:

    >>> list(unique([1, 2, 1, 'cat', 3, 'cat', 'son', 3])) == [1, 2, 'cat', 3, 'son']
    True
    >>> list(unique([1, 0, 1, 3, 0, 1, None], exclude=[None, 0])) == [1, 3]
    True
    """
    seen = set(exclude)
    for x in seq:
        if x in seen: continue
        seen.add(x)
        yield x


def recurring(items: Iterable) -> Generator:
    """
    Produce generator of recurring elements from the given iterable over hashable items
    :param items: hashable
    :return: generator iterating over the recurring items
    """
    found = set()
    for x in items:
        if x in found:
            yield x
        found.add(x)


def map_by_type(values: Collection, types: dict[str, type]):
    """
    Given values and mapping {name: type} create mapping {name: value}
    by establishing value -> type correspondence with isinstance(value, type).

    Requires uniqueness of types in the mapping.

    :param values: a collection of values
    :param types: mapping name -> type

    :return: mapping name -> value for every value from values

    :raises: TypeError if types are not unique or type of the value not found
    """
    names = {t: n for t, n in types.items()}  # invert dictionary
    if len(names) < len(types):
        raise TypeError("Not unique types in the name -> type mapping")

    assigned = {}
    for v in values:  # guess name by its type
        for n, t in types.items():  # by checking all the types
            if isinstance(v, t):
                assigned[n] = v
                break
        else:
            raise TypeError(f"Unexpected type {type(v)} of element {v}")
    return assigned


def select_from(namespace: dict, names: Iterable, strict=True,
                default=UNDEF, factory: Callable[[str], Any] | None = None) -> dict[str, Any]:
    """
    Selects given names from given namespace (dict)
    :param namespace: dictionary like
    :param names: names of the variables to return
    :param strict: if ``True`` names must be in namespace or raise ``KeyError``.
                   if ``False``, use only available names from the namespace
    :param default: if `strict` is ``True`` and `default` is defined,
                    this value will be returned for not found keys.
    :param factory: *instead* of ``default`` can provide function creating default from name
    :return: dict with selected items
    """

    if strict:
        if default is UNDEF:
            if factory is None:
                return {k: namespace[k] for k in names}
            else:
                return {k: factory(k) if (v := namespace.get(k, UNDEF)) is UNDEF else v
                        for k in names}
        else:
            assert factory is None
            return {k: namespace.get(k, default) for k in names}
    else:
        assert default is UNDEF and factory is None
        return {k: namespace[k] for k in names if k in namespace}


def all_satisfied(conditions: List[Callable[[Any], bool]]) -> Callable[[Any], bool]:
    """From collection of condition testing functions ``cond(x): bool``
    compose a function which returns True if ALL the tests are passed

    :param conditions: boolean Callables
    :return: boolean function which tests ALL the conditions
    """

    def func(v):
        return all(map(lambda c: c(v), conditions))

    return func


def to_dict(d: dict) -> dict:
    """Translate dict-like nodes of hierarchical object into pure dict.

    :param d: The object, possibly a combination of dict and :class:`Box`.
    :returns: it's modified self
    """
    if hasattr(d, 'items'):
        if hasattr(d, 'to_dict'):
            return d.to_dict()
        else:
            try:
                return dict(d)
            except:
                pass
        for k, v in d.items():
            d[k] = to_dict(v)
    return d


def transform_node(dct: dict,
                   condition: Callable[[Any], bool], condition_on: Literal['key', 'value'],
                   transformation: Callable[[Any], Any], transformation_on: Literal['key', 'value']):
    """
    Recursively transforming dictionary singular nodes given desired condition.
    Those transformation, as mentioned, applies on non dictionary nodes.

    For instance:

    Value Transformation:
    >>> d = {'a': 4}
    >>> inc = lambda _: _.__add__(1)
    >>> dd = transform_node(d, lambda _: isinstance(_, int), condition_on='value',
    ...                     transformation=inc, transformation_on='value')
    This results in:
    >>> dd = {'a': 5}

    Key Transformation

    >>> d = {'a': 4}
    >>> change_name = lambda _: str.__add__(_, 'wesome')
    >>> dd = transform_node(d, lambda _: isinstance(_, str), condition_on='key',
    ...                     transformation=change_name, transformation_on='key')
    This results in:
    >>> dd = {'awesome': 4}

    The transformation can be done either on the key or value.
    The function does NOT do anything but calling the condition and transformation passed.
    The responsibility of compatibility between the operation and the data is on the user.

    :param dct: The given dictionary.
    :param condition: A condition to apply.
    :param condition_on: Whom the condition applied on.
    :param transformation: A transformation to apply.
    :param transformation_on: Whom the tranformation applied on.
    :return: New dictionary after the transformation.
    """
    _dct = dict()
    on_key = transformation_on == 'key'
    for k, v in dct.items():
        if not v:
            dct[k] = None
        else:
            if isinstance(v, dict):
                v = transform_node(v, condition, condition_on, transformation, transformation_on)
            if condition(k if on_key else v):
                _dct[transformation(k) if on_key else k] = v if on_key else transformation(v)
            else:
                _dct[k] = v

    return _dct


def issubset_report(a: Collection, b: Collection,
                    on_empty: Type[Exception] | Callable = None,
                    on_diff: Type[Exception] | Callable = None) -> bool:
    """
    Checks whether two input collections contain same set of hashable elements
    and optionally report on certain cases using mechanism provided by
    the corresponding argument:
     - calling it if its values is ``Callable``
     - raising it if it is an ``Exception`` type

    :param a: First collection.
    :param b: Second collection.
    :param on_empty: report type if both are empty (return ``True`` if not raises)
    :param on_diff: report kind if difference is found (return ``False`` if not raised)

    :return: ``False`` if collections contain different sets, otherwise ``True``
    """

    def report(s, rep):
        if callable(rep):
            rep(s)
        elif isinstance(rep, type) and issubclass(rep, Exception):
            raise rep(s)

    if not (a or b):
        report("Both collections are empty", on_empty)
    elif differences := set(a).symmetric_difference(b):
        report(f"There are {differences=} between {a} and {b}", on_diff)
        return False
    return True


class Filter:
    """
    Class allows to define a composition of conditions to be checked for a
    Mapping objects (labels), expressed in terms of its keys.

    Its main functionality is represented by __call__ method to evaluate
    given labels object.

    Conditions must be provided either as a
        - single object, ``Callable[[Dict], bool]`` or expression string

          >>> f1 = Filter('x > 2 * y')
          >>> def func(d) -> bool:
          ...    return d['x'] > 2 * d['y']
          >>> f2 = Filter(func)   # same functionality as f1
          >>> labels = dict(x=2, y=3, z=5)
          >>> assert f1(labels) == f2(labels)

        - dictionary, with string keys, which either

           - starts with 'condition' with values as in single object case, or
           - a valid label key (found in labels to be filtered).
             In this case values represent expected label values,

             either a `scalar` (including ``str``), or collection of scallars.

             Mappings are used only as a set of their keys!
    Example
    -------
    >>> filters = dict(
    ...     conditions = 'height * 2 < width - 10',     # 1. general condition
    ...     name = ['Sam', 'David'],                    # 2. allowed values
    ...     age = int(18).__le__,    # 18 <= age        # 3. callable
    ...     side = 'right'                              # 4. allowed scalar value
    ... )
    >>> cond = Filter(filters)
    >>> cond_keys = {'height', 'width', 'name', 'age', 'side'}
    >>> labels = dict(height=10, width=4, name='Nick', age=100, side='top')
    >>> assert cond_keys.issubset(labels.keys())
    >>> assert cond(labels) is False
    >>>
    >>> from pandas import DataFrame
    >>> df = DataFrame([
    ...     dict(height=10, width=4, name='Nick', age=100, side='top'),
    ...     dict(height=10, width=40, name='Sam', age=10, side='right'),
    ...     dict(height=10, width=35, name='David', age=20, side='right')
    ... ])
    >>> assert cond_keys.issubset(df.columns)
    >>> assert df[df.apply(cond, axis=1)].index.item() == 2
    >>>
    """

    def __init__(self, filters: str | dict[str, str | Collection | Any], strict=True):
        """
        Create filter object from collection of filtering conditions to
        be joined by AND operator.

        Conditions are represented as dict items with keys as:
            1. labels categories, with values as   (key-conditin)
                * allowed value
                * allowed collection of values
            2. string started with "condition*", than the value is  (general condition)
                * string with python expression in term of labels categories evaluated to bool
                * a callable function recieving labels as keyword arguments

        :param filters: single condition or dict of them
        :param strict: fail if labels during the filtering don't include catagories used in key
        """
        self.strict = strict

        def valid_condition(cond):
            if isinstance(cond, str):
                from toolbox.utils.fnctools import express_to_kw_func
                return express_to_kw_func(cond)
            if isinstance(cond, Callable):
                return cond
            raise TypeError(f"Invalid condition type {type(cond)}, expected Callable or str!")

        self._filters = filters
        self.general_conditions = {}
        self.key_conditions = {}
        if filters:
            if isinstance(filters, (str, Callable)):
                filters = {'condition': filters}
            assert isinstance(filters, dict)

            for key, val in filters.items():
                if key.startswith('condition'):
                    self.general_conditions[key] = valid_condition(val)
                else:
                    if isinstance(val, Callable):
                        condition = val
                    elif isinstance(val, Collection) and not isinstance(val, str):
                        allowed = set(val)
                        condition = allowed.__contains__
                    else:  # a scalar
                        condition = lambda x: x == val
                    self.key_conditions[key] = condition

    def __call__(self, labels: dict) -> bool:
        """
        Return True if labels match the filters
        :param labels: a mapping with all filters arguments in its keys
        :return: True if ALL the conditions met
        """
        UND = object()
        return all(cond(labels) for cond in self.general_conditions.values()) and \
            all(cond(v) for k, cond in self.key_conditions.items()
                if (v := labels.get(k, UND)) is not UND or self.strict is False)

    def detail_conditions(self, labels):
        """Return dictionary of conditions results per condition.
        Mainly for debugging purposes.
        """
        return {**{k: cond(labels) for k, cond in self.general_conditions.items()},
                **{k: cond(labels[k]) for k, cond in self.key_conditions.items()}}

    def __repr__(self):
        lst = lambda s: '\n' + ', '.join(s) if s else ''
        return f"{self.__class__.__name__} conditions:" \
               f"{lst(self.general_conditions)}" \
               f"{lst(self.key_conditions)}"
