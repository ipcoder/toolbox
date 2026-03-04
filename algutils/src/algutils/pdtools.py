from __future__ import annotations

import logging
import re
import warnings
from collections import namedtuple
from dataclasses import dataclass
from functools import cmp_to_key
from io import StringIO
from numbers import Number
from pathlib import PurePath
from typing import Union, Collection, Iterable, Sequence, Any, Callable, Literal, Generator, TYPE_CHECKING

if TYPE_CHECKING:
    from .label import Labels

import numpy as np
import pandas as pd
from IPython import display as ipy_disp
from pandas.core.dtypes.common import is_list_like

from . import strings as stt, codetools as cdt, nptools as npt
from . import wrap
from .datatools import select_from, issubset_report
from .short import as_list, unless_subset

# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
# Copy-on-Write will become the new default in pandas 3.0. This means than chained indexing will never work.
# As a consequence, the SettingWithCopyWarning won’t be necessary anymore. See this section for more context.
# We recommend turning Copy-on-Write on to leverage the improvements with:
pd.options.mode.copy_on_write = True
pd.options.display.precision = 3
pd.options.display.float_format = '{:.3}'.format
pd.options.display.max_colwidth = 40
pd.options.display.width = 120

TCol = Union[str, tuple[str]]
PTable = Union[pd.DataFrame, pd.Series]  # pandas table-like types
DTable = Union['DataSeries', 'DataTable']  # derived in this module Data* Table like types
StringS = Union[Sequence[str], str]

AxisID = Literal[0, 1, 'index', 'columns']
AxisT = Union[AxisID, Collection[AxisID]]

_ALL = slice(None)
NA = cdt.NamedObj('NA')


def toy_table(rows: int = 4, cols: int | Collection[str] = 2, index: int | Collection[str] = 2):
    """
    Create a sample DataTable for experiments
    :param rows: number of rows
    :param cols: number of columns or their names
    :param index: number of index levels or their names
    :return: DataTable
    """
    if isinstance(index, int):
        index = [f"i{_}" for _ in range(index)]

    if isinstance(cols, int):
        cols = [chr(ord('a') + i) for i in range(cols)]

    data = [{c: 10 * i + j for i, c in enumerate(as_list(index) + as_list(cols))}
            for j in range(rows)]

    data = DataTable(data)
    return data.set_index(index) if index else data


@dataclass
class Parallel:
    """
    Parallelization for DataFrame processing (apply) supporting different
    methods (packages): swifter, joblib, no parallel (pandas)

    Objects are initialized by the parallelization parameters:
    """
    swift: bool = False
    jobs: int = False
    pbar: bool = True
    strict: bool = False

    methods = dict(swifter='swift', joblib='jobs')

    def __post_init__(self):
        from importlib import import_module

        def check(method):
            par = getattr(self, name := self.methods[method])
            if par:
                if not isinstance(par, (int, bool)):
                    raise ValueError(f'Field {name} must be bool or int')
                try:
                    import_module(method)
                except ModuleNotFoundError as e:
                    if self.strict:
                        raise e
                    setattr(self, name, False)
                    logging.getLogger().warning(
                        f'Package `{e.name}` is missing, trying another option')
                return 1
            return 0

        if sum(map(check, self.methods)) > 1 and self.strict:
            raise RuntimeError('Only one parallelization method must be selected')

    @classmethod
    def from_flag(cls, parallel: Parallel | str | bool):
        """Create Parallel object from flag if it's
        either method's name or True or threads number.
        Default method is swifter.

        If ``not bool(parallel)`` return ``None``

        If ``isinstance(parallel,  Parallel)`` return ``parallel``

        Otherwise, raise ValueError.
        """
        if not parallel:
            return None
        if isinstance(parallel, Parallel):
            return parallel
        if isinstance(parallel, (bool, int)):
            return Parallel(swift=parallel, jobs=True, strict=False)
        if isinstance(parallel, str):
            if parallel not in cls.methods.values():
                raise ValueError(f"Unsupported parallel method {parallel}")
            return cls(**{parallel: True})
        raise ValueError("Invalid parallel request")

    def __bool__(self):
        return any(self.methods.values())

    @staticmethod
    def applicable(row_fnc):
        def fnc_to_apply(row):
            """prepare wrapped function to be used in apply"""
            if isinstance(row, pd.DataFrame):  # refuse attempt to process multiple rows
                raise TypeError('Signal of no vectorization support (to swifter)')
            res = row_fnc(row)
            if isinstance(res, (dict, Sequence)):
                return pd.Series(res)
            return res

        return fnc_to_apply

    def swift_apply(self, df: pd.DataFrame, func: Callable[[pd.Series, ...], pd.Series],
                    args: tuple = (), **kws):
        """
        Parallel apply provided function on the rows of the given DataFrame using swifter package.

        Function must:
            1. receive Series as first argument
            2. return Series, even if a single value calculated per row
            3. rise in attempt to provide a DataFrame (deny vectorization support).

        Such function can be constructed from a function returning dict or sequence
        using ``applicable`` decorator of the ``Parallel`` class.

        :param df:
        :param func:
        :param args: tuple with additional func arguments
        :param kws: DataFrame.apply AND func keyword arguments
        :return: table with results
        """
        return (df
                .reset_index(drop=True)  # Dask does not support MultiIndex Dataframes.
                .swifter.allow_dask_on_strings()  # activate swifter
                .set_npartitions(self.swift is True and None or self.swift)  # default is CPUS * 2
                .set_dask_threshold(0).set_dask_scheduler('threads')
                .apply(func, args=args, **kws)  # apply the function
                .set_index(df.index))

    def jobs_apply(self, df: pd.DataFrame,
                   func: Callable[[pd.Series, ...], dict | Sequence], args=(), **kws):
        """
        Implements parallel apply of per-row function on a ``DataFrame`` using ``joblib`` package.
        Supplied ``row_func`` must be able to receive a row (pd.Series) as a first argument,
        and return either a dict (to produce multiple columns) or a Sequence with one element.

        :param df:  DataFrame object to apply on
        :param func: function to be applied on a single row
        :param args: optional additional arguments to the row_func
        :param kws: optional keyword arguments to the row_func
        :return: resulted table (same class as df)
        """
        import joblib as jb
        from .events import tqdm_joblib

        par = self.jobs if isinstance(self.jobs, dict) else dict(
            backend='loky',
            batch_size='auto',
            n_jobs=self.jobs is True and 16 or self.jobs,
            verbose=0
        )
        with tqdm_joblib(desc="Processing rows in parallel", total=len(df), disable=not self.pbar):
            res = jb.Parallel(**par)(jb.delayed(func)(it[1], *args, **kws) for it in df.iterrows())
        return df.__class__.from_records(res, index=df.index)

    def apply(self, df: pd.DataFrame, func: Callable[[pd.Series, ...], dict | Sequence],
              axis=1, result_type='expand', args=(), **kws):
        """
        Parallel apply of per-row function on a ``DataFrame`` using active method.
        Supplied ``row_func`` must be able to receive a row (pd.Series) as a first argument,
        and return either a dict (to produce multiple columns) or a Sequence with one element.

        :param df:  DataFrame object to apply on
        :param func: function to be applied on a single row
        :param args: tuple with optional additional arguments to the row_func
        :param kws: optional keyword arguments to the row_func
        :param axis: must be 1!
        :param result_type: must be 'expand'!
        :return: resulted table (same class as df)
        """
        if not axis == 1 and result_type == 'expand':
            raise NotImplementedError("Currently only support: axis=1, result_type='expand'")

        if self.jobs:
            return self.jobs_apply(df, func, *args, *kws)

        opt = dict(func=self.applicable(func), axis=1, result_type='expand', args=args, **kws)
        if self.swift:  # jobs and swift has been verified above to be mutually exclusive
            return self.swift_apply(df, **opt)
        return df.apply(**opt)


def col_args_row_func(fnc, *, pos: StringS = None, kwcol: StringS = None,
                      alias: dict[str, str] = None, out: StringS = True,
                      cols: StringS = None):
    """
    Wraps given function ``fnc`` accepting regular positional and keyword arguments
    into ``row_fnc`` accepting dict-like object (Series) and redicts its fields
    into ``fnc`` arguments according to the mappings directives defined by
    ``pos``, ``kwcol`` and ``alias``.

    **Warning**: if no columns -> arguments correspondence are provided ALL the columns
    are passed as positional arguments!

    Also outputs of the ``fnc`` may be named (or renamed if its dict) using strings in ``out``,
    (must match amount of the outputs).

    Resulted function may be applied to an arbitrary DataFrames with given all the
    columns referred in the arguments are available.
    (That can be verified if optional ``cols`` argument provides list of columns names)

    **Note**! Output of the ``DataFrame.apply`` with ``result_type='expand'`` depends
    on the type of the ``row_fnc`` output:
        - ``DataFrame`` if it's a ``dict`` with columns named as its keys
        - ``DataFrame`` with enumerated columns if it's a ``list``
        - ``Series`` if it's a scalar (other objects including numpy array)
    So, to make ``apply`` produce a DataFrame even if ``fnc`` returns a scalar,
    provide ``out=name`` and ``row_fnc`` will return ``{name: scalar}`` instead.

    Default ``out=True`` will automatically generate ``{fnc.__name__: scalar}``
    IFF ``fnc`` produces scalars.

    To avoid output manipulation set ``out`` to ``True`` or ``None``.

    :param fnc: function to apply, may return either a scalar or dict or tuple
    :param pos: list of columns names to use as positional arguments
    :param kwcol: fnc keyword arguments which are same as columns names
    :param alias: mapping from columns corresponding fnc kw args {col: kw}
    :param out: if list (type) is passed ensures "vectorized" output even if
                scalar (anything but dict, list, tuple)
                results of fnc "name(s) of the resulted columns. If function returns a scalar and
            out is not defined, apply will return a Series!
    :param cols: Optional list of columns names expected in the DataFrame to apply to
                 for consistency validation.

    :return: row_fnc(row: dict, *args, **kws) -> dict | ...
    """
    pos, kwcol, out = map(as_list, [pos, kwcol, out])
    alias = alias or {}

    groups = [pos, kwcol, alias]
    assert not any(set(groups[i]).intersection(groups[i - 1]) for i in range(len(groups))), \
        "Same column appears in different argument categories!"

    assert all(map(lambda s: isinstance(s, str), [*pos, *kwcol, *alias, *alias.values()]))
    mapping_defined = any(map(len, groups))

    if cols is None:
        mapping_defined or warnings.warn(
            "Neither columns->arguments mappings nor expected columns been defined."
            "Will use ALL the available columns as arguments when applied")
    else:
        assert all(map(lambda x: set(cols).issuperset(x), groups)), "Not a column name!"

    if out is True:
        out_name = fnc.__name__

    def row_fnc(row: pd.Series, *args, **kws):
        """Wraps function input and output"""
        if mapping_defined:
            pos_args = row[pos]
            kwd_args = {k: row[c] for c, k in alias.items()}
            kwd_args.update(**row[kwcol], **kws)
        else:
            pos_args = row.values
            kwd_args = {}

        res = fnc(*pos_args, *args, **kwd_args)

        if not out:
            return res
        if out is True:
            return isinstance(res, (tuple, list, dict)) and res or {out_name: res}
        # here out is always a not empty list
        if isinstance(res, dict):
            return dict(zip(out, res.values(), strict=True))
        if len(out) == 1 and not (hasattr(res, '__len__') and len(res) == 1):
            res = [res]
        return dict(zip(out, res))

    return row_fnc


def apply_col_args(df: pd.DataFrame, fnc: Callable[[...], dict | tuple | Any],
                   *, pos: StringS = None, kwcol: StringS = None,
                   alias: dict[str, str] = None, out: StringS = None,
                   parallel: Parallel = False, args=(), **kws) -> pd.DataFrame:
    """
    Apply function on DataFrame by rows using specific columns as arguments.

    Place results into specified columns or return as a new data-frame.

    Columns can be used as positinal or keyword arguments,
    with supported translation from columns to keyword names.

    Parallel application of the function is supported using either
    `swifter` or `joblib` packages. (falls back to reqular apply if not installed).

    >>> def f(aleph, beth): return aleph + beth
    >>> df = pd.DataFrame({'x': [1, 2], 'y': [3, 4], 'z': [5,6]})
    >>> res = apply_col_args(df, f, pos=['x','z'], out='sum'); res  # selects x, z as positional
       sum
    0    6
    1    8
    >>> def g(aleph, beth): return {'sum': aleph + beth}
    >>> apply_col_args(df, g, alias={'x': 'aleph','z': 'beth'}).equals(res)  # map columns toi args names
    True
    >>> apply_col_args(df[['x', 'z']], g, swift=True).equals(res)  # use all provided columns as positional arguments
    True
    >>> apply_col_args(df['x'], lambda x: x+10, out='inc10', swift=True)
       inc10
    0     11
    1     12
    >>> def min_max(*a): return {'min': min(a), 'max': max(a)}
    >>> apply_col_args(df, min_max, jobs=2)
       min  max
    0    1    5
    1    2    6

    :param df: The source data frame (or Series)
    :param fnc: function to apply, may return either a scalar or dict or tuple
    :param pos: list of columns names to use as positional arguments
    :param kwcol: fnc keyword arguments which are same as columns names
    :param alias: mapping from columns corresponding fnc kw args {col: kw}
    :param out: name(s) of the resulted columns. If function returns a scalar and
    out is not defined, apply will return a Series!
    :param parallel: Parallel object to define parallel execution
    :param args: additional positional arguments for `fnc` appended after pos
    :param kws: additional keyword arguments for `fnc`

    :return: source `df` if `inplace` or new data-frame with results
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()

    row_fnc = col_args_row_func(fnc, pos=pos, kwcol=kwcol, alias=alias, out=out, cols=df.columns)

    parallel = Parallel.from_flag(parallel) or df.__class__
    return parallel.apply(df, row_fnc, axis=1, result_type='expand', *args, **kws)


def index_like(index: pd.Index, *, as_tuple=True, **kws) -> pd.MultiIndex:
    """
    Create multi-index item based on the input one, with specific levels
    replaced by the values in the arguments.

    :param index: index item from multi-index (first is used)
    :param as_tuple: return as tuple
    :param kws: {level_name: new_level_value} - to replace
    :return: Multi-index of same structure as `index`
    """
    dix = index.to_frame(False).T[0].to_dict()
    idx = tuple(kws.get(k, v) for k, v in dix.items())
    return idx if as_tuple else \
        pd.MultiIndex.from_tuples([idx], names=index.names)


def set_index_levels(index: pd.MultiIndex, *, pass_named=True, **kws) -> pd.MultiIndex:
    """In the given index set specific levels to provided values.

    The values could be scalar, vector of index length or callable
    receiving a tuple of index values of the corresponding index row.

    May set existing or add new levels to the index.

    :param index: index to alter
    :param pass_named: if True, pass index tuple as named using index names
    :param kws: {level_name: val|fnc}
    :return: New MultiIndex
    """
    callables = [*map(lambda v: isinstance(v, Callable), kws.values())]
    if fnc_num := sum(callables):
        index = (
            map(namedtuple("Index", [*index.names]), index) if pass_named
            else [*index] if fnc_num > 1
            else index
        )  # avoid redundant namedtuple creations

    f = index.to_frame(False)  # operate on columns is more efficient
    for call, (name, val) in zip(callables, kws.items()):
        f[name] = [*map(val, index)] if call else val

    return pd.MultiIndex.from_frame(f)


def redundant_levels(index: pd.MultiIndex):
    """
    Return list of levels with one unique value
    :param index:
    :return:
    """
    return [i for i in range(index.nlevels) if index.unique(i).size == 1]


def squeeze_levels(index: pd.MultiIndex):
    """
    Return multiindex with removed levels with a single unique value
    :param index:
    :return:
    """
    redundant = redundant_levels(index)
    return index.droplevel(redundant) if redundant else index


class IndexModifier:
    """
    Utility class to replace values of selected levels in multi index
    """
    IndexForm = Literal['list', 'tuple']
    LevelValue = Union[Callable, str, int, float]
    Change = tuple[int, LevelValue]

    def __init__(self, names: Sequence[str] | pd.MultiIndex):
        if isinstance(names, pd.MultiIndex):
            names = names.names
        self.names = names
        self._pos = dict(zip(names, range(len(names))))
        self.name_index_tuple = namedtuple('IndexTuple', names)._make

    def __repr__(self):
        return f"{self.__class__.__name__}[{', '.join(self.names)}]"

    @staticmethod
    def separate_calls(mixed: dict[str, Any | Callable]) -> tuple[dict[str, Any], dict[str, Callable]]:
        """Split the dict by separating items with Callable values"""
        changes = mixed.copy()
        calls = {}
        for k, v in mixed.items():
            if hasattr(v, '__call__'):
                calls[k] = changes.pop(k)
        return changes, calls

    @staticmethod
    def unbound_modifiers(group_name: str = '', call_named=True, named=True,
                          **named_changes: dict[str, LevelValue]
                          ) -> Callable[[IndexModifier, tuple], tuple[tuple, ...]]:
        """
        Given multiple (group) named dicts of changes ``{change_name: {level: value}}``
        to be applied to the specified levels of a multi-index item,
        create a function ``indexer(index: tuple)`` applying every one of them on its argument (index tuple)
        and returning list or named tuple of resulting indices named accordingly.

        :param group_name: [positional argument!] Name of this group of indexes
        :param call_named: callable require named index tuples
        :param named: create indexers producing named tuples
        :param named_changes: dict of groups of changes {ch_name: {lv_name: lv_val}}
        :return: function: amend(index) -> namedtuple(changed_index, ...)
        """
        changes_calls_list = [*map(IndexModifier.separate_calls, named_changes.values())]
        changes_list, calls_list = zip(*changes_calls_list)
        has_calls = any(calls_list)
        if not (has_calls or any(changes_list)):
            raise ValueError("Required changes may not be empty")

        def create_index_calls(imod, index, changes, calls):
            changes = changes | {k: call(index) for k, call in calls.items()}
            make_tuple = imod.name_index_tuple if named else tuple
            return make_tuple(map(changes.get, imod.names, index))

        def create_index(imod, index, changes):
            make_tuple = imod.name_index_tuple if named else tuple
            return make_tuple(map(changes.get, imod.names, index))

        if len(changes_calls_list) == 1:  # no_groups
            _changes, _calls = changes_calls_list[0]
            if has_calls:  # create different functions to optimize for each case (~30%)
                def replace(idx_mod: IndexModifier, index: tuple):
                    if call_named and not hasattr(index, '_fields'):  # convert to named if needed
                        index = idx_mod.name_index_tuple(index)  # as call requires
                    return create_index_calls(idx_mod, index, _changes, _calls)
            elif named:  # no calls no groups
                def replace(idx_mod: IndexModifier, index):  # no callables here
                    return create_index(idx_mod, index, _changes)
            else:  # not named tuples, no calls, no groups - the fastest implementation
                def replace(idx_mod: IndexModifier, index):  # no callables here
                    return tuple(map(_changes.get, idx_mod.names, index))
        else:
            make_group_tuple = wrap.namedtuple(group_name, named_changes)._make if group_name else tuple
            if has_calls:
                def replace(idx_mod: IndexModifier, index: tuple):
                    """Version supporting callables"""
                    if call_named and not hasattr(index, '_fields'):  # convert to named if needed
                        index = idx_mod.name_index_tuple(index)  # as call requires
                    return make_group_tuple(create_index_calls(idx_mod, index, changes, calls)
                                            if calls else create_index(idx_mod, index, changes)
                                            for changes, calls in changes_calls_list)
            elif named:
                def replace(idx_mod: IndexModifier, index):  # no callables here
                    return make_group_tuple(create_index(idx_mod, index, changes)
                                            for changes in changes_list)
            else:
                def replace(idx_mod: IndexModifier, index):  # no callables here
                    return make_group_tuple(tuple(map(changes.get, idx_mod.names, index))
                                            for changes in changes_list)

        return replace

    def replace_at(self, index: tuple, items: Iterable[Change]) -> tuple:
        """
        Return tuple with given positions set to new values
        :param index: tuple with index values
        :param items: tuple of (position in the index tuple value to set)
        :return: updated index tuple

        Example:
        >>> self.replace_at()

        """
        r = list(index)

        for i, v in items:
            if hasattr(v, '__call__'):
                if not hasattr(index, '_fields'):  # skip transform if already namedtuple
                    index = self.name_index_tuple(index)
                r[i] = v(index)
            else:
                r[i] = v
        return type(index)(r)

    def validate_modifiers(self, mods: dict, fail=False):
        """
        Check validity of the ``changes``: it must be a dict with keys from known names.
        Return boolean or rise exception depending on ``fail`` argument.
        :param mods: dict with changes
        :param fail: raise on invalid if True
        :return: True if valid
        """
        try:
            if not isinstance(mods, dict):
                raise TypeError(f'Changes ({type(mods)}) must be dict in {self}!')
            if inv := set(mods).difference(self.names):
                raise KeyError(f'Unknown levels {inv} in {self}')
            for k, v in mods.items():
                if not isinstance(v, (Callable, Number, str)):
                    raise TypeError(f"Invalid label '{k}' modifier type {type(v)}")
        except Exception as ex:
            if fail: raise ex
            return False
        return True

    def iter_named(self, indices: Iterable[tuple]) -> Iterable[tuple]:
        """Convert an iterable over index items in form of tuples (like a MultiIndex)
        into an iterator over named tuples, with names as provided to ``__init__``.
        """
        return (self.name_index_tuple(ii) for ii in indices)

    def replace_levels(self, index: pd.MultiIndex | tuple, *changes: Iterable[Change],
                       **kws: LevelValue) -> pd.MultiIndex | tuple:
        """
        Replace values of specific levels in MultiIndex or a single Multiindex item.
        Return correspondingly a copy of the index or of the items
        with selected levels set to new values.

        Values may be either a scalar, or vector with length of the index, or function,
        in which case values be calculated for every index row by passing it to
        the function as a namedtuple.

        :param index: MultiIndex where levels will be replaced
        :param changes: positions in the tuples of the index to set
        :param kws: alternatively {name: value} provide levels values by their names
        :return: MultiIndex with updated levels values
        """

        changes = [*changes, *self.iter_changes(**kws)]
        if isinstance(index, tuple):  # a single tuple to turn
            return self.replace_at(index, *changes)

        kws = {self.names[i]: val for i, val in changes} | kws
        return set_index_levels(index, **kws)

    def iter_changes(self, **alterations: LevelValue):
        """From key-value dict create iterator over items for an amendment."""
        return zip(map(self._pos.get, alterations), alterations.values())

    def changes(self, **alteration: LevelValue) -> tuple[Change, ...]:
        """Pack description of changes provided in form of
        keywords dictionary {level_name: level_val} into form of
        tuple of items (level_id, level_val) expected by ``update_positions`` method.
        """
        return tuple(self.iter_changes(**alteration))

    def set_fields(self, index, changes):
        # make_tuple = self.name_index_tuple._make if named else tuple
        return tuple(map(changes.get, self.names, index))

    def modify(self, index: tuple, changes: dict[str, LevelValue], *,
               named=False, call: Literal['named', True, False] = True):
        """Replace specified fields in the provided index tuple according to the
        ``changes`` keyword arguments, with keys indicating to which levels to
        assign the new values.

        If a value is ``Callable``, result of its evaluation is assigned instead,
        (unless that is disabled by setting argument ``call=False``)

        Those calls are initiated with ``index`` as argument:
          - if ``call == True`` it is passed to the callable as is,
          - if ``call == "named"`` it is converted into ``IndexTuple``
          (a named tuple with MultiIndex names), unless it IS *a* (!) named tuple.

        :param index: (named?)tuple with values of all the index levels
        :param named: control over type of the output: True for named tuple
        :param call: method to process callable values:
            - named - ensure index_item is passed as named tuple into the callables
            - True - index_item is passed to callable as is
            - False - assign callables as other values, not call them
        :param changes: {level_name: new_value}
        """

        if call:
            if call == 'named' and not getattr(index, '_fields'):
                index = self.name_index_tuple(index)

            itr = map(lambda k, old: (
                new(index) if hasattr(new := changes.get(k, old), '__call__') else new
            ), self.names, index)
        else:
            itr = map(changes.get, self.names, index)

        return self.name_index_tuple(itr) if named else tuple(itr)

    def group_indexers(self, group_name: str = '', *, named=False, call_named=False,
                       **named_changes: dict[str, LevelValue]
                       ) -> Callable[[tuple], tuple[tuple, ...]]:
        """
        Given multiple named sets of changes ``{change_name: {level: value}}``
        to be applied to the specified levels of a multi-index item,
        create a function ``indexer(index: tuple)`` applying every one of them on its argument (index tuple)
        and returning list or named tuple of resulting indices named accordingly.

        :param group_name: Name for the namedtuple type for the groups, or return regular tuple
        :param named: create indices as named tuples (slower)
        :param call_named: changes contain functions requiring named index tuple (slower)
        :param named_changes: dict of the group of changes
        {change_name: {level_name: level_value | Callable}}
        :return: function: amend(index) -> namedtuple(changed_index, ...)
        """
        unbound = self.unbound_modifiers(group_name, named=named, **named_changes, call_named=call_named)
        return lambda index: unbound(self, index)

    def indexer(self, *, named=False, call_named=False, **changes: LevelValue):
        """
        Return function applying given modification to a single index tuple.
        Performance of the function depends on the selected arguments as indicated below.

        :param named: index will be created as a named tuple (slower)
        :param call_named: changes contain functions requiring named index tuple (slower)
        :param changes: dict of changes {level_name: level_value | Callable}
        :return: Function to modify index: func(index)
        """
        unbound = self.unbound_modifiers(changes=changes, named=named, call_named=call_named)
        return lambda index: unbound(self, index)

    def to_index(self, tuples: Iterable[tuple]):
        """Convert iterable of index tuples into MultiIndex"""
        return type(self.index).from_tuples(tuples, names=self.index.names)


def iter_rows_dicts(df: pd.DataFrame):
    """Iterator over rows of a dataframe as dict items with columns as keys"""
    return map(lambda x: x._asdict(), df.itertuples(False))


class TablePainter:
    def __init__(self, rev=False, disp=True, axis=1, cmap='RdYlGn', name=None, cap=None, prec=None, **kws):
        """ Create Style Painter for DataFrame or Series.

        :param rev: reverse colormap sequence to match meaning of high/low
        :param disp: return display object when painting or styling output
        :param axis: gradient color along this axis
        :param cmap: colormap to use
        :param name: (IGNORED if not painting Series) name of the series
        :param cap: Caption text
        :param prec: precision
        :param kws: other styler keyword arguments
        """
        self.name = name
        self.prec = prec
        self.cap = cap
        self.rev = rev
        self.disp = disp
        self.kws = dict(axis=axis, cmap=cmap, **kws)
        self.kws.update(kws)

    def __call__(self, rev=None, disp=None, name=None, cap=None, prec=None, **kws):
        """Return painter updated by provided arguments

        :param rev: reverse colormap sequence to match meaning of high/low
        :param disp: return display object when painting or styling output
        :param name: (ONLY if painting Series) name of the series
        :param cap: Caption text
        :param prec: precision
        :param kws: other styler keyword arguments
        :return:
        """
        return TablePainter(
            disp=self.disp if disp is None else disp,
            prec=self.prec if prec is None else prec,
            cap=cap or self.cap,
            name=name or self.name,
            rev=rev or self.rev,
            **{**self.kws, **kws}
        )

    def __rrshift__(self, df: Union[pd.DataFrame, pd.Series]):
        """Apply style on the given table according to self configuration.

        :param df: Table to paint (DataFrame or Series)
        :return: display or style
        """

        if self.rev:
            kws = self.kws.copy()
            cmap = kws['cmap']
            kws['cmap'] = cmap[:-2] if cmap.endswith('_r') else cmap + '_r'
        else:
            kws = self.kws

        if isinstance(df, pd.Series):
            df = df.to_frame(name=self.name or df.name).T

        styled = getattr(df, 'nominal_value', df).style.background_gradient(**kws)

        if self.cap:
            styled.set_caption(self.cap)

        if self.prec is not None:
            styled.set_precision(self.prec)

        if not self.disp:
            return styled
        ipy_disp.display(styled)

    def __repr__(self):
        return str(f"{self.__class__.__qualname__}{'[rev]' if self.rev else ''}{self.kws}")


def side_tables(*tables, caps=None, hide_index=None, painter=None, **kws):
    """ In Notebook display side by side multiple tables
     Example:

     >>> side_tables(df, ['A', 'B']) == side_tables(df.A, df.B, ['A', 'B'])

    :param tables: DataFrames or styled tables
    :param caps: optional captions for each
                 if SINGLE DataFrame is given and MULTIPLE captions produce
                 tables as caption attributes from the one.
    :param hide_index: if True hide indices from the second table on
    :param painter: optional TablePainter object to paint tables with
    :param kws: keywords to TablePainter IF tables are DataFrames
    """
    from functools import reduce
    if caps is None:
        caps = [None] * len(tables)
    if len(tables) == 1 and len(caps):
        tb = tables[0]
        tables = (getattr(tb, cap) for cap in caps)
        if hide_index is None:
            hide_index = True

    if kws or painter:
        painter = (painter or TablePainter())(disp=False, **kws)
        tables = (tb >> painter for tb in tables)

    def iter_styled(tables_it, captions):
        for i, (table, cap) in enumerate(zip(tables_it, captions)):
            if isinstance(table, pd.DataFrame):
                table = table.style
            elif not isinstance(table, pd.io.formats.style.Styler):
                raise TypeError(f"Argument {i + 1} is neither Table nor Styler but {type(table)}!")

            if hide_index:
                if i == 0:
                    index = table.index.copy()
                    index.names = [None] * len(index.names)
                    columns = table.columns[[0]]
                    t = pd.DataFrame(data=np.arange(len(table.index)), index=index, columns=columns).style
                    yield t.hide_columns([*columns]), None
                yield table.hide_index(), cap
            else:
                yield table, cap

    tables = (tb.set_table_attributes("style='display:inline'")
              .set_caption(cap)._repr_html_()
              for tb, cap in iter_styled(tables, caps))

    ipy_disp.display_html(reduce(str.__add__, tables), raw=True)


def sample(dt: DTable, selection: slice | int | list[int] | float,
           *, shuffle=False, groups=False):
    """
    Return subsample of table rows.

    Selection, in its full form is described by a ``slice`` or list of indices.
    Supported shortcuts:
     -  ``int`` > 0  : selection → slice(0, selection)
     - ``float`` < 1: selection → slice(0, size * selection)

    **Notice** that ``float(1.)`` == `size` samples, when ``int(1)`` == `1` sample!

    :param dt: Data Frame or Series
    :param selection: int or slice or list of integer indices, or float for proportion from 0 to 1
                      For selection == 1, as sampling mostly comes in groups, and sample one row
                        is a less occurring UC, we create float from it to make it 100% prop.
                        For 1 sample, you can create a slice - slice(1)
    :param shuffle: if True random shuffle before sampling
    :param groups: if True sample groups of rows rather than rows themselves.
        In this case indexing is with respect to groups, not rows
    :return: new table
    """
    if shuffle and not np.issubdtype(type(selection), np.number):
        raise TypeError("Shuffling requires number of items as slc argument,"
                        "or a proportion float number from 0 to 1.")

    if selection is None:
        return dt

    def as_slice(s, size):
        """creating slice from input s - either float, int, slice or a list"""
        if np.issubdtype(type(s), np.floating):
            if s == 1.0:
                warnings.warn("Creating sampling of 1. (100%) part of the table! Use 1 for 1 row")
            s = round(s * size)
        if isinstance(s, int):
            s = slice(0, s)  # works with -, fails on > size!
        elif isinstance(s, tuple):
            s = slice(*s)
        if isinstance(s, slice) and not (s.stop is None or s.stop <= size):
            raise ValueError(f"Out of range for {size=} and slice {s}")
        else:  # input is either slice or list of nums to indicate rows
            return s

    groups = groups and as_list(groups) or False
    if groups:
        gid = '__gid__'
        idf = dt.reset_index()  # groupby can't handle NA in Multiindex!
        idf[gid] = idf.groupby(groups, dropna=False).ngroup()  # column with group ids
        groups_num = idf[gid].max() + 1
        slc = as_slice(selection, groups_num)
        gids = np.random.permutation(groups_num) if shuffle else np.arange(groups_num)
        selected = set(gids[slc])  # set of integer indices of selected groups
        slc = idf.index[idf[gid].apply(selected.__contains__)]
    else:
        slc = as_slice(selection, len(dt))
        if shuffle:
            slc = np.random.permutation(len(dt))[slc]

    return dt.iloc[slc]


def index_fillna(db: DataTable, fill=None, **kws):
    """
    Fill NaN values inside index or multiindex (not data), inplace.
    If fill is provided - using it for every NaN.
    If fill == None, using **kws as a dict, where each key is the name of
    the level to change NaN, and each value is the fill_value to change to
    """
    if isinstance(db.index, pd.MultiIndex):
        db.index = pd.MultiIndex.from_frame(
            db.index.to_frame().fillna(**kws) if not fill else
            db.index.to_frame().fillna(fill)
        )
    else:
        db.index = db.index.fillna(**kws) if not fill else db.index.fillna(fill)


# ToDo: leave only one version of add_row
# def add_row(s, data, index, name='data'):
#     """
#     Add row to series
#     :param s:
#     :param data: data to add
#     :param index: pd.Multiindex type
#     :param name: name of the column
#     :return:
#     """
#     if isinstance(index, pd.Index):
#         return pd.concat([s, pd.Series([data], index=index, name=name)])
#     else:
#         return pd.concat([s, pd.Series([data], index=[index], name=name)])


def add_row(fs: DTable | pd.DataFrame | pd.Series, data: Number | str | dict | Sequence,
            index: Number | tuple | str | dict) -> pd.DataFrame | pd.Series:
    """
    Add a new row to the data frame or series.
    Expands index levels and columns if new data requires.

    If ``fs`` is Series and new data does not bring columns of its own, series is returned.

    :param fs: data frame or series
    :param data: data to add, a value or some collection if there are multiple columns
    :param index: index value, tuple of multi-index values (matching existing levels) or dict for new levels
    :return:  DataFrame or Series with new row added
    """
    if isinstance(index, dict):
        index = pd.MultiIndex.from_frame(pd.DataFrame([index]))
    elif isinstance(index, tuple):
        index = pd.MultiIndex.from_tuples([index], names=fs.index.names)
    elif not isinstance(index, pd.Index):
        index = [index]
    new_names = fs.index.names + index.names.difference(fs.index.names)

    is_series = isinstance(fs, pd.Series)
    if is_series:
        fs = fs.to_frame()
    columns = fs.columns

    if isinstance(data, dict):  # dict named columns
        data = [data]
        columns = None
    # from here - unnamed columns
    elif isinstance(data, Sequence) and not isinstance(data, str) and len(columns) > 1:  # multiple columns
        data = [tuple(data)]
    elif (isinstance(data, np.ndarray) and data.ndim == 2) \
            or (isinstance(data, Sequence) and len(columns) == 1):  # sequence data with only one column
        data = [[data]]
    else:  # scalar
        data = [data]

    df = pd.DataFrame(data, columns=columns, index=index).reset_index()
    df = pd.concat([fs.reset_index(), df]).set_index(new_names)

    return df.iloc[:, 0] if is_series and len(df.columns) == 1 else df


def sort_index(df: pd.Series | pd.DataFrame, *, reorder_levels=True, drop_levels=False,
               fail_missing=True, **levels_order):
    """
    Sort multi-index of the given series or frame according to the
    order of values provided for every level.

    All the values not mentioned in the levels order lists are pushed to the end

    Example:

    >>> sort_index(df, color=['red', 'green'], shape=['triangle', 'square', 'circle'])


    :param df: series of data-frame to sort
    :param reorder_levels: if True, reorder index levels according to provided levels order
    :param drop_levels: if True, drop index levels not mentioned in the levels order
    :param fail_missing: raise KeyError if requested level is not if the index
    :param levels_order: {level_name: list of values in required order}
    :return:
    """
    if inv := set(levels_order).difference(df.index.names):
        if fail_missing:
            raise KeyError(f"Index names {df.index.names} does not include {inv}")
        else:
            for k in inv: del levels_order[k]

    sort_levels = list(levels_order)
    rest_levels = df.index.names.difference(sort_levels)
    if drop_levels:
        df._drop_labels_or_levels(rest_levels)

    if reorder_levels:
        df.reorder_levels(sort_levels + rest_levels)

    # From levels_order: {'a': [2, 1], 'b': [20, 10, 30], 'c': [200, 300]} ->
    # to order_map:      [{2: 0, 1: 1}, {20: 0, 10: 1, 30: 2}, {200: 0, 300: 1}]
    order_maps = [{k: p for p, k in enumerate(order)} for order in
                  (levels_order.get(lvl, []) for lvl in df.index.names)]

    def _cmp_idx(i1: tuple, i2: tuple):
        for lvl, (v1, v2) in enumerate(zip(i1, i2)):
            if v1 == v2: continue
            om = order_maps[lvl]
            if not om: continue
            mx = len(om)
            w1 = om.get(v1, mx)
            w2 = om.get(v2, mx)
            if w1 == w2:
                continue
            else:
                return -1 if w1 < w2 else 1
        return 0

    return df.loc[sorted(df.index, key=cmp_to_key(_cmp_idx))]


Scalar = int | float | str
Vector = Sequence[Scalar]


def _expand_vec_values(pairs: list[tuple[Scalar, Scalar | Vector]], prev_expanded=()
                       ) -> Generator[list[tuple[Scalar, Scalar]]]:
    """
    Perfroms external multiplication on the vector values

    >>> [*_expand_vec_values([(1, [10, 20]), (3, 0), (2, ['ok', 'no'])])]
    [[(1, 10), (3, 0), (2, 'ok')],
     [(1, 10), (3, 0), (2, 'no')],
     [(1, 20), (3, 0), (2, 'ok')],
     [(1, 20), (3, 0), (2, 'no')]]

    :param pairs: list of pairs: (key, scalar | list[scalar])
    :return: generator of scalar pairs: (key, scalar)
    """
    if pairs:
        (key, values), remained_pairs = pairs[0], pairs[1:]
        for val in as_list(values):
            yield from _expand_vec_values(remained_pairs, [*prev_expanded, (key, val)])
    else:  # empy pairs list - last most inner recursion iteration
        yield prev_expanded  # return as received - start aggregation


def _row_queries_gen(labels, levels, levels_map) -> Generator[list[tuple[int, Scalar]]]:
    """
    Generator of mult-index row queries in form of:
    ::
        list[tuple[level_num, level_value]]
    """
    for comp_query in labels:
        name_vec_pairs: Iterable[tuple[str, Scalar | Vector]]
        if isinstance(comp_query, dict):
            name_vec_pairs = comp_query.items()
        elif int.__eq__(
                l1 := len(comp_query := as_list(comp_query)),
                l2 := len(levels)
        ):
            name_vec_pairs = zip(levels, comp_query)
        else:
            raise ValueError(f"Mismatch in lengths ({l1} != {l2}) of query "
                             f"levels={list(levels)} and values={comp_query}!")

        i_vec_pairs: list[tuple[int, Scalar | Vector]] = [
            (levels_map[k], vec) for k, vec in name_vec_pairs
        ]
        # expansion of a compressed query which contains multiple actual single row queries
        # and yield them one by one
        for i_val_pair in _expand_vec_values(i_vec_pairs):
            yield i_val_pair


def select(d: DTable, labels: Sequence[dict[str, Any] | tuple[Any, ...]],
           levels=None, *, first_found=False,
           keep_levels: bool | str | list[str] = True):
    """
    From table with multi-index select rows matching given sequence of ``labels``.

    Form of the Result
    ------------------
    Every query item the ``labels`` sequence must match:
      - *exactly one* row in the table, if ``first_found is Flase``,
      - *at least one* row, if ``first_found is True``, (only first is returned!)
    Otherwis, exception is raised.

    **Resulting rows mirrors order as the query labels.**

    Forms of the Inputs
    -------------------

    Every label can be in form of a dict: `{level_name: value}` or only a sequence of values.
    Single dict can be passed to get a single row:

    >>> select(ds, {'a': 10, 'c': 'ok'})

    Later case matches levels names by position from ``levels`` argument, or from ``index.names``.

    For example, there are two options to define same labels for index with levels names `'a', 'b', 'c', 'd'`

    >>> labels = [{'a': 10, 'c': 'ok'},
    ...           {'b': 2, 'd':1, 'c': 'no'}]
    or
    >>> labels = [(10, NA, 'ok'),
    ...           (NA, 2, 'no', 1)]

    Here ``NA`` is ``pdtools.NA``

    Compressed Queries
    ------------------

    Repetitive levels requests may be compressed, in both dict-based and tuples queries:

    >>> ds = DataSeries(range(6), index=pd.MultiIndex.from_product(
    ...     [['image', 'disp', 'conf'], ['R', 'L']], names=['kind', 'view']))

    >>> res = select(ds, [
    ...             ('disp', 'L'),
    ...             ('disp', 'R'),
    ...             ('image', 'L'),
    ...             ('image', 'R')
    ... ])
    >>> res   #doctest: +NORMALIZE_WHITESPACE
                    None (Series)
    kind  view
    disp  L                 3
          R                 2
    image L                 1
          R                 0

    May be written as:

    >>> res2 = select(ds, [(['disp', 'image'], ['L', 'R'])])
    >>> assert (res == res).all()

    Notice, that nested lists are expanded from the last to first.

    When labels are provided as values (without keys), then `levels`

    :param d: ``DataFrame`` or ``Series``
    :param labels: sequence  of dict: {level1: value1, level2: value2}
    :param first_found: if ``True`` returns first found match for every label
    :param levels: Sequence of levels names used for implicit labels values.
    :param keep_levels: ``True`` - to keep the original index levels,
                        ``False`` - to use ``levels`` argument (if provided or fall back to True)
                        ``list|str`` - explicit levels to keep.
    :return: table with sub-set of selected rows
    :raises: ``KeyError`` if not found, ``LokupError`` if number of matches > 1
    """
    levels_names = d.index.names
    is_multi = len(levels_names) > 1

    if isinstance(labels, (dict, tuple)):
        labels = [labels]

    if not (levels is None or (levels := as_list(levels))):
        raise ValueError(f"Invalid {levels=}")

    match keep_levels:
        case True:
            levels = levels or levels_names
        case False:
            if levels:
                keep_levels = levels
            else:
                keep_levels = True
                levels = levels_names
        case [*_levels]:  # levels are provided through keep_levels
            levels = levels or _levels
        case str(x) | int(x):  # levels are provided through keep_levels
            levels = levels or [x]
        case _:
            raise ValueError(f"Invalid {keep_levels=}")

    # ---------------------------------------------
    def row_queries():
        """Generates row queries with accompanying info string"""
        levels_map = {n: i for i, n in enumerate(levels_names)}
        for qi, _query in enumerate(_row_queries_gen(labels, levels, levels_map)):
            yield _query, f"selection {_query} ({levels=})"

    indices = []
    for query, info in row_queries():  # each query run over all the indices
        match_row = -1  # initialize match row number "not existing"
        for row_i, levels_values in enumerate(d.index):  # match the query with every index row
            for lvl_i, label_val in query:  # ALL query labels values must match!
                # if not MultiIndex lvl_i represents level's value
                lvl_val = levels_values[lvl_i] if is_multi else levels_values
                if not (label_val is NA or lvl_val == label_val):
                    break  # level value DOES NOT match - skip row
            else:  # no break - all levels of this row have matched
                if match_row == -1:  # this is the first time this query matched
                    match_row = row_i
                    if first_found: break  # skip verifying that no other matches for the query
                else:  # here first_found is False and second match found!
                    raise LookupError(f'Multiple matches ({match_row, levels_values}, ...) for {info}')

        if match_row == -1:  # finished matching all the index - without success!
            raise KeyError(f'{info} not found in index:\n {d.index}')
        indices.append(match_row)

    sel = d.iloc[indices]

    if is_multi and keep_levels is not True:  # here if keep_levels is True or [...]
        sel = sel.keep_levels(keep_levels)
    return sel


class TableFormats:
    _reg_name = {}
    _reg_type = {}
    _reg_cond = []
    _reg_obj = None

    _html_as_text = False

    @classmethod
    def html_as_text(cls, as_text=True):
        cls._html_as_text = as_text

    @classmethod
    def register(cls, *conditions: str | type | np.dtype | Callable[[pd.Series], bool]):
        """
        Decorator to register DataFrame cells formatting function, by associating it
        with column name, column type or a particular condition on its Series:

         - string with specific name of a column - first condition to check
         - name of a ``numpy`` supported type OR ``numpy.dtype`` - next condition is type association
         - 'string'
         - Literal ``object`` - a fallback formatter for any ``dtype[object]``  column


        :param conditions:
        :return:
        """
        dtypes = {'int', 'int8', 'int16', 'int32', 'int64',
                  'uint', 'uint8', 'uint32', 'uint64',
                  'bool', 'complex', 'complex64', 'complex128',
                  'float', 'float32', 'float64'}
        string = pd.StringDtype()
        if not conditions: raise ValueError("Missing format condition")

        def decorator(fnc):
            for cond in conditions:
                if isinstance(cond, str):
                    if cond in dtypes:
                        cls._reg_type[np.dtype(cond)] = fnc
                    elif cond in {'str', 'string'}:
                        cls._reg_type[string] = fnc
                    else:
                        cls._reg_name[cond] = fnc
                elif cond == string:
                    cls._reg_type[string] = fnc
                elif issubclass(cond, str):
                    cls._reg_type[string] = fnc
                elif isinstance(cond, np.dtype):
                    cls._reg_type[cond] = fnc
                elif cond is object:
                    cls._reg_obj = fnc
                elif isinstance(cond, type):
                    cls._reg_type[np.dtype(cond)] = fnc
                elif isinstance(cond, Callable):
                    cls._reg_cond.append((cond, fnc))
            return fnc

        return decorator

    @classmethod
    def formatters(cls, df: pd.DataFrame):
        def match_cond(name):
            for cond, fnc in cls._reg_cond:
                if cond(df.get(name)): return fnc

        return tuple(cls._reg_name.get(name, None) or
                     cls._reg_type.get(dtype, None) or
                     match_cond(name) or cls._reg_obj
                     for name, dtype in df.dtypes.items())


@TableFormats.register(object)
def obj(x, max_col_width=20):
    if isinstance(x, dict):
        if x.__str__ is dict.__str__:
            return f"{{{stt.short_form(str(x)[1:-1], max_col_width - 2)}}}"
    if isinstance(x, np.ndarray):
        return npt.array_info_str(x, stats=0)
    if isinstance(x, PurePath):
        return path(x, max_col_width=max_col_width)

    clean_line = re.compile(r"\s{2,}|(\n\r?)").sub(' ', str(x))
    return stt.short_form(clean_line, max_col_width)


@TableFormats.register('path', 'folder', 'file', 'filename')
def path(x, max_col_width=40):
    return f"🖿{stt.short_form(str(x), tail=max_col_width - 2)}"


@TableFormats.register('transforms', 'transformed', 'align_trans')
def trans(s, max_col_width=20):
    if not isinstance(s, dict):  # safety if called for non-dict cell
        return str(s)

    def args_str(a, names=True):
        if not hasattr(a, 'items'):
            return str(a)
        ars = ','.join(f"{k}={v}" if names else f"{v}"
                       for k, v in a.items())
        return f"({ars})" if ars else ""

    ss = ', '.join(f"{name}{args_str(args)}" for name, args in s.items())
    if len(ss) > max_col_width:
        ss = ', '.join(f"{name}{args_str(args, False)}" for name, args in s.items())
    return ss


def _invert_levels(levels, names):
    if isinstance(levels, (str, int)):
        levels = [levels]
    if isinstance(levels[0], int):
        return [x for x in range(len(names)) if x not in levels]
    else:
        return names.difference(levels)


class _TableMixIn:

    def as_labels(self, index=True, data=False, squeeze=False) -> Labels | list[Labels]:
        """
        Convert into list of datacast.labels.Labels objects.
        :param index: include all (`True`) or specified index levels into the Labels objects
        :param data:  include all (`True`) or specified columns into the Labels objects
        :param squeeze: from 1 row return its Labels object instead of list with it
        :return: list of or single Labels representation of the row(s)
        """
        from .label import Labels
        return Labels.from_frame(self, index=index, data=data, squeeze=squeeze)

    def squeeze_levels(self: DTable, levels=None, *, keep=None, axis=0):
        """
        Leave only name level and optionally other levels required for
        unique identification of a multi-label category defined by name.

        For example in some benchmark algorithm version and or config are
        redundant, and doesn't include different info for different rows

        :param levels: list of levels to try to squeeze, default - all the levels
        :param keep: keys to keep even if they are squeezable
        :param axis: axis to work with
        """
        index = self.axes[axis]
        keep = set(as_list(keep))
        if unknown := keep.difference(index.names):
            raise NameError(f"{unknown = } levels in {axis = }")

        levels = as_list(levels) or index.names
        levels = set(levels) - keep  # levels to try to squeeze
        drop = [level for level in levels if index.unique(level).size == 1]
        return self.droplevel(drop, axis=axis)

    def find_level(self: DTable, lv_name, axis=None):
        """Return True if name found in specific axis
        :param axis: if None - search in all the axes
        """
        if axis is None:
            axis = [0, 1] if self.ndim == 2 else [1]
        else:
            axis = as_list(axis)

        for ax in axis:
            if lv_name in self.axes[ax].names:
                return True
        return False

    def levels_in(self, levels):
        """
        Return list of levels from the given list actually found in self
        :param levels:
        :return:
        """
        return [*filter(self.find_level, levels)]

    def all_levels_names(self) -> set:
        """Return set of all the levels in both index and columns"""
        all_names = set(self.index.names)
        if self.ndim > 1:
            all_names.update(self.columns.names)
        return all_names

    def named_levels(self: DTable, levels: Union[int, str, Collection[Union[int, str]]],
                     axis=0, *, exclude=False) -> list[str]:
        """Return levels names given different forms: one or more int or str
        Also as exclusion from all the levels names in the index.

        * levels may contain names not in the ``index.names`` without raising error!

        :param levels: one or more levels as int or str
        :param axis: axis to examine
        :param exclude: invert and return list of levels NOT in the names
        :return: list of names
        """
        all_names = self.axes[axis].names
        if isinstance(levels, int):
            levels = all_names[levels]
        elif isinstance(levels, str):
            levels = [levels]
        else:
            levels = [all_names[lvl] if isinstance(lvl, int) else lvl for lvl in levels]
        return [*set(all_names).difference(levels)] if exclude else levels

    def unstack_but(self: DTable, level, strict=False, *, dropna=True, **kws):
        """Unstack all the levels except the given ones.
        Remaining levels reorder as defined in level argument.

        :param level: one or more levels in any form (int, str)
        :param strict: if True ensures ALL the requested levels are found in
               index, stack them from columns if needed or raise
        :param dropna: drop NA columns which could appear as unstack artefact
        :param kws: unstack kwargs
        """
        if not level:
            return self
        if strict and (inv_levels := set(as_list(level)).difference(self.all_levels_names())):
            raise KeyError(f"Not valid levels: {inv_levels}")

        levels = self.named_levels(level)
        index_names = set(self.index.names)

        tbl = self
        if strict:
            missing = set(levels).difference(index_names)
            if missing:
                tbl = tbl.stack([*missing])  # will raise her if not found in columns
                index_names.update(missing)

        tbl = tbl.unstack([*index_names.difference(levels)], **kws)
        if dropna and tbl.ndim > 1:
            tbl = tbl.dropna(axis=1, how='all')

        if tbl.index.nlevels > 1:
            levels = [lvl for lvl in levels if lvl in index_names]  # leave existing keep the order
            return tbl.reorder_levels(levels)
        return tbl

    def stack_but(self, level, strict=False, **kws):
        """Stack all the levels except the given ones.

        :param level: one or more levels names (or ids) to leave in columns
        :param strict: if True ensures ALL the requested levels are in place
              move them from index if needed or rise if not found
        :param kws: stack kw args
        """
        levels = self.named_levels(level, 1) if self.ndim > 1 else as_list(level)
        col_names = set([] if self.ndim == 1 else self.columns.names)
        tbl = self  # type: pd.DataFrame
        if strict:
            missing = set(levels).difference(col_names)
            if missing:
                try:
                    tbl = tbl.unstack([*missing]).dropna(axis=1,
                                                         how='all')  # remove possible artefacts of unstacking
                except KeyError:
                    raise KeyError(f"Levels {missing} not found in index {tbl.index.names}")
                col_names.update(missing)

        tbl = tbl.stack([*col_names.difference(levels)], **kws)  # Need remove NULL rows?

        if tbl.ndim > 1 and tbl.columns.nlevels > 1:  # if there are col levels
            levels = [lvl for lvl in levels if lvl in col_names]
            return tbl.reorder_levels(levels, 1)
        return tbl

    def rmi(self: PTable, key=None, level=None, *, axis=0, split=False, **levels_keys) -> PTable:
        """Remove rows matching given conditions, which could be
        - keys in given levels `DataFrame.xs` style
        - index, then equivalent to df.loc[df.index.difference(key)]

        Examples:
        ::
            df.rmi('L', 'view')

        Attempts to remove not existed index lead to return of the
        input data intact.

        :param self: DataFrame or Series to filter rows from
        :param key: key(s) or index
        :param level: levels of the given keys (omit if key is index)
        :param axis: 0 or 1 for columns
        :param split: if True return also the removed part
        :param levels_keys: (level=key) form instead of (key, level)
        :return: DataFrame/Series with removed rows
        """
        assert axis in {0, 1}
        inv = lambda x: x.T if axis == 1 else x

        if levels_keys:
            if key or level:
                raise ValueError("Can't provide both positional args and keyword level keys")

        self = inv(self)
        try:
            if levels_keys:
                if self.index.nlevels == 1:
                    raise ValueError("Keyword level keys only supported for MultiIndex")
                # Start with a mask of all True, then reduce it
                mask = pd.Series(True, index=self.index)
                for lvl, val in levels_keys.items():
                    mask &= self.index.get_level_values(lvl).isin(as_list(val))
                rm_idx = self.index[mask]
            elif isinstance(key, pd.Index):
                assert level is None
                rm_idx = key
            elif self.index.nlevels == 1:  # Not a MultiIndex
                assert level in {None, 0, self.index.name}
                rm_idx = as_list(key)
            elif not is_list_like(level) and is_list_like(key):
                rm_idx = self.index[self.index.isin(key, level)]
            else:
                rm_idx = self.xs(key, 0, level=level, drop_level=False).index
            rm = self.index.isin(rm_idx)
            return (inv(self[~rm]), inv(self[rm])) if split else inv(self[~rm])

        except KeyError:
            return inv(self)

    def add_labels(self, axis=0, pass_named=True, **kws):
        """
        Add labels to a specific axis
        :param pass_named: if True, pass index tuple as named using index names
        :param axis: axis to add labels to
        :param kws: {level_name: val|fnc}
        :return:
        """
        if axis == 0:
            self.index = set_index_levels(self.index, pass_named=pass_named, **kws)
        elif axis == 1:
            self.columns = set_index_levels(self.columns, pass_named=pass_named, **kws)

    Mismatch = Literal['from_items'] | Exception | Collection[str] | dict[str, Any] | Any

    def add_items(self, items: dict | Iterable[dict],
                  join: Literal['inner', 'outer', 'items', 'original'] = 'outer'):
        """
        Add item(s) provided as iterable of dicts or a single one.

        Keys found in the `items` are compared with all the keys in `table`
        (index levels + columns).

        When both sets of keys are the same, the new items just appended to the bottom of the table.

        :param items: dict-like item or Iterable of items
        :param join: controls behaviour when items keys are different from the table keys
        :return: table with items added as new rows
        """
        # -----------------------
        if isinstance(items, pd.Series):
            items_df = items.to_frame()
            if isinstance(items.columns, pd.MultiIndex):
                items_df = items.T
        elif isinstance(items, pd.DataFrame):
            items_df = items
        else:
            if isinstance(items, dict):
                items = [items]
            elif isinstance(items, Iterable):
                def check_type(x):
                    if isinstance(x, dict):
                        return x
                    raise TypeError(f'Expected dict-like type, received {type(x)}!')

                items = map(check_type, items)
            items_df = DataTable(items)

        def invalid_index_names(t):
            not_valid = not all(names := t.index.names)
            if not_valid and len(names) > 1:
                raise IndexError("Not supported index with mixed named and None levels")
            return not_valid

        items_df = items_df.reset_index(drop=invalid_index_names(items_df))
        table = self.reset_index(drop=invalid_index_names(self))

        if (_join := join) in ('items', 'table'):
            _join = 'outer'
        df = pd.concat([table, items_df], join=_join)

        missing_keys = table.columns.difference(items_df.columns)
        extra_keys = items_df.columns.difference(table.columns)

        if join == 'items' and not missing_keys.empty:
            drop_columns = missing_keys
        elif join == 'table' and not extra_keys.empty:
            drop_columns = extra_keys
        else:
            drop_columns = None

        index_levels = self.index.names
        if drop_columns is not None:
            df.drop(columns=drop_columns, inplace=True)
            index_levels = index_levels.difference(drop_columns)

        if all(index_levels):
            df.set_index(index_levels, inplace=True)
        return df

    def keep_levels(self: PTable, level, *, strict=True):
        """
        Leave only given level(s) in the *given order* in the axis 0 index levels.

        :param level: name or sequence of names of levels
        :param strict: if True raise if unknown level is provided.
                       otherwise just ignore it.
        :return: Table with changed index.
        """
        current_levels = set(self.index.names)
        levels = as_list(level)
        if strict:  # raise if unknown levels found
            unless_subset(current_levels, levels)
        else:
            levels = [l for l in levels if l in current_levels]
        # now we have only valid levels to be kept
        self = self.droplevel([*current_levels.difference(levels)])
        return self.reorder_levels(levels) if len(levels) > 1 else self

    @wrap.doc_from(select)
    def select(self: DTable, labels: Sequence[dict[str, Any] | tuple[Any, ...]] | dict[str, Any],
               levels=None,
               *, first_found=False, keep_levels: bool | str | list[str] = True):
        """From table with multi-index select rows matching given list of labels.

        Every label can be in form of a dict: `{level_name: value}` or only a sequence of values.

        Later case matches levels names by position from ``levels`` argument, or from ``index.names``.

        For example, if ``d.index.names == ['a', 'b', 'c', 'd']`` there are two options to define same labels:
        ::
            labels = [
                {'a': 10, 'c': 'ok'},
                {'b': 2, 'd':1, 'c': 'no'},
            ], or as
            labels = [
                (10, NA, 'ok'),
                (NA, 2, 'no', 1)
            ]

        Here ``NA`` is ``pdtools.NA``

        Rows in the resulting table follow order of the list of provided query labels.

        Every query of labels must match exactly one row in the table or exception is raised.
        Single dict can be passed to get a single row:

        >>> ds.select({'a': 10, 'c': 'ok'})

        **Compressed Queries**

        Repetitive levels requests may be compressed, in both dict-based and tuples queries:

        >>> ds = DataSeries(range(6), index=pd.MultiIndex.from_product(
        ...     [['image', 'disp', 'conf'], ['R', 'L']], names=['kind', 'view']))

        >>> res = ds.select([
        ...             ('disp', 'L'),
        ...             ('disp', 'R'),
        ...             ('image', 'L'),
        ...             ('image', 'R')
        ... ])
        >>> res   #doctest: +NORMALIZE_WHITESPACE
                        None (Series)
        kind  view
        disp  L                 3
              R                 2
        image L                 1
              R                 0

        May be written as:

        >>> res2 = ds.select([(['disp', 'image'], ['L', 'R'])])
        >>> assert (res == res).all()

        Notice, that nested lists are expanded from the last to first.

        ``IndexError`` is raise if number of matches found for any label is not 1.

        :param d:
        :param labels: sequence  of dict: {level1: value1, level2: value2}
        :param first_found: if ``True`` returns first found match for every label
        :param levels: Sequence of levels names used for implicit labels values.
        :param keep_levels: ``True`` - to keep the original index levels,
                            ``False`` - to use ``levels`` argument (if provided or fall back to True)
                            ``list|str`` - explicit levels to keep.
        :return: table with sub-set of selected rows
        """
        return select(self, labels, levels=levels,
                      first_found=first_found, keep_levels=keep_levels)

    def qix(self: PTable, *anonymous,
            drop_level: bool | Collection = False, keep: Collection = None,
            axis: AxisT = None, key_err=True, **named) -> PTable:
        """
        Filter a dataframe or series using fuzzy query of its multi-index.
        Arguments in different forms describe the index values to be searched for.

        Some index levels in the output can be dropped by specifying either
        ``drop_level`` or ``keep`` (mutually exclusive) arguments.

        Use ``drop_level=True`` to drop all the redundant levels used in the query, or
        just provide one (or list) of specific levels to drop.
        Would not drop if the level is important for indexing the data (there is no redundancy),
        even if specified in the drop_level list

        :param self: The table to filter
        :param anonymous: list of values from one of the index levels
                     - will try all until first is found or raise IndexError
        :param drop_level: if True - drop found levels from the result if there is redundancy
                           or a collection of specific levels to drop
        :param keep: a collection of levels to keep - excludes using drop_level
        :param axis: if specified query only this axis
        :param key_err: raise KeyError if index not found or return empty
        :param named:  {level: value} - specific levels to find,
                            eliminates exhaustive search in all levels
        :return: Filtered table

        Tutorial:
        Using this DataTable:

        >>> from algutils.tests.test_pdtools import sdt_general
        >>> dt= sdt_general()
        >>> dt # doctest: +NORMALIZE_WHITESPACE
        z   data1          data2
        w       a  b  c  d     a  b  c  d
        x y
        1 3     1  2  3  4     5  6  7  8
          4     1  2  3  4     5  6  7  8
          5     1  2  3  4     5  6  7  8
          6     1  2  3  4     5  6  7  8
        2 3     1  2  3  4     5  6  7  8
          4     1  2  3  4     5  6  7  8
          5     1  2  3  4     5  6  7  8
          6     1  2  3  4     5  6  7  8

        Example 1: specific levels to find

        >>> dt.qix(w=['a','b'],x='1') # doctest: +NORMALIZE_WHITESPACE
        z   data1    data2
        w       a  b     a  b
        x y
        1 3     1  2     5  6
          4     1  2     5  6
          5     1  2     5  6
          6     1  2     5  6

        We can see the redundant level X. We can drop it with drop_level
        Example 2 - same, with drop level:

        >>> dt.qix(w=['a','b'],x='1',drop_level=True) # doctest: +NORMALIZE_WHITESPACE
        z data1    data2
        w     a  b     a  b
        y
        3     1  2     5  6
        4     1  2     5  6
        5     1  2     5  6
        6     1  2     5  6

        We can also search for levels without knowing their name.
        Example 3 - Anonymous search:

        >>> dt.qix('a','3',drop_level=True) # doctest: +NORMALIZE_WHITESPACE
        z  data1  data2
        x
        1      1      5
        2      1      5
        """
        # ToDo: add option to search not only in the index
        # FixMe: self.empty check twice!
        # if self.empty: return self.iloc[0:0]
        if self.empty:
            return self
        if not (anonymous or named):
            return self.iloc[0:0]

        def merge_repeating_levels_into_list(level_value: list[tuple], seen):
            for lvl, vals in level_value:
                if prev := seen.get(lvl, None):  # lvl is already in named
                    if not isinstance(prev, list):
                        seen[lvl] = [prev, vals]
                    elif vals not in prev:  # vals is not in the list
                        prev.append(vals)
                else:
                    seen[lvl] = vals
            return seen

        axes = self.axes_as_list(axis)
        assert not keep or drop_level is True, "keep levels makes sense only with drop is True"

        if anonymous:  # associate anonymous values with levels and add them to kws
            assoc = self.associate_levels(anonymous, axes)
            named = merge_repeating_levels_into_list(assoc, named)
            if not named:
                raise IndexError(f"Values {anonymous = } not found")

        # build lists of sets of levels for the required axes (could be only 1)
        axs_kws = [select_from(named, idx.names, strict=False) for idx in self.axes]
        if len(axes) > 1 and (both := select_from(axs_kws[0], axs_kws[1], strict=False)):
            raise IndexError(f"Requested levels {list(both)} appear in both axes")

        try:
            if len(named) > sum(map(len, axs_kws)):
                raise IndexError(f"Requested unknown levels: {set(named) - set(axs_kws[0] | axs_kws[1])}")

            slices = tuple(slicer(index, kws) if kws else _ALL for index, kws in zip(self.axes, axs_kws))
            res: DTable = self.loc[slices]
        except KeyError as err:
            if key_err:
                raise err
            logging.debug(f'qix:\n{err}')
            return self.iloc[0:0]

        if drop_level:
            to_set = lambda x: {x} if isinstance(x, (str, int)) else set(x)
            for ax, (kws, axis) in enumerate(zip(axs_kws, res.axes)):
                if drop_level is not True:  # if specific drop
                    kws = select_from(kws, drop_level, strict=False)
                kws_left = kws.keys() - keep if keep else kws.keys()  # if specific keep
                # check if there are only one value in a level name in res, if True - Drop
                drop = [n for n in to_set(axis.names) & kws_left if len(axis.unique(n)) < 2]
                if drop:
                    res = res.droplevel(list(drop), ax)
        return res

    def axes_as_list(self, axis: AxisT = None):
        named_axes = dict(index=0, columns=1)
        if not axis and self.ndim == 1:
            axis = 0
        if isinstance(axis, str):  # 'index' or 'columns'
            axes = [named_axes[axis]]
        elif axis is None:
            axes = named_axes.values()
        elif axis in named_axes.values():  # 0 or 1
            axes = [axis]
        elif all(named_axes.__contains__, axis):
            axes = [named_axes[k] for k in axis]  # [0, 1]
        elif all(named_axes.values().__contains__, axis):
            axes = [*axis]
        else:
            raise ValueError(f"Invalid axis: {axis}")
        return axes

    def associate_levels(self, anonymous: Collection, axes: list[int]):
        """ Associate anonymous level values with level names
        :param self: DataTable
        :param anonymous: Collection of anonymous values to get their level name
        :param axes: Specific axis to check, if requested by the user

        :return: list of tuples [(name,value),...] found and paired
         """
        assoc = []
        for ax in axes:
            if not anonymous: break  # spare unneeded calculations
            index = self.axes[ax]
            levels_vals = [*zip(index.names, index.levels)] if index.nlevels > 1 else [(index.name, index)]
            not_found = []
            for item in anonymous:
                found_level = None
                for lvl, vals in levels_vals:
                    if item not in vals: continue
                    if found_level:
                        raise IndexError(f"{item} found in more then one levels: {found_level, lvl}")
                    found_level = lvl

                if found_level:
                    assoc.append((found_level, item))
                else:
                    not_found.append(item)
            anonymous = not_found  # to look for them in
        return assoc

    def mean(self: DTable, axis=0, skipna=True, level=None, numeric_only=None):
        """ Return the mean of the values over the requested axis / level.

        *Uncertainties are properly calculated if available!

        :param axis : {index (0), columns (1)}
            Axis for the function to be applied on.
        :param skipna : default True, Exclude NA/null when computing the result
        :param level : int or level name, default None
            If the axis is a MultiIndex (hierarchical), count along a
            particular level, collapsing into a Series.
        :param numeric_only : bool, default None
            Include only float, int, boolean columns. If None, will attempt to use
            everything, then use only numeric data. Not implemented for Series.
        """
        series = isinstance(self, pd.Series)
        kws = dict(axis=axis, level=level, numeric_only=numeric_only)
        mean = (pd.Series if series else pd.DataFrame).mean(self.nominal_value, **kws)
        if not (hasattr(mean, 'index') and hasattr(self, 'nominal_value')) or self.std_dev.sum().sum() == 0:
            return mean

        kws.update(level=level)
        num = self.std_dev.count(**(dict(level=level) if series else kws))
        sd = ((self.std_dev ** 2).sum(**kws) / num) ** 0.5
        from uncertainties.unumpy import uarray
        return mean.__class__(uarray(mean, sd), index=mean.index,
                              **({} if mean.ndim == 1 else {'columns': mean.columns}))

    def weighted_mean(self: DTable, ws: Union[pd.Series, pd.DataFrame, Sequence],
                      level=None, unc=True):
        """Calculate mean of DF columns weighted by ws vector of same length.
        :param self: table with columns to average
        :param ws: weights vector in any supported form
        :param level: separate averaging over those levels
        :param unc: uncertainty calculation control:
                    False - do not calculate
                    True - calculate and fill resulted table with objects
                            from uncertainty package
                    <str> - each column to produces two: 'avr', <str>
                            second with standard deviation values
        """

        if isinstance(ws, Sequence):
            ws = DataSeries(ws, index=self.index if level is None else
            self.unique_level_values(level=level))
        elif isinstance(ws, (pd.DataFrame, pd.Series)):
            self = self.unstack_but(ws.index.names)  # align indices
            assert ws.shape[0] == self.shape[0], f"{ws.shape=} vs {self.shape=}"

        ws = ws / ws.sum(level=level)
        avr = (self.multiply(ws, 0)).sum(level=level).unstack_but(level)
        if unc:
            from uncertainties.unumpy import uarray
            sd = ((self.sub(avr) ** 2).multiply(ws, 0).sum(level=level)) ** 0.5
            if unc is True:
                data = uarray(avr.values, sd.values)
                return DataSeries(data, index=avr.index) if avr.ndim == 1 else \
                    DataTable(data, index=avr.index, columns=avr.columns)
            else:
                return pd.concat([avr, sd], keys=['avr', unc], names=['ws']).reorder_levels(
                    [*avr.index.names, 'ws'])
        return avr

    def __add__(self, other):
        index_names = self.index.names
        return pd.concat([self.reset_index(), other.reset_index()]).set_index(list(index_names))

    def __or__(self: DTable, other: dict):
        (out := self.copy()).__ior__(other)
        return out

    def __ior__(self: DTable, other: dict):
        new_levels = list(other)
        self[new_levels] = [*other.values()]
        self.set_index(new_levels, append=True, inplace=True)
        return self

    def freeze(self, state: bool = True):
        old_state = getattr(self, '_hash', None)
        self._hash = id(self) if state else None
        return old_state

    def __hash__(self):
        if getattr(self, '_hash', None) is None:
            super().__hash__()
        else:
            return self._hash

    def hash_str(self, n=0, *, index=True) -> str:
        """Create stable hash string of required length using sha256

        :param n: length of hash str (0 - use all 64 hex chars sha256 returns)
        :param index: include index to hash calculation
        """
        from hashlib import sha256
        hash_value = sha256(pd.util.hash_pandas_object(self, index=index).values)
        return hash_value.hexdigest()[-n:]


def slicer(index: pd.Index | pd.MultiIndex, kws):
    """ Slicing the index with kws requested
    Args:
        index: Multi (or standard) index to be sliced
        kws: the level values requested which define the slice

    Returns: tuple of slices to be used to filter the datatable
    """
    if index.nlevels > 1:  # MultiIndex
        return tuple(q if (q := kws.get(lvl, _ALL)) is _ALL
                     else index.isin(as_list(q), lvl)  # map of found level values
                     for lvl in index.names)

    (level, val), *not_existing_levels = kws.items()
    assert level == index.name
    assert not not_existing_levels
    return as_list(val)


def path_fixer(root, fixer_name, cols='path') -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Create function to transform path column in a given DataFrame
    by either add to or crop from it the root path.

    :param root: root path to add or crop
    :param fixer_name: name of the transformation to perform: 'add' | 'crop'

    :return function object receiving and producing a dataframe
    """
    if not root:
        return None

    cols = set(as_list(cols))

    from . import filesproc as proc
    fixer = {
        'add': proc.root_adder,
        'crop': proc.root_cropper
    }[fixer_name]

    def fix_path(df):
        if columns := [c for c in df.columns if c in cols]:
            df[columns] = df[columns].applymap(fixer(root))
        return df

    return fix_path


def _index_from(s, idx):
    return idx if idx is not None else \
        s.index if isinstance(s, pd.Series) else \
            pd.RangeIndex(len(s))


def kron(s1: Union[pd.Series, Collection], s2: Union[pd.Series, Collection],
         index1: pd.Index = None, index2: pd.Index = None):
    """Create Series or DataSeries (depending on the input types)
        from kron multiplication of two Series

    :param s1: Input Series or Collection (if later, index1 must be supplied)
    :param s2: Input Series or Collection (if later, index2 must be supplied)
    :param index1: Optional replacing index in s1 or providing if missing
    :param index2: Optional replacing index in s1 or providing if missing
    """
    Series = DataSeries if isinstance(s1, DataSeries) or isinstance(s1, DataSeries) else pd.Series
    return Series(np.kron(s1, s2), index=pd.MultiIndex.from_product(
        [_index_from(s1, index1), _index_from(s2, index2)]))


def outer(s1: Union[pd.Series, Collection], s2: Union[pd.Series, Collection],
          index1: pd.Index = None, index2: pd.Index = None):
    """Create 2D Table (DataFrame or DataTable, depending on input types)
    from outer multiplication of two Series

    :param s1: Input Series or Collection (if later, index1 must be supplied)
    :param s2: Input Series or Collection (if later, index2 must be supplied)
    :param index1: Optional replacing index in s1 or providing if missing
    :param index2: Optional replacing index in s1 or providing if missing
    """
    Table = DataTable if isinstance(s1, DataSeries) or isinstance(s1, DataSeries) else pd.DataFrame
    return Table(np.outer(s1, s2), index=_index_from(s1, index1), columns=_index_from(s2, index2))


class DataSeries(_TableMixIn, pd.Series):
    #
    # def kron(self, pd.Series):
    #     weights = pdt.DataSeries(np.kron(acr_ws, [*rgn_ws.values()]),
    #                          index=pd.MultiIndex.from_product(
    #                              [acr_ws.index, pd.Index(rgn_ws.keys(), name='region')]
    #                          ))

    @property
    def _constructor(self):
        return DataSeries

    @property
    def _constructor_expanddim(self):
        return DataTable

    def __getattr__(self, item):
        if isinstance(self.index, pd.MultiIndex):
            for lvl in self.index.levels:
                if item in lvl:
                    return self.xs(item, level=lvl.name)
        return super().__getattr__(item)

    def _prep_repr(self):
        return self.to_frame(name=f'{self.name} (Series)')

    def _repr_html_(self):
        return self._prep_repr()._repr_html_()

    def __repr__(self):
        return self._prep_repr().__repr__()

    def __str__(self):
        return self._prep_repr().__str__()


class DataTable(_TableMixIn, pd.DataFrame):
    @property
    def _constructor(self):
        return DataTable

    @property
    def _constructor_sliced(self):
        return DataSeries

    def unique_level_values(self, level: Union[int, str], axis=0):
        """
        Return unique levels values for given level in given MultiIndex axis
        :param level: level's name or id
        :param axis: 0 - index, 1 - columns
        :return: list of values
        """
        idx = self.axes[axis]
        if idx.nlevels == 1:
            assert level == 0 or idx.name == level
            return idx
        lid = idx._names.index(level) if isinstance(level, str) else level
        return idx._levels[lid]

    def __getattr__(self, item):
        get = pd.DataFrame.__getattribute__
        columns = get(self, 'columns')
        index = get(self, 'index')
        xs = get(self, 'xs')

        for ax, idx in [(1, columns), (0, index)]:
            if isinstance(idx, pd.MultiIndex):
                for lvl in idx.levels:
                    if item in lvl:
                        return xs(item, axis=ax, level=lvl.name)
            elif item in idx:
                return xs(item, ax)
        return super().__getattr__(item)

    def _repr_html_(self):
        if TableFormats._html_as_text:
            text = self.__repr__()
            return f"""<pre style="font-size: 12px">{text}</pre>"""

        if self._info_repr():
            buf = StringIO("")
            self.info(buf=buf)
            # need to escape the <class>, should be the first line.
            val = buf.getvalue().replace("<", r"&lt;", 1)
            val = val.replace(">", r"&gt;", 1)
            return "<pre>" + val + "</pre>"

        get_option = pd._config.get_option
        if get_option("display.notebook_repr_html"):
            dfm = pd.io.formats.format
            formatter = dfm.DataFrameFormatter(
                self,
                columns=None,
                col_space=None,
                na_rep="NaN",
                formatters=TableFormats.formatters(self),
                float_format=None,
                sparsify=None,
                justify=None,
                index_names=True,
                header=True,
                index=True,
                bold_rows=True,
                escape=True,
                max_rows=get_option("display.max_rows"),
                min_rows=get_option("display.min_rows"),
                max_cols=get_option("display.max_columns"),
                show_dimensions=get_option("display.show_dimensions"),
                decimal=".",
            )
            return dfm.DataFrameRenderer(formatter).to_html(notebook=True)
        else:
            return None

    def __repr__(self):
        get_option = pd._config.get_option
        # print(f'=====repr({id(self)})')
        buf = StringIO("")
        if self._info_repr():
            self.info(buf=buf)
            return buf.getvalue()
        width = get_option("display.expand_frame_repr") and \
                get_option("display.width") or None

        self.to_string(
            buf=buf,
            float_format=':.5'.format,
            formatters=TableFormats.formatters(self),
            max_rows=get_option("display.max_rows"),
            min_rows=get_option("display.min_rows"),
            max_cols=get_option("display.max_columns"),
            show_dimensions=get_option("display.show_dimensions"),
            line_width=width,
            max_colwidth=get_option("display.max_colwidth"),
        )
        return buf.getvalue()

    def item_labels(self, idx: int, index=True, data=False):
        """"""
        from .label import Labels
        return Labels.from_frame(self.iloc[[idx]], index=index, data=data)[0]


def as_table(x: PTable, series: bool | None = None) -> DTable:
    """
    Ensure result is of a Table type (DataTable or DataSeries),
    according to the input form, if ``series`` is `None`.

    Otherwise, try to cast into requested form or raise ``TypeError``.

    :param x: any table type (pandas or derived)
    :param series: if True | False - return specifically `DataSeries` or `DataTable`.
    :return: table or series
    """
    is_series = isinstance(x, pd.Series)
    is_frame = isinstance(x, pd.DataFrame)

    # neither series nor frame, no conversion needed, ty to create directly whats required
    if not (is_series or is_frame):
        return DataSeries(x) if series else DataTable(x)

    # first bring into proper series / frame form, not necessary a table
    if series is True:  # requested series
        if is_frame:  # frame -> series
            if len(x.columns) > 1:
                raise TypeError('Can''t convert frame with columns: {x.columns} into series')
            x = x[x.columns[0]]
    elif series is False and is_series:  # requested frame
        x = x.to_frame()  # but it's a series
    # otherwise no conversion needed

    if isinstance(x, _TableMixIn):
        return x
    if isinstance(x, pd.DataFrame):
        return DataTable(x)
    if isinstance(x, pd.Series):
        return DataSeries(x)
    raise TypeError(f"Can't cast {type(x)} to a Table class")


def append_col(df: pd.DataFrame,
               col: Union[TCol, list[TCol]],
               values: Union[pd.DataFrame, Collection],
               levels: Collection[str] = None) -> pd.DataFrame:
    """
    Append column(s) to the DataFrame (may also overwrite)
    and match columns levels if needed.
    :param df:
    :param col: str | tuple | list
                str   - a regular name, then place it at the top level
                tuple - of str for multiple levels - then start
                        filling the existing columns levels from the top
                        and extend them if needed
                list  - list of str or tuple - to add multiple columns
    :param values: values of new columns(s)
                   if multiple columns - same size collection of 1D arrays
                   if value is a DataFrame, its columns' levels are prepended
                   by cols and the added into  `df`
    :param levels: optionally provide names for the levels,
                   must be same number of them as the final levels
    :return: return the input df (the operations are inplace)
    """

    def add_levels(n):
        df.columns = pd.MultiIndex.from_product([df.columns, *[*[['']] * n]])

    if isinstance(values, pd.DataFrame):
        assert isinstance(col, (str, tuple)), "With values as DataFrame, col must be str or tuple"
        col = as_list(col)
        for vc in values.columns:
            df = append_col(df, (*col, *as_list(vc)), values[vc], levels)
        return df

    if isinstance(col, list):
        if len(values) != len(col) and (
                hasattr(values, 'shape')
                and values.ndim == 2
                and values.shape[1] == len(col)):
            values = values.T

        for c, vals in zip(col, values):
            df = append_col(df, c, vals, levels)
        return df

    levels_num = len(df.columns.names)
    if isinstance(col, str):
        if levels_num > 1:
            col = (col, *[''] * (levels_num - 1))
    elif isinstance(col, tuple):
        if levels_num > len(col):
            col = (*col, *[''] * (levels_num - len(col)))
        elif levels_num < len(col):
            add_levels(len(col) - levels_num)
    if levels:
        df.columns.names = levels
    df[col] = values
    return df


def apply_index_level(df, level, func):
    """Alter index by applying a function on specific level of index"""
    idx_df = df.index.to_frame()
    idx_df[level] = idx_df[level].apply(func)
    df.index = idx_df.set_index(df.index.names).index
    return df


def key_based_mapper(func_map: dict[Any, Callable]):
    """Creates functions to use to map apply to DataFrames based on its
    columns index keys - different function per key.

    :param func_map: dictionary of functions
    :return: function to use in DataFrame.apply
    """

    def mapper(x):
        return pd.Series({k: func_map[k](v) for k, v in x.items()})

    return mapper


def unc_split(df, axis=1, names=('val', 'sd')):
    """ Split uncertainty objects in the table into two columns
    :param df:
    :param axis: along with axes to concatenate `val` and `std` arrays
    :param names: names of the value and std column
    :return: data frame with separate rows (or columns) for val and std
    """
    return pd.concat([df.nominal_value, df.std_dev], axis=axis, keys=names)


def add_levels(df: Union[pd.Series, pd.DataFrame], pos=0, **levels) -> Union[pd.Series, pd.DataFrame]:
    """
    Add level to multi-index of the DataFrame or Series.

    :param df: DataFrame or Series to add the level to its index
    :param pos: position among the original levels, first by default: 0
                None does no rearrangements
    :param levels: key: value pairs as in `DataFrame.assign`
    :return: DataFrame or Series with updated index
    """

    if isinstance(df, pd.Series):
        name = df.name
        return add_levels(df.to_frame(), pos=pos, **levels).iloc[:, 0]

    df = df.assign(**levels).set_index([*levels], append=True)

    if pos not in (None, -1):
        pos = pos + 1 if pos < 0 else pos
        original = df.index.names[:-len(levels)]
        df = df.reorder_levels([*original[:pos], *levels, *original[pos:]])
    return df


def split_col(df, col: Union[str, Sequence[str]], val=False) -> tuple:
    """Split dataframe into two by columns.

    :param df: the DataFrame to split
    :param col: name(s) of columns to split into a new dataframe
    :param val: instead of data frames return their values (as arrays)
    :return: tuple of two dataframes or arrays of their values
    """
    x, y = df.drop(columns=col), df[as_list(col)]
    return (x.values, y.values) if val else (x, y)


def plot_hists(df, m):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(int(np.ceil(len(df.columns) / m)), m, figsize=(18, 4))
    plt.tight_layout()
    axes = [*axes.flat]
    for col, axis in zip(df.columns, axes):
        df.hist(column=col, bins=100, ax=axis)
    return axes


def group_iter(df: pd.DataFrame, group: Union[str, list], index: Union[str, list] = None,
               data: Union[str, Iterable] = None, out: str = None, progress: Union[bool, dict] = None,
               islice: int | tuple | None = None, shuffle=False
               ) -> Iterable:  # Iterator[tuple[Union[str, tuple[str]], Union[DataTable, DataSeries]]]
    """
    Create iterator over groups of paths with flexible organization,
    producing tuples::
        (group_index, group_data)

    - `group_index` - tuple of (or single) values of categories used for grouping
    - `group_data` - grouped data elements in form of DataFrame

    Specific structure of the group is controlled by arguments:

        * by which categories the items are grouped as iteration entities:
        * the content and structure of the `group_data` collection (DataFrame)

    Multi-level indexing is supported for one axis of the frame, by default for the rows.
    That allows to combine hierarchical access to the labels values and iteration.

    Example:

    >>> ds.categories == ['scene', 'alg', 'kind', 'side', 'time', 'path', 'transforms']
    >>> (scene, alg), group = next(
    ...     ds.iter(['scene', 'alg'], index=['kind', 'side'], data='path'))
    ... # Notice drop of all the categories not explicitly requested as index or data,
    ... # like 'ext', 'transforms', but 'time' remains to ensure unique indexing
    ... scene, alg == 'Piano', 'NU4'
    ... group.index == ['kind', 'side', 'time']
    ... group.columns == ['path']

    :param df: DataFrame with items to group
    :param group: one or list of categories to group by.
    :param index: list of categories to be used as multi-index levels
                  of the resulting groups (also defines their order)
    :param data: keep those categories in the data (as colums)
                  if `None`: produce `DataFrame` with all original data categories as columns
    :param out: form of the output (None|"frame"|"series"|"pivot")
                  if "frame": return DataFrame with columns as data categories
                  if "series": return `Series` - only if single data column is defined!
                  if "pivot": return DatFrame in pivoted form - to have data-columns as rows,
                  and the `index` argument as columns
                    - column-based multi-level indexing allows hierarchical access:
                        `data.level1.level2 == value`
                    - row-based may be more convinient to `data.itertuples()`:
                        `for (level1, level2), values in data.itertuples():`
                  if None (default): operates as "frame" or "series" if single data column
    :param islice: arguments for ``itertool.islice`` (start, [stop, [step]])
    :param shuffle: reorder groups randomly
    :param progress: show progress bar if True or dict with args for `tqdm`
    :return: iterator over tuples: ((grouped by categories), group)
    """
    assert out in {None, "frame", "series", "pivot"}
    # get input dataframe indexs and data names
    db_index = set(df.index.names)
    db_data = set(df.columns)
    if not data:
        data = db_data
    df = df.reset_index()
    cats = set(df.columns)  # all the categories in the df
    group = as_list(group)  # as list of labels!

    index = as_list(index) or as_list(db_index.difference({*group, *data}))  # requested index or default
    data = as_list(data) or as_list(db_data.difference(set(index)))  # and data categories

    unless_subset(cats, index, "`index` requires not existing categories: {inv}")
    unless_subset(cats, data, "`data` requires not existing categories: {inv}")

    if len(set(index).intersection(set(data))) > 0:
        raise "requested index and data can't share values"
    if out is None:
        out = "series" if len(data) == 1 else "frame"
    if out == "series":
        if len(data) == 1:
            data = data.pop()
        else:
            raise ValueError(f'Multiple requested data columns: {data}\n'
                             f'incompatible with argument out="series"!')

    def prep_group(item):
        idx, grp = item
        grp = grp.set_index(index, append=True).droplevel(0)
        return idx, grp.sort_index()[data]

    groups = df.groupby(group if len(group) > 1 else group[0], dropna=False)
    total = groups.ngroups

    # ------------ requests for reshaping the data -----------------------
    if islice:
        from itertools import islice as slice_iter

        start, stop, step = _norm_slice(islice)
        total = int(np.ceil((min(stop, total) - start) / step))
        groups = slice_iter(groups, start, stop, step)

    if shuffle:
        import random
        groups = list(groups)
        random.shuffle(groups)

    iterator = map(prep_group, groups)

    # -------------------- show progress bar -----------------------------
    if progress is None and total >= 20 or progress is True:
        progress = {}

    if isinstance(progress, dict):
        from tqdm.auto import tqdm
        progress = {'leave': False, 'total': total} | progress
        iterator = tqdm(iterator, **progress)

    return iterator


def _norm_slice(slc):
    match slc:
        case int(stop) | (stop, ):
            return 0, stop, 1
        case (start, stop):
            return start, stop, 1
        case (_, _, _):
            return slc
        case _:
            raise ValueError(f"Invalid slice: {slc}")


def filter_full_groups(df, group) -> (pd.DataFrame, int):
    """
    Filter df to leave only items comprising full groups
    :param df:  the DataFrame to filter
    :param group: column(s) to group by
    :return: tuple(filtered df, number of full groups)
    """
    group = as_list(group)
    mv_idx = df.index.names.difference(group)
    df = df.reset_index(mv_idx)  # leave only 'scene' in the index
    full_groups = df.groupby(group).apply(lambda x: issubset_report(x.kind, ))
    df = df.loc[full_groups[full_groups].index]
    bad_scenes = (full_groups == False).sum()
    groups_num = len(full_groups) - bad_scenes
    if bad_scenes:
        logging.getLogger().warning(f'Filtered {bad_scenes} (of {len(full_groups)}) '
                                    f'incomplete scenes from collection')
    return df, groups_num


def sorted_index_levels(idx) -> dict[str, pd.Index]:
    """Dictionary { name: unique level's index} sorted by index length

    :param idx: Multi or nor Index
    :return: dict sorted by index len
    """
    if isinstance(idx, pd.MultiIndex):
        names = idx._names
        values = idx._levels
        by_len = np.argsort([*map(len, values)])
        return {names[i]: values[i] for i in by_len}
    else:
        return {idx.name: idx}


def all_bool(s: pd.Series):
    bool_or_null = lambda x: isinstance(x, bool) or x is None or x is np.nan
    return s.dtype.hasobject and all(map(bool_or_null, s.values))


def all_str(s: pd.Series):
    str_or_null = lambda x: isinstance(x, str) or x is None or x is np.nan
    return s.dtype.hasobject and all(map(str_or_null, s.values))


def series_has_types(s: pd.Series, *types):
    """Check if series contains elements of give `object` types, ignoring Nans"""
    of_types_or_null = lambda x: isinstance(x, types) or x is None or x is np.nan
    return s.dtype.hasobject and all(map(of_types_or_null, s.values))


def missing_associates(db: pd.DataFrame, select: dict, assoc: dict) -> tuple[pd.MultiIndex, pd.MultiIndex]:
    """
    From selected subset of labels db find items with no associates in the db.

    Labels are represented by the multiindex.
    Labels of associates are determined by replacing values of the items labels
    according to the `assoc` dict.

    :param db: DataFrame with MultiIndex encoding the data labels
    :param select: {label: value} - dict used to select items to find associates with
    :param assoc: {label:  new_value} replacement rules to build for a given label an associated one

    :return: missing labels, their associates
    """
    sel_idx = db.qix(**select).index
    rep_index = set_index_levels(sel_idx, **assoc)
    # missing_mask = ~rep_index.isin(db.index)  # slow for large db
    idx_set = set(db.index)
    missing_mask = ~np.array([*map(idx_set.__contains__, rep_index)], dtype=bool)
    return rep_index[missing_mask], sel_idx[missing_mask]
