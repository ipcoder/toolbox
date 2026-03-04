from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Callable, Iterable, Literal, Any, Collection, get_args

import numpy as np
import pandas as pd

from algutils import as_list, pdtools as pdt
from algutils.fnctools import O, Namespace
from algutils.label import Labels
from algutils.wrap import name_tuple

EMPTY_CELL = (None, pd.NA, np.nan)


def transforms() -> dict[str, Callable]:
    """Return namespace of transforms defined in transform module."""

    from . import transforms
    _ns = vars(transforms)
    return {name: _ns[name] for name in transforms.__all__}


def applicable(fnc, *pos_names, **kws_map):
    """
    Given a function creates its wrap usable in pd.DataFrame.apply
    Some of its arguments to be associated to the pd.DataFrame columns,
    the rest must be either default or passed as `**kwargs` in `apply`.

    Example:
    >>> def f(x, y=0, *, z, w=None): pass
    >>> af = applicable(f, 'data', z='weight')  # data as first arg
    >>> df = pd.DataFrame(dict(data=[1,2,3], weight=[0,1,2]))
    >>> df['new'] = df.apply(af, axis=1, y=2, w=100)

    :param fnc: function to be wrapped into the new interface
    :param pos_names: column names of the positinal parameters
    :param kws_map: mappings between the kw arguments and columns names
    :return: function used for apply
    """
    assert pos_names or kws_map, "At least one argument is required"

    def apply_func(dct, **kwargs):
        if hasattr(dct, '__getitem__'):
            args = (dct[p] for p in pos_names)
            kws = {par: dct[field] for par, field in kws_map.items()}
            return fnc(*args, **kws, **kwargs)
        else:
            return fnc(dct, **kwargs)

    return apply_func


def apply_to(df: pd.DataFrame, func: Callable, out, *args, **kwargs):
    """Create new dataframe with Apply func to the data-frame into `out` column.

    :param df: the data-frame
    :param func: function to be passed to `df.apply`
    :param out: column where the result is placed
    :param args: arguments passed to `func`
    :param kwargs: keyword arguments passed to `func`
    :return The modified input `df`
    """
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        ndf = df.copy()
        ndf[out] = df.apply(func, *args, axis=1, **kwargs)
    return df


@dataclass
class Col:
    """Standard columns names"""
    path: str = 'path'
    data: str = 'data'
    ext: str = 'ext'
    desc: str = 'desc'
    description: str = 'description'
    transforms: str = 'transforms'
    transformed: str = 'transformed'
    align_trans: str = 'align_trans'
    read_trans: str = 'read_trans'


def apply_seq(df: pd.DataFrame, *trans, inp=Col.data, out=Col.data):
    """Apply sequence of transformations to the `df` columns
    :param df:
    :param trans: sequence of transformation stage instructions:
                  `(fnc, inp, out), (fnc, inp), (fnc, )` or `fnc`
    :param inp: default inp column for the first stage. Others if their
                `inp` is not specified use `out` from the previous stage.
    :param out: default output column of any stage with unspecified `out`

    Example:
    >>> apply(df, [f1, f2, f3], inp=x, out=y)
    means: `df[x] -> f1 -> df[y] -> f2 -> df[y] ->f3 -> df[y]`
    >>> apply(df, [(f1, y1, x), (f2, y2), f3], out=y)
    means: `df[x] -> f1 -> df[y1] -> f2 -> df[y2] -> f3 -> df[y]`,
    So that in this case all the intermediate results are kept in y1, y2
    """
    for t in trans:
        fnc, _out, _inp = (*t, *(out, inp)[:3 - len(t)]) \
            if isinstance(t, (list, tuple)) else (t, out, inp)
        inp = _out  # default inp for next stage is previous out
        if not fnc:
            continue
        df[_out] = df[_inp].apply(applicable(fnc, _inp), axis=1)


def apply_transforms(df: pd.DataFrame, *, inp=Col.path, out=Col.data,
                     trans=Col.transforms, ns=None,
                     keep: Union[str, Iterable] = ()) -> pd.DataFrame:
    """
    Apply sequence of data initialization transforms on the df.
    Full sequence is:
      [inp] -> read -> [out] -> {transforms} -> [out]

    Stage `read` is applied only if defined (default is `imread`)

    :param df:  columns will be processed in this data frame
    :param inp: name of first stage input column (if read then path)
    :param out: name of output column
    :param keep: `trans`, `inp` are dropped unless any is listed here
    :param trans: name of column with transforms
    :param ns: module or dict-like container with transform functions
    :return:
    """
    if ns is None:
        ns = transforms()

    trans = as_list(trans)

    def row_transform(row: pd.Series):
        return O.comp(*map(row.get, trans), ns=ns, skip=EMPTY_CELL)(row.get(inp))

    drop = {*trans, inp}.difference({out}.union(as_list(keep)))
    df = df.copy()  # if df is a view, then df[out] = assigns to cached copy

    if set(trans).issubset(df.columns):
        df[out] = df.apply(row_transform, axis=1)
    df.drop(drop, axis=1, inplace=True, errors='ignore')
    return df


def apply_column_transform(df: pd.Series | pd.DataFrame, *,
                           inp: str | Collection[str] = Col.path,
                           out: str = Col.data,
                           trans: str = Col.read_trans,
                           recover: bool = False,
                           args: Literal["series", "pos", "kws"] = 'pos',
                           fallback: Any = pd.NA,
                           drop: str | Collection[str] | bool = True,
                           inplace=False,
                           **kwargs) -> pd.DataFrame:
    """
    Apply (by row) transforms stored in ``trans`` column on the ``inp`` column.

    Default configuration is set for `read_trans` transformation to load data.

    **See also**:
        ``compile_read_transforms`` to create ``read_trans`` with functions.

        ``apply_transforms`` as more general version of this function

    By default, drop (True), all `used` columns (``trans`` and ``inp``).
    Otherwise, specific columns to drop may be also specified, or None to keep.

    Transform may receive arguments in three different forms:
     - `pos` - as positional arguments ordered as in `inp` collection. `trans(*pos)`.
     - `series` - row series as produced by the pandas `apply`. `trans(series)`
     - `kws` - as keyword arguments with columns as keys:  `trans(**kws)`

    Transform failed on specific rows can be recovered by re-applying on the item
    provided "fallback" function, or just producing given "error" value.

    :param df: group (Series or Frame) usually produced
    :param inp: name of the input column,
    :param trans: column with transform functions objects
    :param out: output column name
    :param args: from in which arguments are passed into the `trans`
    :param recover: if True - recover per row from failed transform using `fallback`
    :param fallback: alternative callable or fallback value used to recover
    :param drop: True to drop used columns, or specific columns to drop
    :param kwargs: passed to `trans` function as additional kw args
    :param inplace: update given ``DataFrame``, and return it
    :return: transformed table
    """
    # first make sure we work with frame
    if isinstance(df, pd.Series):
        if inplace:
            raise TypeError("Can't inplace transform Series!")
        df_out = df.to_frame()
    elif not inplace:
        df_out = df.copy()
    else:
        df_out = df

    inputs = as_list(inp) or df_out.columns

    if args == 'pos':
        func = lambda g: g[trans](*(g[attr] for attr in inputs), **kwargs)
    elif args == 'kws':
        func = lambda g: g[trans](**{attr: g[attr] for attr in inputs}, **kwargs)
    elif args == 'series':
        func = lambda g: g[trans](g if inp is None else g[inputs], **kwargs)
    else:
        raise ValueError(f'Unknown {args=}')

    if recover:  # wrap func into (try - except) to ensure row-based resilience
        if not isinstance(fallback, Callable):
            error_value = fallback
            fallback = lambda _: error_value
        _func = func

        def func(g):
            try:
                return _func(g)
            except:
                return fallback(g)

    df_out[out] = df_out.apply(func, axis=1)

    if drop is True:
        drop = set(inputs) - {out} | {trans}
    if drop:
        if not inplace:
            df.drop(drop, axis=1, inplace=True)  # drop used columns in the original df, if returning new df
            return df_out.drop(drop, axis=1)
        df_out.drop(drop, axis=1, inplace=True)  # drop used columns in the inplace df
    return df_out


def row_func_prepend_transform(fnc: Callable, trans=Col.read_trans, inp=Col.path, out=Col.data):
    """Prepend transform to the given row function

    :param fnc: function to wrap with pre-transform
    :param trans: name of the column with transform
    :param inp: name of the columns used as input to the transform
    :param out: name of the column to write transform results
    """
    if not trans:
        return fnc

    def trans_fnc(row: pd.Series, *args, **kws):
        row[out] = row[trans](row[inp])
        return fnc(row, *args, **kws)

    return trans_fnc


def _encode_transforms(series: pd.Series, invert=False) -> tuple[list[tuple[str]], dict[tuple[str], dict]]:
    """
    Encode series with transforms (dict of dict or nan) in the cells into a two components data structure:
     - series-size array of tuples with names of transforms, and
     - map from such a tuple into the original transforms of the corresponding cell, or their inverse

    More specifically:
     - every cell of the series may contain dict of transforms ``C = {tr_name: tr_dict}``
     - aggregated transform merges of all the cells dicts into one ``A``
     - inverse cell transform ``IC`` is all the transforms in ``A`` except of those in ``C``: ``IC = A - C``

    **ASSUMPTION**

    Inversion implementation assumes that transforms are uniquely identified by their ``tr_name``

    Return array of such `code names` and a map from name to the actual transforms, which
    contains reference to cells with no transforms in the entry: ``"EMPTY": {}``.

    :param series: a Series with transforms as dict of dict (or NONE)
    :param invert: if TRue - request to construct map with inverse transforms
    :return: array of code names per original cell, and map from the names to associated transforms
    """
    series = series.reset_index(drop=True)  # same index is used by the all_names np.array!
    defined = series[~series.isna()]  # work with cells actually containing transforms

    if not all(filter(lambda x: isinstance(x, dict), defined)):
        raise TypeError("Transforms must be dictionaries")
    empty = 'EMPTY'
    all_names = np.full(series.size, empty, dtype=object)  # results initialized by empty names
    cells_map = {empty: {}}  # cell hash:tuple(names) -> transforms:dict
    trans_map = {}  # trans hash:str -> item:(func, {par}) - needed for INVERT case

    for idx, cell_trans in defined.items():
        items = [*cell_trans.items()]  # list transform items in this cell
        cell_names = tuple(map(str, items))  # tuple of items str (hashable!) in this cell
        if cell_names not in cells_map:  # new type of cell is found
            cells_map[cell_names] = cell_trans  # update mappings of cells and
            invert and trans_map.update(zip(cell_names, items))
        all_names[idx] = cell_names

    if invert:  # notice, cells names remain unconverted but still unique!
        all_trans_names = set(trans_map)  # crashes all transforms names together, ignores arguments!
        cells_map = {cell: dict(map(trans_map.get, all_trans_names.difference(cell)))
                     for cell in cells_map}
    return name_tuple(names=all_names, map=cells_map)


def compile_read_transforms(df: pd.DataFrame, *, col=Col, ns=None):
    """
    From collection (usually a transformed column pd.Series) with dicts
    describing transforms which have been applied on the data (in this row)
    build a collection of transforms required to align the corresponding data
    to same total transformation for entire collection.

    Namely, in set terms:
      ``t_i -> T - t_i``,    (where ``T = sum[ t_i ]``)

    Assumes `commutability` of all the transformations!

    If namespace with functions is provided, use it to substitute
    dicts describing transforms to Callables from this namespace.
    Otherwise, return ``np.ndarray`` with dicts or nans

    :param df:
    :param col: has attributes:  transforms, transformed with names of those columns
    :param ns: namespace with function to substitute from
    :return: align_trans
    """

    ns = ns and Namespace(ns) or {}
    reader = (read_by_format, 'read')

    # Prepare only transforms available in the columns
    has_tm, has_td = map(df.columns.__contains__, [col.transforms, col.transformed])
    if not (has_tm or has_td):  # NO transforms columns - just use reader for all the cells
        return np.full(len(df), O.comp(reader, ns=ns))

    coded = dict(  # hash only existing columns
        td=has_td and _encode_transforms(df[col.transformed], invert=True),
        tm=has_tm and _encode_transforms(df[col.transforms])
    )

    if has_td and has_tm:  # BOTH transform types are defined
        iterator = zip(coded['td'].names, coded['tm'].names)  # iter over codes: (td-name, tm-name)
        td_map, tm_map = coded['td'].map, coded['tm'].map
        trans = lambda k: td_map[k[0]] | tm_map[k[1]]  # combine from transforms and inversed transformed
    else:  # ONLY ONE of transform columns is defined
        iterator, trans_map = coded[has_td and 'td' or 'tm']  # iter over codes: t[d|m]-name
        trans = lambda k: trans_map[k]  # function return transform given code

    def trans_gen():
        """ Generator of transforms for all the rows."""
        cache = {}  # Cache previously seen transforms to avoid expensive creations
        for key in iterator:
            if (t := cache.get(key, None)) is None:
                t = O.comp(trans(key), reader, ns=ns)
                cache[key] = t
            yield t

    return np.array([*trans_gen()])


def is_data(item):
    """Check if item could be any data object:

    `isinstance(item, Iterable) or not pd.isna(item)`
    """
    return isinstance(item, Iterable) or not pd.isna(item)


# noinspection PyTypeChecker
def cond_read_trans(s: pd.Series, check_data: bool | Callable = is_data, check_file=True):
    """
    Conditional variant of ``read_trans`` function `applicable` to DataFrame with
    columns: `Col.data`, `Col.path`, `Col.read_trans`.

    Calls the original `s[Col.read_trans]` if two conditions are satisfied:
     - `check_data(s[Col.data]) is False` AND
     - `check_file` is False OR `s[Col.path].is_file`

    :param s: pd.Series
    :param check_data:
    :param check_file: return ``pd.NA`` instead raising on not existing files
    :return:
    """
    if check_data and 'data' in s:
        if (is_data if check_data is True else check_data)(data := s.data):
            return data  # return existing data

    if path := s[Col.path]:
        if check_file and not Path(path).is_file():
            return pd.NA  # file not found
        read_trans: Callable = s[Col.read_trans]
        return read_trans(path)  # read file

    return pd.NA  # path is not defined


def read_by_format(path):
    """
    Read data from file by determining its format from its name or labels.
    :param path:
    :return: data from the file if format is found or None
    """
    from algutils.io.imread import imread, SUPPORT_READ, imread_stereo
    path = Path(path)
    ext = path.suffix.lower()
    if "StereoImage__Channel_" in path.name and ext == '.tif':  # combined L and R from NU40
        return imread_stereo(path, cam_info=True)  # raise exception if there is no cam_info
    elif ext in SUPPORT_READ:
        return imread(path)

    if path.name.lower() == 'calib.txt':
        from algutils.io.special import middlebury_calib
        return middlebury_calib(path)
    raise NotImplementedError(f"File's {str(path)} {ext=} is not among supported ({SUPPORT_READ=})")


# TODO: remove all the references, including docs
# def align_transformed(df, *, inp=Col.data, out=None,
#                       align_trans=Col.align_trans, transformed=Col.transformed,
#                       keep=(), lib=None) -> pd.DataFrame:
#     """
#     Align all the data items in the column to ensure that same
#     transforms are applied to all of them.
#
#     Note: transforms must commute (applied in arbitrary order)
#
#     :param df: a pd.DataFrame containing the data column to align
#     :param inp: name of the column to align
#     :param out: name of the columns with aligned data (default=`inp`)
#     :param keep: temporal columns to keep, otherwise `transformed`, 'align_trans', `inp`
#     :param align_trans: column name with alignment transforms
#     :param transformed: column describing transforms been applied
#                         to the `inp` by now
#     :param lib:  dictionary of transforms functions, by default from transforms()
#     :return: pd.DataFrame with aligned column
#     """
#     from functools import reduce
#     from box import Box  # to make dicts hashable for sets
#     out = out or inp
#     keep = as_list(keep)
#     if transformed in df:
#         tran_set = lambda t: {*Box(t, frozen_box=True).items()} \
#             if isinstance(t, dict) else set()  # as set of tuples {(func, kwargs), }
#         combined = reduce(set.union, map(tran_set, df[transformed]))  # all transforms
#
#         dif = lambda t: dict(combined.difference(tran_set(t)))  # create column with
#         df[align_trans] = df[transformed].apply(dif)  # missing transforms
#
#         df = apply_transforms(df, trans=align_trans, read=None, keep=keep,
#                               inp=inp, out=out, ns=lib or transforms())  # apply missing transforms
#         if transformed not in keep:
#             df.drop(transformed, axis=1, inplace=True)  # drop temporary column
#     else:
#         if out != inp:  # trivial inp -> out transform
#             df[out] = df[inp]
#         if inp not in keep:
#             df.drop(inp, axis=1, inplace=True)
#     return df
#
#
# def grp_code_regions(g: pd.DataFrame, codes='rgn_codes', data='data', out='series') -> pd.Series:
#     """Transform group by splitting `data` column coded image into Regions
#     object according to mapping in `code` column (usually 'rgn_code').
#     If such column is not found - return the input group
#
#     :param g: group with 'data' and codes columns
#     :param codes: name of the codes mapping column (dict content)
#     :param data: name of the data column (usually 'data')
#     :param out: 'series' | 'frame' | None
#     :return: pd.DataFrame with Regions in decoded rows of data column
#              and with `codes` column dropped
#     """
#     if 'rgn_codes' in g.columns:
#         from algutils.image import Regions
#         data_cols = ['data', codes]
#         is_rgn = g.rgn_codes.notna()
#         g.loc[is_rgn, data] = g.loc[is_rgn, data_cols].apply(lambda p: Regions(*p), axis=1)
#         g = g.drop(codes, 1)
#     return g[data] if out == 'series' else g

_FETCH_RES = Literal['data', 'all', 'essen']


class Fetchable:
    # ToDo: @Ilya - Create Task: g.qix().fetch(), g.select().fetch() keep data in g!

    def fetch(self: pdt.DTable, *, check_file=False,
              out: _FETCH_RES = 'data',
              reuse: bool | Callable = True,
              inplace=True) -> CollectTable | CollectSeries:
        """
        Return table with data fetched according to the read transform of every item.

        Resulted table may be a data-only `series` or `frame` with columns depending
        on the ``out`` argument:
          - 'data' - `series` with only data,
          - 'all' - `frame` with all the original + data,
          - 'essen' - `frame` with essential columns: `all` except those used by transform.
        (`frame` will be produced also in special case when source is series with only `data`)

        If source already has valid data, data is not reloaded if ``reuse is True``.

        The default condition of *validity* (``toolbox.datacast.transtols.is_data``)
        can be replaced by providing alternative function: ``reuse=new_data_cond``.

        If not ``inplace``, fetch returns a new table without changing the source,
        so that subsequent fetching from same source requires loading data again.

        Using ``inplace=True`` allows to avoid reloading in certain scenarios.

        In this case data column is added to the source table.
        Then, only if requested format of the output table is different, a new table is produced.

        It modifies self by loading into 'data' column, AND ALSO returns the data in the requested form.

        Therefore, in this example only first ``fetch()`` will require loading:

        >>> g2 = g1.fetch()                 # loading
        >>> g3 = g2.qix(**labels).fetch()   # no loading
        >>> g1.loc[index].fetch()           # no loading
        ```
        However, that does not work if used AFTER querying:

        >>> g2 = g1.qix(**labels).fetch()   # loading
        >>> g2.fetch()                      # no loading
        >>> g1.loc[index].fetch()           # loading again

        :param self: table
        :param out: columns to return,
        :param check_file: if file in `Col.path` not exists don't raise just return ``pd.NA``
        :param reuse: don't fetch data if already available in `Col.data`
        :param inplace: update self ``DataFrame`` instead of fetching into a copy.
        :return: with selected rows and loaded data as values or a data object
        """
        assert out in get_args(_FETCH_RES)
        trans_labels = [Col.path, Col.read_trans]

        if self.empty:
            self['data'] = None
            return self

        if not isinstance(self, pd.DataFrame):  # Series, must be already with data
            if reuse and self.name == 'data':
                return self if out == 'data' else pdt.as_table(self, series=False)
            raise IndexError(f"Fetch requires {trans_labels} or 'data' columns!")

        if not set(trans_labels).issubset(self.columns):
            if reuse and 'data' in self.columns:
                return self.data if out == 'data' else self
            raise IndexError(f"Fetch requires {trans_labels} or 'data' columns")

        # Consider: use apply_column_transform for cases above to verify data?

        if reuse or check_file:  # change to conditional read if required
            kws = {'check_file': check_file} | (reuse and {'check_data': reuse} or {})
            trans = '__cond_trans__'
            self[trans] = O(cond_read_trans, **kws)
            trans_args = dict(drop=trans, args='series', inp=None)  # series with all columns
        else:
            trans = Col.read_trans
            trans_args = dict(drop=False)

        res = apply_column_transform(self, out=Col.data, trans=trans, inplace=inplace, **trans_args)
        match out:
            case 'data': return res.data
            case 'essen': return res.drop(trans_labels)
            case 'all': return res


class CollectSeries(pdt.DataSeries, Fetchable):
    _metadata = pdt.DataSeries._metadata + ["collection"]

    @property
    def _constructor(self):
        return CollectSeries

    @property
    def _constructor_expanddim(self):
        return CollectTable


class CollectTable(pdt.DataTable, Fetchable):
    _metadata = pdt.DataTable._metadata + ['collection']

    @property
    def _constructor(self):
        return CollectTable

    @property
    def _constructor_sliced(self):
        return CollectSeries


class DataItem(Labels):

    def fetch(self, data='data', trans='read_trans', inp='path',
              reuse=True, check_file=True,
              drop: str | Collection[str] | bool = True):
        if missing := {trans, inp}.difference(self.keys()):
            raise KeyError(f"Fetch requires keys which are {missing=}")

        if reuse or check_file:
            from .transtools import cond_read_trans
            trans = cond_read_trans()

        res = self.copy()
        if not (reuse or data in res):
            res[data] = res[trans](res[inp])
        if drop is True:
            drop = [trans, inp]
        if drop:
            res.drop(drop)
