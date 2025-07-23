from __future__ import annotations

from toolbox.utils.events import Timer

with Timer(f" ← {__file__} imports", "timing", min=0.1, pre=f' → importing in {__file__} ...'):
    from collections import namedtuple
    from pathlib import Path
    from typing import Iterable, Sequence, Tuple, Callable, Union, Literal, Any, TYPE_CHECKING, Collection
    import pandas as pd

    # utils import
    from toolbox.utils import as_list, logger, drop_undef, as_iter
    import toolbox.utils.pdtools as pdt
    from toolbox.utils.cache import CacheMode, Cacher

    # internal imports
    from . import transtools as tr, CollectTable
    from .transtools import Col, DataItem

    if TYPE_CHECKING:
        import numpy as np
        from .caster import DataCaster
        from toolbox.datacast.models import DatasetRM, CollectionRM, SchemeRM

        _DS = Union[DatasetRM, DataCaster, str]

PathT = Union[Path, str]

StringS = pdt.StringS

_log = logger('datacast.collect')

__all__ = ['DataCollection', 'SinkRepo']


class DataCollection:
    """
    Container for labeled data collected potentially from multiple related datasets.

    Description
    ===========
    Merging Datasets
    ----------------
    Practically only datasets sharing same `categories` should be merged
    into collections. Common categories are usually used as index in the
    collection's multi-level indexing structure.

    Separation of categories into index and data is arbitrary, and
    is completely controlled by user, but to reduce configuration
    complexity some default assumptions are made:
        - default data categories: `path`, `data`, `transformed`, `transforms`
        - all other categories used as index
        - categories without variations of values and `ext` are dropped

    Accessing Collection Data Structure
    -----------------------------------
    `DataCollection` provides two levels of access to its data:
        - low level access to `.db` <DatFrame> attribute
        - several high level functions supporting specific use cases:

            - smart iteration with `iter`
            - query based functions: `filter`, `select`
            - information attributes:  `.categories`, `dataset`
            - operations: `drop_non_unique`, `apply_transforms`, ...

    Iterations
    ~~~~~~~~~~
    Method `iter` is aimed to simplify coding of the most common use case
    when processing on every iteration requires access to certain
    semantic context selected from the whole dataset.

    It provides great flexibility by allowing to:
        - iterate over groups of elements united by custom criteria
        - a group structure can be specified
        - default transformation applied on the fly

    Data Transformation
    -------------------
    Some data sets include data transformation instructions
    as part of their schemes in form of `transforms` and `transformed`
    labels assigned to specific data items.
    `transforms` (T) define set of operations to be applied on the item.
    `transformed` (Td) describes operations which has been applied.

    This is especially relevant to make sure data in different
    items is aligned for its size, scale, range, etc.

    The assumption is that `transforms` would project data into the
    common _reference_ state, and `transformed` defines how the data
    been transformed from this state.

    The alignment procedure then includes two steps:    \n
    1.. Apply all the `transforms` T on the items where they are defined \n
    2.a. Apply inverse transforms Td^(-1) where Td are defines OR \n
    2.b. Apply forward transform Td on the rest of the items

    Problem of Inverse Transformations
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Option 2.a is considered to be preferable, if inverse is defined,
    otherwise 2.b. is applied.

    In the case when both direct and inverse transform can be defined
    the choice of marking `transformed` items with Td or the rest of
    the items in the dataset with Td^(-1) is not quite symmetric,
    as aforementioned reference state may have some natural meaning,
    shared between different datasets.

    Since every dataset description must be self-contained, in order
    to allow merging between datasets certain semantic convention on
    the data must be maintained, making option 2.a even more desirable.

    If forced towards 2.b the proper approach would be to apply Td on
    all the data elements in DataCollection _after_ datasets are merged.
    Its not always clear which specific elements require this correction.
    For example cropping can be be universally applied to all the images,
    but values transformation only to the data of same *kind*.

    Approach to Inverse Transformation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    To avoid all the complications for step 2 it can be delegated to
    the algorithms sensitive to those transformation which would access
    information in the `transformed` label and act accordingly.
    """

    _TRANS_COL = {Col.transforms, Col.transformed}
    _NO_CAT_COL = {Col.read_trans} | _TRANS_COL  # non category columns

    DATA_COL = (Col.path, Col.read_trans)

    class _HistoryItem(namedtuple('HistoryItem',
                                  ['oper', 'desc', 'args', 'kws', 'before', 'after'],
                                  defaults=['', '', (), None, None, None]
                                  )):
        def __repr__(self):
            from toolbox.utils.strings import dict_str
            change = "" if (self.before is None or self.after is None) \
                else f" {self.before}⮕{self.after}"
            desc = self.desc or ','.join(filter(None, [
                ','.join(as_list(self.args)),
                dict_str(self.kws)
            ]))
            return f"{self.oper}({desc}){change}"

    class Group(namedtuple('Group', ['gid', 'grp'])):
        """ Namedtuple Group(gid, grp), also can be initialized by a
        tuple-like Group((gid, grp))
        """

        def __new__(cls, *args):
            if len(args) == 2:
                gid, grp = args
            else:
                group, gid, grp = args
                if len(group) > 1:
                    gid = DataItem(zip(group, gid))
            if not isinstance(grp, CollectTable):
                grp = CollectTable(grp)
            return super().__new__(cls, gid, grp)

        def __repr__(self):
            return f"<Group> gid: {self.gid}, grp:\n{self.grp}"

        def fetch(self, **kws):
            self.grp = self.grp.fetch(**kws)
            return self

    @property
    def db(self):
        return self._db

    # attributes which could be inherited when modifying a collection
    _inherit_attrs = ['name', 'casters', 'bundle', 'cacher']

    @classmethod
    def from_db(cls, db, *, like: DataCollection = None,
                history: list[str | tuple] | [str | tuple] = None,
                name='', casters=None, bundle=None, cacher=None):
        """
        Create a DataCollection object from given db and optionally other
        attributes.

        :param db:  DataTable(Frame) object to build the collection from
        :param like: DataCollection object to inherit some of its attributes
        :param history: list of history records or single history record,
        in which case append to the history in the `like` collection.
        :param name: override that in like. Required if like is None!
        :param casters: if provided override that in like
        :param bundle: if provided override that in like
        :param cacher: if provided override that in like
        :return: a new DataCollection
        """
        hist_item = lambda h: isinstance(h, str) and cls._HistoryItem('stage', h) or cls._HistoryItem(*h)

        if like is None and not isinstance(name, str):
            raise ValueError("Provide either name or like")
        casters = casters or []
        bundle = as_list(bundle) or []

        dc = cls.__new__(cls)
        # TODO insert here handling more than db
        dc._db = db if isinstance(db, CollectTable) else CollectTable(db)

        if isinstance(history, (str, tuple)):
            history = hist_item(history)
            dc._history = like and like._history + [history] or [history]
        elif isinstance(history, list):
            dc._history = list(map(hist_item, history))
        elif not history:
            dc._history = like and like._history or []
        else:
            raise TypeError("Invalid history argument")

        lc = locals()
        for attr in cls._inherit_attrs:
            if not (val := lc[attr]) and like:
                val = getattr(like, attr)
            setattr(dc, attr, val)
        dc.cacher is True and dc._create_cacher(True)
        return dc

    def __init__(self, name: str | CollectionRM | _DS = None, *,
                 datasets: Union[_DS, Iterable[_DS]] = None,
                 label_datasets: bool = None,
                 query: str | dict | Tuple[str, str | dict] = None,
                 bundle: list[str] = None,
                 # below are not CollectionRM arguments
                 unique=False,  # Consider: not needed argument?
                 data: str | Iterable[str] = DATA_COL,
                 drop: str | Iterable[str] = None,
                 cache: CacheMode | bool | Literal['LOAD', 'SAVE', 'KEEP', 'PASS'] = None,
                 temp_cache: bool | Path | str = None,
                 calc_cache: PathT | Cacher | bool = False,  # Consider: remove calc_cache* and may be calc!
                 calc_cache_rel: PathT | bool = True,
                 progress: bool = False,
                 description: str = None):
        """ Create collection of labeled paths from multiple schemes
        extracting labels from data's folders structure (or another DataCollection).

        :param name: optional name identifying the dataset and its **scheme**
        :param datasets: the desired datasets to collect in the form of
        `DataCaster` or `DatasetRM`.
        :param query: optional query to select only a subset of the collection
                      in str or dict form
        :param data: which labels categories to consider as data (others - index)
        :param unique: leave only labels with multiple unique values
        the schemes-defined transforms to replace them in read_trans column.
        :param drop: try to drop those categories if that keeps the index unique
        :param cache: optionally override caster's caching mode (`None` - don't)
        :param temp_cache: optionally override caster's `temp` argument (`None` - don't)
        :param calc_cache: (for calculations over the DS (not scheme!), values:
                    - path to cache folder or
                    - Cacher instance or
                    - False to disable cacher
                    - True for automatic path (self.common_root / .cache)
        :param calc_cache_rel: store cached paths tables relative to this folder
               which can be provided explicitly or:
                - False | None | '' - store absolute
                - True | 'common' - use this self.common_root folder
                Ignored if cache is Cacher object
        :param description: String that describes the collection.
        """
        from .caster import DataCaster
        from .models import DatasetRM, CollectionRM

        # --- deal with different supported arguments formats ----
        if isinstance(name, (DatasetRM, DataCaster)) and datasets is None:
            name, datasets = None, [name]
        elif not (name is None or isinstance(name, (str, CollectionRM))):
            raise ValueError("DataCollection init requires either "
                             "name (str or collection) or datasets arguments")
        cfg = locals().copy()  # turn arguments into CollectionRM kwargs

        #  ------------------ Build DataCasters and collect data --------------------------
        changes = drop_undef(cache=cache, temp_cache=temp_cache)  #

        def prep_caster(ds):
            if isinstance(ds, DataCaster):
                return DataCaster(**(ds.config.dict() | changes)) if changes else ds
            return DataCaster(**(ds if isinstance(ds, dict) else {'name': ds}), **changes)

        if not datasets:  # first *find* config with datasets from the collection name - then build casters
            self.config = CollectionRM.from_config(cfg, ignore=True, undefined=False)
            self.casters = [prep_caster(d) for d in as_iter(self.config.datasets)]
        else:  # first build casters from the *provided datasets* - then create a new config
            self.casters = [prep_caster(d) for d in as_iter(datasets, no_iter=(DataCaster, DatasetRM))]
            cfg['datasets'] = [ds.config for ds in self.casters]  # update dataset from casters config
            self.config = CollectionRM.from_config(cfg, ignore=True, undefined=False)

        if not self.casters:
            raise ValueError("Can't create collection without datasets")
        from pandas import concat
        db = CollectTable(concat(cst.collect(progress=progress) for cst in self.casters))

        # ----------------------  set principal attributes -----------------
        self.name = self.config.name
        self.description = description
        self._history: list[DataCollection._HistoryItem] = []
        bundle = as_list(bundle) or sum(
            filter(None, map(lambda s: s.bundle, self.casters)), []
        )
        self.bundle = list(set(bundle))

        if db.empty:
            self._db = db
            return

        #  ---------------------- Filters ----------------------------
        # Here all the categories found in the data are in the columns
        # Filter query may use categories which are removed later
        sq, dq = ('', query) if isinstance(query, dict) else (query, {})
        db: CollectTable = self._select(db, sq, **dq)
        if unique:  # if requested to remove categories with same value
            db = db.drop(columns=db.columns[(db.nunique(dropna=False) == 1)])

        # --------- Setup data and index columns
        data: set = as_list(data, collect=set)  # explicitly requested columns
        db = self._init_transforms(db, data)
        self._db = self._init_index(db, data, drop)

        self.cacher: Cacher = self._create_cacher(calc_cache, calc_cache_rel)

    def drop_flat_categories(self, *drop, keep=None, only_nan=False, fail=False,
                             keep_bundle: bool | str | Collection[str] = True):
        """
        Remove uninformative categories from the index.

        - If `drop` is provided, select only remove from those.
        - Categories in `keep` are excluded from the drop.

        :param drop: if provided, only categories from this list are verified
        :param keep: exclude from verification
        :param only_nan: if `True`, drop only if all category values are `None`
        :param keep_bundle: if `True`, *bundle* categories are excluded,
           otherwise bundle may be reduced. by dropped categories,
           unless *all* the bundle is 'flat' in this case it is kept.
        :param fail: raise `ValueError` on errors or just return without changes
        :return:
        """

        def _exit(msg):
            _log.error(msg)
            if fail:
                raise ValueError(msg)

        db = self._db
        all_cats = set(db.index.names)
        drop, keep = (as_list(_, collect=set) for _ in (drop or all_cats, keep))

        if isinstance(keep_bundle, bool):
            keep_bundle = self.bundle if keep_bundle else []
        else:
            keep_bundle = as_list(keep_bundle, collect=set)
            if invalid := keep_bundle.difference(self.bundle):
                return _exit(f"Invalid bundle categories {invalid}")
        keep = keep.union(keep_bundle)

        cats = all_cats.difference(keep).intersection(drop)

        if (sz := len(db)) < 2:
            return _exit(f"Can't determine flat categories in collection with {sz} rows")

        drop_cats = [name for name in cats if (
                (unq := db.index.unique(name)).size == 1
                and (not only_nan or pd.isna(unq[0]))
        )]

        if not drop_cats:
            return
        elif len(drop_cats) == len(all_cats):
            return _exit("Can't drop ALL the index levels!")

        bundle = set(self.bundle).difference(drop_cats)
        if not bundle and self.bundle:
            return _exit("Dropping categories eliminates the bundle, "
                         "consider constraining with keep_bundle")

        self._db = db.droplevel(list(drop_cats))
        self.bundle = list(bundle)

    def _init_index(self, db, data: set, drop):
        """
        Move categories (columns) into index, except of:
         - explicitly defined as `data`
         - columns containing any object type except of ``bool``, ``str``
        """
        data = data.intersection(db.columns) | {c for c, s in db.items() if s.dtype.hasobject and
                                                not pdt.series_has_types(db[c], (str, bool))}
        drop: set = as_list(drop, collect=set)
        db.drop(columns=list(drop & data), inplace=True)  # drop requested if they are not in data
        if index_labels := list(db.columns.drop(data)):  # all the non-data labels
            db.set_index(index_labels, inplace=True)  # create index
            if db.index.unique().size != db.index.size:
                raise IndexError("Collection index is not unique, invalid scheme?")
            db = self.drop_non_unique(db, drop, inplace=True)  # drop requested index levels only if unique
        return db

    def _init_transforms(self, db, data: set):
        if Col.read_trans in data:
            db[Col.read_trans] = tr.compile_read_transforms(db, ns=tr.transforms())
        db.drop(columns=list(self._TRANS_COL - data), errors='ignore', inplace=True)
        return db

    def _create_cacher(self, cache: Cacher | PathT | bool, cache_rel: PathT = True) -> Cacher:
        """ Create cacher object from different forms of definitions
        (in case of Cacher input return it ignoring other arguments).

        :param cache: path to cache folder or Cacher object
                - False to disable cacher
                - True for automatic path (self.common_root / .cache)
        :param cache_rel: store cached paths tables relative to this folder
               which can be provided explicitly or:
                - False | None | '' - store absolute
                - True | 'common' - use this self.common_root folder
                Ignored if cache is Cacher object
        :return: created Cacher object
        """
        from toolbox.utils.cache import CacheMode, Cacher
        if isinstance(cache, Cacher):
            return cache

        if cache_rel in (True, 'common'):
            cache_rel = self.common_root

        mode = CacheMode.PASS
        if cache:
            mode = CacheMode.KEEP
            if isinstance(cache, bool):
                cache = self.common_root / '.cache'
            else:
                cache = Path(cache).expanduser().absolute()
            cache.mkdir(parents=True, exist_ok=True)  # make sure cache folder is valid
            _log.debug(f'Caching folder for {self.__class__.__name__} "{self.name}": {str(cache)}')

        return Cacher(folder=cache, mode=mode,
                      pack=pdt.path_fixer(cache_rel, 'crop'),
                      unpack=pdt.path_fixer(cache_rel, 'add'),
                      load=pdt.pd.read_pickle, save=lambda f, df: df.to_pickle(f))

    def measure_data(self, msr_fnc: Callable[[...], dict | Sequence], *, df: pd.DataFrame = None,
                     pos: StringS = Col.data, kwcol: StringS = None,
                     alias: dict[str, str] = None, out: StringS = None,
                     cache: Cacher | bool | dict = False, parallel: pdt.Parallel | str | bool = False,
                     direct: bool = False, reindex: bool = True, args: tuple = (), **kws) -> pd.DataFrame:
        """
        Perform measurements on the internal (default) or provided DataFrame.

        Measurements are per row, and are defined by the provided ``msr_fnc`` which MUST
          - receive `at least one positional argument` (data), and
          - produce results in either named (dict-like) or not form (list-like).

        ``out`` argument may be used provide (or override) names of the results,
        which defines also the names of the resulting columns in the returned DataFrame.

        Additional arguments of ``msr_fnc`` may come either from columns,
        or are common for all the rows (``args`` and ``kws`` arguments).

        In current implementation to use index levels as inputs, they should be first reset into columns.

        ``pos``, ``kwcol`` and ``alias`` arguments relate columns names with ``msr_fnc`` arguments,
        and MUST include `data` column in one of them.

        Caching of the calculations results is configured by `cache` argument.
        Since index is not used, it's not cached either, and is attached to the columns
        loaded from the cache.

        Parallelization control follows ``pdtools.Parallel.from_flag()``

        :param msr_fnc:  function to
        :param df: if provided used instead of the internal db
        :param pos: name[s] of the columns to be used as msr_fnc positional arguments
        :param kwcol: name[s] of the columns same as msr_fnc keyword arguments
        :param alias: mapping of columns names into the corresponding msr_fnc names
        :param out: names of the resulting columns
        :param cache: a Cacher object | True to use default | dict to update default's parameters
                    str as shortcut {'name': str} | False or None to disable
        :param parallel: 'swift'|'jobs'|int|True|False|None
        :param direct: directly perform pre-built transformations
        :param reindex: reindex the resulting frame to source index
        :param args: additional positional arguments passed into msr_fnc
        :param kws: additional keyword arguments passed into msr_fnc
        :return:
        """
        # --- Pre-process Arguments
        pos, kwcol, out = map(as_list, [pos, kwcol, out])
        alias = alias or {}

        df = df if df is not None else self.db
        engine = pdt.Parallel.from_flag(parallel) or type(df)  # parallel or DataFrame apply-engine

        def cache_config():
            """return dict with caching parameters:"""

            def auto_name(name=''):
                db_hash = self.hash_str(columns=sorted(use_col), index=False)
                return '_'.join(filter(bool, [getattr(msr_fnc, '__name__', ''), name, db_hash]))

            if not cache:
                return {'mode': False}  # switch cacher OFF
            if isinstance(cache, dict):
                return cache | {'name': cache.get('name', auto_name())}  # update cacher
            if isinstance(cache, (bool, str)):
                return {'name': auto_name('' if cache is True else cache)}
            raise TypeError(f"Unexpected {type(cache)=}")

        def req_trans():
            """Find transform columns required to produce missing columns or raise if impossible!
            Return name of the column with transform and set of columns used by the measurement
            """
            avl_cols = [*df.columns]  # columns available for measurement function
            req_cols = {*pos, *kwcol, *alias}
            if missing := req_cols.difference(avl_cols):  # from required by the mappings
                trans_inp, trans_out = {Col.path, Col.read_trans}, {Col.data}  # supported transform
                if missing == trans_out and trans_inp.issubset(avl_cols):  # transform can provide missing
                    return Col.read_trans, (req_cols - trans_out).union(trans_inp)
                raise LookupError(f"Missing columns defined as function arguments: {missing}.")
            return None, req_cols  # not needed

        if not direct:
            # --- Construct measurement function: first by-row version, then prepend transform, then apply
            trans, use_col = req_trans()
            row_fnc = pdt.col_args_row_func(msr_fnc, pos=pos, kwcol=kwcol, alias=alias, out=out)
            row_fnc = tr.row_func_prepend_transform(row_fnc, trans=trans)  # does nothing if not trans
        else:
            row_fnc = tr.row_func_prepend_transform(msr_fnc)  # use pre-built transformations

        if not isinstance(cache, Cacher):
            cache = self.cacher.context(**cache_config())

        @cache  # make measurement cachable if 'mode' is active
        def measure(_df, _args, **_kws):  # expose arguments caching must be sensitive to
            return engine.apply(_df, row_fnc, axis=1, result_type='expand', args=_args, **_kws)

        return measure(df.reset_index(), args, **kws).set_index(df.index) if reindex \
            else measure(df.reset_index(), args, **kws)

    def changelog(self, oper='', desc='', args=(), kws=None, before=None, after=None):
        """Add to the changelog of the data collection.
        if exactly one of `before` or `after` is provided, the other is taken from the current state.

        :param oper: name of the operation led to the change
        :param desc: description of the operation, could be parameters
        :param args: arguments the operation function was called with used
        :param kws: keyword arguments the operation function was called with used
        """
        if before is None and after is not None:
            before = len(self._db)
        elif before is not None and after is None:
            after = len(self._db)
        self._history.append(self._HistoryItem(oper, desc, args, kws or {}, before, after))

    def copy(self):
        """Return a copy of the object , deep if requested"""
        return self.from_db(self.db, like=self)

    @property
    def caster(self):
        """Caster in the base of the collection or None if multiple"""
        return len(self.casters) == 1 and self.casters[0] or None

    @property
    def common_root(self) -> Path:
        """Common root of all the schemes in the collection."""
        if len(self.casters) == 1:
            common = self.casters[0].root
        else:
            from os.path import commonpath
            common = commonpath(s.root for s in self.casters)
        return Path(common)

    @property
    def categories(self):
        """Set of categories, in both index and data sections."""
        return set(self.db.columns).union(filter(None, self.db.index.names)) - self._NO_CAT_COL

    def category(self, name):
        return self.db.index.get_level_values(name).unique()

    def index_info(self, *, width=70, out=None):
        """
        Collect and return info about the database index in different forms
        :param out: None - print, str - same as string, dict - {level: unique values}
        :param width: line width in characters
        :return: one of the formats or None if `out` is None
        """
        import pandas as pd

        def repr_seq(total, seq):
            if isinstance(seq, str): return seq
            sep_len = len(sep := ', ')
            res = ''
            for s in map(str, seq):
                if len(s) + len(res) + sep_len < total:
                    res = res and f"{res}{sep}{s}" or s
                else:
                    res = res + f'… [{len(seq)}]'
                    break
            return res

        index = self.db.index
        levels = {
            lvl or '∅': f"Range[{index.start}:{index.stop}:{index.step}]"
            if isinstance(index, pd.RangeIndex) else
            index.unique(lvl) for lvl in index.names
        }
        if out is dict:
            return levels

        names_width = max(map(len, levels))
        fmt = f"{{:>2}}:{{:<{names_width}}} : {{}}"
        width -= (names_width + len(fmt.format(0, '', '')))
        info = '\n'.join(fmt.format(lid, lvl, repr_seq(width, vals))
                         for lid, (lvl, vals) in enumerate(levels.items()))
        return info if out == str else print(info)

    def __len__(self):
        return len(self.db)

    def __str__(self):
        return f"📚Collection '{self.name}' [{len(self)}] ({len(self.casters)} schemes)"

    def __repr__(self):
        if self.empty:
            return f"📚Collection '{self.name}' is EMPTY!"
        bundle = set(self.bundle)
        mark_bundle = lambda c: c in bundle and f"✓{c}" or c
        index_cats = ','.join(mark_bundle(c) for c in self.db.index.names if c)
        data_cats = ','.join(self.db.columns)

        hsep = ' ▷ '
        hpfx = "➿ Altered: "
        history = hsep.join(map(str, self._history))
        if len(self._history):
            if len(history) > 70:
                history = ('\n  ' + hsep).join([hpfx, *map(str, self._history)]) + '\n'
            else:
                history = hpfx + history + '\n'

        return f"{str(self)}\n" \
               f"{history}" \
               f"🔖 Index[{index_cats}] ⮕ Data[{data_cats}]" \
               f"\n{self.index_info(width=80, out=str)}"

    def keep_levels(self, level: str | Sequence[str], *, strict=True):
        """
        **Change internal** ``.db`` by leaving only given level(s) in the *given order*
        in the axis 0 index levels.

        :param level: name or sequence of names of levels
        :param strict: if True raise when unknown level is provided.
                       otherwise just ignore it.
        :param: return self with updated db.
        """
        self._db = self._db.keep_levels(level, strict=strict)
        return self

    @staticmethod  # Consider: move to pdtools
    def drop_non_unique(src: pdt.DTable, drop: str | Iterable[str], *, inplace=True):
        """Try to drop categories in drop if that keeps index unique.
        :param src: data frame
        :param drop: categories to drop if all the values are not unique
        :param inplace: if True operate on the input collection db
        :return: updated collection
        """
        trg = src if inplace else src.copy(True)
        for cat in as_list(drop):
            if cat in trg.index.names:
                new_idx = trg.index.droplevel(cat)
                if len(new_idx.unique()) == len(new_idx):
                    trg.index = new_idx
        return trg

    @staticmethod
    def _select(df: pd.DataFrame, query: str = None, **kws) -> pd.DataFrame:
        """ Select from the given DataFrame a subset according to a query in
        string or dict form.
        :param df: DataFrame
        :param query: str may include an arbitrary python expression supported by
                      DataFrame.query() in terms of the categories of the labels:
                                ``"cat1 == value1 or cat2 * 2 < value"``
        :param kws:    query in category=value pairs selects data
                            where ``cat1 == value1 and cat2 == value and ...``
                       Either kws OR string query may be provided
        :return: resulted selected as a DataFrame
        """
        if kws and not query:
            query = kws
        elif query and kws:
            raise ValueError("Both query AND keywords arguments are provided!")
        if not query:
            return df
        if isinstance(query, dict):
            query = _dict_to_query(query)
        if not isinstance(query, str):
            raise TypeError('Query must be a string!')
        return df.query(query, engine='python')

    def sample(self, selection: slice | int | float,
               comment: str = None, shuffle=False, bundle=True, inplace=False):
        """
        Sample the db using the slice.

        :param selection: slice to use (slc:int means [0:scl])
        :param comment: describe this operation for logging
        :param shuffle: if True - random shuffle before sampling
        :param bundle: list of index levels to group and sample from groups.
            ``True`` - use `self.bundle`.
            ``False`` - sampling rows.
        :param inplace: change the db itself
        :return: changed object or copy
        """
        bundle = self.bundle if bundle is True else bundle
        bundle_info = "" if not bundle else bundle == self.bundle and "<bundle>" or bundle

        db = pdt.sample(self._db, selection=selection, shuffle=shuffle, groups=bundle)
        before, after = len(self._db), len(db)
        if before == after:
            return self

        comment = comment or f'[{selection}]{shuffle and " Shuffle" or ""} {bundle_info}'
        history = self._history + [self._HistoryItem('sample', comment, (selection,), {}, before, after)]
        if inplace:
            self._db = db
            self._history = history
            return self
        return type(self).from_db(db, history=history, like=self)

    # Consider: rename to query?
    def select(self, query: str = None, **kws) -> pd.DataFrame:
        """ Select from the internal db.
        :param query: str may include an arbitrary python expression supported by
                      DataFrame.query() in terms of the categories of the labels:
                                ``"cat1 == value1 or cat2 * 2 < value"``
        :param kws:    query in category=value pairs selects data
                            where ``cat1 == value1 and cat2 == value and ...``
                       Either kws OR string query may be provided
        :return: resulted selected as a DataFrame
        """
        return self._select(self.db, query, **kws)

    # Consider: why both filter and select?
    def filter(self, query: str = None, *, comment: str = None, inplace=False, **kws):
        """ Select from the internal db.
        :param query: str may include an arbitrary python expression supported by
                      DataFrame.query() in terms of the categories of the labels:
                                ``"cat1 == value1 or cat2 * 2 < value"``
        :param comment: description of the meaning of this filter
        :param inplace: if True change own data-base and return self
        :param kws:    query in category=value pairs selects data
                            where ``cat1 == value1 and cat2 == value and ...``
                       Either kws OR string query may be provided
        :return: DataCollection Filtered
        """
        trg = self if inplace else self.copy()
        if not (query or kws):
            return trg

        args = ((query,), kws)
        if kws and not query:
            query, kws = _dict_to_query(kws), {}

        db = trg.select(query=query, **kws)
        before, after = len(trg._db), len(db)
        if before != after:
            trg.changelog('filter', str(comment or query), *args, before, after)
        trg._db = db
        return trg

    def qix(self, *args, drop_level: bool = False, trans=False,
            axis=None, key_err=True, as_dc=False, **kws) -> CollectTable | DataCollection:
        """Fuzzy query of data collection index and return either filtered
        version of the data collection (as_dc argument) or
        filtered `db` table, optionally in *transformed* formed (data loaded).

        Different query types are supported:
         - by

        :param args: list of values from one of the index levels
                     - will try all until first is found or raise KeyError
        :param drop_level: if True - drop found levels from the result
        :param trans: if True apply data transforms when producing results
        :param axis: if specified query only this axis
        :param key_err: if False ignore key errors
        :param as_dc: results are returned as a new DataCollection
        :param kws:  {level: value} - eliminates exhaustive search in all levels

        :return: Transformed selection's data as DataSeries object.
        """
        import pandas as pd
        # support of usage of direct index instead of smart queries
        if trans and as_dc:
            raise ValueError("Transforms can not be applied if producing data collection")

        # a single unnamed argument could be an index or list of indices
        if not kws and len(args) == 1 and isinstance(args[0], (list, tuple, pd.Index)):
            if isinstance(idx := args[0], tuple):
                idx = list(idx) if isinstance(idx[0], tuple) else [idx]
            sub = self.db.loc[idx]
            if len(sub) != (len(idx) if isinstance(idx, list) else 1):
                raise IndexError("Invalid index passed to qix")
        else:  # query based indexing
            sub = self.db.qix(*args, drop_level=drop_level, axis=axis, key_err=key_err, **kws)

        if trans:
            import pandas as pd
            sub = tr.apply_column_transform(sub)
            if isinstance(sub, pd.DataFrame):
                sub = sub[Col.data]

        if not as_dc:
            return sub

        # ----  prepare new DataCollection parameters ----
        before, after = len(self.db), len(sub)
        if before == after:
            return self

        if bundle := self.bundle:
            bundle = [*filter(sub.index.names.__contains__, self.bundle)]

        from toolbox.utils.strings import dict_str
        join = lambda seq: seq and [','.join(map(str, seq))] or []
        desc = join(join(args) + (kws and [dict_str(kws)] or []))[0]
        return type(self).from_db(sub, like=self, bundle=bundle,
                                  history=('qix', desc, args, kws, before, after),
                                  name=isinstance(as_dc, str) and as_dc or self.name)

    def fetch(self, labels: Sequence[dict[str, Any]], *, levels=None, first_found=False):
        return self._db.select(labels, levels=levels, first_found=first_found).fetch()

    def add(self, other: pd.DataFrame | DataCollection, *,
            other_name='DataFrame', desc=None, inplace=False, **kws):
        """Add another data collection or bare db DataFrame
        to append to this one or form a new.

        :param other: a dc or db to add
        :param other_name: used if other is DataFrame
        :param desc: to use in changelog
        :param inplace: if True update this
        :return: new DC of this updated
        """
        import pandas as pd
        # when other is None or empty
        if other is None or (isinstance(other, DataCollection) and other.empty):
            return self
        assert isinstance(other, (DataCollection, pd.DataFrame))
        is_dc = isinstance(other, DataCollection)
        assert is_dc or isinstance(other, pd.DataFrame)
        # when self is empty but other have new data
        if self.empty:
            return other if is_dc else self.from_db(other, like=self)
        odb, other_name = (other.db, other.name) if is_dc else (other, other_name)
        name = f"{self.name}+{other_name}"

        db = pd.concat([self.db.reset_index(), odb.reset_index()])
        idx1, idx2 = self.db.index.names, odb.index.names
        db.set_index(idx1.union(idx2.difference(idx1)), inplace=True)
        before = len(self), len(odb)

        if is_dc:
            def rename_hist(obj):
                rename_item = lambda h: h._replace(oper=f"<{obj.name}>{h.oper}")
                return [*map(rename_item, obj._history)]

            upd = dict(
                name=name,
                casters=self.casters + other.casters,
                bundle=list(set(self.bundle + other.bundle)),
                history=sum(map(rename_hist, [self, other]), []),
                cacher=self.cacher or other.cacher or None
            )
            kws.update({k: v for k, v in upd.items() if k not in kws})
        desc = desc or kws.get('name', name)

        if inplace:
            dc = self
            self._db = db
            any(setattr(self, k, v) for k, v in kws.items() if k != 'cacher')
        else:
            dc = type(self).from_db(db, like=self, **kws)
        dc.changelog('Add', desc, before=before, after=len(db))
        return dc

    def __iadd__(self, other):
        return self.add(other, desc='+=', inplace=True)

    def __add__(self, other):
        return self.add(other, desc='+')

    @classmethod
    def mix(cls, col_sel: Iterable[tuple[DataCollection, Union[float, int, tuple]]],
            shuffle=True, bundle: tuple[list[str]] | list[str] | bool = True) -> DataCollection:
        """
        Returning new DataCollection with a sample of each input dc with its correspondent proportion.

        :param col_sel: iterable over tuples: (data collections, sampling params), (or `dict` of such)
        :param shuffle: if True - random shuffle before sampling for each dc.
        :param bundle: The bundle for the sampling
             - ``True`` to each use its own `dc.bundle`, or list of bundles per
             or user defined bundle for each dc
             - ``list[str]`` bundle - for a mutual bundle for all
                                                    None - sampling rows
        :return: DataCollection with mix of collections.
        """
        if isinstance(col_sel, dict):
            col_sel = col_sel.items()
        if not isinstance(bundle, list):
            from itertools import repeat
            bundle = repeat(bundle)

        return sum(
            (dc.sample(selection=prop, shuffle=shuffle, bundle=bnd)
             for (dc, prop), bnd in zip(col_sel, bundle)),
            cls.from_db(CollectTable())
        )

    def __getitem__(self, req: int | slice | dict):
        if isinstance(req, int) or isinstance(req, slice):
            return self.db.iloc[req]
        if isinstance(req, dict):
            records = self.select(**req).to_dict(orient='records')
            if not len(records):
                raise KeyError(f'Items with key {req} not found in {self.__class__}')
            return records if len(records) > 1 else records[0]
        raise TypeError(f'Unsupported key type {type(req)}!')

    def __iter__(self):
        return self.db.iterrows()

    def apply(self, *trans, inp=Col.data, out=Col.data):
        """Apply sequence of transformations to the dataset columns
        :param trans: sequence of transformation stage instructions:
                        `(inp, fnc, out)` or `(inp, fnc)` or `fnc`
        :param inp: default inp column for the first stage
        :param out: default output column of any stage if out is missing
        """
        from .transtools import apply_to
        apply_to(self.db, *trans, inp=inp, out=out)

    def groups_num(self, group: str | list[str] = None, **kws):
        """Count number of specific groups in the database.

        :param group: categories to group by
        :param kws: other DataFrame.groupby arguments
        """
        if not (group := group or self.bundle):
            raise ValueError("Neither group nor bundle is defined")
        return self.db.groupby(as_list(group), **kws).ngroups

    class Iter:
        def __init__(self, iterator, tuples: bool, size=NotImplemented):
            self.iter = iterator
            self.tuples = tuples
            self.size = size

        def __iter__(self):
            return self.iter.__iter__()

        def __next__(self):
            return self.iter.__next__()

        def __length_hint__(self):
            return self.size

        def __mod__(self, filter_func: Callable):
            return self.__class__(
                (it for it in self.iter if filter_func(*(
                    it if self.tuples else (it,)
                ))),
                tuples=self.tuples)

        def __truediv__(self, apply_func: Callable):
            return self.__class__(
                (it.__class__(it.gid, apply_func(it.grp)) for it in self.iter)
                if self.tuples else
                (it.__class__(apply_func(it)) for it in self.iter),
                tuples=self.tuples,
                size=self.size)

    def iter(self, group: str | Sequence[str] = None, *,
             index: str | Sequence[str] = None,
             data: str | Sequence[str] = None, gid=True,
             trans: bool | str | Tuple[str, str] = False,
             islice=None, shuffle=False,
             out: str = None, progress: bool | dict = None) -> Iter:
        """
        Wraps `toolbox.utils.pdtools.group_iter` to wrok on the internal
        DataFrame (self.db), also can transform and pivot the groups.

        Docs from `toolbox.utils.pdtools.group_iter`:
        -----------------------------------------
        Create iterator over groups of paths with flexible organization,
        producing either tuples (if `gid` is True): (group_index, group_data)
        or just `group_data` items (DataFrame or CollectionTable)

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

        :param shuffle:
        :param group: one or list of categories to group by.
        :param index: list of categories to be used as multi-index levels
                      of the resulting groups (also defines their order)
                        if 'db_index' use the index of the db
                        if ``None`` - keep original index, or
                        remove from it ''group'' levels if ``gid`` is True
        :param data: keep those categories in the data (as colums)
                      if `None`: produce `DataFrame` with all data categories as columns
        :param gid: if True return namedtuple `GID(gid, grp)` otherwise only `grp`
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
        :param trans: a tuple (inp, out), or just out, equivalent to (out, out) -
                       columns names to apply transforms and alignments (in-place):
                       `inp -> apply_transforms -> out -> align_transformed -> out`
                       If more flexibility is required, disable automatic transfomations
                       trans = False or None, and apply explicitely calling
                       `from ds.transform: apply_transforms, align_transformed`
        :param progress: Show progress bar if `True` or `dict` with args for `tqdm`,
                        `None` - decide automatically
        :param shuffle: shuffle order of groups
        :param islice: arguments for ``itertool.islice`` (start, [stop, [step]])
        :return: iterator over tuples: ((grouped by categories), group)
                 or, if gid is False - over group CollectionTable ( or DataSeries)
        """
        import pandas as pd
        if not (group := group or self.bundle):
            raise ValueError("Neither group nor bundle is defined")

        if trans:  # various options to define dict(inp=..., out=...)
            if trans is True:
                trans = [Col.path, Col.data]
            elif isinstance(trans, str):
                trans = [trans] * 2
            if not isinstance(trans, dict):
                trans = dict(zip(('inp', 'out'), trans))
            assert len(trans) == 2 and {'inp', 'out'}.issubset(trans)

        data = (trans['out'] if trans else list(self.db.columns)
                ) if data is None else as_list(data)
        group = as_list(group)
        if not index:
            index = self.categories.difference([*self.db.columns, *(group if gid else [])])
        elif index == 'db_index':  # ToDo: rename to 'keep'
            index = self.db.index.names

        def prep_group(item: Tuple[Tuple, pd.DataFrame]) -> Tuple[Tuple, pd.DataFrame]:
            idx, grp = item
            if trans:
                grp = tr.apply_column_transform(grp, **trans)
            grp = grp.sort_index()
            if isinstance(grp, pd.DataFrame):
                grp = grp[data]
            if out == "pivot":
                grp = grp.T.squeeze()

            return self.Group(group, idx, grp) if gid else grp

        return DataCollection.Iter(map(prep_group, pdt.group_iter(
            self.db, group=group, index=index, shuffle=shuffle, islice=islice,
            out=out, progress=progress
        )), tuples=gid)

    @property
    def empty(self):
        return self.db.empty

    def hash_str(self, fmt='{name}_{shape}_{hash}', *, columns=None, index=True, n=6, **kws):
        """Calculate string uniquely representing this data collection.

        Content of the string can be controlled by including tags: ``{name}, {shape}, {hash}``
        representing correspondingly: collection name, its db shape, and db content hash.

        Length of the hash is controlled by argument ``n``, when 0 means all of it.

        :param fmt: format string with tags to substitute
        :param n: length of hash str from internal db
        :param columns: list of columns to include in hash calculation
        :param index: use or not index for hash calculation
        :param kws: additional tags with values to use in format (external)
        """
        df = self.db[columns] if columns else self.db
        hash_val = '{hash}' in fmt and df.hash_str(n, index=index) or None
        return fmt.format(name=self.name, hash=hash_val,
                          shape='x'.join(map(str, self.db.shape)), **kws)

    @staticmethod
    def collect(labels_iter: Iterable[dict], progress=True) -> pd.DataFrame:
        """ Gathers files categories into a database supporting :met:`__len__`
        and queries by categories' values.

        May be overridden to support different database or caching.
            Then method :met:`query` must be overridden too.

        :param labels_iter: iterable over dicts of labels
        :param progress: control the display of progress
        :returns: database object
        """
        import pandas as pd
        if progress:
            from tqdm import tqdm
            from os.path import split
            loc = split(getattr(labels_iter, 'root', 'file structure'))[-1]
            labels_iter = tqdm(labels_iter, f"Parsing {loc}", unit=' files')

        return pd.DataFrame(list(labels_iter))

    @staticmethod
    def query_description(query: str | dict) -> str:
        """Try to identify selection description from the query"""
        if not query:
            return ''
        return _dict_to_query(query) if isinstance(query, dict) else query


class SinkRepo:
    """
    Repository supporting addition of new items, but not reading.
    Data is saved according to the given path scheme and with
    specific formats of files depending on the labels.
    """

    SAVE_KWS_TAG = 'imsave_args'
    _labels_tag = 'DATA_LABELS'

    def __init__(self, dataset_or_scheme: DatasetRM | SchemeRM | dict | str, root: str = None,
                 data='data', select: str | dict = None, labels: dict = None, create_dir=True):
        """
        Construct write only repository according to given `dataset` model or `scheme`.

        Scheme can be provided as a model, name or yml location, or formatting pattern
        and requires provisioning of `root` folder.

        :param dataset_or_scheme: `scheme` (or its location or name) OR `dataset`
        :param root: folder to save under or `None` if `dataset` in first argument
        :param data: key of the label with data (image) (default: 'data')
        :param select: conditions on labels as dict or eval string
        :param create_dir: if False and don't exist raise `NotADirectroryError`
        """
        from .caster import Labeler
        from .models import SchemeRM, DatasetRM

        if root is None and isinstance(dataset_or_scheme, str):
            dataset_or_scheme = DatasetRM(dataset_or_scheme)

        if isinstance(dataset_or_scheme, DatasetRM):
            root = root or dataset_or_scheme.source.root
            dataset_or_scheme = dataset_or_scheme.scheme

        self.scheme = SchemeRM(dataset_or_scheme)
        self.scheme.reverse = True  # required to build paths
        self.labeler = Labeler(**dict(self.scheme))  # unlike x.dict() keeps internal models!
        self.root = Path(root)
        if not self.root.is_dir():
            msg = f'SinkRepo folder does not exist: {str(self.root)}'
            if not create_dir:
                raise NotADirectoryError(msg)
            _log.warning(msg)

        self._data_key = data
        self._filter = _dict_to_query(select)
        self.labels = labels or {}

    def __str__(self):
        return f"<{type(self).__name__}>({self.root}){self.labeler._pather.form.str}"

    def __repr__(self):
        from toolbox.utils.strings import indent_lines as ind
        seq = lambda _: ', '.join(_)

        return f"<{type(self).__name__}>({self.root}):\n" + \
            ind(self.labeler._pather.form.str,
                ind(f"- categories: {seq(self.labeler.categories)}",
                    f"- anonymous: {seq(self.labeler.anonymous)}"),
                f"sceheme: {self.scheme.name}",
                self.scheme.description or '')

    @property
    def dataset(self):
        """Dataset model of the repository"""
        from .models import DatasetRM
        return DatasetRM(scheme=self.scheme, source=self.root)

    def _prepare_path(self, labels: dict, no_tag=None, exist_ok=None) -> Path:
        path = self.path(no_tag=no_tag, **labels)
        if path.exists():
            if exist_ok is False:
                raise FileExistsError(path)
            if exist_ok is None:
                _log.warning(f'Overwriting {path}')
        else:
            path.parent.mkdir(mode=0o777, parents=True, exist_ok=True)
        return path

    def path(self, no_tag: dict | bool | None = None, **labels):
        """
        Construct path according to the repo scheme based on the given labels.

        Raise ``KeyError`` if required categories are missing in labels.
        Except of 'date' which is then generated automatically.

        `no_tag` controls content of possible unnamed TAG in the pattern.
         - if a `dict` - used if unnamed TAG is defined or fails.
         - if `True` - uses for that all the unknown labels (or similarly fails)
         - if `False` or `None` - ignores unknown tags in the `labels`

        :param no_tag: labels for unnamed tag or permission to take unknown from the `labels`
        :param labels: {cat: val}
        :return: path to the file
        """
        labels |= self.labels

        if 'date' in self.labeler.categories and 'date' not in labels:
            from datetime import datetime
            labels['date'] = datetime.now().strftime('%y%m%d-%H%M%S')

        # remove label
        if 'data' in labels:
            labels.pop('data')

        return self.labeler.path(self.root, no_tag=no_tag, **labels)

    def save_tables(self, labels: dict = None, exist_ok=True, **tables: pd.DataFrame | pd.Series):
        """
        Save pandas tables: DataFrames or Series.
        Format (hdf or xls) is defined by extension in scheme's path

        Tables are found by type among the labels and each is stored in
        separate sheet / node in the data, named by its key.
        Other labels are also saved as auto-created 'labels' Series

        All the labels are updating those provided in constructor.

        :param labels: dict with labels defining the data
        :param exist_ok: if False and file exists raises `FileExistsError`
                         if True - ignores existed
                         if None: warning
        :param kws: additional labels
        :return:
        """
        import pandas as pd
        labels = self.labels | (labels or {})  # combine labels into single dict
        path = self._prepare_path(labels, exist_ok=exist_ok)

        is_sub = lambda x, cls: type(x) is not cls and issubclass(type(x), cls)
        to_base = lambda x: (pd.DataFrame(x) if is_sub(x, pd.DataFrame) else
                             pd.Series(x) if is_sub(df, pd.Series) else x)

        with Timer(f"Table saved in {{:.3}}s to {path}", _log.info):
            ext = path.suffix.lower()
            if ext in ('.hdf', '.hdf5'):
                if any('.' in name for name in tables):
                    import warnings
                    from tables import NaturalNameWarning
                    warnings.filterwarnings('ignore', category=NaturalNameWarning)

                with pd.HDFStore(path, mode='a') as store:
                    if labels:
                        store.put(self._labels_tag, pd.Series(labels))

                    for key, df in tables.items():
                        store.put(key, to_base(df))  # pytabels failing on pandas subclasses

            elif ext in '.xls':  # TODO: add index description meta-data
                with pd.ExcelWriter(path) as store:
                    if labels:
                        pd.Series(labels).to_excel(self._labels_tag, sheet_name=self._labels_tag)

                    for key, df in tables.items():
                        df.to_excel(store, sheet_name=key, merge_cells=True)
            else:
                raise NotImplementedError(f'Not supported saving DataFrame in {path.suffix} format')

            self.read(labels=labels)

    @staticmethod
    def _default_compression(a: np.ndarray, path: str):
        """Deduce default compression kwargs from path (ext) and data (dtype)"""
        if path.rsplit('.', 1)[-1] in ('tif', 'tiff'):
            if a.dtype.kind == 'f':
                return {"compression": 'zstd', 'predictor': 3}
            else:
                return {"compression": "jpeg2000"}
        return {}

    def save(self, items: pd.DataFrame | Iterable[dict] | pd.Series | dict, *,
             data: str | np.ndarray = None, compress: bool | dict | None = False,
             labels: dict = None, exist_ok=None, no_tag: dict | bool = None) -> int:
        """
        Saves data item(s) to the repository according to its labels structure.

        Every data item contains labels for identification and establishing
        its naming, location and formatting in the repository.

        A single item may be passed in form of dict-like object with
        optional data field (ndarray) and named 'data' or as indicated by
        `data` argument if provided.
        Alternatively data (ndarray) may be passed in the `data` argument.

        Multiple items passed as:
         - DataFrame - row per item, considering columns and named levels
           in multi-level index as labels, or
         - Iterable of dict-like objects
        In this case `data` argument may be used to specify data field name

        `no_tag` controls content of possible unnamed TAG in the pattern.
         - if a `dict` - used if unnamed TAG is defined or fails.
         - if `True` - uses for that all the unknown labels (or similarly fails)
         - if `False` or `None` - ignores unknown tags in the `labels`

        **Compression**

        Optional (loseless) of data can be controlled by setting `compress=True`
        or providing dict with arguments for `skimage.io.imsave` or `tifffile.imwrite`,
        depending on the file extension, `False` or `None` diables altogether.

        If `True`, compression parameters are first looked for in provided
        `labels[SinkRepo.SAVE_KWS_TAG]`, if not found, then
        `self._default_compression(data, path)` is called to determine the optimal.

        Examples:
        ---------
        >>> sink.save(df, data='image')  # data in 'image' column
        >>> sink.save(df.iloc[10])       # Series from a single row
        >>> sink.save([dict(a=10, b=20, data=rand(10,20)),
        ...            dict(a=10, b=22, data=rand(30,20))])
        >>> sink.save(dict(a=10, b=20), rand(10,20))

        :param items: one or multiple set of data labels.
                      - dict, Series
                      - DataFrame, container or iterator overt dicts
        :param data: alternative name for data key in `items`, or
                     data array itself, then items contains only labels
        :param compress: dict with compression arguments for
                        OR `True` for auto, `Fasle` - disable
        :param labels: optionally common labels for all the items.
                       (could be overidden by the items labels)
        :param exist_ok: if False and file exists raises `FileExistsError`
                         if True - ignores existed
                         if None: warning
        :param no_tag: labels for unnamed tag or permission to take unknown from the `labels`
        :return: number of data items saved
        """
        from toolbox.io.imwrite import imsave
        from toolbox.io.imread import imread
        import numpy as np
        import pandas as pd
        import os.path
        from tqdm import tqdm

        is_str = lambda _: isinstance(_, str)
        count = 0

        compress = compress or {}
        if not (compress is True or isinstance(compress, dict)):
            raise TypeError(f"Invalid {compress=}")

        def save_item(lbs: dict, a: np.ndarray):
            nonlocal count
            path = self._prepare_path(labels | lbs, no_tag=no_tag, exist_ok=exist_ok)
            path = str(path)
            if os.path.isfile(path):
                if exist_ok:
                    # skipping writing existing files
                    return count
            if not isinstance(a, np.ndarray):
                if os.path.isfile(a):
                    a = imread(a)
                else:
                    raise TypeError(f"Received {type(a)} instead of expected numpy array! or path to file!")

            if (kws := compress) is True:  # if auto compression is requested
                if (kws := lbs.get(self.SAVE_KWS_TAG, None)) is None:  # and not defined in labels
                    kws = self._default_compression(a, path)  # try to determine default

            imsave(path, a, **(kws or {}))
            _log.debug(f"Saved {kws or ''} in {self}: {path}")
            count += 1
            return count

        # -----------------------------------
        labels = labels or {}
        if isinstance(data, np.ndarray):
            return save_item(items, data)

        data = data or self._data_key

        if isinstance(items, dict):
            return save_item(items, items[data])
        if isinstance(items, pd.Series) and all(map(is_str, items)):
            items = items.to_dict()
            return save_item(items, items[data])

        if isinstance(items, (pd.Series, pd.DataFrame)):
            named_levels = list(filter(bool, items.index.names))
            # leave only required items
            for _, row in DataCollection._select(
                    items.reset_index(named_levels), query=self._filter
            ).iterrows():
                row = row.to_dict()
                save_item(row, row[data])
        elif isinstance(items, Iterable):
            for item in tqdm(items):
                if isinstance(item, dict):
                    save_item(item, item[data])
                else:
                    if count:
                        raise KeyError("Attempt to save multiple data items under same labels")
                    save_item({}, item)
        else:
            raise TypeError

        return count

    def read(self, keys: str | Sequence[str] = None, *,
             labels: dict = None, path: str = None, out_labels=False):
        """Read data from repository given labels
        (preferable) or path (hack)

        :param keys: name(s) of the fields to read
        :param labels: identifying item in the repo
        :param path: instead of labels path of the file
        :param out_labels: if True return additionally this item's labels dict
        :return DataFrame if ``key`` is a str, otherwise namedtuple
        """
        import pandas as pd
        assert path is None or labels is None
        path = path or self.path(**labels)
        if path.suffix.lower() in {'.hdf', '.hdf5'}:

            outputs = lambda d: (
                d, hdf.get(self._labels_tag)
                if self._labels_tag in hdf.keys() else None
            ) if out_labels else d

            with pd.HDFStore(path, mode='r') as hdf:
                if isinstance(keys, str):  # return specific key
                    return outputs(hdf.get(keys))

                res = {k: hdf.get('/' + k) for k in
                       (keys or (k[1:] for k in hdf.keys()))}
                res.pop(self._labels_tag, None)

                return outputs(res)

        raise NotImplementedError

    def __call__(self, *args: dict, exist_ok=None, **kwargs):
        """Function form interface for saving data.
        Calls to `save` or `save_table` method depending on the sink type

        :param *args: dict-like data items
        :param exist_ok: if True - ignores existed
                         if False - raise error
                         if None: warning
        """
        if self.labeler._pather.ext.lower() in {'xls', 'hdf', 'hdf5'}:
            return self.save_tables(*args, exist_ok=exist_ok, **kwargs)
        else:
            return self.save(*args, exist_ok=exist_ok, **kwargs)


def _dict_to_query(dq: str | dict):
    """Convert dict to query string with and between items"""
    if not dq or isinstance(dq, str):
        return dq
    if not isinstance(dq, dict):
        raise TypeError("Invalid query type")

    def to_str(it):
        k, v = it
        return f"{k}=='{v}'" if isinstance(v, str) else f"{v}"

    return ' and '.join(map(to_str, dq.items()))
