from __future__ import annotations

import os
from copy import deepcopy
from typing import Iterator, Callable, Optional, Iterable

import pydantic
import regex as re  # Using more advanced regex package!
import yaml

from toolbox.param import YamlModel
from toolbox.utils import logger, as_list
from toolbox.utils.cache import CachedPipe, CacheMode, Pickle, CacheInvalidError, NoSerial
from toolbox.utils.datatools import Filter
from toolbox.utils.events import Timer
from toolbox.utils.filesproc import Path, root_cropper, root_adder, normalize
from toolbox.utils.fnctools import express_to_kw_func
from . import models as md

__all__ = ['DataCaster', 'CacheMode']

_log = logger('datacast')
DEFAULT_MODE = CacheMode.KEEP


def path_to_win(lbs):
    """Convert POSIX path into WIN"""
    lbs['path'] = lbs['path'].replace('/', '\\')
    return lbs


# Consider: do we really need this class, given its overlap with from SchemeRM?
class Labeler(YamlModel, hash_exclude=['reverse', 'categories']):
    labels: dict
    mappings: dict
    reverse: bool
    categories: Optional[list]

    _pather = pydantic.PrivateAttr()
    _processor = pydantic.PrivateAttr()

    def __init__(self, *, labels, mappings, reverse, search: md.GuideScan, namespace=None, **_):
        super().__init__(labels=labels, mappings=mappings, reverse=reverse)

        self._pather = search._pather
        cat_merger, self.categories = _create_key_merger(self._pather.regex.categories)
        cat_mapper, reverse_mapper = _create_values_mappers(mappings, reverse)
        cond_labels = [*map(parse_conditional_key, _to_dict(labels).items())]
        cond_labeler = _create_labeler(cond_labels, namespace or {})

        self._processor = lambda lbs: cond_labeler(cat_mapper(cat_merger(lbs)))

        self._pather.mapper = reverse_mapper
        self.categories.update(cat for _, lbs in cond_labels for cat in lbs)

    @property
    def anonymous(self):
        return tuple(self._pather.regex.anonym_groups)

    @property
    def processor(self):
        return self._processor

    def path(self, root, *, no_tag: dict = None, **labels):
        """Create full path from given labels relative to the given root"""
        root = normalize(root)
        return root / self._pather(no_tag, **labels)


class DataCaster:
    """
    Casts data in files organized and named according to certain **layout** into
    a generic form of *labeled data*.

    See `./docs/data_casting.md`

    Iterator's Processing Flow
    ---------------------------
    ::

                         paths  ⎡ match ⎤    ⎡filter ⎤
        folder -> [walk] -----> ⎣pattern⎦ -> ⎣matched⎦ -> [extract] -> labels ->

           ⎡   merge  ⎤    ⎡  rename  ⎤    ⎡filter by⎤    ⎡add label:⎤
        -> ⎣categories⎦ -> ⎣categories⎦ -> ⎣condition⎦ -> ⎣transforms⎦ -> labels

    """
    CacheMode = CacheMode  # allow PathScheme.CacheMode instead of scheme.CacheMode

    def __init__(self, name: str | md.DatasetRM = None, *,
                 source: md.DataSourceRM | str = None,
                 scheme: md.SchemeRM | str = None,
                 filters: dict = None, transforms: dict = None,
                 sample: md.DSample = None,
                 labels: None | dict = None,
                 # -- Not DatasetRM params --
                 temp_cache=False, cache: Optional[CacheMode | bool] = True,
                 progress: dict | bool = None):
        """
        >>> DataCaster('KnownName')
        >>> DataCaster(md.DatasetRM(source='/', scheme='*'))
        >>> DataCaster(source='/', scheme='*')
        Create data caster semantic parser for particularly structured files tree.
        ::
            tree -> [pattern] -> files -> [label] -> [filter] -> labeled
        """
        self.config = md.DatasetRM.from_config(locals(), undefined=False, ignore=True)

        scheme = self.config.scheme
        search = scheme.search
        labeler = Labeler(labels=scheme.labels | (self.config.labels or {}),
                          mappings=scheme.mappings, search=search,
                          reverse=scheme.reverse, namespace=self.labeling_namespace)

        self.categories = labeler.categories
        self.bundle = scheme.bundle  # ! MUST be after compile_schemes! validates categories

        # --------------------------- building the pipeline -------------------------
        is_win = os.name == 'nt'
        # There is a challenge to support both POSIX and Windows, since two things are potentially
        # system dependent: search patterns provided by scheme and cached paths.
        # In selected approach both are in POSIX,
        # to allow using same schemes and cached data from everywhere.
        # To implement that, the processing pipeline should be adjusted to the current system:
        #  1. The first scanning stage is configured to convert WIN->POSIX BEFORE matching
        #     the pattern, so that POSIX pattern works AND resulted matched path is POSIX.
        #     Note, that only works when matching not the full path, where are problems with disks, etc.
        #  2. Since step 1 ensures path label is cached and propagated as POSIX,
        #     on Windows additional final stage is added to convert generated labels['path'] to Win

        stages = [
            CachedPipe.Source(search.scanner(self.root, is_win), 'wlk', cfg=search.to_yaml()),
            CachedPipe.Map(labeler.processor, 'lbl', cfg=labeler.to_yaml()),
        ]
        if filters := self.config.filters:
            stages.append(CachedPipe.Filter(Filter(filters), 'flt', cfg=yaml.dump(filters)))
        if is_win:  # Create non-cachable stage converting path from stored POSIX into Win
            stages.append(CachedPipe.Map(path_to_win, 'win', serial=NoSerial(), mode=CacheMode.PASS))

        self.cached_pipe = CachedPipe(
            stages=stages, folder=Path(self.root) / '.cache', mode=CacheMode(cache), temp=temp_cache,
            serial=PickleRelativePath(self.root, safe=False), progress=progress
        )

    @property
    def labeling_namespace(self):
        from toolbox.utils.strings import hash_str
        from pandas import NA, isnull
        return dict(hash_str=hash_str, name=self.config.name, NA=NA, isnull=isnull)

    @property
    def bundle(self):
        return self._bundle

    @bundle.setter
    def bundle(self, names: str | list[str]):
        names = as_list(names)
        if dif := set(names).difference(self.categories):
            KeyError(f"Bundle labels {dif} are not defined in scheme!")
        self._bundle = names

    @property
    def root(self):
        return self.config.source.root

    def __repr__(self):
        _list = lambda _: f"[{', '.join(_)}]" if _ else ""
        info = dict(
            root=str(self.root),
            bundle=_list(self.bundle),
            cats=_list(self.categories),
            regex=self.config.scheme.search._pather.regex.regex.pattern
        )
        items = (f"{k}: {v}" for k, v in info.items() if v)
        return '\n\t'.join([str(self), *items])

    def __str__(self):
        return f"<{self.__class__.__name__}: {self.config.name}>"

    def __copy__(self):
        return deepcopy(self)

    def __iter__(self):
        return self.iter()

    def iter(self, *, progress: CachedPipe.ProgressT = None, ignore_sampling=False) -> Iterator:
        """Create iterator over files described by the scheme.

        This method unlike __iter__ allows tuning default behaviours:

        *Progress Bar*

        Progress can be shown during the iterations if argument is:
         - ``Callable(Iterable) -> Iterator`` like ``tqdm.tqdm``
         - ``True`` - then ``tqdm`` is used
         - ``dict`` with arguments for ``tqdm``
        :param ignore_sampling: explicitly disable sampling when iterate over dataset with sample defined
        :param progress: True, False, None (auto - show only if cache not found)
        """
        if self.config.sample and not ignore_sampling:
            raise NotImplementedError("Iterating over sampled dataset is not defined! "
                                      "Use collect_table() method!")
        if progress is not None:
            self.cached_pipe.progress_bar(dict(delay=.8, unit='file', desc=f'{self}'))
        return self.cached_pipe

    def collect(self, progress=True):
        """
        Collect dataset into the table.
        Apply additional labels and sampling.

        :param progress:
        :return:
        """
        from .transtools import CollectTable
        ds = CollectTable(self.iter(progress=progress, ignore_sampling=True))
        if ds.empty:
            _log.warning(f"{self} found no matching files!")

        if cfg := self.config.sample:
            from toolbox.utils.pdtools import sample
            ds = sample(ds, **cfg.dict())
        return ds

    def copy(self):
        return self.__copy__()

    def spellcheck(self, words: Iterable[str] = (), *, warn_thresh=85.):
        """Check that defined labels are not just misspelled keyword.

        List of words to avoid similarity with includes:
         - reserved words: 'description', 'pattern', 'mappings'
         - standard labels names (defined in .transfrom.Col),
         - and may be extended by ``words`` argument.

        :param words: additional words to avoid
        :param warn_thresh: similarity threshold
        :return: number of found close matches
        """
        from rapidfuzz.process import extractOne
        from .transtools import Col
        keywords = list(v for k, v in vars(Col).items() if not k.startswith('_'))
        keywords += ['description', 'pattern', 'mappings', *words]
        exclude = {'dataset'}

        count = 0
        for cat in self.categories:
            closest, score, _ = extractOne(cat, keywords)
            if warn_thresh < score < 100 and cat not in exclude:
                count += 1
                _log.warning("Label '%s' is %d% similar to keyword '%s' in %s.",
                             cat, score, closest, self)
        return count


def _path_to_win(labels_gen: Iterable[dict]):
    """Generator converting path label into windows form"""
    for lbs in labels_gen:
        lbs['path'] = lbs['path'].replace('/', '\\')
        yield lbs


def _to_dict(d: dict) -> dict:
    """Translate dict-like nodes of hierarchical object into pure dict.

    :param d: The object, possibly a combination of dict and :class:`Box`.
    :returns: it's modified self
    """
    if hasattr(d, 'items'):
        if hasattr(d, 'to_dict'):
            return d.to_dict()
        for k, v in d.items():
            d[k] = _to_dict(v)
    return d


# Consider: Make it part of Pather
def _create_key_merger(keys: Iterator, *, join: Callable[[Iterator], str] = '_'.join) \
        -> tuple[Callable[[dict], dict], set]:
    """Create function for INPLACE merging dict items with enumerated keys
        (Utility function used in parse file-name info.)

    Example:
        >>> dict(x=0, s_1='oh', s_2='my', y='ok', s_4='god') -> dict(x=0, y='ok', s='oh_my_god')

    :param keys: iterable over the keys treat (usually those parsed)
    :param join: function to glue the parts (default: "_".join(parts))
    :returns: a function object: `func(dict) -> dict`, new labels categories
    """
    dp = re.compile(r'((\w+?)_?\d+)$')  # categories ending with number: scene_1
    splits = filter(dp.fullmatch, keys)  # leaves only split categories
    splits = dict(m.groups() for m in map(dp.match, splits))  # produces {'name_id': 'name'}
    if not splits:
        return (lambda info: info), set(keys)

    merged_keys = {k: [] for k in set(splits.values())}  # group split names by their real name
    for k, v in splits.items():  # transpose k <-> v: splits -> groups
        merged_keys[v].append(k)  # form: groups = {name: [name_0, name_1, ...]}

    new_keys = set(keys).difference(splits).union(merged_keys)

    def merger(info: dict):
        for name, split_names in merged_keys.items():
            info[name] = join(info.pop(sn) for sn in split_names)
        return info

    return merger, new_keys  # Consider: return inverse to the merger - splitter (for path formatting)


# @staticmethod
def _create_labels_filter(filters):  # Consider: Needed?
    if not filters:
        return {}

    if isinstance(filters, str):
        filters = {'condition': filters}
    for k, v in filters.items():
        if k.startswith('condition'):
            filters[k] = express_to_kw_func(v)
        elif isinstance(v, dict):
            raise TypeError('Scheme label condition must be callable, sequence or scalar')
    return filters


ValMapFunc = Callable[[dict], dict]


# @staticmethod
def _create_values_mappers(trans: dict[str, Callable | str | dict],
                           reverse: bool) -> tuple[ValMapFunc, ValMapFunc]:
    """Given a dictionary of label mappings create a function
    which would translate values of its input labels dict
    using transform with matching key if found.

    Transforms could be either a dict or callables.

    Return such values mapper function and reverse function, restoring the original
    values from the transformed labels (Works correctly only for dict transforms!)

    :param trans: transforms by label's category {category: transform},
        with transform as a dict, Callable, or str evaluable into Callable.
    :param inverse: request reverse mapping
    :return: map_func, rev_map_func
    """

    def map_values(dct, *, fnc_map={}, default_fnc=lambda _: _):
        """
        Map dict by applying on every value a functions selected by the corresponding key:
        dict(key, value) -> dict(key, fnc_map[key](value))

        :param dct: dictionary to convert
        :param fnc_map: dict of functions with keys as in dct
        :param default_fnc: function to use if key not found in fnc_map
        """
        return {k: fnc_map.get(k, default_fnc)(v) for k, v in dct.items()}

    def mapping_fnc(transform):
        # converts different allowed forms of transform into functional form
        if isinstance(transform, dict):
            return lambda v: transform.get(v, v)
        if isinstance(transform, str):
            return eval(transform)
        if isinstance(transform, Callable):
            return transform
        raise ValueError(f"Unsupported mapping type {type(transform)}")

    def reverse_map(d):
        """Create reverse transform from a direct one
        defined for a particular category in the mappings.

        Notice, that from 3 types of transforms supported (see map_func)
        only dict can be reversed (assuming no repeated values!)

        So currently this function supports only that,
        and replaces everything else by str casting.

        Consider: think about real use cases
        """
        return {v: k for k, v in d.items()} if isinstance(d, dict) else str

    def mapper(transforms: dict) -> Callable[[dict], dict]:
        """Creates function to map dict of labels according to the given transformation rules.

        Values which keys are missing from the ``transforms`` are passes as is!
        """
        mappings = map_values(transforms, default_fnc=mapping_fnc)  # make all transforms callable
        return lambda labels: map_values(labels, fnc_map=mappings)

    # reverse labels transformations rules: mapper(rev_trans)(mapper(trans)(labels)) == labels
    reverse_mapper = mapper(map_values(trans, default_fnc=reverse_map)) if reverse else None
    return mapper(trans), reverse_mapper


# @staticmethod
def _create_labeler(cond_labels: list[tuple[Callable, dict]], namespace: dict) -> Callable[[dict], dict]:
    """Create function which updates provided labels with additional labels
    if the provided labels meet certain conditions.

    The argument ``cond_labels`` is a source of those conditions with associated additional labels.
    """
    from types import CodeType

    def add_labels(current_labels):
        for cond_test, labels in cond_labels:
            if cond_test(**current_labels):
                # evaluate labels using the context provided by the current labels
                labels = {
                    k: eval(v, namespace, current_labels)
                    if isinstance(v, CodeType) else v
                    for k, v in labels.items()
                }  # now labels are fully defined
                current_labels |= labels  # replacing some labels if they are already there
        return current_labels

    return add_labels


def parse_conditional_key(key_item: tuple[str, dict | str]) -> tuple[Callable[[...], bool], dict]:
    """
    Extract a condition from a scripted form of `key` string
    in the given label tuple ``(key, value)``.

    Return a tuple of
        - extracted condition (if found), as a callable
          (or universal ``true_cond`` function)
        - pure label, at a dict {pure key: value}

    `key` string templates:
        1. ``<key>``             - unconditional key (value is the item)
        2. ``<key> if <cond>``   - conditional key (value is the item)
        3. ``if <cond>``         - condition only (label(s) are the item(s))

    In case of 3 check that the value is a dict.

    :param key_item:
    :return: cond, label
    """
    import re
    cond_exp = re.compile(r'(\w+)?(?:\s*if (.+))?', re.IGNORECASE)
    fstr_exp = re.compile(r'f[\'"].*[\'"]')
    pyth_exp = re.compile(r'\b(if|else|isnull)\b|[{}()=|\'\"]')  # Consider: use actual python parser instead

    def compile_value(v):
        """Return the input as is or compile if it is a dynamic expression"""
        return compile(v, 'scheme_dynamic', 'eval') \
            if isinstance(v, str) and (
                fstr_exp.fullmatch(v) or
                pyth_exp.search(v)
        ) else v

    def true_cond(**_):  # always True
        return True

    key, item = key_item
    try:
        key, cond = cond_exp.fullmatch(key).groups()
    except AttributeError:
        raise SyntaxError(f'Configuration has invalid key "{key}"')

    cond = express_to_kw_func(cond) if cond else true_cond  # (2,3) or 1

    if key:
        label = {key: compile_value(item)}  # label for cases 1, 2
    elif isinstance(item, dict):  # case 3.  `if cond: {key: value}`
        label = {k: compile_value(v) for k, v in item.items()}  # item is a dict {key: value, ...}
    else:
        raise SyntaxError(f'Configuration under condition: {cond}\n'
                          f' expected label (a dict), but instead received:\n{item}')
    return cond, label


class PickleRelativePath(Pickle):
    """Serializer of labels used by cached pipeline to save relative paths in the cache."""

    def __init__(self, root: Path | str, *, safe):
        """
        :param root: relative to this folder
        :param safe: True to use safe but slow method validating that paths indeed contain root.
        """
        assert root, f'{self.__class__.__name__} definition failed, supply root'
        self.add_root = root_adder(root, as_str=not safe, out_str=True)
        method = 'relative' if safe else 'crop'
        self.crop_root = root_cropper(root, method, out_str=True)
        self._msg = f"[{method}] paths relative to {root} in {{:.2f}}sec"

    def save(self, file: Path, records: list[dict]):
        def crop_path_copy(item):
            item = item.copy()
            item['path'] = self.crop_root(item['path'])
            return item

        with Timer(f"Save cache {file.name} {self._msg}", _log.debug):
            records = list(map(crop_path_copy, records))
            super().save(file, records)

    def load(self, file: Path):
        with Timer(f"Load cache {file.name} {self._msg}", _log.debug):
            records = super().load(file)
            if records and Path(p := records[0]['path']).is_absolute():
                raise CacheInvalidError(f"Cache {file.name} must contain relative paths\n[found: {p}]")
            for item in records:
                item['path'] = self.add_root(item['path'])
            return records
