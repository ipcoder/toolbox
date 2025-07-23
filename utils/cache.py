from __future__ import annotations

import abc
from enum import IntEnum, auto
from functools import wraps
from pathlib import Path
from typing import List, Callable, Iterable, Union, get_type_hints, Iterator, Type

from joblib import load, dump
from tqdm import tqdm

import toolbox.utils.logs
from inu.env import EnvLoc
from toolbox.utils import logger
from toolbox.utils.events import Timer
from toolbox.utils.filesproc import Locator
from toolbox.utils.strings import hash_str

_log = logger('caching')
__all__ = ['CacheMode', 'CachedPipe', 'filecached', 'Cacher', 'Pickle', 'CacheInvalidError', 'AStage']


class CacheError(Exception):
    pass


class CacheInvalidError(CacheError):
    pass


# ------------   Define Serialization Protocols   --------------------
class Serial(metaclass=abc.ABCMeta):
    """Base Abstract class to define Serializers.

    A serializer must define ``save`` and ``load`` static methods,
    and ``ext`` attribute (extension string without dot).
    """

    def __repr__(self):
        return f"{self.__class__.__name__}(.{self.ext})"

    @classmethod
    def io(cls, meth):
        return cls.time_io(staticmethod(meth))

    @classmethod
    def time_io(cls, f):
        @wraps(f)
        def wrapper(file, *args):
            if not _log.isEnabledFor(_log.DEBUG - 1):
                return f(file, *args)

            mode = '[v] save' if args else '[^] load'
            with Timer(f'{mode} in {{:.3f}}sec {str(file)}', _log.debug):
                return f(file, *args)

        return wrapper

    @property
    @abc.abstractmethod
    def ext(self): ...

    @staticmethod
    @abc.abstractmethod
    def save(file: Path, records: List[dict]): ...

    @staticmethod
    @abc.abstractmethod
    def load(file: Path): ...


class NoSerial(Serial):
    """Dummy Serializer ensures it is never called"""
    ext = 'xxx'

    @staticmethod
    def save(file: Path, records):
        raise RuntimeError()

    @staticmethod
    def load(file: Path):
        raise RuntimeError()


class Parquet(Serial):
    _column_name = '___dumb_list_name__'
    ext = 'pqt'

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    @staticmethod
    @Serial.io
    def save(file: Path, records: List[dict]):
        from pandas import DataFrame
        if not isinstance(records[0], dict):
            records = {Parquet._column_name: records}
        file.parent.mkdir(exist_ok=True)
        DataFrame(records).to_parquet(file)

    @staticmethod
    @Serial.io
    def load(file: Path):
        from pandas import read_parquet
        df = read_parquet(file)
        if Parquet._column_name in df:
            return df[Parquet._column_name].values
        return df.to_dict('records')


class Pickle(Serial):
    ext = 'pkl'

    @staticmethod
    @Serial.time_io
    def save(file: Path, records):
        import pickle
        file.parent.mkdir(exist_ok=True)
        with open(file, 'wb') as fh:
            pickle.dump(records, fh, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    @Serial.time_io
    def load(file: Path):
        import pickle
        with open(file, 'rb') as fh:
            return pickle.load(fh)


# ----------------------------------------------------------
class CacheMode(IntEnum):
    """Modes controlling caching behaviour:

     - NONE: undefined mode, Typed equivalent of None
     - PASS: bypass caching mechanism
     - LOAD: load instead of processing
     - SAVE: save processed
     - KEEP: use available and create if missing
     - CLEAR: clear caches and set to SAVE
    """
    NONE = 10000  # undefined mode, Typed equivalent of None, start far from small numbers
    PASS = auto()  # bypass
    LOAD = auto()  # load instead of processing
    SAVE = auto()  # save processed
    KEEP = auto()  # use available and create if missing
    CLEAR = auto()  # clear caches and set to SAVE

    def __repr__(self):
        return self.name

    def __bool__(self):
        return self.value is not CacheMode.NONE

    @classmethod
    def _missing_(cls, value):
        if value is True: return cls.KEEP
        if value is False: return cls.PASS
        if isinstance(value, str):
            return cls.__members__.get(value, None)

    __str__ = __repr__


class AStage(metaclass=abc.ABCMeta):
    """
    Abstract cached pipeline stage definition.

    Subclasses may extend stage definition parameters (`__init__`)
    and *must* implement `next_func` factory method.

    This method implements the essential logic of the particular subclass,
    and must satisfy those requirements:
      - receive previous cached stage object (`prev`) and
      - returns a closure, which
      - on every call pulls from the `prev` next data item
    """

    @abc.abstractmethod
    def next_func(self, prev: CachedStage) -> Callable: ...

    def __repr__(self):
        return f"<{self.__class__.__name__}>[{self.name}]"

    def __init__(self, name: str, *_, serial: Serial = Pickle(),
                 mode: CacheMode = None, copy: Callable = None, cfg: str = None):
        """
        Define pipeline stage
        :param name: unique name of the stage (used in cache file)
        :param serial: serialization class subclassing ``Serial``
        :param mode: default cache mode or None to inherit from Pipe
        :param copy: optional function to create copy of data items when placing them into
        caching buffer - may be needed if stage yields unstable data items.
        """
        assert not _, "Only defines interface for subclasses"
        self.name = name
        self.serial = serial
        self.mode = mode or CacheMode.NONE
        self.copy = copy or (lambda _: _)
        self.cfg = cfg


class CachedStage:

    def __init__(self, st: AStage, prev: CachedStage,
                 folder: Path, mode=CacheMode.PASS, serial: Serial = None):
        """
        Initialize a processing stage layer positioned after given previous layer
        serializing data in the given folder.

        :param st: stage description in form of AStage
        :param prev: previous cache level object in the pipeline
        :param folder: location for cache files
        :param mode: enumerated by CacheMode: [PASS] | LOAD | SAVE | KEEP
                     PASS = [default] to bypasses caching mechanism
                     LOAD - use available cache files, raise if not found!
                     SAVE - saves FULL iterations results into the file
                     KEEP - same as LOAD if file is found otherwise SAVE
        """
        if hasattr(st, '__length_hint__'):
            self.__length_hint__ = st.__length_hint__

        self.mode = CacheMode(mode if st.mode is CacheMode.NONE else st.mode)
        use_cache = self.mode not in (CacheMode.NONE, CacheMode.PASS)
        name = f"{st.name}_{hash_str(st.cfg, 3)}" if use_cache and st.cfg else st.name

        self.type = type(st)
        self.name = name
        self.serial = serial or st.serial
        self.next_func = st.next_func
        self._copy = st.copy
        self._loaded = None  # content loaded from cache file
        self.next = None
        self.prev = prev

        if prev:
            self.prev.next = self
            name = f"{prev.file.stem}_{name}"  # file name to include all the previous stages
        self.file = Path(folder, f"{name}.{self.serial.ext}")

        if use_cache:
            # If needed verify and preload cache content
            if self.mode is CacheMode.KEEP:  # if not possible switch into SAVE mode!
                self._loaded = self._try_load_file(fail=False)  # includes case of empty file
                self.mode = CacheMode.LOAD if self._loaded else CacheMode.SAVE
            elif self.mode is CacheMode.LOAD:
                self._loaded = self._try_load_file(fail=True)  # Could return [] if found file is empty

            if self.mode is CacheMode.SAVE:
                if not is_folder_writable(folder):
                    raise PermissionError(f"{folder} is not writable!")
                if st.cfg:
                    Path(folder, f"{self.name}_cfg.yml").write_text(st.cfg)

        _log.debug(f"+ Stage: {str(self):<30}")

    def _try_load_file(self, *, fail=True) -> list | None:
        """
        Attempts to load the file and return buffer content as a list of items.

        On error if `fail` is ``True`` raise ``FileNotFoundError`` otherwise return ``None``.
        """
        try:
            if not self.file.exists():
                raise FileNotFoundError(f"Cache file not found {self.file}")
            if not self.file.stat().st_size:
                raise FileNotFoundError(f"Cache file {self.file} is empty!")
            buf = self.serial.load(self.file)
            if len(buf) == 0:
                _log.warning(f"Cache file {self.file} is empty!")
            return buf
        except Exception as ex:
            if fail: raise ex
            _log.info(f"{ex} in {self}")
            return None

    def __bool__(self):
        return True

    def __repr__(self):
        return f"<{self.type.__name__}> '{self.name}' [{self.mode}]"

    def __iter__(self):
        if self.mode is CacheMode.LOAD:  # ignore previous stages!
            self._get_next = iter(self._loaded).__next__
        else:
            self._get_next = self.next_func(self.prev)
            if self.mode is CacheMode.SAVE:
                self._buf = []

        return self

    def __next__(self):
        save_mode = self.mode == CacheMode.SAVE
        try:
            item = self._get_next()
            save_mode and self._buf.append(self._copy(item))
            return item
        except StopIteration:
            if save_mode:
                self.serial.save(self.file, self._buf)  # logged by save!
            raise StopIteration

    def __length_hint__(self):
        if self.mode is CacheMode.LOAD:
            return len(self._loaded)
        if self.prev:
            return self.prev.__length_hint__()
        return _src_default_length


def pipestage(cls):
    """Decorator to define Stage classes for CachedPipe.

    Such class must:
     - define its `essential`` attribute representing stages functionality
     - implement ``next_func(self, prev: CachedStage)`` method

    This method (given previous stage) must:
     - initializes its iterator, and
     - return a special function object (`nexter`)

    Call to this object (`nexter`) must perform three actions:
     1. iterate over the created ``prev`` iterator (if prev exists)
     2. invoke ``essential`` functionality
     3. return result

    Implementation of a typical processing stage:

    >>> @pipestage
    ... class ProcessingStage:
    ...     proc: Callable
    ...
    ...     def next_func(self, prev):
    ...         it = iter(prev)    # initialize iterator over prev
    ...         proc = self.proc   # prepare processing
    ...         return lambda: proc(next(it))  # create function (1,2,3)
    """
    from dataclasses import dataclass, make_dataclass

    @dataclass
    class Stage:
        name: str
        serial: Serial = Pickle()
        mode: CacheMode = CacheMode.NONE
        copy: Callable = (lambda _: _)  # function used to copy a data into buffer

        def __repr__(self):
            return f"<{self.__class__.__name__}>[{self.name}]"

        def __post_init__(self):
            for name, tp in get_type_hints(type(self)).items():
                atr = getattr(self, name)
                if not isinstance(atr, tp):
                    raise TypeError(f'Expected type of {name} is {tp}, not {type(atr)}!')

    def wrap(cls):
        cls = dataclass(cls)
        assert isinstance(cls.next_func, Callable), "Stage must define `next_func(self, prev)` method!"
        dc = make_dataclass(cls.__name__, [], bases=(Stage, cls))
        return dc

    return wrap(cls)


_src_default_length = 10


class CachedPipe:
    """Pipeline envelops cached processing stages into Iterable.
    ::
      Pipeline: S1 -> S2 -> S3 ->...

    On each iteration an item is pulled from the last stage of the pipeline,
    causing it to pull next from the previous stage, and so on.

    Caching is implemented for every stage separately to keep all the data items passed through the stage.

    If valid cache is found for stage Si, data items are pulled from there,
    skipping execution of this and all the earlier stages.

    CachePipe provides three most common types of stages: ``Source``, ``Map``, and ``Filter``,

    Additional may be created by sub-classing ``AStage`` class.
    """

    ProgressT = Union[Callable[[Iterable], Iterator], bool, dict]

    class Source(AStage):
        """
        A simple ``Source`` stage built around an Iterable `src` object.

        Its ``__next__`` just calls  ``next(src)``.
        """

        def __init__(self, src: Iterable, name: str, *, serial: Serial = Pickle(),
                     mode: CacheMode = None, copy: Callable = None, cfg=None):
            if copy is None:
                copy = dict.copy
            super().__init__(name=name, serial=serial, mode=mode, copy=copy, cfg=cfg)
            self.src = src

        def __iter__(self):
            self.it = iter(self.prev)
            return self

        def __next__(self):
            return next(self.it)

        def next_func(self, _) -> Callable:
            it = iter(self.src)
            return it.__next__

        def __length_hint__(self):
            for len_attr in ['__len__', '__length_hint__']:
                if len_fnc := getattr(self.src, len_attr, None):
                    return len_fnc()
            return _src_default_length

    class Filter(AStage):
        """
        A ``Filter`` stage  is built around a boolean condition function.

        Its ``__next__`` method pulls and discards items from the previous stage,
        and yields an item only when ``condition(item)`` is satisfied.
        """

        def __init__(self, cond: Callable, name: str, *, serial: Serial = Pickle(),
                     mode: CacheMode = None, copy: Callable = None, cfg=None):
            super().__init__(name=name, serial=serial, mode=mode, copy=copy, cfg=cfg)
            self.cond = cond

        def __iter__(self):
            self.it = filter(self.cond, self.prev)
            return self

        def __next__(self):
            return next(self.it)

        def next_func(self, prev) -> Callable:
            it = filter(self.cond, prev)
            return it.__next__

    class Map(AStage):
        """
        ``Map`` stage built around a ``proc`` function mapping an input item into any other item.
        """

        def __init__(self, proc: Callable, name: str, *, serial: Serial = Pickle(),
                     mode: CacheMode = None, copy: Callable = None, cfg=None):
            super().__init__(name=name, serial=serial, mode=mode, copy=copy, cfg=cfg)
            self.proc = proc

        def next_func(self, prev) -> Callable:
            it = iter(prev)
            proc = self.proc
            return lambda: proc(next(it))

    StageType = Type[Union[Source, Filter, Map]]

    def __init__(self, stages: List[AStage], *, folder, temp: Path | str | bool = False,
                 mode=CacheMode.PASS, serial: Serial = None, progress: ProgressT = None):
        """Create cache manager for a scheme keeping files in given
        folder if given or in scheme's root folder'

        Initialize pipeline with give stages in required mode:
        ::
            PASS - bypass caching
            SAVE - save stages results to cache files
            LOAD - load from latest (by order) available stage cache
            KEEP - LOAD in stages with cache SAVE otherwise
            CLEAR - PASS but clears all the caches

        Progress can be shown during the iterations if argument is:
         - ``Callable(Iterable) -> Iterator`` like ``tqdm.tqdm``
         - ``True`` - then ``tqdm`` is used
         - ``dict`` with arguments for ``tqdm``

        :param stages: list of processing stages as ``AStage`` objects.
        :param folder: cache files location
        :param temp: use of temporal folder as fallback
        :param mode: one of CacheMode enumerated modes, could have different
            effect depending on stage's cache file (as described above)
        :param progress: show progress when iterating.
                Disabled if ``None`` or ``False``
        """
        self.folder = Path(folder).expanduser().absolute()
        _log.debug(f"<{type(self).__qualname__}> {len(stages)} stages {mode=} in {str(self.folder)}")
        clear = mode is CacheMode.CLEAR and (mode := CacheMode.PASS)  # CLEAR -> PASS

        folders = [self.folder]
        if temp:
            if temp is True:
                import tempfile
                temp = Path(tempfile.gettempdir(), 'cache')
            folders.append(Path(temp))

        latest_load = None
        for attempt, folder in enumerate(folders):
            if attempt and latest_load: break  # skip another attempt if already loaded
            latest_load = self.first = self.last = None
            try:
                for i, stage in enumerate(stages):
                    self.last = CachedStage(stage, prev=self.last, folder=folder, mode=mode, serial=serial)
                    if self.last.mode is CacheMode.LOAD:  # cached stage found and ready to load
                        latest_load = (i, self.last)
                    if not self.first:
                        self.first = self.last
            except PermissionError as ex:
                if attempt == len(folders) - 1: raise ex
                toolbox.utils.logs.error(f"{ex} - Trying fallback {str(folders[-1])}")

        if latest_load and _log.isEnabledFor(_log.DEBUG):
            n, (i, stage) = len(stages), latest_load
            _log.debug(f'Loading from cache [{i - n}|{n}]-th stage {stage} shadows preceding ({i})')

        clear and self.clear()
        self.progress = None
        self.progress_bar(progress)

    def progress_bar(self, progress: ProgressT):
        """Set progress bar and return its previous state"""
        prev = self.progress
        if progress in (None, False) or isinstance(progress, Callable):
            self.progress = progress
        elif isinstance(progress, dict):
            self.progress = lambda it: tqdm(it, **progress)
        elif progress is True:
            self.progress = tqdm
        else:
            raise TypeError(f"Invalid progress argument {progress}")
        return prev

    def stages(self, reverse=False):
        """
        Return tuple of all the stages in requested order
        :param reverse: True to reverse from last to first
        :return:
        """
        return tuple(self.stages_iter(reverse=reverse))

    def stages_iter(self, reverse=False):
        """Iterator over stages from first to last or reversed"""
        direct, stage = ('prev', self.last) if reverse else ('next', self.first)
        while stage:
            yield stage
            stage = getattr(stage, direct)

    def clear(self):
        _log.info(f"Clearing cache in {self.folder}")
        stage = self.last
        while stage:
            (found := stage.file.exists()) and stage.file.unlink()
            _log.debug(found and f"[x] delete {stage.file}..." or f"not found {stage.file}")
            stage = stage.prev

    def exists(self):
        """If at least one stage cache is found"""
        return self.last.file.exists()

    def __iter__(self):
        if self.progress:
            return iter(self.progress(self.last))
        return iter(self.last)

    def __length_hint__(self):
        return self.last.__length_hint__()

    def __repr__(self):
        lines = [f"<{self.__class__.__name__}> with stages:", *map(str, self.stages_iter())]
        return '\n\t'.join(lines)

    def cached_stage(self):
        """Return latest stage with usable cache or None"""
        for stage in self.stages_iter(reverse=True):
            if stage.mode is CacheMode.LOAD:
                return stage
        return None


def file_namer(name: str | Callable):
    if isinstance(name, (str, Path)):
        def name_cache_file(*_, **__):
            return name
    elif isinstance(name, Callable):
        name_cache_file = name
    else:
        raise TypeError(f'Unsupported {type(name)=} for cache file name')
    return name_cache_file


def filecached(
        func=None, *, file_name=None, mode: CacheMode = CacheMode.KEEP,
        pack: Callable = None, unpack: Callable = None,
        protocol='pickle', load=None, save=None):
    """
    Decorator to warp function calls into file-based caching.
    :param func: function to cache its calls
    :param file_name: either full path to the file or a function to create such
            (it must accept all the arguments passed to `func`)
    :param mode: CacheMode supported: KEEP, PASS, LOAD
    :param pack: optional function to pack data before saving
    :param unpack: optional function to unpack data after loading
    :param protocol: 'pickle' | 'pandas' - defines load and save functions for the caching.
    :param load: override cache loading function of the protocol
    :param save: override cache saving function of the protocol
    :return: cached function or maker of cache function initialized by given arguments
    """
    namer = file_namer(file_name)

    if protocol == 'pickle':
        from io.special import savep as prot_save, loadp as prot_load
    elif protocol == 'pandas':
        from pandas import read_pickle as prot_load, to_pickle as prot_save
    else:
        raise NotImplementedError(f'Unknown protocol {protocol=}')

    load = load or prot_load
    save = save or prot_save
    _load = unpack and (lambda f: unpack(load(f))) or load
    _save = pack and (lambda f, d: save(f, pack(d))) or save

    def cached_dec(_func):
        if mode == CacheMode.PASS:
            return _func

        @wraps(_func)
        def cached_func(*args, **kws):
            cache_file = Path(namer(_func, *args, **kws))

            if cache_file.is_file():
                if mode in (CacheMode.LOAD, CacheMode.KEEP):
                    _log.debug(f'Loading {_func.__name__} results from {cache_file!s}')
                    return _load(cache_file)
                elif mode is CacheMode.CLEAR:
                    cache_file.unlink()
            elif mode is CacheMode.LOAD:
                raise FileExistsError(f"LOAD cache required but not found: {cache_file!s}")

            data = _func(*args, **kws)

            if mode in (CacheMode.KEEP, CacheMode.SAVE):
                _log.debug(f'Saving {_func.__name__} results to {cache_file!s}')
                cache_file.parent.mkdir(exist_ok=True)
                _save(cache_file, data)

            return data

        return cached_func

    return cached_dec if func is None else cached_dec(func)


def name_by_call(func, *args, **kws):
    from pickle import dumps
    from hashlib import md5

    def hashable_args(*all_args):
        for arg in all_args:
            try:
                yield md5(dumps(arg)).hexdigest()
            except:
                pass

    hashes = tuple(hashable_args(*args, *kws.items()))

    args_hash = md5(dumps(hashes)).hexdigest()
    name = f"{func.__qualname__}_{len(hashes)}_{args_hash}"
    _log.debug(f'Generate {name=} using {len(hashes)}(of {len(args) + len(kws)}) hashable arguments')
    return name


class Cacher:
    """ ``Cacher`` class maintains context for otherwise stateless function ``filecache``.
    It can be used by itself, or as a super-class, then providing support for caching of method calls.

    Alternatively this class can be used to cache function calls while
    separating chaing parameters initialization,
    from wrapping a particular function into its own caching context.

    >>> cacher = Cacher(folder='./tmp')
    >>> cacher
    <Cacher>[name=name_from_func, folder=./tmp] mode:KEEP, pack:None, unpack:None, protocol:pickle, load:None, save:None
    >>> def func(x): return 10*x
    >>> cacher(func)(2)
    20
    """

    # ----------------------------------------------------
    def __init__(self, *, name=name_by_call, folder=None,
                 mode: CacheMode | bool = CacheMode.KEEP,
                 pack: Callable = None, unpack: Callable = None,
                 protocol='pickle', load=None, save=None):
        self.name = name
        self.folder = folder
        self._cache_par = dict(mode=mode, pack=pack, unpack=unpack,
                               protocol=protocol, load=load, save=save)

    def __copy__(self):
        return self.context()

    def __repr__(self):
        form = lambda x: getattr(x, '__qualname__', str(x))
        par = '\n\t'.join(f"{k}:{form(v)}" for k, v in self._cache_par.items())
        return f"<Cacher>[name={form(self.name)}, folder={self.folder}]\n\t{par}"

    def update(self, *, mode: CacheMode | bool = None, name=None, folder=None, **par):
        """Update cacher parameters"""
        if isinstance(mode, bool):
            mode = mode and CacheMode.KEEP or CacheMode.PASS

        mode is not None and par.update(mode=mode)
        folder is not None and setattr(self, 'folder', folder)
        name is not None and setattr(self, 'name', name)

        if unknown := [*filter(lambda p: p not in self._cache_par, par)]:
            raise NameError(f"Caching parameters {unknown} not among supported: {[*self._cache_par]}")
        self._cache_par.update(**par)
        return self

    def context(self, *, name=None, folder=None, mode: CacheMode | bool = None, **par):
        """Create a new caching context inheriting this cacher (self) parameters
        and updating it with provided arguments.

        The original (self) cacher remains intact.

        >>> cacher = Cacher(folder='./tmp/.cache')
        >>> special_cacher = cacher.context(folder = './tmp/.special_cache', name='special_func')
        >>> special_func = special_cacher(lambda x: x*2)
        >>> special_func(2)
        4
        """
        kws = vars(self).copy()
        cache_par = kws.pop('_cache_par')
        return self.__class__(**kws, **cache_par).update(name=name, folder=folder, mode=mode, **par)

    def cached(self, func, *, name=None, folder=None, mode: CacheMode | bool = None, **par):
        """Unless  `mode` is ``False`` return cached version of the function
        with optionally updated caching parameters, otherwise return the original function.

        Arguments with ``None`` values are not updated!

        >>> Cacher(folder='./tmp').cached(min, mode=CacheMode.PASS)(1, 2)
        1
        """
        if mode is False:
            return func

        if not par and all(map(lambda _: _ is None, [name, folder, mode])):
            return filecached(func, file_name=self._file_name(), **self._cache_par)
        return self.context(name=name, folder=folder, mode=mode, **par)(func)

    def __call__(self, func=None, *,
                 name=None, folder=None, mode: CacheMode | bool = None, **par) -> Cacher | Callable:
        """Syntactic sugar, calls ``context`` if func is None, ``cached`` otherwise."""
        if func is None:
            return self.context(name=name, folder=folder, mode=mode, **par)
        return self.cached(func, name=name, folder=folder, mode=mode, **par)

    def _file_name(self):
        from pathlib import Path
        name = self.name
        folder = self.folder

        try:  # construct file name
            if not name:
                if folder or not (file_name := Path(self._cache_par['file_name'])):
                    raise RuntimeError()
            else:
                if folder := (folder or self.folder):
                    folder = Path(folder)

                if isinstance(name, Callable):  # if name is a function creating a name
                    file_name = name  # the folder is optional and is used only
                    if folder:  # if explicitly provided, then expecting
                        def file_name(_func, *args,
                                      **kws):  # name() to produce only name part
                            return folder / name(_func, *args, **kws)
                else:  # Otherwise, folder is mandatory!
                    file_name = folder / name
        except:
            raise RuntimeError(f'Cache path is not initialized: {folder=}, {name=}')
        return file_name

    def is_loading(self, fnc=None, *args, **kws):
        """Simulates filecached logic to see if cache state will attempt to load"""
        if not self._cache_par['mode'] in [CacheMode.KEEP, CacheMode.LOAD]:
            return False
        name = self._file_name()
        if isinstance(name, str):
            name = Path(name)
        if isinstance(name, Path):
            return name.exists()
        return Path(file_namer(name)(fnc, *args, **kws)).exists()


def is_folder_writable(folder: Path):
    """Check folder is writable by touching a file in it.
    If folder is missing it is created.
     """
    folder = isinstance(folder, Path) and Path(folder) or folder

    try:
        folder.mkdir(parents=True, exist_ok=True)
        try_file = folder / '_file_can_be_created_.check'
        try_file.touch()
        try_file.unlink()
    except:
        return False
    return True


if __name__ == '__main__':
    _log.setLevel(_log.DEBUG - 1)


    def build(num=10, k=2, inc=1, flt=4):
        return [
            CachedPipe.Source(range(num), f'source_{num}'),
            CachedPipe.Map(lambda x: x * k, f'mul_{k}'),
            CachedPipe.Filter(lambda x: x % flt == 0, f'flt_{flt}'),
            CachedPipe.Map(lambda x: x + inc, f'add_{inc}')
        ]


    fld = '/tmp/_cache'
    res1 = [*CachedPipe(build(10, 2, 1), fld)]
    print(res1)


class PersistCache:

    @classmethod
    def cachable(cls, *, namer=None, mode='rw', loc: Locator = EnvLoc.ALG_CACHE, enable=True):
        """decorator around function to cache its calls"""
        from functools import wraps

        def decorator(func):
            if enable is False:
                return func
            cacher = PersistCache(func, namer=namer, loc=loc, mode=mode)

            @wraps(func)
            def wrapper(**kwargs):
                return cacher._cached_call(**kwargs)

            return wrapper

        return decorator

    def __init__(self, proc, *, loc: Locator = EnvLoc.ALG_CACHE, mode: str = '',
                 namer: Callable = None, pre_save: Callable = None, post_load: Callable = None):
        self._mode = mode or ''
        self._proc = proc
        self._pre = pre_save
        self._post = post_load
        self._loc = loc
        self._namer = namer or self._default_namer

        if self._mode:
            if not set(self._mode).issubset('rw'):
                raise ValueError(f"Invalid {mode = }, expecting a combination of 'rw'")

            if not list(loc.defined()):
                raise NotADirectoryError(f"No cache locations are defined by {loc}")

    @staticmethod
    def _default_namer(**kwargs):
        return hex(hash(tuple(kwargs.items()))) + 'cached'

    def _cached_call(self, **kwargs):
        """
        :param proc:
        """
        name = self._namer(**kwargs) if self._mode else None
        if 'r' in self._mode:
            if file := self._loc.first_file(name):
                res = load(file)
                _log.debug("Result of %s loaded from %s", self._proc, file)
                return self._post(res) if self._post else res

        res = self._proc(**kwargs)
        if 'w' in self._mode:
            for cache_folder in self._loc.defined():
                try:
                    filename = cache_folder / name
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    dump(self._pre(res) if self._pre else res, filename)
                    break
                except Exception as ex:
                    if filename.exists(): filename.umlink()
                    _log.warning(f"Error dumping cache for {name}: {ex}")
            else:
                raise NotADirectoryError(f"Failed dumping algo cache for {name}")

        return res
