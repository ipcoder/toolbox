""" Module contains utilities for automation of files processing.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Union, Iterable, Callable, Literal, Type, Any, Generator

from .datatools import unique as unique_filter
from .short import as_iter
from toolbox.utils.events import timed, Timer
import logging

PathT = Union[Path, str]
UNDEF = object()


def represents_path(v: PathT, required='/\\', max_len=256):
    """Checks if given object MAY represent path:
      1. if its `Path` instance, or
      2. if is a str which contains at least one of the `smb` characters
    :param v: object to test
    :param smb:  string with collection of path-indicating characters
    """
    if isinstance(v, str):
        if required:
            for s in required:
                if s in v: break
            else:
                return False

        if len(v) > max_len or re.search(r'[\t\n?^@&\'\"]|\s{2,}', v):
            return False
        return True
    return isinstance(v, Path)


def package_folder(dotted: str):
    """
    Given the `import name` of a package or module
    return folder of its location.

    :param dotted: dotted name of the package or module
    :return: Path object of its folder
    Example:
    ::
        package_folder('toolbox.datacast')
        PosixPath('/home/user/code/algodev/inu/datacast')

    :param p:
    :return:
    """
    from importlib.util import find_spec
    spec = find_spec(dotted)
    return Path(spec.origin).parent


def root_cropper(root: PathT, method: Literal['crop', 'replace', 'relative'] = 'replace',
                 *, out_str=str) -> Callable[[PathT], PathT]:
    """
    Create a function that crops a root folder from path.
    May choose from those methods:
     * crop    -  just removes beginning of the path str with length of root.
                  Use if sure that all the paths include root
     * replace -  use ``str.replace`` to find and remove (once) the root substring.
                  Rise error if not found, so less dangerous but slower (~x2)
                  Still makes assumptions about normalization of paths.
     * relative - use Path.relative_to to find relative path to the root.
                  Safest option, but much slower (~x50 than crop)
    :param root: path to a root folder
    :param out_str: if True the function would return str otherwise Path object
    :param method: supported 'crop' | 'replace' | 'relative'

    :return: function object receiving path and returning str of relative to root
    """
    root = normalize(root)
    if method == 'relative':
        return \
            (lambda s: str(Path(s).relative_to(root))) if out_str else \
                (lambda s: Path(s).relative_to(root))

    root = str(root)
    if not root.endswith(os.sep):
        root += os.sep

    if method == 'replace':
        return out_str and \
               (lambda s: s.replace(root, '', 1)) if out_str else \
            (lambda s: Path(s.replace(root, '', 1)))
    if method == 'crop':
        root_len = len(root)
        return \
            (lambda s: s[root_len:]) if out_str else \
                (lambda s: Path(s[root_len:]))

    raise NotImplementedError(f'Not supported path cropping {method=}')


def root_adder(root, *, as_str=True, out_str: bool = None) -> Callable[[PathT], PathT]:
    """
    Create a function that prepends root folder to a relative path.
    :param root: path to a root folder
    :param as_str: use root as string and append using string manipulation
                   alternatively uses pathlib (~x20 slower, but more robust)
    :param out_str: True of False for the func return str.
                    if None follows as_str.

    :return: function object receiving path and returning str of relative to root
    """
    out_str = as_str if out_str is None else out_str
    root = normalize(root)
    if as_str:
        root = str(root)
        if not root.endswith(os.sep):
            root += os.sep
        return \
            (lambda s: root + s) if out_str else \
                (lambda s: Path(root + s))
    else:
        return \
            (lambda s: str(root / s)) if out_str else \
                (lambda s: root / s)


def compress_numpy_file(file, clean=True):
    """ Convert numpy file into compressed npz form (loss-less)
    :param clean: remove the sources
    """
    import numpy as np
    from toolbox.io.imread import imread
    file = Path(file)
    np.savez_compressed(file.with_suffix('.npz'), **imread(file, out=dict))
    if clean:
        os.remove(file)


def prepare_parent_folder(file_or_path, mode=0o777):
    """
    Make sure given folder (or parent folder of given file) exists.

    :param file_or_path:
    :param mode: mode for the _new_ folder
    """
    normalize(file_or_path).parent.mkdir(mode=mode, exist_ok=True)


def empty_folder(folder_path):
    """ Create empty folder if folder in folder_path not exist or clean clean existing one.

    :param folder_path: resulting empty folder
    """
    from shutil import rmtree
    import stat

    if os.path.exists(folder_path):
        rmtree(folder_path)
    os.makedirs(folder_path)
    os.chmod(folder_path, stat.S_IWUSR)


def valid_folder(folder):
    """Normalize folder and throw FileExistsError if path not exists"""
    folder = normalize(folder, Path)
    if folder.is_dir() and folder.exists():
        return folder
    raise FileExistsError(f'Folder {folder} does not exist!')


def rename_files_by_ref(files: Union[Iterable, str], ref_files: Union[Iterable, str], *,
                        pat=r'_([\d]+)\.[\w]+$',
                        ref_pat=None, rename=True) -> List[Tuple[str, str]]:
    """ Rename files by changing their ids as appear in /some_file_name_<id>.ext
    by using ids from another list of files.
    Match is done by lexicographical sort!
    :param files: iterable over files' names to rename or glob pattern
    :param ref_files: iterable over files' names to use as reference or glob pattern
    :param pat: regexp pattern with a single group to replace in the original files
                by default - '_<numbers>.ext'
    :param ref_pat: pattern with a single group to replace with from the reference.
                    if None - same as pat
    :param rename: if False don't really change the files
    :return list of tuples [(old, new)]
    """
    from glob import glob
    import re

    if isinstance(files, str):
        files = glob(files)
    if isinstance(ref_files, str):
        ref_files = glob(ref_files)

    if len(files) != len(ref_files):
        raise ValueError(f'Files lists are of different size: {len(files)} != {len(ref_files)}')

    pat = re.compile(pat)
    ref_pat = pat if ref_pat is None else re.compile(ref_pat)

    def replace_id(nm1, nm2):
        id1 = pat.search(nm1).groups[0]
        id2 = ref_pat.search(nm2).groups[0]
        return nm1.replace(id1, id2)

    old_new_pairs = [(f, replace_id(f, r)) for f, r in
                     zip(sorted(files), sorted(ref_files))]
    if rename:
        for old_new in old_new_pairs:
            os.rename(*old_new)
    return old_new_pairs


def normalize(path: PathT, out: Type[Path] | Type[str] | None = Path):
    """
    Translate path into canonical form expanding user but not resolving links.
    Mainly created since ``pathlib``'s resolve does path normalization
    AND link resolution, which in some cases is not desirable.

    Example:
    >>> normalize('~'), normalize('~', out=None), normalize(Path('~'), out=None)
    Path('/home/user'), '/home/user', Path('/home/user')

    :param path: str or Path object
    :param out: output type, None - keep as input
    :return: Path | str
    """
    norm = os.path.normpath(os.path.expanduser(path))
    if out is None and not isinstance(path, str):
        out = type(path)
    return out(norm)


def transform_path(full_path: str | Path, *, new_root: str = None, new_ext: str = None,
                   old_root: Path | str = None) -> str:
    """
    Transforming the input path based on the supplied parameters.
    The user must supply BOTH the full path and the old root.
    The user must supply AT LEAST one of the two:
    1. new root - this root will replace the old root.
    2. new_ext - this extension will replace the old extension

    Example:
    >>> supplied_full_path = f'{old_root}/.../file.old_ext'
    >>> transformed = transform_path(supplied_full_path, new_root, new_ext, old_root)

    The new path structure is:
    >>> f'{new_root}/.../file.{new_ext}'

    :param full_path: desired file absolute full_path
    :param new_root: desired root
    :param new_ext: desired extension
    :param old_root: old root

    :return: post transformation string that altered by the user desired paramaters
    """
    assert full_path, 'full_path must supplied'
    assert new_root or new_ext, \
        'New path must differ in at least one segment, overwrite is not supported'
    if new_root: assert old_root, 'Root replacements demands old root to be supplied.'

    if new_ext and '.' not in new_ext:
        new_ext = f'.{new_ext}'

    new_path = str(Path(full_path).with_suffix(new_ext)) if new_ext else str(full_path)

    if new_root:
        if old_root != Path('/'):  # if there is an old root
            new_path = new_path.replace(Path(old_root).__str__(), Path(new_root).__str__())
        else:  # special case where old root is '/' - include the new root at the start
            new_path = Path(new_root).joinpath(Path(new_path[1:]))
    return new_path


class Locator:
    """
    Manages multiple locations in the file system and provides methods to find and validate them.

    Each of the locations can be defined:
     - explicitely as a path,
     - as environment variable
     - as another Locator

    Main functionality resides in ``Locator.existing_folders()`` method.
    By default it returns iterator over actually existing folders at the time of the call.
    """
    _PT = Union[PathT, 'Locator']
    _PsT = Iterable[_PT] | _PT

    _log = logging.getLogger('env')

    AlarmTypes = bool | Callable[[str], Any] | Exception | Type[Exception]

    @staticmethod
    def _win_bug_safe(safe: bool):
        if os.name == 'nt' and safe:
            Locator._log.info(f'Disabling {safe=} mode (not supported in Windows)')
            return False
        return safe

    @staticmethod
    def _validate_alarm_type(alarm: AlarmTypes) -> AlarmTypes:
        if (isinstance(alarm, type) and not issubclass(alarm, Exception)
                or not isinstance(alarm, (bool, Exception, Callable))):
            raise TypeError(f"Invalid {type(alarm)=}")

        if os.name == 'nt' and alarm:
            Locator._log.info(f'Disabling validation {alarm=} (not supported in Windows)')
            return False
        return alarm

    @staticmethod
    def _normalize(p: _PT) -> _PT:
        """If provided location is path and not ``Locator`` normilize its representation"""
        return p if isinstance(p, Locator) else normalize(p)

    def __init__(self, *folders: PathT | Locator | None, envar: str = None, sub: PathT = None, order='EIA',
                 alarm: AlarmTypes = False, safe: bool | None = None, timeout=.5, caching=True):
        """
        Define locations to check for usually known before the run-time.

        Locations are searched in order is defined by the string of following letters denoting:
          - *E*: folder from the Env variable
          - *I*: Internal fallback folders
          - *A*: Additional folders

        Every letter must be used no more than once.
        Valid orders: '*IA*', '*E*', '*AIE*'.

        Non-responsiove File System
        ---------------------------

        Sometimes a network mount may become non-responsive freezing file system queries to its content.

        (*Validation and safety mechanism is currently only supported in Linux.*)

        Validation
        ==========

        ``Locator.validate()`` allows to discover such situation and remove problematic
        folder from the locator or raise exception depending on the `alarm` argument:
          - ``False`` - don't validate
          - ``True`` - remove problematic folder silently
          - ``Exception`` - given exception object is given is raisen
          - ``Exception class`` - construct an object of this type with info message and raise
          - ``Callable`` - called with info message

        Safely Checking Existence
        =========================

        It's an alternative meachanism based on a non-hanging alternative for `Path.is_dir`.

        It can be activated by setting `safe`=``True`` in constructor, or as argument
        in some method like ``existing``, ``first_existing``.

        I not set explicitely, default state of ``Locator.safe`` depends on validation:
            - ``Locator(...,alarm,safe=None)`` sets `safe = bool(alarm)`
            - ``Locator.validate(...,alarm,safe=None)`` resets `safe = not alarm`

        :param folders:     list of paths, or ``None`` - empty strings and ``None`` are ignored!
        :param envar:       environment variable with search folder
        :param default:     default search folder
        :param sub:         sub-path relative to those defined in envar and defaults
        :param order:       order of searching locations
        :param alarm:       validate accessibility of all the deinfed folders
        :param safe:        (de)activate slower but not not-hanging existence chek of folders
        :param timeout:     timeout in seconds for safe and validation operations
        """
        # ---- safe checking mechanism ---
        self.alarm = self._validate_alarm_type(alarm)
        self.safe = bool(self.alarm) if safe is None else safe
        self._cache = {}
        self.timeout = timeout
        self.set_check_opt(caching=caching)

        # ---- core logic ------------
        self.envar = envar
        self.folders = list(self._norm_folders(folders))  # remove possible Nones, etc
        self.order = self._valid_order(order)
        self._initial = {'envar': envar, 'folders': self.folders}

        self._show_existing = False  # control if  __repr__ does not checks firts_existing

        if isinstance(sub, str):
            if (sub := sub.strip()) == '':
                raise ValueError(f"Invalid value of argument {sub=}!")
            sub = Path(sub)
        if sub and sub.is_absolute():
            raise ValueError(f"Argument {sub=} must be relative!")
        self.sub = sub

    def set_check_opt(self, *, caching: bool | None | Literal['clear'] = None,
                      safe: bool = None, timeout=None) -> bool | None:
        """
        Set options for safe checking of "non-reponding" path mechanism
        used in `validate` and `existing` methods.

        (``None`` argument leaves the corresponding option unchanged)

        A path can be checked by 3 different methods:
          1. ``Path.is_dir`` - not a safe method, the system call may freeze under certain conditions
          2. *safe* check (in another process killed after timeout) - expensive (hundreds ms)
          3. retrive *cached* results of previous safe check - fast, but not upto date

        :param safe: (de)activate safe `existence` checks
        :param caching: bool - enable or disable, or 'clear' (also enables)
        :param timeout: indicator of "non-responsiveness"
        """
        if safe is not None:
            self.safe = safe

        if timeout is not None:
            self.timeout = timeout

        if caching == 'clear' or caching is True and self._cache is None:
            self._cache = {}
        elif caching is False:
            self._cache = None

    def _check_path(self, path, timeout=None, safe=None):
        if safe is None:
            safe = self.safe
        if not safe:
            return Path.is_dir(path)

        if timeout is None:
            timeout = self.timeout

        path = str(path)
        caching = self._cache is not None

        if not caching or UNDEF is (res := self._cache.get(path, UNDEF)):
            res = check_path(path, timeout=timeout, log=self._log.info)
            caching and self._cache.setdefault(path, res)
        elif caching:
            self._log.debug(f'check cache hit: `{path}`')
        return res

    @property
    def safe(self):
        return self._safe

    @safe.setter
    def safe(self, val: bool):
        self._safe = self._win_bug_safe(val)

    def reconfigure(self, *folders: PathT | Locator, envar: str = UNDEF, order=None):
        """
        Reset configuration of the locator by replacing folders AND/OR environment variable IF provided.

        Usefull for alter locations for objects already initialized by this locator object without
        reinstantiation or reconfiguring them.

        :param folders: new folders
        :param envar: new name of environment variable
        :param order: new order
        """
        if envar is not UNDEF:
            self.envar = envar
        if folders:
            self.folders = list(self._norm_folders(folders))
        if order:
            self.order = self._valid_order(order)

    @staticmethod
    def _norm_folders(folders):
        return map(Locator._normalize, filter(None, folders))

    def validate(self, alarm: AlarmTypes | None = None, *, safe=None, timeout=None) -> Locator:
        """
        Validate locator by removing invalid problematic folders and envar (or raise ),
        depending on the `alarm` argument:
          - ``None`` - follow ``self.alarm`` value
          - ``False`` - don't validate
          - ``True`` - remove problematic folder silently
          - ``Exception`` - given exception object is given is raisen
          - ``Exception class`` - construct an object of this type with info message and raise
          - ``Callable`` - called with info message

        Generally, since validation makes *safe* queries unnecessary, by default it sets \n
        ``self.safe=False`` (`if alarm` and `safe==None`)

        :param alarm:  optionally (if not ``None``) used instead of ``self.alarm``
        :param safe: optionally (if not ``None``) set ``self.safe`` to this value.
        :param timeout: optional (if not ``None``) to use insted own timeouts
        :return: ``self``
        """
        if alarm is None:
            alarm = self.alarm
        else:
            self._validate_alarm_type(alarm)

        if safe is None:  # default beghaviour
            safe = False if alarm else self.safe  # disable if validated or leave untoched
        self.safe = safe

        if not alarm:
            return self

        def _timeout(base):
            return base if timeout is None else timeout

        def rmv_msg(obj):
            f"Removing from {self} \n\t not responding for {_timeout(self.timeout)}sec {obj}"

        def _validate(folder: Path | Locator) -> Path | Locator | None:
            """Return validated locator or folder if OK, or ``None`` or raise depending on `validate`.         """
            nonlocal alarm  # otherwise (..., alarm=alarm) can't work
            if isinstance(folder, Locator):  # if locator - recreate it recursively validated
                return Locator(*folder.folders, envar=folder.envar, sub=folder.sub, order=folder.order,
                               safe=folder.safe, alarm=alarm, timeout=_timeout(folder.timeout))

            if check_path(
                    folder / self.sub if self.sub else folder,
                    timeout=_timeout(self.timeout)
            ) is not None:  return folder  # folder is OK

            # ---------------  now process the failure  --------------
            if alarm is True or isinstance(alarm, Callable):
                self._log.warning(msg := rmv_msg(folder))
                alarm is not True and alarm(msg)
                return None

            if isinstance(alarm, type) and issubclass(alarm, Exception):
                alarm = Exception(f"{folder=} is not accessible!")
            if isinstance(alarm, Exception):
                raise alarm
            raise TypeError(f"Invalid {type(alarm)}")

        self.folders = list(filter(_validate, self.folders))

        if self.env_path and check_path(
                self.env_path,
                timeout=_timeout(self.timeout)
        ) is None:  # invalid
            self._log.warning(rmv_msg(f"envar ${self.envar}={self.env_path}"))
            self.envar = None  # remove it

        return self

    def __repr__(self):
        sub = f"|'/{self.sub}'" if self.sub else ''

        INDT = '\n\t\t\t'
        folders = list(map(str, self.folders))
        n = len(folders)
        if any('\n' in f for f in folders):  # multiline already - prepend \t to all the lines
            folders = INDT.join(f.replace('\n', INDT) for f in folders)
        else:
            sep = INDT if n > 1 or sum(map(len, folders)) + 2 * n > 80 else ', '
            folders = sep.join(folders)
        folders = f'folders:{INDT}{folders}' if sep == INDT else f"[{folders}]"
        env = f"${self.envar}={self.env_path or ''}" if self.envar else ''
        if self._show_existing:
            found = '🗸' if self.first_existing() else '❌'
        else:
            found = ''

        safe = ('☔' if self.safe else '🌂') if self.alarm else ('☂' if self.safe else '')
        return f"{type(self).__name__}<{self.order}{sub}>{env} {folders}{found}{safe}"

    def repr(self, existing=True):
        """Allows explicitely request expensive show of existing folder"""
        keep = self._show_existing
        if existing is not None:
            self._show_existing = existing
        res = self.__repr__()
        self._show_existing = keep
        return res

    def __bool__(self):
        return bool(self.folders or self.envar)

    def __add__(self, folders: _PsT) -> Locator:
        return Locator(*self.folders, *as_iter(folders), envar=self.envar, sub=self.sub)

    def __iadd__(self, folders: _PsT):
        self.folders.extend(self._norm_folders(as_iter(folders)))
        return self

    def _rm_folders_iter(self, folders: _PsT):
        """Return iterator over folders remained in ``self`` after removing given folders.

        If ``self.folders`` contain ``Locator`` objects, they are replaced by
        a copies with given folders removed from them as well.
        """
        folders = set(self._norm_folders(as_iter(folders)))
        for p in self.folders:
            if isinstance(p, Locator):
                p = p - folders
            elif p in folders:
                continue
            yield p

    def __sub__(self, folders: _PsT) -> Locator:
        return Locator(*self._rm_folders_iter(folders), envar=self.envar, sub=self.sub)

    def __isub__(self, folders: _PsT):
        self.folders = list(self._rm_folders_iter(folders))
        return self

    def __truediv__(self, sub: PathT) -> Locator:
        """Create new locator with a more deep sub relative to the former sub.
        The path in the divisor is an extension to the current sub.
        Locator("/tmp", sub="one") / "two" == Locator("/tmp", sub="one/two")
        """
        if isinstance(sub, str) and not (sub := sub.strip()):
            return self
        if self.sub:
            sub = Path(self.sub) / sub
        return Locator(*self.folders, envar=self.envar, sub=sub)

    @staticmethod
    def _valid_order(order):
        if not isinstance(order, str):
            raise TypeError(f'Expected str argument {order=}')
        order_set, codes = set(order := order.upper()), set('AEI')
        if not order_set or len(order_set) < len(order) or order_set - codes:
            raise f"Invalid {order=}. Use {codes=} no more than once each."
        return order

    def _locations_sequence(self, *folders: PathT, order=None) -> Generator[Path]:
        """Generator of locations in certain order coded using those letters:

           - *E*: folder from the Env variable
           - *I*: Internal fallback folders
           - *A*: Additional folders

        Every letter must be used no more than once.
        Valid orders: '*IA*', '*E*', '*AIE*'.

        **Note!** The additional folders are not appended by ``sub``!

        :param order: if provided overrides default order
        :param *folders: additional locations to try
        """

        def prep(p, sub=None):
            if isinstance(p, str):
                p = Path(p)
            if sub:
                p = p / sub
            return p

        def folders_gen(fs, sub=None):
            for i, loc in enumerate(fs):
                if isinstance(loc, Locator):
                    yield from loc._locations_sequence(order=order)
                else:
                    yield prep(loc, sub)

        def env_gen(sub):
            if enval := self.env_path:
                yield prep(enval, sub)

        # ---------------------------------------------
        order = self._valid_order(order or self.order)
        for src in order:
            yield from (
                env_gen(self.sub) if src == 'E' else
                folders_gen(self.folders, self.sub) if src == 'I' else
                folders_gen(self._norm_folders(folders))
            )

    @property
    def env_path(self):
        return self.envar and os.getenv(self.envar)

    def defined(self, *folders, order=None, unique=True) -> Generator[Path]:
        """
        Generator of all the defined (not necessary existing!) locations.

        Order is defined by the string of following letters denoting:

          - *E*: folder from the Env variable
          - *I*: Internal fallback folders
          - *A*: Additional folders

        Every letter must be used no more than once.
        Valid orders: '*IA*', '*E*', '*AIE*'.

        **Note!** The additional folders are not appended by ``sub``!

        **Note!**
            Current implementation treats ``Locator`` objects found amoung `folders`
            as source of folders without distinguishing thier internal ``envar``.

            Therefore environment-defied folders will be intermixed in their order
            within other folders defined by the internal locator.

            However, the `order` argument is still passed recursively
            to order internal locators.

        :param order: if provided overrides default order
        :param fail: if ``True`` raise ``NotADirectoryError`` for any not existing folder
        :param *folders: additional locations to try
        :param unique: if True filters out repeated appearences of same folders
        """
        seq = self._locations_sequence(*folders, order=order)
        return unique_filter(seq) if unique else seq

    def existing(self, *folders, order=None, unique=True, fail=False, safe=None) -> Generator[Path]:
        """
        Generator of ONLY existing folders checked in the given order.

        Order is defined by the string of following letters denoting:

          - *E*: folder from the Env variable
          - *I*: Internal fallback folders
          - *A*: Additional folders

        Every letter must be used no more than once.
        Valid orders: '*IA*', '*E*', '*AIE*'.

        **Note!** The additional folders are not appended by ``sub``!

        **Note!**
            Current implementation treats ``Locator`` objects found amoung `folders`
            as source of folders without distinguishing thier internal ``envar``.

            Therefore environment-defied folders will be intermixed in their order
            within other folders defined by the internal locator.

            However, the `order` argument is still passed recursively
            to order internal locators.

        :param order: if provided overrides default order
        :param fail: if ``True`` raise ``NotADirectoryError`` for any not existing folder
        :param *folders: additional locations to try
        :param unique: if True filters out repeated appearences of same folders
        :param safe: override `safe` flag of the oject for this call
        """
        seq = map(normalize, self._locations_sequence(*folders, order=order))

        for folder in unique_filter(seq) if unique else seq:
            if ret := self._check_path(folder, safe=safe):
                yield folder
            elif fail:
                raise NotADirectoryError(f"{'Not-responsding' if ret is None else 'Missing'} {folder=}")

    def first_existing(self, *folders: PathT, order=None, safe=None) -> Path | None:
        """
        Return first existing folder in the given order.

        Order is defined by the string of following letters denoting:
          - *E*: folder from the Env variable
          - *I*: Internal fallback folders
          - *A*: Additional folders

        **Notice!** Nones and '' are filtered out from `folders` before the processing!

        :param folders: additional "fall back" options to check
        :param order: optionally override order of searching the locations
        :param safe: override `safe` flag of the oject for this call
        """
        for folder in self.existing(*filter(None, folders), order=order, safe=safe):
            return folder
        return None

    def find_file_iter(self, name: PathT, *, order=None, safe=None) -> Generator[PathT]:
        """Generator of all the found paths to the file with given name.

        :param name: the last parts of the file path
        :param order: if provided ('EIA' codes) use insted of the current order
        :param safe: override `safe` flag of the oject for this call
        """
        for folder in self.existing(order=order, safe=safe):
            full_path = folder / name
            if full_path.exists():
                yield full_path

    def first_file(self, name, *, order=None) -> None | Path:
        """
        Search for the first occurance of the file in all the locations.

        Example:
        >>> Locator('/path1', '/path2/sub').first_file('par/name.ext')
        # Path('/path2/sub/par/name.ext') - first location found

        :param name: file name with possible its parents folders parts
        :param order: if provided ('EIA' codes) use insted of the current order
        :return: None or the first found file full path
        """
        for path in self.find_file_iter(name, order=order):
            return path

    @property
    def first(self) -> Path | None:
        """First DEFINED (not cheking existence) location by the currently set order"""
        for folder in self.defined():
            return folder

    def first_of(self, order=None) -> Path | None:
        """First DEFINED (not checking existence) by the given or current order

        :param order: override the current order using 'EIA' codes.
        """
        for folder in self._locations_sequence(order='I'):
            return folder

    def reset_folders(self):
        self._log.warning(f'Attention! Erasing {self} internal folders.')
        self.folders = []


def glob_folders(folders, file_pattern) -> Iterable[str]:
    """
    Find files using `glob` in MULTIPLE folders.
    With a single folder works as iterable version of `glob`
    :param folders: a folder or iterable of folders
    :param file_pattern: glob pattern for path under the folder
    :return: Iterator over the files found
    """
    from toolbox.utils import as_list
    from more_itertools import collapse

    return list(collapse(
        glob(str(Path(folder, file_pattern).expanduser()))
        for folder in as_list(folders)))


class PyModuleLocator:
    """
    Allows to locate modules in specific folder and determine their package name.
    """
    _Path = Path | str

    def __init__(self, folder: _Path):
        """
        Create module locator
        :param folder:
        :param paths:
        :param sys_paths:
        """
        import sys

        self._cache = {}
        self.package = None
        self.folder = Path(folder).resolve()

        if self.folder.is_file():
            raise NotADirectoryError(f"Found file insted {str(self.folder)}")

        # find the longest package root to include the folder
        self.sys_path = max((os.path.commonpath([self.folder, p])
                             for p in sys.path if self.folder.is_relative_to(p)
                             ), key=len, default=None)
        sep = '\n\t'
        paths_msg = f"max common {self.sys_path} from import paths:{sep}{sep.join(sys.path)}"
        if self.sys_path is None:
            raise ModuleNotFoundError(f"Folder {str(self.folder)} is not in under {paths_msg}")
        elif (_log := logging.getLogger('env')).isEnabledFor(logging.DEBUG):
            _log.debug(f"Found under {paths_msg}")

        package = '.'.join(self.folder.relative_to(self.sys_path).parts)
        if self._is_valid_route(self.sys_path, package, sys_path=True):
            self.package = package

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self.folder)})[{self.package}]"

    @classmethod
    def add_sys_path(cls, *paths, check_exists=True):
        """Normalize and and add given path to sys.path if not already there.
        :param paths: str or Path to add
        :param check_exists: vefify that it exists
        """
        for path in paths:
            path = normalize(path)
            if check_exists and not path.is_dir():
                raise NotADirectoryError(f"Can't add to sys.path not existring dir {path}")

            if not (path := str(path)) in set(sys.path):
                sys.path.append(path)

    @staticmethod
    def _is_valid_route(root, route: str, sys_path) -> bool:
        """Validate that given route describes valid package or module located under the `root` folder.

        That is a folder exists on every dot level with '__init__.py', unless:
          - `route` == '' or points or module directly under the `root`.

        :param root: parent folder relative to which the `route` is defined
        :param route: dotted package or module name (under the root)
        :param sys_path: True if root is in sys.path and does not need '__init__.py'

        :return True if valid
        """
        if route == '':
            return True

        root = Path(root)
        path = root.joinpath(*route.split('.'))  # for sure at least 1 part in the split

        if path.with_suffix('.py').is_file():  # route points to a module
            path = path.parent  # set to the folder of this module

        while path != root:  # up from the deepest path check for '__init__.py'
            if not path.joinpath('__init__.py').is_file():
                return False
            path = path.parent

        if sys_path:
            return True
        return root.joinpath('__init__.py').is_file()

    def module_import_params(self, module, abs: bool | None = False) -> tuple[str, str | None]:
        """Return a tuple of ``import_module(module, package)`` args given
        an absolute or relative (to `self.package`) module name.

        Depending on the value of the `abs` argument:
         - ``True`` - module must be absolute
         - ``False`` - must not be absolute
         - ``None`` - no special reuests

        Given `module` is validated by locating the corresponding file.

        Failure to satisfy any validation raises ``ModuleNotFoundError``.

        Return (`module`, `package`)
        """
        if module.startswith('.'):
            if abs is True:
                raise ModuleNotFoundError(f"Relative {module=} provided instead of absolute")
            if self.package is None:
                raise ModuleNotFoundError(f"Relative {module=} can't be located without package")

            if self._is_valid_route(self.folder, module, sys_path=not self.package):
                return module, self.package  # if self.package == '' self.folder is in sys.path

            raise ModuleNotFoundError(f"No {module=} in the package {self.package}")
        # continue with absolute module
        if abs is False:
            ModuleNotFoundError(f"Expected relative {module=}")
        if not self.is_valid_absolute(module):
            raise ModuleNotFoundError(f"No sys.path to the absolute {module=}")

        return module, None

    @classmethod
    def is_valid_absolute(cls, route) -> bool:
        """Validate given absoulte route can be imported baseed on the current sys.path"""
        import sys
        for p in sys.path:
            if cls._is_valid_route(p, route, sys_path=True):
                return True
        return False

    def file_to_module(self, file: Path, relative=False):
        """Convert file path into its dotted module name.

        **Example**

        ``PyModuleLocator`` instance wth:
        ::
            self.folder:            /some/path/package/folder
            self.package_root:      /some/path/package

            file:                   /some/path/package/folder/sub/module.py
            return:
                if relative:        sub.module
                else:               package.folder.sub.module

        :param file: path to the file
        :param relative: not full name in the package, relative to the ``self.folder`` location
        """
        if self.package is None:
            raise ModuleNotFoundError(f"{str(self.folder)} does not belong to any package")

        section = Path(file).relative_to(self.folder)
        pfx = '' if relative else self.package
        return '.'.join([pfx, *section.parent.parts, section.stem])  #

    @staticmethod
    def modules_under(folder: Path, deep: int = 0, sys_path: bool = False,
                      include: str = None, exclude: str = '[._].*') -> Generator[Path]:
        """Generate Path of module files found under specified folder.

        :param folder: Path to search under
        :param deep: number of levels deep to go, 0 - only the `folder`.
        :param sys_path: if folder is in sys_path: '__init__.py' is not required there
        :param include: re pattern for file names to include
        :param exclude: re pattern for file names to exclude (default, starting with '_'
        """
        included_name = lambda p: (
                (not include or re.fullmatch(include, p.name)) and
                (not exclude or not re.fullmatch(exclude, p.name))
        )

        if not included_name(folder) or (  # excluded folder
                (deep or not sys_path)  # check one every depth (except 0 if sys_path)
                and not folder.joinpath('__init__.py').is_file()
        ): return

        for file in folder.glob('*.py'):  # by files on specific depth
            if included_name(file):
                yield file

        if deep:
            for sub in folder.glob('*/'):
                yield from PyModuleLocator.modules_under(sub, sys_path=False, deep=deep - 1,
                                                         include=include, exclude=exclude)


def _probe_path(path, out_queue):
    """
    Return (by putting to the `out_queue`)
        - ``True`` if path exists
        - ``False`` if not
        - ``None`` if check fails

    :param path:
    :param out_queue:
    :return:
    """
    try:
        out_queue.put(os.path.exists(normalize(path)))
    except Exception:
        out_queue.put(None)


def check_path(path: Path | str, *, timeout=1.0, log: Callable[[str],] = None) -> bool | None:
    """
    Safely check if the given `path` exists (``True``|``False``)
    or there are mounting or access problems (``None``)

    :param path:
    :param timeout: path is considered not responding if delay > timeout
    :param log: optional log function (reports only if delay > 0.1)

    :return: ``True|False|None``
    """
    with Timer(f"Safe-checked '{path}' in", min=0.1, active=bool(log), out_func=log):
        import queue, multiprocessing

        q = multiprocessing.Queue(1)
        p = multiprocessing.Process(target=_probe_path, args=(path, q))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join()
            return None
        try:
            return q.get(False)
        except queue.Empty:
            return None  # no result sent by the child → treat as failure
