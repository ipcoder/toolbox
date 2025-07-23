from __future__ import annotations

import fnmatch
import re
from enum import Flag, auto
from pathlib import Path
from typing import Iterable, Any, Optional, Union, Sequence, Type, Literal, Callable, NamedTuple

from pydantic import BaseModel, Field, DirectoryPath, PrivateAttr, validator

from toolbox.utils import as_list, logger
from toolbox.utils.regexp import is_regex
from toolbox.utils.wrap import name_outputs

_log = logger('format')
Desc = lambda s: Field(default=None, description=s, required=False)
PathT = Union[Path, str]
SupportT = Union[Callable[[...], bool], type]
Strings = Union[str, list[str]]


class Content(Flag):
    UNDEF = 0

    BIN = auto()  # any binary data
    META = auto()  # contains meta data
    CONFIG: Content = auto()  # contains configuration (tree-like)

    IMAGE = auto()  # image kind of binary data
    VIDEO = auto()  # video kind of binary data

    DATA = IMAGE | VIDEO | BIN  # add more binary kinds as introduced
    MIXED = DATA | META  # contains both data and meta-data

    def __repr__(self: Flag):
        from math import log2
        if not (name := self.name):
            name = '|'.join(
                m.name for m in type(self) if
                (val := m.value) and log2(val).is_integer() and m in self)
        return f"{name}[{bin(self.value)[2:]}]"


CT = Content


def sfx_of(file_name_like: PathT):
    """
    Return lower case suffix (the most right part separated by the '.')
    from the string or Path, or '' if not found.
    """
    return (m := re.search(r'\.\w+$', str(file_name_like))) and m[0].lower() or ''


class MetaFormat(type):
    desc: str
    content: Content
    case: bool
    data_sup: None
    meta_sup: None
    specificity: int

    _formats: list[type(FileFormat)] = []

    def valid_content(cls, content):
        return content is CT.UNDEF or cls.content & content

    def _auto_desc(cls, *, abstract=False):
        return cls.__name__.lower().replace('format', '') + (
            ' (Abstract)' if abstract else '')

    def _str_head(cls):
        return f'<{cls.__qualname__}|+{cls.specificity}>'

    def __repr__(cls: type(FileFormat)):
        if not hasattr(cls, '_patterns'):
            return super().__repr__()

        width = 70
        len_ind = 6
        indent = ' ' * len_ind

        desc = getattr(cls, 'desc', '')
        desc = ' ' if cls._auto_desc() == desc else f' "{desc}" '
        head = cls._str_head() + desc
        len_head = len(head)

        patterns = ', '.join(f"{pi}" for pi in cls._patterns)
        patterns = f"[{patterns}]"
        len_pat = len(patterns)

        if len_head + len_pat > width:
            head += '\n' + indent
            if len_pat > width:
                from textwrap import wrap
                subs = wrap(patterns, width=width - 2 * len_ind, subsequent_indent=indent + ' ',
                            break_long_words=False, break_on_hyphens=False)
                patterns = '\n'.join(subs)

        return head + patterns

    @classmethod
    def find_handler(cls, filename: PathT = None, *, write=None, meta=None,
                     out: Literal['any', 'best', 'all', 'all_bests'] = 'best'
                     ) -> FileFormat | list[FileFormat]:
        """Given a filename returns a matched handler with best score or None.

        If called without a filename return list of all the handlers sorted by suffix.

        Depending on the ``best`` argument return only **one** or list of matched handlers.

        Select by support of ``meta`` and ``write`` following convention for the corresponding values:
         - ``True``, search for the supporting handles
         - ``False``, search for handles not supporting ``write``
         - ``None`` - ignore support when searching
         - ``<data>`` - search for handlers capable writing this data object

        Use ``best`` argument to control possible multiplicity of found handlers:
         - '`best`': the *first* **one** of those with the *best* score
         - '`any`': *any* **one** independent of score
         - '`all_best`': ``list`` of all handlers with the same *best* score
         - '`all`': ``list`` of all matched handlers (ignore scores)

        :param filename: filename to match the supported patterns
        :param write: if ``True`` verifies if ``write`` is implemented
        :param out: which matched handlers toi return:

        :param meta: if True or False filter by the support of metadata
        :return: one format handler (out in 'any'|'best') or list of them
        """

        data_flag = lambda flag: (None, flag) if flag in (  # flag: write | meta
            True,  # supported
            False,  # not supported
            None  # does not matter
        ) else (flag, True)  # flag is actually a data, set flag to True
        data, write = data_flag(write)
        meta_data, meta = data_flag(meta)

        def supports(f):
            check_support = lambda flag, sup: \
                True if flag is None else \
                    not sup if flag is False else sup
            # Consider: optimize supports check
            return check_support(write, f.supports_write(data=data, meta=meta_data)) \
                and check_support(meta, f.supports_meta(meta_data))

        # --------------------------------------------------------------
        supporting_formats = sorted(filter(supports, cls._formats), key=lambda f: f.specificity)
        if not filename:
            return supporting_formats

        if not (m := re.search(r'[^\\/:]+$', str(filename))):
            raise NameError(f"Not a valid file name {str(filename)}")
        name_ext = m.group()

        matching_formats = [f for f in supporting_formats if f.regex.fullmatch(name_ext)]

        nm = len(matching_formats)
        if nm == 0:
            return None if out in ('any', 'best') else []
        if nm == 1 or out == 'any':
            return matching_formats[0]

        # more than 1 here
        if out == 'best':
            return max(matching_formats, key=lambda f: f.score(name_ext))
        if out == 'all':
            return matching_formats
        if out == 'all_best':
            scores = [f.score(name_ext) for f in matching_formats]
            best_score = max(scores)
            return list(f for s, f in zip(scores, matching_formats) if s == best_score)
        raise ValueError(f'Invalid argument {out=}')

    @classmethod
    def find_class(cls, name: str):
        """
        Find format class by name (Format substring may be omitted)
        """
        name = name.lower().replace('format', '')
        reg = re.compile(f'(format)?{name}(format)?', re.I)
        for handlers in cls._formats.values():
            for h in handlers:
                if reg.fullmatch(h.__class__.__name__):
                    return h

    @staticmethod
    def _check_support(obj, criteria: SupportT):
        return criteria is None or \
            isinstance(criteria, Callable) and criteria(obj) or \
            isinstance(obj, criteria)


class FileFormat(metaclass=MetaFormat):
    """
    Base class to define and manage file formats.

    New file formats are introduced by subclassing ``FileFormat`` and

    1. Providing additional sub-classing arguments:

      - ``desc``    : Format description string (class name wo "Format" part by default )
      - ``content`` : a combination of ``Content`` flags
      - ``patterns``: one or list of string or regular expressions matching supported file names
      - ``case``    : set ``True`` to treat the patters as case-sensitive, ``False`` by default
      - ``data_sup``: optional information about supported data, its type or Callable(data) -> bool
      - ``meta_sup``: optional information about supported meta-data, its type or callable

    2. Implementing methods:

     -  ``read``  - mandatory (virtual)
     -  ``write`` - optinally, to support writing
     -  ``supports_write`` - to additionally consider ``data`` and ``meta``.

    Example::

        >>> class MiddleburyCalib(FileFormat, desc='Middlebury Camera Calib',
        ...                       patterns="calib.txt", content=CT.CONFIG):
        ...     @classmethod
        ...     def read(cls, filename: PathT):
        ...         return middlebury_calib(filename)
        ...
        ...     @classmethod
        ...     def supports_write(cls, *, data=None, meta=None):
        ...         \"""Illustration obly - super returns false since write is not defined\"""
        ...         return super().supports_write() and isinstance(data, dict)
    """

    class PatternInfo(NamedTuple):
        pattern: str
        regex: re.Pattern
        score: int
        name: str
        sfx: str

        def __repr__(self):
            return f"{self.pattern}({self.score})"

    _patterns: list[PatternInfo]
    regex: re.Pattern
    specificity = 0

    # https://regex101.com/r/pOoTXO/2
    __pat_parser = re.compile(r"^(:?[\\/])?"  # sep
                              r"((?(1)\.?|)[\w*?]+)?"  # name
                              r"((?:\.\w+)+)?$")  # sfx
    # https://regex101.com/r/ZIMelq/1
    __rex_sfx = re.compile(r'(?:(?<!\\)\\\.(\w+|\(\w+(\|\w+)+\)))+$')

    @classmethod
    def _parse_pattern(cls, pat: str | re.Pattern, case: bool) -> PatternInfo:
        r"""
        Split glob-like pattern into file name and sfx.

        Format pattern is a non-empty string with the following structure:
        ::
            format:     [sep][name][sfx]            # where:
            sep:        '\' | '/'                   # '/' may (and should) be used also in Windows
            name:       '*' | (letter | '?' )       # "*", "fun", "f?n"
            sfx:        ('.' letter)+               # ".cfg", ".bmk.yml"
            letter:     [A-Za-z_0-9]                # an ascii letter or number or '_'

        **Notice**:
         - Pattern is not supposed to describe the folder - only name and extension.
         - The optional leading path separator should be used only to resolve the ambiguity
         - Extension may be *multipart*: ``name.some.sfx`` (sfx: ``.some.sfx``).
         - Symbol ``* `` can't be used in extension and interpreted as ``\w+`` (at least 1 letter)
         - Symbol ``?`` can't be used in extension and is interpreted as ``\w`` (some 1 letter)
         - Pattern starting with '.' is treated as extension, with name='*' (``.txt = *.txt``)
         - To define name with leading '.' start pattern with separator '/' (``/.name.sfx``)

        Alternatively pattern may be defined as a regular expression (``re.Pattern`` or ``str``).
        In this case allowed suffices must be either literal or multichoice:
        ::
                Output_Video_(Left|Right)_\d+\.(bmp|png|tif|jpg)
                StereoImage_\d+\.inu\.tif
                Image\.png

        Returns detailed description of the pattern in form of
        ::
            class PatternInfo(NamedTuple):
                pattern: str
                regex: re.Pattern
                score: int
                name: str
                sfx: str

        :param pat: glob-like regexp
        :return ``PatternInfo`` structure for this pattern with fields:
        """
        def score(s: str):
            return sum(map(bool, s.split('.')))  # 1 for any . separated name segment

        if is_regex(pat, compile_check=True):
            pat = re.compile(pat)
        if isinstance(pat, re.Pattern):
            rex, pat = pat, pat.pattern
            if (m := cls.__rex_sfx.search(pat)) and (
                    sfx := m.group().replace(r'\.', '.')) and (
                    name := pat[:m.start(0)]):
                return cls.PatternInfo(pattern=pat, regex=rex,
                                       score=score(sfx) + 2 * (len(name) > 1),
                                       name=name, sfx=sfx)
            raise NameError("In regex pattern must have name and suffix be a regular string!")
        elif pat and (m := cls.__pat_parser.fullmatch(pat)):
            # give a higher score as name is more specific
            _, name, sfx = m.groups()
            rex = (name or '*') + (sfx or '')
            rex = rex.replace('*', r'\w+').replace('?', r'\w').replace('.', r'\.')
            return cls.PatternInfo(pattern=pat, regex=re.compile(rex, flags=0 if case else re.I),
                                   score=score(pat) + (name and len(name) > 1 and 2 or 0),
                                   name=name, sfx=sfx)

        raise NameError(f"Invalid {cls.__name__} pattern {pat}")

    def __init_subclass__(cls, desc: str = None, *,
                          patterns: str | Iterable[str] = None,
                          data_sup: SupportT = None, meta_sup: SupportT = None,
                          content: Content = None, specific=0, case=False, **kws):
        super().__init_subclass__()

        def set_sup(name, val):
            if not (val is None or isinstance(val, Callable) or isinstance(val, type)):
                raise ValueError(f'Invalid subclassing argument {name}={val} for {cls.__qualname__}')
            setattr(cls, name, val)

        set_sup('data_sup', data_sup)
        set_sup('meta_sup', meta_sup)
        cls.desc = desc or cls._auto_desc(abstract=not patterns)
        cls.specificity = cls.specificity + specific

        cls.content = content or Content.UNDEF

        if not patterns:
            return  # ----Abstract format - nothing to register

        if cls.is_abstract:    # read has not been implemented
            raise NotImplementedError(f"{cls} with defined patterns must implement read method")

        cls._patterns = [cls._parse_pattern(pat, case) for pat in as_list(patterns)]
        cls.regex = re.compile('|'.join(  # Consider: is this grouping needed in regex?
            f"({_.regex.pattern})" for _ in cls._patterns))

        if not hasattr(cls, 'content'):
            raise AttributeError(f"Content type not defined for {cls}")
        cls._formats.append(cls)  # add to the common FileFormat registry
        _log.debug('(+) %s', cls)

    @classmethod
    def score(cls, name_ext: str):
        """Return score of the matched regex pattern for this name or None if no match"""
        for info in cls._patterns:
            if info.regex.fullmatch(name_ext):
                return info.score

    @classmethod
    @property
    def patterns(cls):
        return [_.pattern for _ in cls._patterns]

    @classmethod
    @property
    def suffixes(cls):
        return [_.sfx for _ in cls._patterns]

    @classmethod
    @property
    def match_pattern(cls) -> str:
        """Regex pattern matching *only the ending* part of the the path.

        Suppose format is ``*.txt``, then resulting regexp should match any
        number of parent folders containing such file, like: ``/some/path/name.txt``
        """
        return cls.regex.pattern

    @classmethod
    def supports_meta(cls, meta=None):
        return bool(CT.META & cls.content
                    ) and (
                meta is None or cls._check_support(meta, cls.meta_sup))

    @classmethod
    def supports_data(cls, data):
        """Return True if data is None or specific supported object"""
        return data is None or cls._check_support(data, cls.data_sup)

    @classmethod
    @property
    def is_abstract(cls):
        return cls.read.__func__ is FileFormat.read.__func__

    @classmethod
    def supports_write(cls, *, data=None, meta=None):
        """
        Reports on support for writing for this format,
        and, optionally, for the given data and metadata.

        Default implementation only checks if ``write`` is implemented.
        :param data:
        :param meta:
        :return:
        """
        return cls.write.__func__ is not FileFormat.write.__func__ \
            and (data is None or cls._check_support(data, cls.data_sup)) \
            and (meta is None or cls.supports_meta(meta))

    @classmethod
    def read(cls, filename: PathT, *, content: Content, **kws) -> Any | tuple[Any, Any]:
        """Read from file data and (optionally) meta-data.

        :param filename:
        :param content: which part of the content to return, a combination of Content flags
        :param kws: additional format specific keyword arguments
        :return data | meta | tuple(data, meta) if meta is requested
        """
        raise NotImplementedError

    @classmethod
    def write(cls, filename: PathT, data, *, meta=None, **kws):
        """Write given (non None) content of data or meta into the file

        :param filename:
        :param data: format specific data
        :param meta: format specific meta-data
        :param kws: additional format specific keyword arguments
        """
        raise NotImplementedError

    @classmethod
    def forget_formats(cls):    # Consider: specify which formats to forget
        cls._formats.clear()


Keys = Optional[Union[Sequence[str], str]]
Pattern = Union[str, re.Pattern]
PatternsForms = Union[Pattern, tuple[Pattern], list[Union[Pattern, tuple[Pattern]]]]

FormatHandler = Union[MetaFormat, FileFormat]
MetaReaderT = Union[str, Callable[[PathT, str, ...], dict], Type[FileFormat], Type[BaseModel]]

read_outputs = name_outputs(['data', 'meta'], adjust=None)


def is_filename(s: str, sfx=r'\w+'):
    """Return true if a string contains path separator or specific file extension.

    Example:

    >>> is_filename('some_name')
    False
    >>> is_filename('some_file.extension')
    True
    >>> is_filename('some_file.extension', 'png|jpg')


    :param s: string to test
    :param sfx: string containing '|' separated extensions
    """
    pat = rf'[\\/:]|(\.{sfx})'
    return bool(re.search(pat, s, re.I))


def subs_vars(mapping: dict[str, str | Callable]):
    """Create validator replacing variables in str fields according to provided mapping.

    If mapping is to a Callable it is called and cast into str during the substitution.
    """

    def substitute(v):
        for src, trg in mapping.items():
            if isinstance(trg, Callable):
                trg = str(trg())
            v = v.replace(src, trg)
        return v

    return substitute


def valid_path(v):
    if isinstance(v, str):
        from toolz import comp
        v = comp(*DirectoryPath.__get_validators__())(Path(v))
    return v


class MetaSrc(BaseModel):
    pattern: str
    reader: Any = Desc("Callable or Format/Model class or their name")
    require: Strings = Desc("Required fields in the meta-data")
    accept: Strings = Desc("Fields allowed if found in the meta-data")
    _rex: re.Pattern = PrivateAttr()  # final compiled regex to match paths

    @validator('reader', always=True)
    def init_reader(cls, reader, values):
        """
        Initialize reader from different supported definitions:
         1. ``Callable`` -> ``Callable``
         2. Format -> ``Format.read`` or ``Format.read_meta``, depending on ``meta`` field
         3. Model -> Model.parse_file
         4. pattern -> Format -> (2)
         5. Format name -> Format -> (2)

        Cases except of (1, 3) may need to select between ``read`` and ``read_meta``.

        This decision could be derived from ``Format``'s support for meta,
        unless explicitly defined through ``meta`` field, in this case
        raise if requested support is missing.

        :param reader: One of the supported description of the reader
        :param values:  the rest of the fields
        :return: Callable(filename) -> dict
        """
        meta = values.get('meta', None)

        def read_method(obj):
            """Select ``read_meta`` or ``read`` method of the ``obj`` depending
            on its declared support for meta-data and ``meta`` field setting.

            Prefer ``read_meta`` in case of ambiguity
            """
            support = getattr(obj, 'supports_meta', None)
            support = support and support()

            _meta = support if meta is None else meta
            if _meta is None:
                _meta = True  # case of io.format.read_meta
            elif _meta is True and support is False:
                raise NotImplementedError(f"Format {obj} does not support meta data")

            if _meta:
                return getattr(obj, 'read_meta', None) or getattr(obj, 'read', None)

        if reader is None:  # guess from pattern
            if handler := FileFormat.find_handler(values.get('pattern')):
                reader = read_method(handler or FileFormat)
        elif isinstance(reader, str):
            if not (fmt_cls := FileFormat.find_class(name := reader)):
                raise NameError(f"Can't find Format '{name}'")
            reader = read_method(fmt_cls)
        elif isinstance(reader, type):
            if issubclass(reader, FileFormat):
                reader = read_method(reader)
            elif issubclass(reader, BaseModel):
                assert meta is None, "Using Model reader incompatible with meta"
                reader = reader.parse_file
            else:
                raise TypeError(f"Invalid reader {type(reader)}")
        elif isinstance(reader, Callable):
            assert meta is None, "Using reader function incompatible with meta"
        else:
            raise TypeError(f"Unsupported {type(reader)=}")
        return reader

    def __init__(self, cmd=None, *, glob: bool = False,
                 pattern: str = None, reader: MetaReaderT = None,
                 require: Strings = None, accept: Strings = None):
        """
        Initialization can be done either by a *shortcut command* or by providing
        fields values, in particular the mandatory ``pattern``.

        A single value initialization provides useful shortcuts:
        >>> MetaSrc(True)
        >>> MetaSrc("calib.txt") == MetaSrc(pattern="calib.txt", glob=True)
        >>> MetaSrc("*.yml") == MetaSrc(pattern=".*\\.yml", glob=False)
        >>> MetaSrc("MiddleburyCalib")
        >>> MetaSrc(MetaSrc)

        Optinal flag ``glob`` indicateds if patter is of ``glob`` or ``regex`` type.
        If not provided it is guessed from the ``pattern``, which is not always straightforward.

        :param cmd: a high level command to be auto-expanded into fields:
        """
        if cmd is not None:
            if any((glob, pattern, reader)): raise ValueError("Either val or attributes")
            if cmd is True:
                pattern = '.*'
                reader = None
            # what if cmd is False? If that scenario is not supported - why bool ?
            elif isinstance(cmd, type(self)):  # copy constructor
                fields_dict = {}
                for at in cmd.__fields__:
                    fields_dict[at] = getattr(cmd, at)
                super().__init__(**fields_dict)
                self._rex = re.compile(self.pattern)
                return
            elif isinstance(cmd, str) and (is_regex(cmd) or is_filename(cmd)):
                pattern = cmd
                reader = None
            else:
                pattern = '.*'
                reader = cmd

        if glob is None:  # try to determine if pattern is regex or glob
            glob = re.search(r'(?<!\\)\.(?![*{+?])', pattern) or pattern.startswith('*')
            if glob and re.search(r'[({\<]', pattern):
                raise ValueError(f"{pattern=} does not seem neither glob nor regex")
        if glob: pattern = fnmatch.translate(pattern)

        super().__init__(pattern=pattern, reader=reader, require=require, accept=accept)
        self._rex = re.compile(self.pattern)

    def read(self, filename):
        data = self.reader(filename)
        if isinstance(data, BaseModel):
            data = data.dict(exclude_unset=True)
        if not isinstance(data, dict):
            raise TypeError("Data type returned by the reader must be dict!")
        if missing := self.require and self.require.difference(data):
            raise LookupError(f"Missing meta data in file {filename}: {missing}")
        if self.accept:
            for key in set(data) - self.accept:
                del data[key]
        return data


class MetaForms:

    def __init__(self, *sources: Callable | MetaSrc | str,
                 fail: bool | Type[Exception] = True, **attrs):
        """
        Define meta-data reading from files, including the reader
        and optional labels to require or accept in the read meta-data.

        Examples:
        ========
        >>> MetaForms() == MetaForms(False)    # no meta-data processing
        >>> MetaForms(True) == MetaForms(pattern='.*', reader=True)
        >>> MetaForms('calib.txt', 'config.yml')  # read only from files names so
        >>> MetaForms(pattern='some/**/file.*', glob=True, require=['focal', 'view'])
        >>> MetaForms(
        ...     MetaSrc(pattern='some/**/file.*', glob=True, require=['focal', 'view']),
        ...     MetaSrc('.*/calib\.txt', glob=False, reader='MiddleburyCalib'),
        ...     'config.yml')
        ... )


        :param sources: `MetaSrc`es or string a MetaSrc can be created from
        :param fail: True|False to (en|disp)able raising exceptions on read errors.
                     Or a subclass of ``Exception`` to raise specif exception type.
        :param attrs: kws arguments (see specifically below) to MetsSrc constructor
                      (not together with ``sources``!)

        :param pattern: regular expression describing file name template
        :param reader: Function returning a dict, or
                       FileFormat returning dict or pydantic Model, or
                       Pydantic Model
        :param require: list of labels raising LookupError if not found in the data
        :param accept:  list of labels to accept from the data
        :param kws:
        """
        if attrs:
            if sources: raise ValueError("Provide either attrs or sources")
            sources = [MetaSrc(**attrs)]
        elif len(sources) == 1 and sources[0] in (False, None):
            sources = []
        self.sources = list(map(MetaSrc, sources))  # this calls copy constructor if sources has MetaSrc
        self.fail = fail

    def __bool__(self):
        return bool(self.sources)

    def __call__(self, path: str):
        compatible = [src for src in self.sources if src._rex.fullmatch(path)]

        if len(compatible) == 1:  # one found
            return compatible[0].read(path)
        elif len(compatible) == 0:  # nothing found
            compatible = self.sources

        for src in compatible:
            try:
                content = src.read(path)
                return content
            except Exception as e:  # try to mitigate exception type
                continue

        if self.fail:
            if issubclass(self.fail, Exception):
                raise self.fail(f'Given path - {path} is not supported by {self.sources}')
            raise TypeError(f'Given path - {path} is not supported by {self.sources}')
