from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Literal, Sequence

import regex as re

from toolbox.utils.regexp import format_to_regex, regex_to_format, is_regex, partition


def ng(name=None, exp=None, /, **kw):
    """Construct named group regex using argument key and value
    ::
        ng('tag', '.*') == ng(tag='.*') == (?P<tag>.*)
    """
    if exp is None: name, exp = next(iter(kw.items()))
    return f'(?P<{name}>{exp})'


class AnoGroup:
    """
    **Groups of Unknown Labels**

    Anonymous groups allow definitions of more general patterns
    requred when some labels are not known in advance.

    Pattern marks locations of groups of such anonymous labels,
    identified by the group's tag:
    ::
        pattern = r"{dataset}/{*:inp}/{kind}{_*:out}?_{alg}{_*}?.tif"

    | In this example there are 3 groups of anonymous labels:
    |    - two tagged: `inp, out`, and
    |    - one untagged (only one such is allowed).

    |
    | **Definition in Patterns**

    The anonymous groups are defined according to the formatter:
    ::
        {<pfx>*<sfx>:<tag>}?, where each of

    ``<pfx>, <sfx>`` - represents an optional separator symbol from ``{_.- }``.

    Optional ``?`` symbol at the end allows the group to be empty.

    |
    | **Path Builder Arguments**

    Arguments to the ``build_path`` defining the anonymous grous are
    distinguishable by their ``dict``type (pseudocode):
    ::
        path = pather(notag,
            **labels: dict[key, value],     # regular labels
            **anonymous_groups: dict[name_fmt, dict[key, value]]
        )

    Example:
    ::
        path_build = PathBuilder(
            pattern=r"{dataset}/{*:inp}/{kind}{*:out}?_{alg}{_*}?.tif",
            mapper=lambda _: _, root='/tmp')
        path = path_build(
            {'norm': 'batch'},  # untagged anonymous group
            dataset='Driving',  # a regular label
            inp=dict(           # anonymous group tagged "inp"
                dir='back',     # an anonymous label
                speed='fast',   # an anonymous label
                focal=35),      # an anonymous label
            kind='disp',        # a regular label
            alg='RAFT',         # a regular label
            out={'view': 'L'}   # anonymous group tagged "out"
        )

    ** Parsing Paths **

    The *same* pattern can parse paths with different anonymous labels:
    For pattern - {dataset}/{*:inp}/{kind}{*:out}?_{alg}{_*}?.tif
    ::
        1) MIDS/Piano/disp[view=L]_SENSE.tif
           # extracted labels:
                        dataset: MIDS
                        scene: Piano
                        kind: disp
                        view: L
                        alg: SENSE

        2) Driving/[dir=back,speed=fast,focal=35]/disp[view=L]_RAFT[norm=batch].tif
           # extracted labels:
                        dataset: Driving
                        dir: back
                        speed: fast
                        focal: 35
                        kind: disp
                        view: L
                        alg: RAFT
                        norm: batch
    """
    # Some naming conventions:
    #   f_*     formatter string, ready for f_*.format(...)
    #   *_rex   some regex pattern to parse paths
    #   *_fmt   some formatter template to build path
    #   *_pat   regex to parse mixed pattern definition format
    #   CAPITAL names of reserved string literals used in regex

    NO_TAG = 'TAG'  # default name for unnamed tag
    SEP = ','  # mandatory separator between key-value pairs

    # FX = r'[_\-\.\# ]'  # suffix or prefix symbols # Consider: do we need more than that?
    FX = r'[^/\\?*{}:]+'  # suffix or prefix symbols - everything except complex constructions
    KEY = '_KEY'  # name of key regex group (in the PAIR)
    VAL = '_VAL'  # name of val regex group (in the PAIR)
    NAME = '_ANG'  # name of group with regex for anonymous group
    INTERNAL = {KEY, VAL, NAME}  # Internal utility groups  - not capturing labels

    # ----------------  PAIR  -------------------
    f_pair = f'{{}}={{}}'  # format of key-value pair for both form AND regex
    pair_rex = f_pair.format(  #
        ng(KEY, r'[a-zA-Z]\w*?'),
        ng(VAL, r'[^:=\[\]\(\)\{\}\*\?]+')
    )

    # ----------------  GROUP  ------------------
    f_group_rex = ng(NAME, rf'\[{{pair}}(?:{SEP}{{pair}})*\]')

    # PREAMBLE contains definition of anonymous group to be referenced by large
    # regular expressions instead of inserting them multiple times verbatim.
    # It consists of definition of PAIR, followed by ANG which refers to PAIR
    preamble_rex = '({pair}){{0}}{ang}{{0}}'.format(
        pair=pair_rex,  # 1-st capturing group defines pair, referred as () by the ang definition
        ang=f_group_rex.format(pair="(?1)")  # use reference to the pair definition (?1)
    )

    gid = re.compile(preamble_rex).groupindex[NAME]  # or we could use 'P>{NAME}' instead
    f_ang_rex = rf'({{pfx}}(?{gid}){{sfx}}){{opt}}'  # full ang search complex

    # named-group parser extracting AnoGroup from the input pattern
    # template `{pfx*sfx}?` where each of (pfx, sfx, ?) is optional:
    group_pat = re.compile(r"\{{{pfx}?\*{sfx}?(?::{tag})?\}}{opt}?".format(
        pfx=ng(pfx=FX),  # supported prefix and postfix symbols
        sfx=ng(sfx=FX),
        tag=ng(tag=r'([a-zA-Z]\w*?)'),
        opt=ng(opt=r'\?')
    ))

    @property
    def group_fmt(self):
        s = f'{self.pfx}{{{self.tag}}}{self.sfx}'
        return f"({s})?" if self.opt else s

    @classmethod
    def format(cls, labels: dict[str, Any]):
        """Produce string representation labels in the given anonymous group
        :param labels: dict of the labels in the group
        :return: formatted labels or empty string if dict is empty
        """
        if labels:
            pairs = cls.SEP.join(cls.f_pair.format(*kv) for kv in labels.items())
            return f'[{pairs}]'
        return ''

    @classmethod
    def pattern_to_format(cls, pattern: str):
        """
        Substitute anonymous patterns in given regex string by their format representation.
        ::
            →  (<pfx>{<tag>_ANG}<sfx>)?
            →   <pfx>{<tag>_ANG}<sfx>
        """
        return cls.group_pat.sub(lambda m: cls(**m.groupdict()).group_fmt, pattern)

    def __init__(self, tag, sfx, pfx, opt):
        self.tag = tag or self.NO_TAG
        self.sfx = sfx or ''
        self.pfx = pfx or ''
        self.opt = opt or ''
        self.rex = self.f_ang_rex.format(**self.__dict__)

    def __str__(self):
        return f"{{{self.pfx}{self.tag}{self.sfx}}}{self.opt}"

    def __repr__(self):
        return f"ANG<{self}>"


class Regex:
    """Regex representation of path pattern.

    Constructed incrementally by appending new anonymous groups with intermediate segments.

    Provides access to all the labels ``categories`` to be extracted from paths.
    """

    def __init__(self, pattern, flags: int = 0):
        self.str = ''
        self.flags = flags
        self.regex: re.Regex = None
        self.categories = None
        self._pyregex = None
        self.anonym_groups = {}

        # regex from the parts each containing some pattern and anonym group:
        # format_to_regex(pattern) + anonym.rex
        for before, match in partition(AnoGroup.group_pat, pattern):
            try:
                self.str += format_to_regex(before, dots=False)
            except Exception as ex:
                raise SyntaxError(f"regex_to_format({before}):\n{ex}")

            if match:  # could be None in the last part
                ang = AnoGroup(**match.groupdict())
                if ang.tag in self.anonym_groups:
                    raise SyntaxError(f'Duplicated anonymous group tag "{ang.tag}"')

                self.str += ang.rex  # g.group for named group
                self.anonym_groups[ang.tag] = ang

        if self.anonym_groups:  # Add preamble (definitions) only if ANG is found
            self.str = AnoGroup.preamble_rex + self.str

        self.compile()

    ANONYM_OUT = Literal['merge', 'nested', 'separate', 'drop']
    MATCH_METHOD = Literal['match', 'search', 'fullmatch']

    def _ang_parse(self, s, *, method: MATCH_METHOD = 'fullmatch', anonym: ANONYM_OUT = 'merge'):
        """Parse string into labels using given matching method"""
        if m := getattr(self.regex, method)(s):
            cap: dict = m.capturesdict()
            labels = dict(zip(cap[AnoGroup.KEY], cap[AnoGroup.VAL]))
            for key, val in m.groupdict().items():
                if key not in AnoGroup.INTERNAL:
                    labels[key] = val
            return labels

    def _basic_parse(self, s, method: MATCH_METHOD = 'fullmatch'):
        """Parse string into labels using given matching method"""
        if m := getattr(self._pyregex, method)(s):
            return m.groupdict()

    def parse(self, s, *, method: MATCH_METHOD = 'fullmatch', anonym: ANONYM_OUT = 'merge') -> dict:
        """Parse string into labels using given matching method"""
        pass  # To be set in compile

    def __copy__(self):
        # workaround compiled regex does not copy well
        new = Regex()
        new.str = self.str
        if self.regex:
            new.compile()
        return new

    def compile(self, flags=None):
        if flags is None:
            flags = self.flags
            if flags is None:
                flags = re.X if '\n' in self.str else 0

        self.regex = re.compile(self.str, flags=flags)
        self.categories = tuple(filter(lambda k: k not in AnoGroup.INTERNAL, self.regex.groupindex))
        self.parse = self._ang_parse

        if not self.anonym_groups:  # python stdlib re can apparently be twice as fast
            import re as pyre  # so we use it when complex regex are not required
            self._pyregex = pyre.compile(self.str, flags=flags)
            self.parse = self._basic_parse

    def __repr__(self):
        comp = 'C' if self.regex and self.regex.pattern == self.str else 'X'
        return f"[{comp}]{self.str}"


class Form:
    """Forms path string from labels.
    Helper class for `TransPath`, not intended to be used outside its context.

    Properly initialized instance functions builds path string from labels using `__call__` method.
    """

    opt_grp_pat = re.compile(r"""
        (?P<_OG>           # recursive definition of optional group (<opt>)?
            \(
            (?P<opt>                    # (named part includes only group's content)
                [^()]*?                 # with arbitrary content with
                (?:                     # at least one
                   (?:                  # named or nested optional subgroup
                     \{(?P<name>\w+)\}      # {name}
                     | \(.+?\)\?            # (?&_OG)              # or nested (...{name}...)?
                   )
                   [^()]*?              # separated by arbitrary symbols
                )+
            )
            \)\?                        # ()? symbold are outside of 'opt' group!
        )
    """, re.X)

    def __init__(self, pattern: str, *, verbose=False, _optional: Sequence | None = None):
        """
        Initialize formatter -
           1. from iterable over (ano_grp, segment) tuples (from `TransPath`)
           2. from a string (recursively from `Form` itself)

        `_nest` MUST be `None` (default) if called outside the `Form`.

        In recursive call provide *set* of optional groups names found in the
        provided `pattern` as discovered by the upper levels.

        :param pattern:
        :param verbose:
        :param _optional: must be `None` unless called by an internal class member
        """
        assert verbose in (None, True, False)
        if _optional is None:
            pattern = regex_to_format(pattern, verbose=verbose)
            pattern = AnoGroup.pattern_to_format(pattern)
        self.str = pattern
        self._optional = set(_optional or ())
        # here self.str is a valid format except of (...)?
        success_parse = self._parse_optional_groups(pattern)
        self.valid = success_parse and self._valid_form()

    def _iter_parts_tree(self):
        """Yield all the parts recursively"""
        for p in self._parts:
            if isinstance(p, Form):
                yield from p._iter_parts_tree()
            else:
                yield p

    def _repr_parts_tree(self, indent=' ' * 4, head='Form:\n'):
        """String representing parsed tree of parts"""
        from toolbox.utils.strings import indent_lines

        return head + '\n'.join(indent_lines(
            p._repr_parts_tree(indent=indent, head=head) if isinstance(p, Form) else p
        ) for p in self._parts)

    def _valid_form(self):
        """A primitive validation, may be improved"""
        s = ''.join(self._iter_parts_tree())
        return not is_regex(s)

    def __repr__(self):
        return f"{self.__class__.__name__}{'🗹' if self.valid else '☒'}<{self.str}>"

    def _parse_optional_groups(self, pattern) -> bool:
        """
        Parse `self.str` pattern into a nested list `self._parts` composed of 2 types of elements:
            - just a sub-string of the original `self.str`
            - a `Form` instance built around a found optional group with named subgroups

        For example, a string with structure like that:
        ::
            sub_1(...{name_1}...{name_2}...)?(...{name3}...(...{name_4})?...)?sub_4
        is parsed into 3-level hierarchy of *Forms*:
        ::
            Form[sub_1,
                 Form([...{name_1}...{name_2}...]),
                 Form([
                        ...{name3}...,
                        Form([...{name_4}]),
                        ...]),
                 sub_4]

        Notice, that (...)? *without* named subgroups are not extracted and treated as regular text.

        :return: False if any created sub Forms is invalid
        """
        self._parts = []
        for before, match in partition(self.opt_grp_pat, pattern):
            if before:  # also as the last part (when match is None)
                self._parts.append(before)

            if match:  # everywhere except the last part after the last match
                part = Form(match.group('opt'), _optional=match.captures('name'))
                self._parts.append(part)
                if not part.valid:
                    return False  # stop parsing if its invalid

        return True

    def __call__(self, labels):
        if not self.valid:
            return None
        labels = {k: AnoGroup.format(v) if isinstance(v, dict) else v
                  for k, v in labels.items()}
        format_str = self._make_format_str(labels)
        try:
            return format_str.format(**labels)
        except KeyError as ex:
            raise KeyError(f"Missing label {ex.args[0]} for {self}")

    def _make_format_str(self, labels):
        """
        Produce format string representing this Forms instance.

        Built recursively by merging nested optional forms.

        Every form is
         - formatted if *all* of its optional labels are provided,
         - removed if *none* are provided
         - raise `KeyError` if only part of its labels is provided.
        """
        if missing := self._optional.difference(labels):  # happens only in subgroups!
            # Consider: treat partially missing labels as fully missing to avoid failing?
            if len(missing) < len(self._optional):
                raise KeyError(f"Partially defined labels in optional group {missing = } in {self}")
            return ''  # all optional are missing, remove the optional group altogether

        return ''.join(p._make_format_str(labels) if isinstance(p, Form) else p for p in self._parts)


class TransPath:
    """
        Path Translator and Builder.

        From the scheme's *definition pattern* creates:
          1. ``regex`` to *parse from* the path strings semantic data labels
          2. ``formatter`` to construct path string from labels (inverse problem)

        First one is accessible as an attribute,
        second is used by the ``__call__`` interface.

        **Supported Path Patterns**

        The input pattern uses mixed notations of python formatter, regex and
        some special syntax to define how labels are encoded into the path structure.

        There are two kinds of labels described by the pattern, with either defined by:
          1. explicit names and extraction algorithms
          2. location in the path string, with unknown names or their number

        Example of defining explicit labels and a corresponding path:
        ::
           pattern: found_{number}_{color}_{fruit}s?
           path:    found_25_red_apples

        Additionally, *tagged* groups of so-called *anonymous* labels
        are defined in the pattern using {...*...: tag} notations:
        ::
            pattern:  meet {name} in {place}{ *:address}{ *:when}

            string 1: meet Bill in Boston [street=Marx,num=17] [day=Monday]
            extracts: name: Bill
                      place: Boston
                      street: Marx              # address: {street: Marx, num: 17}
                      num: 17                   # address: {street: Marx, num: 17}
                      day: Monday               # when: {day: Monday}

            string 2: meet Sarah in school [hour=9,min=30]
            extracts: name: Sarah
                      place: school
                      hour: 9                   # when: {hour: 9, min: 30}
                      min: 30                   # when: {hour: 9, min: 30}
        """

    # {_*-:inp}? --> {tag: inp, sfx: _, pfx: -, opt: ?}

    def __init__(self, pattern: str, *, mapper: Callable = None, flags=None):
        """Since not every valid pattern can be translated into formatter,
        ``valid_form`` flag controls possible cases:
          - ``True``: valid format translation is required, failure raises ``SyntaxError``
          - ``False``: format translation is disabled
          - ``None``: automatic - translation is attempted , failure switches to False

        :param mapper: function mapping parsed groups
        :param flags: regex flags, use re.X for Verbose,
        `None` adds this automatically iof detects multiple line
        """
        flags = (flags or 0) | (re.VERBOSE if '\n' in pattern else 0)

        self.pattern = pattern
        self.mapper = mapper

        self.regex = Regex(self.pattern, flags)  # not matchable definition for references
        self.form = Form(self.pattern, verbose=bool(re.VERBOSE & flags))
        self.expected_tags = set(self.regex.anonym_groups).union(self.regex.categories or ())

    def __copy__(self):
        return deepcopy(self)

    def __str__(self):
        return f"{self.__class__.__name__}({self.regex})"

    def __repr__(self):
        return f"<{self.__class__.__name__}>\n\t{self.form}\n\t{self.regex}"

    def __call__(self, no_tag: dict | bool | None = None, **labels):
        """
        Callable interface to a `TransPath` object to construct
        path according to the scheme's formatter given labels.

        Supports two forms:
        ::
            trans_path(unnamed, tag1=val1, tag2=val2, ...)
        or, in case no unnamed tag is defined:
        ::
            labels = dict(tag1=val1, tag2=val2, ...)
            trans_path(labels)

        `no_tag` controls content of possible unnamed TAG in the pattern.
         - if a `dict` - used if unnamed TAG is defined or fails.
         - if `True` - uses for that all the unknown labels (or similarly fails)
         - if `False` or `None` - ignores unknown tags in the `labels`

        :param no_tag: controls content of the unnamed anonymous group
        :param labels: labels to fill the path formatter
        :return: path string
        """
        if self.form.valid is False:
            raise RuntimeError(f"{self} \n can't build path!")

        if no_tag and not labels:   # support passing labels as dict
            labels, no_tag = no_tag, None

        if no_tag and not self.has_unnamed:
            raise ValueError("Unnamed tag is not defined but requested!")
        if no_tag is True:
            from toolbox.utils.datatools import split_dict
            labels, no_tag = split_dict(labels, lambda k, v: k in self.expected_tags)
            no_tag = no_tag or None
        if no_tag:
            if not isinstance(no_tag, dict):
                raise TypeError(f"{type(no_tag) = } must be dict!")
            labels[AnoGroup.NO_TAG] = no_tag

        # remove labels not used in the formatting
        labels = {k: v for k, v in labels.items() if k in self.expected_tags}
        if self.mapper:
            labels = self.mapper(labels)

        return self.form(labels)  # ang already formatted here

    @property
    def has_unnamed(self):
        """Accepts unnamed anonymous group"""
        return AnoGroup.NO_TAG in self.regex.anonym_groups

    @property
    def ext(self):
        """Extension produced by the path builder"""
        return self.form.str.rsplit('.', 1)[-1]


def valid_path_sep(path, regex=False):
    r"""
    Replace path separators by valid in this os.
    `path` may include sep in any form (`\` or '/') - will be all replaced to os.sep
    (or its regex safe form).
    :param path: path string or regex pattern describing a path string
    :param regex: if `True` assumes that the string represents a regular expression
                       then '/' are replaced by os.sep ('\\' if os.spe == '\')
                       and '\\' by '/' if os.sep == '/'.
    :return: valid path string
    """
    from os import sep
    if regex:  # in this case in valid regex string '\' path sep appears as '\\'
        re_sep = r'\\\\' if sep == '\\' else sep  # sep usable inside regex
        pattern = r'(\\{2}|/)'  # leave safe separators
    else:
        re_sep = sep
        pattern = r'(\\|/)'
    return re.sub(pattern, re_sep, path)
