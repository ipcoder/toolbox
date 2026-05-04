from __future__ import annotations

from typing import Iterable, Generator, Union
import regex as re


def _gpt_strip_verbose(pattern):
    """
    How it works:
     1. Tracking State:
        The code uses two boolean flags:
          - in_char_class to know when it’s inside a character class (so that spaces
            or # characters there are left intact).
          - escaped to ensure that characters preceded by a backslash
            are not processed as potential comment or whitespace markers.

     2. Skipping Comments and Whitespace:
       Outside a character class, when encountering a #,
       the loop skips all characters until the next newline.

       Similarly, whitespace characters outside character classes are not added to the output.

     3. Preserving Important Characters:
       All characters that are significant to the regex
       (including escaped ones and those within character classes)
       are appended to the result list.

    **Caveats**

    May fail on some edge cases, like nested constructs

    :param pattern:
    :return:
    """

    result = []
    in_char_class = False
    escaped = False
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if escaped:
            # If previous char was a backslash, keep this character.
            result.append(char)
            escaped = False
        elif char == '\\':
            # Mark the next character as escaped.
            result.append(char)
            escaped = True
        elif char == '[':
            in_char_class = True
            result.append(char)
        elif char == ']' and in_char_class:
            in_char_class = False
            result.append(char)
        elif char == '#' and not in_char_class:
            # Skip the comment: ignore characters until the end of the line.
            while i < len(pattern) and pattern[i] != "\n":
                i += 1
            # Continue without appending the newline
            # (or add it if you need to preserve line breaks)
        elif char in (' ', '\n', '\t') and not in_char_class:
            pass  # Skip free-space when outside a character class.
        else:
            result.append(char)
        i += 1
    return ''.join(result)


def strip_verbose_regex(pattern: str | re.Regex, *, validate=True, check=False):
    """
    Given pattern or compiled Regex convert it from verbose into plain form

    :param pattern: regex pattern to convert
    :param check: check before stripping it is indeed verbose (is multiline)
    :param validate: try to validate (compile) resulting expression
    """
    if not isinstance(pattern, str):
        pattern = pattern.pattern

    if check and '\n' not in pattern:
        return pattern

    clean_pattern = _gpt_strip_verbose(pattern)

    if validate:
        import regex as _re
        try:
            _re.compile(clean_pattern)
        except re.error:
            try:
                _re.compile(pattern, _re.MULTILINE | _re.VERBOSE)
            except re.error:
                raise ValueError(f"Invalid input {pattern=}")
            else:
                raise ValueError(
                    f"Attempt to strip verbose pattern '{pattern}'\n"
                    f"produced invalid pattern '{clean_pattern}'")
    # In verbose pattern literal # and \s would be preceded by \ - remove it
    return re.sub(r'\\([#\s])', r'\g<1>', clean_pattern)


class _Patterns:
    """Helper class for compiling patterns on demand"""
    def __init__(self):
        self._patterns = {}

    def _get(self, name, s, flags):
        if not (pat := self._patterns.get(name, None)):
            pat = self._patterns[name] = re.compile(s, flags)
        return pat

    @property
    def named_rex(self):
        """Captures groups: 'p_name' """
        return self._get('named_rex', r"""
            \(\?P<
            (?P<p_name>             # named of the named group (captured as 'p_name')
            \w+)>
              (?P<REC>              # matching pattern as REC in form: b(R1)a1...(R2)a2
                [^)(]*+             # `b` (before) part - any not `)(` characters
                (?:                 # non capturing group for repeated `(R)a`
                  \((?&REC)\)       # `R` - reference to group 2 for possible nested braces
                  [^)(]*+           # 'a' (after) part  - any not `)(` characters
                )*                  # `(R)a` repetition is optional
              )                     # end the group 2
            \)                      # end of the named group to match
            """, flags=re.FULLCASE | re.VERBOSE)

    @property
    def named_fmt(self):
        """Captures groups: `f_name`, `spec`, `opt`"""
        return self._get('named_fmt', r"""
            {(?P<f_name>[a-zA-Z]\w*)  # Group captured as 'f_name'
             (?:
               :(?P<pat>           # Group : 'pat' - anything between `:` and `}`
                  (?:               # explicitly describe repetitions \d{1,2}\w{3}
                    [^{}]+          # to allow correct parsing of its nested {}
                    {\d*(?:\,\d*)?}
                  )*
                  [^{}]*
                )?                  # The 'pattern' part may be empty
            )?}(?P<opt>\?)?         # {}?  - question mark indicating 'optional'
            """, flags=re.FULLCASE | re.VERBOSE)

    @property
    def named_rex_or_fmt(self):
        """Captures groups: `p_name` | (`f_name`, `spec`) """
        rex, fmt = self.named_rex, self.named_fmt
        return self._get('named_rex_or_fmt',
                         f"{rex.pattern}|{fmt.pattern}",
                         flags=rex.flags | fmt.flags)


_patterns = _Patterns()  # use to access patterns


def format_to_regex(pattern: str, *, pat=r'\w+?', dots=None, fspec=False):
    r"""Convert format-like string into its regular expression form by replacing
    named groups  `{name:...}` into `(?P<name>{pattern})`.

    Default `{pattern}` is provided by the `pat` argument.

    The default `pat` is `\\w+?` (`?` - not greedy mainly to avoid interference
    with `_` if used as a separator before the next group).

    It also may be specified for every named group if `fspec=False`.

    In this case, the part after `:` is interpreted as regex pattern:
        "frame_{fid:00\\d{4}}" -> "frame_(?P<fid>00\\d{4})"
    Otherwise, when `fspec=True`, format specifier is expected and the default
    `pat` is used for the named group regex (if `pat='\\w+?'):
         "very_{big:08.4g}/{fruit}" - > "very_(?P<big>\\w+?)/(?P<fruit>\\w+?)


    :param pattern: format string with items in curled braces:
                no regular expression symbols are allowed, except of '.'
    :param pat: re sub-pattern to use for groups, default is r'\w+?'.
    :param fspec: interpret part after ':' in {name:fspec} as format specifier
    :param dots: True if literal and require conversion ('.' -> '\.'), ``None`` - auto
    :returns: a regular expression string

    """
    parts = list(partition(_patterns.named_rex_or_fmt, pattern))
    dots = dots or (dots is None and not (
        is_regex(''.join(_[0] for _ in parts))
    ))  # decide if its regex only based on non-group parts

    s = ''
    for before, match in parts:
        if before:
            s += before.replace('.', r'\.') if dots else before
        if match:
            groups = match.groupdict()
            if name := groups['f_name']:    # {name: pat}
                _pat = groups['pat']
                opt = groups['opt'] or ''   # optional '?'
                s += fr'(?P<{name}>{pat if fspec or not _pat else _pat}){opt}'
            else:                           # (?P<name>...)?
                s += match.group()   # just copy the regular expression as is
    return s


def regex_to_format(pattern, verbose: bool | None = None):
    r""" Convert regular expression named group parsing pattern (?P<key>...)
     into an equivalent str.format substitutable string: {key}
    Example::

        regex_to_format(r'/some/fold_(?\P<id>)/(?P<name>\.tif') \
        == '/some/fold_{id}/{name>}.tif'

    :param pattern: regex with (?P<key>...) elements
    :param verbose: consider pattern verbose, `None` - try guess from the pattern
    :return: string with format {} elements.
    """
    if verbose is not False:
        assert verbose in (None, True)
        pattern = strip_verbose_regex(pattern, validate=False, check=verbose is None)
    # don't validate - could be mixed format-regex
    fmt = _patterns.named_rex.sub(r'{\g<p_name>}', pattern)
    fmt = fmt.replace(r'\\', '\\').replace(r'\.', '.').replace('(?:', '(')

    # Remove from string groupings ...(...)... or ...(?:...)...
    # with no additional effects like ()? or ()+.
    fmt = re.sub(r"(?P<pair>\((?:[^\(\)]+|(?&pair))*\))(?![?+*]|\{\d})",
                 lambda _: _.group()[1:-1], fmt)

    def optional_named(m):
        grp = f"{{{m.group('f_name')}}}"
        return f"({grp})?" if m.group('opt') else grp

    fmt = _patterns.named_fmt.sub(optional_named, fmt)
    return fmt


def partition(rex, string: str, *, flags=None,
              ) -> Iterable[tuple[str, re.Match | None]]:
    """
    Given string partition it into sections containing one substring matching
    given regex.

        [prefix]<match>|[prefix]<match>|[prefix]None

    Yield every segment as tuple: `(substring preceding the match, match object)`

    The last tuple *may* contain `None` as second element,
    if the string does not end with a match.

    Allows to provide compiled regular expression to control which regex package is used.


    :param rex: regex as str or compiled
    :param string: string to partition
    :param flags: provide only if `rex` is `str`!
    :return: generator
    """
    if isinstance(rex, str):
        rex = re.compile(rex, flags=flags or 0)
    elif not (flags is None or int(flags) == rex.flags):
        raise ValueError(f"Argument {flags=} conflicts with {rex.flags=}")

    last = 0
    for match in rex.finditer(string):  # for every found optional group
        start, stop = match.span()
        yield string[last:start], match
        last = stop
    if last < len(string):
        yield string[last:], None


def is_regex(s: str, compile_check=False) -> bool:
    """Check if string is a regular expression.
    Looks for special symbols in the string.

    If found, as an additional verification, the compilation can be requested.
    In this case, if failed - return False, even if first test is passed.

    :param s: string to check
    :param compile_check: try to compile to confirm

    :return: ``True`` if all the tests confirmed, otherwise ``False``
    """
    if re.search(r'[$^?\+\*]|(\.[\?\+\*])|({\d+(,\d*)?})', s) is None:
        return False
    if compile_check:
        try:
            re.compile(s)
        except:
            return False
    return True


def regex_parse(string: str, pattern, *, method='any') -> dict:
    """
    Extract tags from string according to the NAMED groups in the regexp pattern.
    :param string:  file name
    :param pattern: regular expression with NAMED groups to extract
    :param method: which part of string to match:
            - end - string ends with the pattern
            - full - full match of the string to the pattern
            - start - start of the string
            - any - find anywhere in the string

    Notice, that '$' is added to the pattern to process 'end' case.
        You may prefer to do that by yourself (and select 'any') to speed it up.

        For file names patterns ambiguity isn't probable and 'any' is enough.
    """
    if method == 'end':  # there is no such re function, so we alter pattern
        current = pattern if isinstance(pattern, str) else pattern.pattern
        if current[-1] != '$' or current[-2] == '\\':  # or '...\$' -> '\$$'
            pattern = current + '$'

    match = _rp_method[method](pattern, string)
    return match.groupdict() if match else {}


_REX = Union[str, re.Pattern]


def filter_regex_matches(regex: _REX | Iterable[_REX], strings: str | Iterable[str], flags=0
                         ) -> Generator[str]:
    """
    Filter given iterable of strings for those matching at least one of provided regular expressions

    :param regex: one or Iterable of regular expressions in object or string form
    :param strings: one or Iterable
    :param flags:
    :return:
    """
    from .short import as_list

    regex = as_list(regex)
    strings = as_list(strings)

    for s in strings:
        for rex in regex:
            if re.fullmatch(rex, s, flags=flags):
                yield s
                break  # at least one match found


_rp_method = dict(
    full=re.fullmatch,
    any=re.search,
    end=re.search,
    start=re.match
)
