"""Strings manipulation utils"""
from __future__ import annotations

import re
from typing import Callable, Sequence, Literal


def fuzzy_find(query: str, strings: Sequence[str], *, case=True,
               limit: int | None = 1, score_cutoff=None,
               out: Literal['string', 'index', 'tuple'] = 'all'):
    """
    Find imperfectly matches to the `query` string in the given sequence of `strings`.

    If `out_str` is `False` for every found match return tuple of
     - `string` from the `strings` matching the `query`
     - `score` of the match [0, 100]
     - `index` of the found sting in the sequence

    Return list of found candidates as tuple `(string, score, index)` or
    element specified by the `out`: 'string' or 'index'.

    :param query: a string to find
    :param strings: sequence of string to scan
    :param case: if `True` - case-sensitive
    :param limit: max number of matched
    :param score_cutoff: drop found with score < score_score_cutoff
    :param out: form of the output items

    :return: list of found strings or tuple
    """
    from rapidfuzz.process import extract
    processor = None
    if not case:
        from rapidfuzz.utils import default_process as processor

    found = extract(query, strings, processor=processor, limit=limit,
                    score_cutoff=score_cutoff)
    if out == 'tuple':
        return found
    i = dict(string=0, index=2)[out]
    return [_[i] for _ in found]


def short_form(s: str, head=0, tail=0):
    """
    Cut head , center or tail of the string if its len exceeds given sizes.
    Replace cut part with '…'.

    If both tail and head are provided, cut from the center

    :param s: string to crop
    :param head: len of head part to leave
    :param tail: len of tail part to leave

    :return: cut str
    """
    n = len(s)
    if n <= head + tail:
        return s
    if head and tail:
        return f"{s[:head]}…{s[-tail:]}"
    if tail:
        return f'…{s[-tail:]}'
    if head:
        return f'{s[:head]}…'
    return s


def smart_warp(s, width=80, **splitters):
    """
    Warp string py applying `splitters` in order to limit each line by requested `width`.

    Every splitter must be a regular expression to use by `re.split(splitter, line, maxsplit=1)`
    If splitters not provided, defaults are used:
    ::

        splitters = dict(
            bullets = r"(?<=.{2})(?=\s+[-*]\s+)",
            dot_capital = r"(?<=[\.\:])\s+(?=[A-Z])",
            long_dot = r"(?<=.{15}\w{2}\.)\s+(?=\w{2})",
            long_capital = r"(?<=.{25})\s+(?=[A-Z]\w{2})",
            space = r"(?<=.{40})\s+(?=.{20})"
        )
    """
    transforms = splitters or dict(
        bullets=r"(?<=.{2})(?=\s+[-*]\s+)",
        dot_capital=r"(?<=[\.\:])\s+(?=[A-Z])",
        long_dot=r"(?<=.{15}\w{2}\.)\s+(?=\w{2})",
        long_capital=r"(?<=.{25})\s+(?=[A-Z]\w{2})",
        space=r"(?<=.{40})\s+(?=.{20})"
    )

    def apply(lines, tfm):
        # print(f'------------[{tfm}]')
        for line in lines:
            remained = [line]
            while remained:  # mau be of size 0 or 1
                if len(line := remained.pop(0)) > width:
                    # print(f"[{len(line):4}] {line}")
                    line, *remained = re.split(transforms[tfm], line, maxsplit=1)
                # print(' '*6, line)
                yield line

    # ---------------------------
    lines_iter = [s]
    for t in transforms:
        lines_iter = apply(lines_iter, t)

    return '\n'.join(lines_iter)


def join_by_groups(ss, sep, mx, mx2=None):
    """
    From sequence of strings produce a sequence with some of
    subsequent string joined with given separator not to exceed max len.
    :param ss: iterable over strings
    :param sep: separator string to when joing sub-groups
    :param mx:   maximal length of a joined string
    :param mx2:  if defined try keep lengths between mx and mx2
    :return: iterator over joined strings
    """
    mx2 = mx2 or mx
    assert mx2 >= mx
    sep_len = len(sep)
    acc = []
    cur = 0

    next_len = lambda: cur + len(s) + (sep_len if acc else 0)

    for s in ss:
        nxt = next_len()
        if (cur >= mx or nxt > mx2) and acc:
            yield sep.join(acc)
            acc = []
            cur = 0
        cur = next_len()
        acc.append(s)

    if acc:
        yield sep.join(acc)


def wrap_sep_split(s, mx, *, mx2=None, sep=', ', jsep=None, newline='\n'):
    """
    Wrap long line by splitting into multiple lines along the separators
    and limited in length by the mx, mx2 (see  `join_by_groups`)
    :param s: the string
    :param mx: maximal length
    :param mx2: hard maximal length
    :param sep: separator regexp
    :param jsep: join separator (same as sep if not defined)
    :param newline: replaces the separator in split locations
    :return: wrapped string
    """
    jsep = jsep or sep
    return newline.join(join_by_groups(filter(len, re.split(sep, s)),
                                       sep=jsep, mx=mx, mx2=mx2))


def join_wrap(seq, max_line=80, *, head='', sep=',', spc=' ', newline='\n', left='') -> str:
    """ Join sequence of strings by wrapping long lines and adding optional left alignment shift
        :param seq: sequence of string
        :param max_line: maximal length of the line before wrapping
        :param head: head of the string to form - will not be separated by the `sep`
        :param sep: separator between strings - used also on the line wrapping
        :param spc: addition to the separator - inside the line
        :param newline: the newline symbol(s)
        :param left: left prefix string for every line
        return:
            The concatenated string
    """
    sep_len = len(sep + spc)
    left_len = len(left)
    head_len = len(head)
    res = head
    line_len = head_len
    for s in seq:
        if len(res) == head_len:
            next_len = len(s)
            next_sep = ''
        else:
            next_len = len(s) + sep_len
            next_sep = sep

        if line_len + next_len > max_line or (next_len + left_len) >= max_line:
            smart_sep = next_sep + newline + left
            line_len = next_len + left_len
        else:
            smart_sep = next_sep + spc
            line_len += next_len

        res = res + smart_sep + s if res else left + s

    return res


def compact_repr(v, max_len=60) -> str:
    """
    Compact representation of values of different types
    :param v:
    :param max_len: maximal length of the resulting string
    :return: str
    """
    from .units import Quantity
    from .nptools import array_info_str, np
    import pandas as pd

    def crop(s, mx):
        if len(s) < mx and '\n' not in s:
            return s
        ss = s.splitlines()
        sfx = ''
        if len(ss) > 1:
            s = ss[0]
            sfx = '...'
        return s.strip()[:(mx - len(sfx))] + sfx

    if isinstance(v, np.ndarray):
        return array_info_str(v, 1e4)

    if isinstance(v, pd.DataFrame):
        if len(v.columns.names) > 1:
            columns = ['.'.join(map(str, col)) for col in v.columns]
        else:
            columns = [*map(str, v.columns)]
        return f"{len(v)}[{' '.join(columns)}]"

    if isinstance(v, pd.Series):
        return f"{len(v)}[{' '.join(v.name)}]"

    if isinstance(v, (list, tuple)) and all(  # repr for list of arrays
            map(lambda x: isinstance(x, np.ndarray), v)):
        ss = ', '.join(map(lambda x: array_info_str(x, 0), v))
        ss = ("[{}]" if isinstance(v, list) else "({})").format(ss)
    elif isinstance(v, dict):
        ss = repr(v if type(v) is dict else dict(v))
    else:
        ss = repr(v)

    s = crop(ss, max_len)
    try:
        return f"[{len(v)}] {s}" if hasattr(v, '__len__') \
                                    and not isinstance(v, (Quantity, str)) else s
    except TypeError:
        return s


def dict_str(d, *, prec: int | str = None, sep=', ', to='=', bracket=None, nested=True):
    """
    Produce formatted dict str.

    >>> dct = dict(a=2, b=3.14562342345689, c='string', d={'x':49452.3345523454567});
    >>> dict_str(dct)
    'a=2, b=3.14562342345689, c=string, d={x=49452.334552345455}'
    >>> dict_str(dct, prec=1)
    'a=2, b=3e+00, c=string, d={x=5e+04}'
    >>> dict_str(dct, prec='1f')
    'a=2, b=3.1, c=string, d={x=49452.3}'
    >>> dict_str(dct, prec=2, to=': ', bracket='<>')
    '<a: 2, b: 3.15, c: string, d: <x=49452.33>>'

    :param d: dict to format
    :param prec: int - precision of general {:.<prec>g} or {:.<prec>} if str
    :param sep: separator between items
    :param to: key: value separator
    :param bracket: None or Sequence of 2 str for opening and closing brackets around the dict
    :param nested: apply recursively to nested dicts values
    :return: resulted string
    """

    def fmt(v):
        if isinstance(v, dict):
            return dict_str(v, bracket=bracket or '{}', prec=prec)
        if isinstance(v, float) and prec is not None:
            return f"{{:.{prec}}}".format(v)
        return str(v)

    if isinstance(prec, int):
        prec = f"{prec}g"

    s = sep.join(f"{k}{to}{fmt(v)}" for k, v in d.items())
    if bracket:
        bra, ket = bracket
        s = f"{bra}{s}{ket}"
    return s


_num_rex = re.compile(r'-?(\d*\.)?\d+([eE]\d+)?')


def is_num_str(v):
    """Return True if given str is a representation of some number"""
    return _num_rex.fullmatch(v)


def indent_lines(*lines, indent: int | str = 4) -> str:
    """Indent all the given lines by the indent str or given number of spaces.
    All the lines provided as the arguments are spilt by '\n', then indented,
    and then joined back by '\n' and return as a single string.

    :param lines: strings each potentially containing multiple \n separated lines
    :param indent: number of spaces or explicit indentation string
    """
    if isinstance(indent, int):
        indent = ' ' * indent

    def iter_lines():
        for line in lines:
            if line:
                yield from line.split('\n')

    return '\n'.join(indent + _ for _ in iter_lines())


class Indent:
    _stack: list[Indent] = []

    __push_depth__: int | None

    @classmethod
    def _push_indenter(cls, indenter: Indent):
        indenter.__push_depth__ = indenter.depth
        cls._stack.append(indenter)
        print(f"Pushed {indenter}")

    @classmethod
    def _pop_indenter(cls, indenter: Indent):
        """
        If indenter is being closed:
            - pop it from the stack
            - and return the new "top" None if stack is empty
        Otherwise, return False
        """
        if indenter.depth == indenter.__push_depth__:
            last = cls._stack.pop()
            if last is not indenter:
                raise RuntimeError
            last.__push_depth__ = None
            return cls._prev_indenter()
        return False

    @property
    def _stack_id(self):
        """Return """
        for sid, x in enumerate(self._stack):
            if x is self: break
        else:
            sid = 'Dead'
        return sid

    def _check_alive(self):
        """Safety check to avoid using indent out of its context"""
        if self.__push_depth__ is None:
            raise RecursionError("Can't use closed indent context!")

    @classmethod
    def _prev_indenter(cls):
        if not cls._stack: return None
        return cls._stack[-1]

    def __init__(self, indent: str | int = 3, *, depth: int = 1, max_depth: int | None = None,
                 width: int | None = 80, crop='...', str_func=repr):
        """
        Create text indenter `context` to represent nested objects.
        Context manager ``__call__`` function generates nested representation

        *Example:*

        >>> s = 'Header'
        ... with Indent('. ') as ind:
        ...     s += ind('item1')
        ...     s += ind('item2')
        ... print(s)

        *Produces:*
        ::
            Header
            . item1
            . item2

        :param indent: level indentation string or number of spaces describing it
        :param width: line width to fit into - crop the rest
        :param crop: str to indicate the cropping
        :param depth: initial indentetion depth
        :param max_depth: nesting level when recursion stops
        :param str_func: function (usually `str` or `repr`) convert objects to str
        """
        self._line_crop_mark = crop  # mark ends of cropped lines
        self._max_depth_mark = "⮷"  # mark ends of lines representing depth-collapsed nodes
        self._ind = indent if isinstance(indent, str) else ' ' * indent

        self._str = str_func
        self.depth = depth - 1
        self.width = None if width is None else max(width, 0)

        from_prev = ((prev := self._prev_indenter())
                     and prev.max_depth is not None
                     and (prev.max_depth - prev.depth))
        self.max_depth = (max_depth if from_prev is None else
                          from_prev if max_depth is None else
                          min(max_depth, from_prev))

        self._depth_indent = self.depth * self._ind
        self._max_marked = prev and prev._max_marked  # track max_depth_mark placement event
        self._push_indenter(self)

    def __enter__(self):
        self.depth += 1
        self._depth_indent += self._ind
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.depth -= 1
        self._depth_indent = self._depth_indent[:-len(self._ind)]

        if prev := self._pop_indenter(self):
            prev._max_marked = self._max_marked  # inherit max mark status from the child

    def __call__(self, obj, nl='\n'):
        """Create indented representation of the object.  By default, prepend a new line."""
        self._check_alive()
        if (mark := self._collapse_depth_mark()) is not None:
            return mark

        if isinstance(obj, str):
            if is_num_str(obj):
                return f"{nl}'{obj}"
            else:
                s = obj
        else:
            s = self._str(obj)

        return nl + '\n'.join(map(self._indent_crop, s.split('\n')))

    @property
    def remained_width(self):
        return None if self.width is None else self.width - len(self._depth_indent)

    def _collapse_depth_mark(self):
        """Check if depth exceeds max_depth and return mark to be printed out or None.

        This mark is a self._max_depth_mark for the top level being collapsed, and then
        just an empty string for the inner collapsed levels.
        """
        if self.max_depth is not None and (over := self.depth - self.max_depth) > 0:
            if over == 1 and not self._max_marked:
                self._max_marked = True
                return self._max_depth_mark
            return ''
        self._max_marked = False

    def _indent_crop(self, line):
        """
        Add indentation and crop if line exceeds the width.

        If cropping has removed the max-depth-mark at the end of the line, restore it.
        """
        if self.width is None: return line
        if len(line := self._depth_indent + line) > self.width:  # cropping is needed
            mark = self._max_depth_mark
            restore = line.endswith(mark) and len(mark)  # size of the mark being cropped (or 0)
            line = line[:self.width - len(self._line_crop_mark) - restore] + self._line_crop_mark
            if restore: line += mark  # restore the cropped mark
        return line

    def __repr__(self):
        return (f"{self.__class__.__name__}[{self._stack_id}]"
                f"(depth: {self.depth}/{self.max_depth})"
                f"{self._max_marked and self._max_depth_mark or ''}")


def smart_quoted(obj, fnc: Callable = str):
    """Return string representation of object inserting quotes to avoid ambiguity
    when describing numbers vs strings representing numbers.

    >>> smart_quoted(10) == "10"
    >>> smart_quoted("2.4e-16") == "'2.4e-16'"
    >>> smart_quoted({'a': 10}) == "{'a': 10}"

    :param obj: str returned as is, unless it's a representation of a number, then quoted
    :param fnc: function used to convert into str,
    """
    if isinstance(obj, str):
        return f"'{obj}'" if _num_rex.fullmatch(obj) else obj
    return fnc(obj)


def camel_to_snake(s):
    return re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", '_', s).lower()


def repr_nested(obj, width: int | None = None, ind: int | None = None, depth: int | None = None):
    """
    Create repr for nested object to fit into given line `width`.

    :param obj: Any object optionally consisting of sub-objects
    :param width: max line width to fit in
    :param ind: indent added to every nested level
    :param depth: maximal nesting depth
    :return: str representation
    """
    from toolbox.utils.short import drop_undef

    def need_multi_lines(ind, vals):
        return any('\n' in vs for vs in vals) or (
                (w := ind.remained_width) is not None and sum(map(len, vals)) > w
        )

    with (Indent(**drop_undef(depth=0, indent=ind, width=width, max_depth=depth)) as indent):
        if isinstance(obj, dict):
            values = [str(v) for v in obj.values()]
            if need_multi_lines(indent, values):
                return '\n'.join(f'{k}: {repr_nested(v, width=indent.remained_width)}'
                                 for k, v in obj.items())
            else:  # one-line dict
                return '{{{}}}'.format(', '.join(f'{k}: {v}' for k, v in zip(obj, values)))
        elif isinstance(obj, list):
            values = [smart_quoted(v) for v in obj]
            if need_multi_lines(indent, values):
                return ''.join(repr_nested(indent('- ' + smart_quoted(v, repr)),
                                           width=indent.remained_width) for v in obj)
            else:
                return f'[{", ".join(values)}]'
        else:
            return smart_quoted(obj, repr)


#
# def repr_nested(obj, width, indent, level: int = None):
#     """
#     Create repr for nested object to fit into give line `width`.
#
#     :param obj: Any object optionally consisting of sub-objects
#     :param width: max line width to fit in
#     :param indent: indent added to every nested level
#     :param level:  Not supported currently
#     :return: str representation
#     """
#     width -= len(indent)
#
#     if width < 12 or level is not None and level < 0:
#         if hasattr(obj, '__len__'):
#             obj = f'{type(obj).__name__}(×{len(obj)} items)'
#         else:
#             obj = str(obj).split('\n')[0][:width] + '...'
#         return obj
#
#     items = None
#     if is_dict := isinstance(obj, dict):
#         items = [f"{k}: " + repr_nested(v, width, indent) for k, v in obj.items()]
#     elif isinstance(obj, list):
#         items = [repr_nested(v, width, indent) for v in obj]
#
#     if items is not None:  # dict or list
#         concat = ''
#         sep = f'\n{indent}'
#         for s in items:
#             if len(s) > width or '\n' in s or len(concat := concat + s) > width:
#                 if is_dict:
#                     return f'{sep}'.join(items)
#                 else:
#                     return f'{sep}- '.join(['', *items]).replace('\n', '\n' + indent)
#
#         return ('{{{}}}' if is_dict else '[{}]').format(', '.join(items))
#     elif not isinstance(obj, str):
#         obj = str(obj)
#     return obj.replace('\n', '\n' + indent)


def plural(word):
    """Make plural form of a noun"""
    # Check if word is ending with s,x,z or is
    # ending with ah, eh, ih, oh,uh,dh,gh,kh,ph,rh,th
    if re.search('[sxz]$', word) or re.search('[^aeioudgkprt]h$', word):
        # Make it plural by adding es in end
        return re.sub('$', 'es', word)
    # Check if word is ending with ay,ey,iy,oy,uy
    elif re.search('[aeiou]y$', word):
        # Make it plural by removing y from end adding ies to end
        return re.sub('y$', 'ies', word)
    # In all the other cases
    else:
        # Make the plural of word by adding s in end
        return word + 's'


ALPHABETS = {
    36: "0123456789abcdefghijklmnopqrstuvwxyz"
}


def hash_str(s, sz: int = None, *, base: int | str = 16):
    """For given string build md5 hash string of requested length.

    By default, result is 16-base (HEX) string.
    Also, 36 based '0123456789abcdefghijklmnopqrstuvwxyz' is supported,

    :param s: string to hash
    :param sz: length of the resulting hash string cropped from the right
    :param base: alphabet or base of convertion of md5 bytecode (default 16, may be 36)
    :return: hash str
    """
    from hashlib import md5
    h = md5(s.encode())
    if base == 16:
        s = h.hexdigest()
    else:
        if isinstance(base, int):
            if not (alphabet := ALPHABETS.get(base, None)):
                if (max_base := max(ALPHABETS)) < base:
                    raise ValueError(f"Invalid {base=}")
                alphabet = ALPHABETS[max_base][:base]
        s = int_to_string(int.from_bytes(h.digest()), alphabet=alphabet, padding=sz)
    return s[-(sz if sz else 0):]


def int_to_string(number: int, alphabet: str | int, padding: int | None = None) -> str:
    """
    Convert a number to a string, using the given alphabet.

    The output has the most significant digit first.
    """
    if not isinstance(alphabet, str):
        alphabet = ALPHABETS[alphabet]
    alpha_len = len(alphabet)

    output = ""
    while number:
        number, digit = divmod(number, alpha_len)
        output += alphabet[digit]
    if padding:
        remainder = max(padding - len(output), 0)
        output = output + alphabet[0] * remainder
    return output[::-1]

