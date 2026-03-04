from ..regexp import strip_verbose_regex, filter_regex_matches, is_regex, format_to_regex, \
    regex_to_format


def test_strip_verbose():

    verb_patterns = [
        r"""
            # Discard whitespace, comments and the escapes of escaped spaces and hashes.
              ( (?: \s+                  # Either g1of3 $1: Stuff to discard (3 types). Either ws,
                | \#.*                   # or comments,
                | \\(?=[\r\n]|$)         # or lone escape at EOL/EOS.
                )+                       # End one or more from 3 discardables.
              )                          # End $1: Stuff to discard.
            | ( [^\[(\s#\\]+             # Or g2of3 $2: Stuff to keep. Either non-[(\s# \\.
              | \\[^# Q\r\n]             # Or escaped-anything-but: hash, space, Q or EOL.
              | \(                       # Or an open parentheses, optionally
                (?:\?\#[^)]*(?:\)|$))?   # starting a (?# Comment group).
              | \[\^?\]? [^\[\]\\]*      # Or Character class. Allow unescaped ] if first char.
                (?:\\[^Q][^\[\]\\]*)*    # {normal*} Zero or more non-[], non-escaped-Q.
                (?:                      # Begin unrolling loop {((special1|2) normal*)*}.
                  (?: \[(?::\^?\w+:\])?  # Either special1: "[", optional [:POSIX:] char class.
                  | \\Q       [^\\]*     # Or special2: \Q..\E literal text. Begin with \Q.
                    (?:\\(?!E)[^\\]*)*   # \Q..\E contents - everything up to \E.
                    (?:\\E|$)            # \Q..\E literal text ends with \E or EOL.
                  )        [^\[\]\\]*    # End special: One of 2 alternatives {(special1|2)}.
                  (?:\\[^Q][^\[\]\\]*)*  # More {normal*} Zero or more non-[], non-escaped-Q.
                )* (?:\]|\\?$)           # End character class with ']' or EOL (or \\EOL).
              | \\Q       [^\\]*         # Or \Q..\E literal text start delimiter.
                (?:\\(?!E)[^\\]*)*       # \Q..\E contents - everything up to \E.
                (?:\\E|$)                # \Q..\E literal text ends with \E or EOL.
              )                          # End $2: Stuff to keep.
            | \\([# ])                   # Or g3of3 $6: Escaped-[hash|space], discard the escape.
        """,
        r"""(?V1)
            \(\?P<(\w+)>    # named group (name captured as group 1): (?P<name>
              (             # matching pattern as group (2) in form: b(R1)a1...(R2)a2
                [^)(]*+     # `b` (before) part - any not `)(` characters
                (?:         # non capturing group for repeated `(R)a`
                  \((?2)\)  # `R` - reference to group 2 for possible nested braces
                  [^)(]*+   # 'a' (after) part  - any not `)(` characters
                )*          # `(R)a` repetition is optional
              )             # end the group 2
            \)              # end of the named group to match
        """,
        r"""
           \d{3}    # Area code
           [\s-]?   # Optional separator
           \d{3}    # First three digits
           [\s-]?   # Optional separator
           \d{4}    # Last four digits
       """
    ]

    correct = [
        r"((?:\s+|#.*|\\(?=[\r\n]|$))+)|([^\[(\s#\\]+|\\[^# Q\r\n]|\((?:\?#[^)]*(?:\)|$))?|\[\^?\]?[^\["
        r"\]\\]*(?:\\[^Q][^\[\]\\]*)*(?:(?:\[(?::\^?\w+:\])?|\\Q[^\\]*(?:\\(?!E)[^\\]*)*(?:\\E|$))[^\["
        r"\]\\]*(?:\\[^Q][^\[\]\\]*)*)*(?:\]|\\?$)|\\Q[^\\]*(?:\\(?!E)[^\\]*)*(?:\\E|$))|\\([# ])",
        r"(?V1)\(\?P<(\w+)>([^)(]*+(?:\((?2)\)[^)(]*+)*)\)",
        r"\d{3}[\s-]?\d{3}[\s-]?\d{4}"]

    for pat, cor in zip(verb_patterns, correct):
        res = strip_verbose_regex(pat, validate=True)
        assert res == cor


def test_is_regexp():
    assert is_regex(r'OK\w{2}')
    assert is_regex(r'yes.*')
    assert is_regex(r'(?<tag>some.*)')


def test_filter_regex():
    s = '__ok'
    assert [*filter_regex_matches(r'__\w*', s)] == [s]

    ss = ['Yes', 'no', 'Yellow', 'Young']
    assert [*filter_regex_matches('Ye.*', ss)] == ss[::2]

    ss = ss + ['__call__', '_repr_latex_']
    assert [*filter_regex_matches(['Ye.*', '_+[^_]+_+'], ss)] == ss[::2]


def test_format_to_regex_to_format():
    fmt = '{some}_big_{exp}ression'
    rgx = r'(?P<some>\w+?)_big_(?P<exp>\w+?)ression'
    assert format_to_regex(fmt) == rgx
    assert regex_to_format(rgx) == fmt

    rgx = r'(?P<kind>(frames_cleanpass)|(disparity))/(?P<subset>\w+)' \
          r'/(?P<scene_1>\w+)/(?P<scene_2>\w+)/(?P<side>\w+)' \
          r'/(?P<scene_3>\w+)\.(?P<ext>(?(3)pfm|png))'
    fmt = '{kind}/{subset}/{scene_1}/{scene_2}/{side}/{scene_3}.{ext}'
    assert regex_to_format(rgx) == fmt
