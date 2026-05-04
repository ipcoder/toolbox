# set of tools for coding convenience
from __future__ import annotations

import ast
import re
from typing import Union, Collection


def importer(name, root_package=False, relative_globals=None, level=0):
    """ We only import modules, functions can be looked up on the module.
    Usage:

    from foo.bar import baz
    >>> baz = importer('foo.bar.baz')

    import foo.bar.baz
    >>> foo = importer('foo.bar.baz', root_package=True)
    >>> foo.bar.baz

    from .. import baz (level = number of dots)
    >>> baz = importer('baz', relative_globals=globals(), level=2)
    """
    return __import__(name, locals=None,  # locals have no use
                      globals=relative_globals,
                      fromlist=[] if root_package else [None],
                      level=level)


def call_args_expr(level=None, *, name: str = None, extend=5):
    """
    From within function in run time find literal expression of the arguments
    used in its call.

    Example:

    >>> def func1(x=10, y=None,  k=0):
    ...     print(call_args_expr())
    ...
    ... func1(1, sin(x**2/pi))   # this call will produce:
    ... ['1', 'sin(x**2/pi)']
    ...
    >>> def func2(x=10, y=None,  k=0):
    ...     def func3():
    ...         print(call_args_expr(name='func2'))  # find by name
    ...
    ...     def func3():
    ...         print(call_args_expr(2))  # specify level other than 1
    ...     func3()
    ...
    ... func2(1, sin(x**2/pi))   # this call will produce:
    ... ['1', 'sin(x**2/pi)']
    ... ['1', 'sin(x**2/pi)']

    Deaful parameters assume using this function just within a function
    which call is being examined.

    :param level: position in this call relative to this call
    :param name: name of the function call to parse
    :param extend: expected maximal number of lines the code may spread.
                Its an initial guess - and incrementally continues up to
                100 line before raising a NameError
    :return:
    """
    import inspect

    def code_gen(frame):
        """
        Generate increasingly large segments of code starting from the a one
        line of context from the frame, adding one line until limit of 100.
        :param frame: frame to analyze
        :return: yield code segment
        """
        code = ''
        lines_in = 0
        context_size = 1
        while lines_in < 100:
            context_size += extend * 2
            fi = inspect.getframeinfo(frame, context_size)  # get +- max_lines around the call
            for line in fi.code_context[fi.index + lines_in:]:
                code = code + line if code else line.lstrip()
                lines_in += 1
                yield code

    if not level and not name:
        level = 1  # default level is 1 if name is not provided

    stack = inspect.stack()
    if level:
        frame_info = stack[level + 1]
        found_name = stack[level].function
        if name != found_name:
            if name is not None:
                raise NameError(f"Found function {found_name} not {name}")
            name = found_name
    else:
        for level, frame_info in enumerate(stack[1:], 1):
            if frame_info.function == name:
                frame_info = stack[level + 1]
                break
        else:
            raise NameError(f"Call to {name} not found!")

    for source in code_gen(frame_info.frame):
        try:
            tree = ast.parse(source, mode='single')
            break
        except SyntaxError:
            pass
    else:
        raise SyntaxError(f"Failed to parse call {frame_info.code_context}")

    for func in ast.walk(tree):
        if isinstance(func, ast.Call) and getattr(func.func, 'id', '') == name:
            break
    else:
        raise NameError(f"Function {name} not found on level {level}")

    func_src = ast.get_source_segment(source, func)
    return [re.sub(r'\n\s*', '', ast.get_source_segment(func_src, arg))
            for arg in func.args]


class IsIn:
    """Checks if value belongs to a hierarchy of objects.

    The root is True, the rest is defined as collection of objects.

    ``__call__`` with a tested value returns True if it belongs

    Examples:
        >>> show = IsIn('one.x', 'one.y', 'two.a', 'two.b')
        >>> assert show('one.y')
        >>> assert show == 'two.b'
        >>> assert show.branch('one') != 'two.a'
        >>> assert show.two == 'a'    # same as show.branch('two') == 'a' TODO: fix !
        >>> assert 'one.x' in show
        >>> show = IsIn(True)
        >>> assert show == 'anything'
        >>> assert show.branch('three') == 'three'
    """

    def __init__(self, *values: Union['IsIn', bool, Collection[str]]):
        """Define the collection of objects.

         :param values: collection of the values OR
                        True - for any, False - for none
        """

        from .short import as_list

        if len(values) == 0:
            self.values = set()
        elif len(values) == 1:
            values = values[0]
            if isinstance(values, IsIn):
                self.values = values.values
            elif values is True:
                self.values = True
            elif values in {None, False, 0}:
                self.values = set()
            else:
                self.values = set(as_list(values))
        else:
            self.values = set(as_list(values))

    def __call__(self, v) -> bool:
        """Return True if v is in"""
        return True if self.values is True else v in self.values

    def __eq__(self, other):
        return self(other)

    def __ne__(self, other):
        return not self(other)

    def __repr__(self):
        return f"IsIn: {self.values}"

    def __bool__(self):
        return bool(self.values)

    def __contains__(self, item):
        return self(item)

    def branch(self, v):
        return IsIn(
            self.values if v in {True, False, None, 0} else
            True if self.values is True or v in self.values else (
                    (vv := (v + '.')) and (n := len(vv)) and
                    {xn or True for x in self.values if x == v or (xn := x[n:]) == vv}
            )
        )

    def __getattr__(self, item):
        return self.branch(item)


class NamedObj:
    def __init__(self, name):
        self.name = f"<{name}>"

    def __repr__(self):
        return self.name
