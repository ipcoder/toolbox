from __future__ import annotations

import collections
import enum
import re
import sys as _sys
from functools import wraps
from typing import Collection, Optional, Type, Callable, Sequence, Literal, Generic, TypeVar


def name_func_outputs(func, names: Collection[str] = None, *,
                      out_type: str | Type[tuple | dict] = None,
                      adjust=False, nest=None):
    """Turns a function returning a tuple of items into one returning
    a named tuple or a dict with items assigned with provided names.

    - Converts into a tuple, if typename str or None (same as str=='<func_name>_out').
    - Converts into dict if typename is ``dict``

    If number of items in the output does not match that of names,
    depending on the value of ``adjust`` argument:

    - ``False`` - raise ``TypeError``
    - ``True`` - uses first ``len(names)`` and raise only if ``len(output)`` is bigger
    - ``None`` - returns output without change

    Special checking is done for the case of nested wrapping attempts,
    which are being tracked and depending on the argument ``nest``:
    - ``False`` - raise ``RuntimeError``
    - ``True``  - allow nested wrapping and return multi-wrapped function
    - ``None``  - ignore nested wrapping request and return the input function

    Example::

    >>> f0 = lambda x: tuple(range(x)) if x > 1 else 0
    >>> name_func_outputs(f0, list('xyz'))(3)
    <lambda_out> x: 0, y: 1, z: 2

    >>> name_func_outputs(f0, list('xyz'), out_type=dict, adjust=True)(2)
    {'x': 0, 'y': 1}

    :param func: the function to wrap
    :param names: collection of names to match the function's outputs or named tuple
    :param out_type: name of namedtuple or namedtuple class
    :param adjust: adjust names length to the output length
    :param nest: allow nested `name_outputs` wrappings
    :return: new function with dict output
    """

    def out_len_error(n):
        raise RuntimeError(f"Cant wrap a {n} outputs of {func_name} in {ln} names")

    @wraps(func)
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        if not isinstance(out, tuple):
            if len(names) == 0 or adjust is None: return out
            if len(names) == 1 or adjust is True: out = (out,)
            if adjust is False: out_len_error(1)
        else:
            if (lo := len(out)) > ln: out_len_error(lo)
            if lo == ln or adjust is True: return construct(out)
            if adjust is None: return out  # from here its always lo < ln
            out_len_error(lo)
        return construct(out)

    func_name = re.sub(r'\W', '', func.__name__)
    if names is not None:
        is_str = lambda x: isinstance(x, str)
        if isinstance(names, str) or not all(map(is_str, names)):
            raise TypeError("names must be a collection of strings")
        if out_type is None:
            out_type = f"{func_name}_out"

        if isinstance(out_type, str):
            construct = lambda out: namedtuple(
                out_type, names[:(len(out) if adjust else None)]
            )(*out)
        elif issubclass(out_type, dict):
            construct = lambda out: dict(zip(names, out))
        else:
            raise TypeError("Given names typename must be a dict class or str or None")
    elif isinstance(out_type, tuple) and hasattr(out_type, '_fields'):
        names = out_type._fields
        construct = lambda out: out_type(*out)
    else:
        raise TypeError("typename must be namedtuple class if names are not provided")
    ln = len(names)

    # check if attempting re-wrap the function in the same way
    attr = dict(names=names, out_type=out_type, prev=None, nest=0)
    if prev_attr := getattr(func, '_name_outputs', None):
        if prev_attr['out_type'] == out_type and prev_attr['names'] == names:
            if nest is None:  # function is already wrapped as requested
                return func
            elif nest is False:
                raise RuntimeError(f'Re-wrapping name_outputs with {attr}')
        attr['prev'] = prev_attr
        attr['nest'] = prev_attr['nest'] + 1
    setattr(wrapper, '_name_outputs', attr)
    return wrapper


def doc_from(src: type | Callable, merge: Literal['append', 'prepend', 'anchor', 'replace'] = 'append', cite=False):

    def merger(obj):
        doc = (f'Doc from {src.__qualname__}:\n' if cite else '') + (src.__doc__ or '')
        if not obj.__doc__ or merge == 'replace':
            obj.__doc__ = doc
        if merge == 'append':
            obj.__doc__ += doc
        elif merge == 'prepend':
            obj.__doc__ = doc + '\n' + obj.__doc__
        elif merge == 'anchor':
            obj.__doc__ = obj.__doc__.replace('<<anchor>>', f'{doc}')
        else:
            raise NameError(f'Unknown doc merging method {merge}')
        return obj

    return merger


@doc_from(name_func_outputs)
def name_outputs(names: Optional[Collection[str]] = None, *,
                 out_type: str | Type[tuple | dict] | None = None,
                 adjust=False, nest=None):
    f"""{name_func_outputs.__doc__}"""

    def convert(func):
        return name_func_outputs(func, names, out_type=out_type, adjust=adjust, nest=nest)
    return convert


name_outputs.__doc__ = name_func_outputs.__doc__


class NamedTupleI:
    """
    Implements Interface of ``NamedTuple`` classes.

    Don't subclass it directly, use instead:
     - either ``namedtuple`` function in this module
     - or ``NamedTupleMeta`` metaclass

    It can be used for type checking though:
    ::
            isinstance(obj, NamedTupleI)

    instead of
    ::
            isinstance(type(obj), NamedTupleMeta)
    """
    from toolbox.utils.nptools import array_info_str

    _fields: list[str]
    __iter__: Callable

    def __init__(self, *args, **kws): ...

    def __repr__(self):
        val_str = lambda v: (NamedTupleI.array_info_str(v, stats=False)
                             if hasattr(v, 'shape') and hasattr(v, 'ndim') else str(v))
        fields_str = (f'{k}: {val_str(v)}' for k, v in zip(self._fields, self))

        end, sep = '<end>', '<sep>'
        s = f"<{self.__class__.__name__}> {end}" + sep.join(fields_str)
        end_str, sep_str = ['\n\t'] * 2 if len(s) > 50 else ['', ', ']
        return s.replace(end, end_str).replace(sep, sep_str)

    def _part(self, *items: str | int, name: Optional[str] = None):
        """Return a part of tuple items as a new namedtuple

        :param name: name of the new namedtuple class
        :param items: fields names to extract (in this order!)
        """
        name = name or 'Part'

        def extractor(elements):
            for elm in elements:
                if isinstance(elm, int):
                    yield self._fields[elm], tuple.__getitem__(self, elm)
                elif isinstance(elm, str):
                    yield elm, getattr(self, elm)
                elif isinstance(elm, slice):
                    for item in zip(self._fields[elm], tuple.__getitem__(self, elm)):
                        yield item
                else:
                    raise TypeError("Invalid reference to tuple element")

        _fields, _items = zip(*extractor(items))
        return namedtuple(name, _fields)(*_items)

    def apply(self, func):
        return self.__class__(*(map(func, self)))

    def __add__(self, other):
        return namedtuple('Combined', [*self._fields, *other._fields])(*self, *other)

    @staticmethod
    def __build_reduced__(typename, fields, values, module):
        return namedtuple(typename, fields, module=module)(*values)

    def __reduce__(self):
        """Override pickle machinery to support dynamically defined or nested named tuples"""
        cls = self.__class__
        return (
            NamedTupleI.__build_reduced__,
            (cls.__name__, self._fields, [*self], cls.__module__),  # passed to __build_reduced__
            {})  # no other state information is used


class NamedTupleMeta(type):
    @staticmethod
    def _find_module(level=1, module=None):
        if module is None:
            try:
                module = _sys._getframe(level).f_globals.get('__name__', '__main__')
            except (AttributeError, ValueError):
                return None
        return module

    def __repr__(cls):
        return f"{cls.__name__}({', '.join(cls._fields)})"

    def __new__(mcs, name, bases, attrs, fields: Sequence[str], defaults: Sequence = None, module=None):
        module = NamedTupleMeta._find_module(module=module)
        if NamedTupleI not in bases:
            bases = (*bases, NamedTupleI)
        bases = (*bases, collections.namedtuple(name, fields, defaults=defaults, module=module))
        cls = super().__new__(mcs, name, bases, attrs)
        cls.__module__ = module
        return cls

    def __init__(cls, name, bases, attrs, **_):
        super().__init__(name, bases, attrs)


def namedtuple(typename: str, fields: Sequence[str], defaults: Sequence = None, module=None):
    """Wraps ``collections.namedtuple`` to add custom __repr__ for some data types:
       numpy arrays, ...


       **Additional manipulation over tuple content**

       Extraction of part of the fields as a new tuple:

        >>> v3 = namedtuple('XYZ', ['x', 'y', 'z'])(10, 20, 30)
        >>> v3._part('z', 'x', name='ZX') == (v3.z, v3.x)
        True
        >>> v3._part(slice(0,2)) == v3[:2]
        True
        >>> v3._part(slice(0,2), slice(2,3)) == v3
        True
        >>> v3._part('x', slice(1,2), 2) == v3
        True

       Concatenation of tuples:

        >>> v3._part('x', 'y') + v3._part('z') == v3
        True

    :param typename: name of the new type
    :param fields: names of tuple fields in this order
    :param defaults: sequence of default valued of the fields in same order
    :param module: module the new class to be associated with
    """
    module = NamedTupleMeta._find_module(level=1, module=module)
    return NamedTupleMeta(typename, bases=(), attrs={}, fields=fields, defaults=defaults, module=module)


def name_tuple(typename='Tuple', **kws):
    """
    Create namedtuple OBJECT from name: value pairs

    :param typename: for the tuple type
    :param kws: names-values pairs
    :return: namedtuple
    """
    return namedtuple(typename, tuple(kws.keys()))(*kws.values())


def double_wrap(dec_func):
    """
    a decorator of decorator, allowing the decorator to be used as:
    @decorator(with, arguments, and=kwargs)
    or
    @decorator
    """

    @wraps(dec_func)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return dec_func(args[0])
        else:
            # decorator arguments
            return lambda real_func: dec_func(real_func, *args, **kwargs)

    return new_dec


class CaseInsEnum(str, enum.Enum):
    """
    Enum of string values allowing to set it with case-insensitive names or values.

    >>> class E(CaseInsEnum):
    ...     min: 'minimal'
    ...     max: 'maximal'
    ... assert E('Min') is E.min and E('MINIMAL') is E.min

    Same can be also created functionally:

    >>> E = CaseInsEnum('E', {'min': 'minimal', 'max': 'maximal'})

    Or just list the keys if values are of no importance:

    >>> En = CaseInsEnum('En', ['min', 'max'])
    """
    @classmethod
    def _missing_(cls, key: str):
        key = key.lower()
        for n, v in cls._member_map_.items():
            if n.lower() == key or isinstance(v, str) and v.lower() == key:
                return v


def enum_str_type(*values, name):
    """
    Create type enumerating provided strings as attributes AND values.

    >>> Colors = enum_str_type('Colors', ['red', 'yellow'])
    >>> assert Colors.yellow == Colors('yellow')

    :param name:
    :param values:
    :return:
    """
    return CaseInsEnum(name, dict(zip(values, values)))


def enum_attr(attr: str, **kwargs):
    """Create ``CaseInsEnum`` class with an extra attribute for every member.

    Examples
    >>> AB = enum_attr('info', a='Member A', b='Member B')
    >>> AB('a') is AB.a
    True
    >>> AB.b.info
    'Member B'

    Special attribute '__call__' is supported, to make the memebres callable by
    providing each with its own function:

    Examples
    >>> Funcs = enum_attr('__call__', len=len, print=print)
    >>> Funcs.len('Hi')
    2
    >>> Funcs.print('Hi')
    Hi

    :param attr: name of the attribute to add
    :param kwargs: dict for all the members {name: attr_value}
    """
    class _EnumNameMatcher:
        if attr == '__call__':
            def __call__(self, *args, **kws):
                return kwargs[self.name](*args, **kws)
        else:
            def __init__(self, m):
                setattr(self, attr, kwargs[self.name])

    return CaseInsEnum('EnumMatch', [*kwargs], type=_EnumNameMatcher)


T = TypeVar("T")
RT = TypeVar("RT")


class classproperty(Generic[T, RT]):
    """
    Class property attribute (read-only).

    This works around deprecation from 3.13 of
    ::
        class C:
            @classmethod
            @property
            def some(cls,): ...

    Same usage as @property, but taking the class as the first argument.
    ::
        class C:
            @classproperty
            def x(cls):
                return 0

        print(C.x)    # 0
        print(C().x)  # 0
    """

    def __init__(self, func: Callable[[type[T]], RT]) -> None:
        # For using `help(...)` on instances in Python >= 3.9.
        self.__doc__ = func.__doc__
        self.__module__ = func.__module__
        self.__name__ = func.__name__
        self.__qualname__ = func.__qualname__
        # Consistent use of __wrapped__ for wrapping functions.
        self.__wrapped__: Callable[[type[T]], RT] = func

    def __set_name__(self, owner: type[T], name: str) -> None:
        # Update based on class context.
        self.__module__ = owner.__module__
        self.__name__ = name
        self.__qualname__ = owner.__qualname__ + "." + name

    def __get__(self, instance: Optional[T], owner: Optional[type[T]] = None) -> RT:
        if owner is None:
            owner = type(instance)
        return self.__wrapped__(owner)

