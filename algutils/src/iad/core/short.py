from __future__ import annotations

import logging
from typing import Iterable, Union, TypeVar, Collection, Type, Callable, get_args

Strings = Union[str, list[str]]

__all__ = ['as_list', 'drop_undef', 'as_iter', 'Strings', 'unless_subset']


def unless(condition, event, action: Union[Callable, BaseException] = ValueError):
    """one-liner expression for checking condition and acting if failed.

    :param condition: an object which can be cased to boolean
    :param event: could be Exception object or just a string
    :param action: optional action:
                    - exception class - used if exception is string
                    - a callable object accepting event as argument
    :return: the condition object

    Example:
        use_outcome = unless([], 'Empty list!', print)   # == Empty list!
    """

    if not condition:
        if isinstance(event, BaseException):
            raise event
        if isinstance(action, type) and issubclass(action, BaseException):
            raise action(event)
        action(event)
    return condition


def unless_subset(the_set: Iterable, sub_set: Iterable,
                  event="{inv} not found in the set: {the_set}",
                  action: Union[Callable, BaseException] = KeyError):
    """
    Triggers event action unless `ss` is a subset of `s`.
    :param the_set: the set
    :param sub_set: expected subset
    :param event: format(able) string with {inv} and {s} or exception to throw
    :param action: optional action to perform (exception or callable)
    :return: condition
    """
    the_set = set(the_set)
    inv = set(sub_set).difference(the_set)
    event = event.format(**locals()) if isinstance(event, str) else event
    return unless(not inv, event, action=action)


def issubset(set_a: Collection, set_b: Collection,
             nonempty: bool = False, fail: bool = False):
    """
    Checks whether set_a is a subset of set_b.
    Overrides python's empty set native behaviour.
    This behaviour relies on empty set - that is a subset of everything in set theory.
    Although this holds, we want to know if one or both of the sets is empty.

    python:
    >>> set([]).issubset(['a']) == True

    ours:
    >>> set([]).issubset(['a']) == False

    The user is in charge to pass set-cast friendly arguments.
    The function gives the user descriptive message on non subset results.

    function flow can be summarized as follows:

    cast -> optional non empty check -> cardinality check -> subset check.

    :param set_a: First collection.
    :param set_b: Second collection.
    :param nonempty: Whether empty sets results on error.
    :param fail: Whether to raise errors, or return a boolean.

    :raises KeyError: If set_a is not a subset of set_b.
            ValueError: If a is empty.

    :return: boolean indicates whether set_a is a subset of set_b
    """
    set_a = set(set_a)
    set_b = set(set_b)
    if nonempty:
        assert set_a, f'Given set_a is empty'
        assert set_b, f'Given set_b is empty'

    if not len(set_a) > 0:
        if not len(set_b) > 0:
            return True
        else:
            msg = f'Given set_a is empty, set_b {set_b} is not'
            if fail:
                raise ValueError(msg)
            logging.warning(msg)
            return False

    if set_a.issubset(set_b):
        return True
    else:
        msg = f'The keys {set_a - set_b} of set_a are missing in set_b {set_b}'
        if fail:
            raise KeyError(msg)
        else:
            logging.warning(msg)
        return False


def compare(set_a, set_b, fail):
    set_a = set(set_a)
    set_b = set(set_b)
    diff1 = set_a.difference(set_b)
    diff2 = set_b.difference(set_a)

    if diff1 or diff2:
        msg = ''
        if diff1:
            msg += f"\nKeys {diff1} are missing in set_b {set_b}"
        if diff2:
            msg += f"\nKeys {diff2} are missing in {set_a}"
        if fail:
            raise ValueError(msg)
        return False
    return True


def as_number(v):
    """
    Return a number given number or string (int or float)
    :param v: number | str
    :return: number
    """
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            return float(v)
    return v


def drop_undef(*keys: str, ns: dict = None, _undef=None, **kwargs) -> dict:
    """
    Drop items with undefined values from a given dict.

    (Useful to exclude undefined kw arguments when passing to functions.)


    Supported Forms
    ---------------
    There are two forms of providing the dict.

    Undefined object can be specified using `undef` argument, ``None`` by default.

    Alternatively, undefined value can be defined per key and passed as kwargs
     (if `ns` argument is defined and contains the dict to filter, otherwise `kwargs` are treated as `ns`)

    Remove Items with `undef` values
    =============================

    The dict to filter can be passed to the function
      1.  as `keyword arguments`                        ``defined_kws(**kwargs)``
      2.  as a single argument `namespace`              ``defined_kws(namespace=dct)``
      3.  selected keys from the `namespace` dict       ``defined_kws(*keys, namespace=dct)``

    So that form ``3`` is a generalized version of ``2``.

    Examples
    ^^^^^^^^
    ::

        dct = dict(x=False, y=None, z=0)
        drop_undef(**dct)                  # drop all None from keyword arguments
        {'x': False, 'z': 0}

        drop_undef(ns=dct)          # same, drop all None from namespace dict
        {'x': False, 'z': 0}

        drop_undef('x', 'y', ns=dct)   # drop all except x, y, and them if None
        {'x': False}

        UNDEF = object()                # define particular UNDEF object to keep Nones
        drop_undef(undef=UNDEF, x=10, y=None, z=UNDEF)
        {'x': 10, y: None}

    Remove items with custom values
    ===============================
    Forms ``2`` and ``3`` allow to describe `undefined` for everey key:
       - as a value, alternative to ``None``
       - as a callble evaluated to ``False`` indicating "undefined"

    If a key is not mentioned it assumes the usual undefined value of ``None``.

        drop_undef(ns=dct, y=0)        # drop y if 0, others if None
        {'x': False, 'y': None, 'z': 0}

    As in ``3`` additional selection by `keys` may be applied:

        drop_undef('x', 'y', ns=dct, x=False, y=0)  #  drop z regardless
        {'y': None}

        drop_undef(ns=dct, x={False, 0}.__contains__)
        {'z': 0}

    """

    def defined(k, v):
        if keys and k not in keys:
            return False

        cond = kwargs.get(k, None)
        return not (
            v is _undef if cond is None
            else cond(v) if callable(cond)
            else cond == v
        )

    keys = keys and set(keys)
    if ns:
        return {k: v for k, v in ns.items() if defined(k, v)}
    else:
        return {k: v for k, v in kwargs.items() if v is not _undef}


T = TypeVar('T', bound=Collection)


def as_list(v: Iterable | object, empty_none=True, *, collect: Type[T] = list,
            no_iter: Type[Iterable] | tuple[Type[Iterable]] = None) -> T:
    """ Converts iterable and scalar objects to list (or requested collection).

    By *scalar* we understand any object which is neither of:
     - instance of ``Iterable``
     - string
     - has attribute ``.shape`` (like ``ndarray``)
     - of type (or tuple of types) provided in ``no_iter`` argument

    `None` is converted to [] unless `empty_none` is False

    :param v: an object or iterable to convert.
    :param empty_none: convert ``None`` into empty collection, otherwise treat as value
    :param collect: type of the resulting collection.
    :param no_iter: type of tuple of types to regard as scalar, inspire they are ``Iterable``.
    """
    if v is None:
        return collect() if empty_none else collect([None, ])

    if isinstance(collect, type) and isinstance(v, collect):
        if no_iter and isinstance(v, no_iter):
            return collect([v])
        return v

    if (isinstance(v, str) or
            hasattr(v, 'shape') or
            not hasattr(v, '__iter__') or
            no_iter and isinstance(v, no_iter)):
        v = [v]

    return collect(v)


def as_iter(v: Iterable | object, empty_none=True,
            no_iter: Type[Iterable] | tuple[Type[Iterable]] = None) -> Iterable:
    """
    For any input ensure the output is ``Iterable``:
     - *scalar* input ``v`` convert to tuple ``(v, )``
     - ``Iterable`` return unchanged
     - ``None`` convert to ``()`` if ``empty_none is True``, otherwise to ``(None, )``

    By *scalar* we understand any object which is neither of:
     - instance of ``Iterable``
     - string
     - has attribute ``.shape`` (like ``ndarray``)
     - of type (or tuple of types) provided in ``no_iter`` argument

    :param v: an object or iterable to convert.
    :param empty_none: convert ``None`` into empty collection, otherwise treat as value.
    :param no_iter: type of tuple of types to regard as scalar, inspire they are ``Iterable``.
    """
    if v is None:
        return () if empty_none else (None,)

    if (isinstance(v, str) or
            hasattr(v, 'shape') or
            not hasattr(v, '__iter__') or
            no_iter and isinstance(v, no_iter)):
        return (v, )

    return v


def validate_literal(val, literal_type, *, msg="literal value") -> set:
    """
    Validate that given value matches given literal type.

    :param val: Value(s) to check
    :param literal_type: Type created using `Literal[...]
    :param msg: description of the type
    :raise: `ValueError` if not.

    return - validated set of literals
    """
    valid = set(get_args(literal_type))
    val = as_list(val, collect=set)

    if inv := val - valid:
        raise ValueError(f"Invalid {msg} {inv}. Allowed: {valid}")

    return val
