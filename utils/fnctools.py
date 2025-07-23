from __future__ import annotations

import dataclasses as dcl
from importlib import import_module
from types import ModuleType
from typing import Callable, get_type_hints

from toolbox.utils import as_list

__all__ = ['Namespace', 'Operator', 'O', 'comp']


class Namespace:
    """
    Create compound namespace from multiple sources.
    Provides read-only dict-like access.

    >>> n1 = Namespace({'x': abs}, {'y': 10, 'w': 20})
    >>> n1
    <Namespace> collected from [2]: {1+2}
    >>> n2 = Namespace({'z': 'ok'}, n1)
    >>> n2
    <Namespace> collected from [3]: {1+1+2}
    >>> n2.keys()
    ['z', 'x', 'y', 'w']
    >>> n1 + n2  # n2 contains n1, so only content of n1 (different order)
    <Namespace> collected from [3]: {1+2+1}
    >>> n2 += {'a': 100, 'b': 101, 'c': 102}; n2
    <Namespace> collected from [4]: {1+1+2+3}
    """

    def __init__(self, *namespaces: dict | ModuleType | str, built=False):
        """
        Create compound namespace from sources of different kinds:
            another namespaces (dict) or modules (as objects or their names)

        :param namespaces: Can be dict, or module or importable module name
        :param built: if True add also __builtin__ as a source
        """

        def flatter(collection):
            for ns in collection:
                if isinstance(ns, self.__class__):
                    yield from flatter(ns._namespaces)
                else:
                    yield ns

        self._namespaces = self._normalize([*flatter(namespaces)], built=built)

    @classmethod
    def _normalize(cls, namespaces, built=False):
        """Remove duplicates and put builtins at the end"""
        built = built or __builtins__ in namespaces

        found = set()  # if duplicated leave the first
        namespaces = [found.add(id(ns)) or ns for ns in namespaces
                      if not (id(ns) in found or ns is __builtins__)]  # exclude builtins
        built and namespaces.append(__builtins__)  # to ensure it is the last
        return list(map(cls._to_mapping, namespaces))

    @staticmethod
    def _to_mapping(ns):
        if isinstance(ns, str):
            ns = import_module(ns)
        if isinstance(ns, ModuleType):
            ns = vars(ns)
        if isinstance(ns, dict):
            return ns
        raise TypeError('Namespace must be provided as dict, module or mudule name')

    def __getitem__(self, item):
        for ns in self._namespaces:
            if item in ns:
                return ns[item]
        raise KeyError(item)

    def keys(self):
        res = []
        for ns in self._namespaces:
            res.extend(ns)
        return res

    def __contains__(self, item):
        for ns in self._namespaces:
            if item in ns:
                return True
        return False

    def __add__(self, other):
        res = self.__class__(self)
        res += other
        return res

    def __iadd__(self, other):
        if isinstance(other, self.__class__):
            self._namespaces.extend(other._namespaces)
        else:
            self._namespaces.append(self._to_mapping(other))
        self._namespaces = self._normalize(self._namespaces)
        return self

    def __repr__(self):
        represent = lambda ns: ns is __builtins__ and 'b' or str(len(ns))
        return f"<Namespace> collected from [{len(self._namespaces)}]: " \
               f"{{{'+'.join(map(represent, self._namespaces))}}}"

    def __bool__(self):
        return bool(self._namespaces)


class Operator:
    """
    Class Operator (alias O) allows building operators from functions and
    use the operator notation to chain functions:
    Instead of:
        sin(cos(min(1, tan(x))))   ->   (O(sin) * cos * O(min, 1) * tan)(x)
    >>> inc_1 = O(lambda _:  _ + 1, o_name='inc_1')
    >>> f = O(min, 1, o_name='min_1') * inc_1
    >>> f(2)
    1
    >>> print(f)
    O|min_1|inc_1|
    >>> f = O(abs)
    >>> f *= inc_1
    >>> f(-2)
    1
    >>> print(abs * inc_1)
    O|abs|inc_1|
    """

    @dcl.dataclass
    class Func:
        fnc: Callable
        name: str = None
        args: tuple = ()
        kws: dict = None
        inp_key: str = None
        inp_pos: int = None

        def resolve(self, ns):
            if isinstance(name := self.fnc, str):
                if '.' in name:
                    module, name = name.rsplit('.', 1)
                    self.fnc = getattr(import_module(module), name)
                else:
                    self.fnc = ns[name]
                if not self.name:
                    self.name = name

        def __post_init__(self):
            if self.inp_key and self.inp_pos:
                raise ValueError("Operand argument can't be described as both positional and keyword")

            if self.inp_pos:
                if not self.args:
                    raise ValueError("Operand's position > 0 assumes other positional arguments")
                self.args = [*self.args[:self.inp_pos], None, *self.args[self.inp_pos:]]

            if isinstance(self.fnc, self.__class__):
                assert not (self.name or self.args or self.kws)
                self.__dict__ = self.fnc.__dict__
                return

            if not self.name:
                self.name = isinstance(self.fnc, str) and self.fnc.rsplit('.')[-1] or self.fnc.__name__
                assert self.name and self.name != '<lambda>'
            self.kws = self.kws or {}

    # work around unresolved postponed types in the new python annotations mechanism
    _func_arg_types = get_type_hints(Func)  # resolved types of the fields

    @classmethod
    def comp(cls, *stages: Callable | str | Func | tuple, ns: dict = None, skip=None):
        """
        Creates an Operator object implementing a composition of functions
        with specific signature, allowing the composition with next function.

        The composition (sequential application) of two functions ``(f1, f2)``
        ::
            def f1(inp: T1, *args1, **kws1) -> T2
            def f2(inp: T2, *args2, **kws2) -> T3
        is performed from right to left, and ``y = comp(F1, F2)(x)`` stands for:
        ::

            t = f2(x, args2, **kws2)
            y = f2(t, args1, **kws1)
        where ``F`` is one of the forms to describe a function ``f``,
        its `name` and `arguments`.

        In the full form F is an instance of ``Operator.Func``:
        ::
            F = Operator.Func(f, 'func_name', args, kws)
        However ``comp`` allows for additional forms, supported for convenience
        by the ``Func`` constructor.

        In addition to that, the function ``f`` may be provided as a string
        referencing its implementation in the supplied `namespace` (dict-like) object.

        Also stages can be `dict` in which case they are flattened into sequence of
        tuples (key, value) with `key` for func, and `value` for its arguments.

        Notice! None or empty elements passed as stages are filtered out.

        >>> f = O.comp('sin', abs, 'cos', ns=Namespace('math'))
        >>> f(10)
        0.7440230792707043

        In particular ``Namespace`` can be used to combine different sources.

        Note, that unless the ``ns`` argument is used,
        the ``comp`` is equivalent to the ``Operators`` chaining
        ::
            comp(F1, F2) == O(F1) * O(F2)

        Alternatively, in the dotted form, it may contain name of the package:
        ``numpy.sin``, ``toolbox.io.imread``.

        The simplest form describes composition when neither function
        arguments nor names are required:

        >>> f = O.comp('math.sin', min, ('numpy.linalg.norm', {'axis':0}))
        >>> print(f)
        O|sin|min|norm|
        >>> f([[1,2,3], [3,0,1]])
        0.9092974268256817

        :param stages: dict of functions definitions in the required order
        :param ns: Namespace with implementations of the functions
        :param skip: a collection of hashable values to skip if found in stages,
                    if None - use all
        :return:
        """
        from toolbox.utils.datatools import map_by_type

        ns = ns or Namespace()
        skip = set(as_list(skip))

        def normalize(seq):
            for x in seq:
                if isinstance(x, dict):
                    for items in x.items():
                        yield items
                elif not hasattr(x, '__len__') and x in skip:
                    continue
                else:
                    yield x

        comp_func = O()

        for func in normalize(stages):
            if isinstance(func, str):
                func = cls.Func(func)
            elif isinstance(func, tuple):
                fnc, *rest = func
                func = cls.Func(fnc, **map_by_type(rest, cls._func_arg_types))
            isinstance(func, cls.Func) and func.resolve(ns)
            comp_func *= func
        return comp_func

    def __init__(self, fnc: Callable | str | Func | tuple[Func, Func] | Operator = None,
                 *args, o_name: str = '', o_key: str = None, o_pos: int = None,
                 ns: Namespace = None, **kws):
        """
        Create operator from the function description.

        >>> f = O()   # Without arguments creates a bypass operator.
        >>> f(10) == 10
        True

        :param fnc: function performed by the operator
        :param args: its additional positional arguments
        :param o_name: name of the operator (uses fnc.__name__ unless fnc is lambda!)
        :param o_key: key name of the operand argument - if not at pos 0
        :param o_pos: position the operand argument - if not 0
        :param ns: namespace in case function is described as a str
        :param kws: its keyword arguments
        """
        if fnc is None:
            self._stages = tuple()
            return

        if isinstance(fnc, tuple):  # from stages, called from __mul__
            assert all(map(lambda f: isinstance(f, Operator.Func), fnc))
            self._stages = fnc
            return

        if isinstance(fnc, Operator):  # COPY CONSTRUCTOR
            assert not args and not kws
            if not o_name:
                self._stages = fnc._stages
                return
            # otherwise, encapsulate Operator as a function with new name

        if not isinstance(fnc, Operator.Func):
            fnc = Operator.Func(fnc, o_name, args, kws, inp_key=o_key)
        ns and fnc.resolve(ns)
        self._stages = (fnc,)

    @property
    def __name__(self):
        return '<'.join(self._names())

    def _names(self):
        return map(lambda f: f.name, self._stages)

    def __call__(self, x):
        #         print('Operator call')
        ret = x
        for f in reversed(self._stages):
            if f.inp_key:
                ret = f.fnc(*f.args, **({f.inp_key: ret} | f.kws))
            elif f.inp_pos:
                f.args[f.inp_pos] = ret
                ret = f.fnc(*f.args, **f.kws)
            else:
                ret = f.fnc(ret, *f.args, **f.kws)
        return ret

    def __mul__(self, other):
        if not isinstance(other, self.__class__):
            other = self.__class__(other)
        return self.__class__((*self._stages, *other._stages))

    def __rmul__(self, other):
        return O(other) * self

    def __imul__(self, other):
        if not isinstance(other, self.__class__):
            other = self.__class__(other)
        self._stages = (*self._stages, *other._stages)
        return self

    def __repr__(self):
        return f"O|{'|'.join(self._names())}|"


O = Operator
comp = O.comp


def express_to_lambda(expression, arg='_'):
    """Produces lambda function with a single argument input executing
    given python expression.

    :param arg: name of the input argument used in the exprerssion ('_')
    :returns: function(arg) - a function expecting keywords arguments
    Example:
    >>> func = express_to_lambda('3 < _ < 2')
    >>> func(4)
    False
    """
    assert arg in expression, f"{arg} must be a variable name in the expression!"
    code = compile(expression, '<string>', 'eval')

    def func(_):
        f"""Function: {expression}"""
        return eval(code, __builtins__, {arg: _})

    func.__qualname__ = f'`{expression}`'
    return func


def express_to_kw_func(expression: str):
    """
    Produces a function executing given python expression with variables
    passed into its namespace either as its keyword arguments or
    a single dict argument.

    Example:

    >>> func = express_to_kw_func('x < 2 or y > 3')
    >>> func(x=1, y=10)
    True
    >>> func(dict(x=1, y=10))
    True

    :param expression: string with python expression using variables names
                       to be passed to the resulting function as arguments
    :returns: function(**kws) or function(kws: dict)
                - a function with keywords arguments or one dict arguments
    """
    code = compile(expression, '<string>', 'eval')

    def func(ns=None, **kws):
        kws = kws if ns is None else {**ns, **kws}
        f"""Function: {expression}"""
        return eval(code, __builtins__, kws)

    func.__qualname__ = f'`{expression}`'
    return func
