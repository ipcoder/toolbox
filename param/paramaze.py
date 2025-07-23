import inspect
from contextlib import contextmanager
from enum import IntEnum, unique
from typing import Union, Callable, Optional, Dict, Sequence

from toolbox.utils import as_list, fnctools as fnt, wrap
from .tbox import TBox

Rule = Union[Callable, str, None]


def sub(*args, **kws):
    func_defs = {f.paramaze.alias: f.paramaze.defaults for f in args}
    return TBox(frozen_box=True, **func_defs, **kws)


@unique
class Validate(IntEnum):
    OFF = 0         # no validation
    REPORT = 1      # report about errors but not raise
    RAISE = 2       # validate, but not report - only raise on errors
    VERBOSE = 3     # raise on errors

    @classmethod
    @contextmanager
    def level(cls, vld_level: Union['Validate', str]):
        """
        Create validation context to execute code with specific validation level
        and restore the current one on exit
        :param vld_level: as name str or `Validate` object
        """
        if isinstance(vld_level, str):
            vld = Validate[vld_level]
        old_vld = FuncPar.validation(vld)
        try:
            yield vld
        finally:
            FuncPar.validation(old_vld)


class FuncPar:
    """
    Class cls::`FuncPar` treats arguments of a function as dictionary of
    parameters with additional properties tools, like
    - defaults
    - validation conditions
    - automatic or explicit validation
    - automatic extraction of parameters found arbitrary processing flows
    - and more
    """

    _validate = Validate.RAISE

    @classmethod
    def validation(cls, state: Optional[Validate] = None) -> Validate:
        """Set new parameters validation level and return the previous one"""
        if state is None:
            return cls._validate
        else:
            old, cls._validate = cls._validate, Validate(state)
            return old

    class Tracker:
        """Helper sub-class of cls::`FuncPar` to track nested calls of
        parametrized functions."""

        def __init__(self):
            self._track = False
            self._frame = None
            self._depth = None
            self._record = {}
            self._stack = []

        reset = __init__

        def start(self):
            """Start tracking of the execution flow for all nested calls"""
            if self._frame is None:
                st = inspect.stack()
                self._frame = st[1].frame
                self._depth = len(st)
            self._track = True

        def stop(self):
            """Stop tracking - but keep collected record (unlike `reset`)"""
            self._track = False

        def restart(self):
            """Clear context and restart the tracking from current frame"""
            self.__init__()
            self.start()

        def report(self) -> Dict[str, 'FuncPar']:
            """Produce dictionary of all calls to functions with `FuncPar`
            with keys as dot separated names of nested called functions.
            """
            return self._record.copy()

        def __bool__(self):
            return self._track

        def track_call(self, arg_par: 'FuncPar'):
            """Track specific call provided its `FuncPar` context"""
            func_name = arg_par.function.__qualname__
            self._stack.append(func_name)
            if self._track:
                st = inspect.stack()
                depth = len(st) - self._depth + 1
                if depth > 0 and st[depth].frame is self._frame:
                    self._record['.'.join(self._stack)] = arg_par

    track = Tracker()

    def __init__(self, func: Callable, *, alias: str = None,
                 validate: Union[Rule, Sequence[Rule]] = None,
                 directives: Dict[str, Union[Rule]]):
        """Construct `FuncPar` instance for specific function using
        provided validation rules and condition directives.

        :param func: function object to associate with
        :param alias: optional (short) version of name to associate with
                      the parameters structure instead of the function name
        :param validate: single or sequence of validation rules in form of
            a python expression using variables named by the parameters
            or callable accepting keywords arguments or those parameters.
        :param directives: dictionary per parameter of validation conditions
            in form of expression using single variable denoted as "{}" or
            callable with a single argument to receive the parameter value.

        Examples:
            Associates conditions with function arguments.
            >>> def func(x=10, y=20): pass
            >>> p_func = FuncPar(func, validate='x < y',\
                            directives={'y': "{} > 0"}).function
            >>> assert p_func(20, 10)    # will fail on x < y
            Traceback (most recent call last):
                ...
            AssertionError
            >>> assert p_func(-10, -2)   # will fail on y > 0
        """
        from functools import wraps

        self._name = func.__qualname__
        self._params = TBox()    # arguments which are parameters with defaults
        self._cond = {}          # parameters with assigned conditions
        self._alias = alias or self._name

        # First parse all the function's arguments to determine:
        # - which have defaults - candidates to be parameters
        # - which are mentioned in the special directives
        all_defaults = {}  # default values for ALL the arguments
        invalid_references = set(directives)   # ensure directives refer to existing arguments

        pos_kws_start = 0  # location of first not position-only argument
        pos_kws = []  # names (ordered) of all the pos_or_kw arguments
        try:
            for i, (name, arg) in enumerate(inspect.signature(func).parameters.items()):
                if arg.kind is arg.VAR_POSITIONAL:
                    raise NotImplementedError("Variadic positional arguments")
                if arg.kind in (arg.POSITIONAL_ONLY, arg.VAR_KEYWORD):
                    continue  # positional and variadic arguments are not parametrized!

                # other kinds with provided default values are be turned in parameters
                # unless explicitly excluded in directives
                if arg.kind is arg.POSITIONAL_OR_KEYWORD:  # keep track on ambiguous arguments
                    if not pos_kws:                  # to separate position-only when called
                        pos_kws_start = i
                    pos_kws.append(name)

                if arg.default is not arg.empty:
                    all_defaults[name] = arg.default  # to implement defaults logistics
                    self._params[name] = arg.default        # may be exclude below
                    if name in directives:  # either provides condition or excludes from toolbox.params
                        invalid_references.remove(name)     # remove valid names from the black list
                        cond = directives[name]
                        if cond:
                            if isinstance(cond, str):
                                cond = fnt.express_to_lambda(cond.format(name), name)
                            self._cond[name] = cond
                        else:  # exclude from params if explicitly set to False or None
                            self._params.pop(name)

            if invalid_references:
                raise ReferenceError(f"Conditions defined for invalid arguments: {invalid_references}")

            self._validators = [v if callable(v) else fnt.express_to_kw_func(v) for v in
                                (as_list(validate) if validate else [])]
        except SyntaxError as e:
            e.msg = f"Invalid validation condition for function <{self._name}>"
            raise e

        def separate_kw_args(args):
            """Separate variadic positional argument into position only and pos_or_kw."""
            found_pos_kws = len(args) - pos_kws_start
            if found_pos_kws > len(pos_kws):
                raise SyntaxError(f"Positional args to <{self._name}> exceed  {len(pos_kws)}!")
            if found_pos_kws < 0:
                raise SyntaxError(f"Missing {-found_pos_kws} positional args in <{self._name}>!")
            # create kw form of the args passed by position
            silent_kw_arg = dict(zip(pos_kws[:found_pos_kws], args[-found_pos_kws:]))
            return silent_kw_arg

        def filter_keys_in(d, seq, *, keep_in=True):
            """Filter dictionary by keys belonging or not to given sequence"""
            seq = set(seq)
            condition = (lambda x: x in seq) if keep_in else (lambda x: x not in seq)
            return {k: v for k, v in d.items() if condition(k)}

        @wraps(func)
        def pz_func(*args, _par: dict = None, **kws):
            # Responsibilities of the wrapper:
            # 1. Optional validation of the parameters values
            # 2. Inform the ArgPar.tracker about the call to collect flow information.
            # 3. Automatically fetch parameters from repository through special argument _par
            # 4. (No Harm) Maintain same syntax as the original function in spite of different signature
            #    - make variadic argument behave like named in the original function
            #    - support the additional (optional) _par argument
            silent_kws = separate_kw_args(args)

            # optionally keyword parameters may be passed inside special _par
            if _par:  # automatic fetch of relevant parameters from the repo
                # dict-like argument as item under the key `func.name` or `func.alias`
                # which `dict`-like value would contain any subset of them.
                _par = getattr(_par, self._name, getattr(_par, self._alias, {}))
                # make sure no arguments are not passed twice from different sources
                if self.validation() > Validate.RAISE:
                    if twice := set(_par).intersection(set(silent_kws).union(kws)):
                        from warnings import warn
                        warn(f"Double {twice} in {self._name}: as args and (ignored) in _par")

                kws = {**filter_keys_in(_par, silent_kws, keep_in=False), **kws}  # _par-silent-kws+kws

            self.validate({**silent_kws, **kws})  # add silent to validation
            FuncPar.track.track_call(self)
            return func(*args, **kws)

        self._func = pz_func
        self.__call__ = pz_func

        setattr(pz_func, 'paramaze', self)
        setattr(pz_func, 'defaults', self.defaults)
        setattr(pz_func, 'validate', self.validate)

    def __repr__(self):
        return f'Params in function `{self.name}`:\n' \
               f'{self._params.to_yaml()}'

    @property
    def function(self):
        return self._func

    @property
    def name(self):
        return self._name

    @property
    def alias(self):
        return self._alias

    @property
    def defaults(self):
        """Return editable copy of the default parameters"""
        return TBox(self._params, frozen_box=False)

    def validate(self, par, throw=True):
        vld = self.validation()
        if vld is Validate.OFF:
            return True
        if vld is Validate.VERBOSE:
            print(f'Validating {self.name} parameters: {tuple(par)}...', end='')

        def try_cond(context, cond, *args, **kws):
            try:
                return cond(*args, **kws)
            except Exception as e:
                raise SyntaxError("Invalid validation condition for function "
                                  f"<{self._func.__qualname__}>\n"
                                  f"{context} {cond.__qualname__}\n{e}")

        failed_pars = [f"{k}={v} fails condition: {self._cond[k].__qualname__}"
                       for k, v in par.items() if k in self._cond and
                       not try_cond(f"for parameter {k}", self._cond[k], v)]

        merged_par = {**self._params, **par}
        failed_rules = [rule.__qualname__ for rule in self._validators if
                        not try_cond("in validation rule", rule, **merged_par)]

        msg = '\n\t'.join(['Invalid Parameters:', *failed_pars]) + '\n' if failed_pars else ''
        msg += '\n\t'.join(['Failed rules:', *failed_rules]) if failed_rules else ''
        if msg:
            msg = f'Failed parameters validation for <{self.name}>:\n{msg}'
            vld is Validate.VERBOSE and print()  # additional line in verbose case
            print(msg, repr(TBox(par)))
            if throw and vld is not Validate.REPORT:
                raise ValueError(msg)
            return False
        if vld is not Validate.RAISE:
            print('OK!')
        return True


@wrap.double_wrap
def paramaze(func, _alias: str = None,
             _validate: Union[Rule, Sequence[Rule]] = None,
             **kws: Rule):
    """
    Decorator wrapping function to associate it with `FuncPar` and
    provide all the related functionality: arguments as parameters with
    defaults, validation conditions, automatic collection of flow parameters,...

    By default all keyword arguments with defaults are considered parameters.
    Decorator mau be used with or without arguments, then no conditions or
    alias is associated with the function, but parameters collections is
    still available.

    :param func: function to wrap
    :param _alias: optional (short) version of name to associate with
                  the parameters structure instead of the function name
    :param _validate: single or sequence of validation rules in form of
        a python expression using variables named by the parameters
        or callable accepting keywords arguments or those parameters.
    :param kws: dictionary per parameter of validation conditions
        in form of expression using single variable denoted as "{}" or
        callable with a single argument to receive the parameter value.

    :return: wrapped function

    Example:

    >>> @paramaze
    >>> def func(x=10, y=20): pass
    >>> assert func.defaults == {'x': 10, 'y': 20}

    >>> @paramaze(y="{} > 0",
    >>>           _validate_type=['x < y', 'x**2 + y**2 < 100'])
    >>> def func(x=10, y=20): pass
    >>> assert func.parz.validate(x=2, y=3)
    """
    return FuncPar(func, alias=_alias, validate=_validate, directives=kws).function
