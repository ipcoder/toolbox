from __future__ import annotations

import abc
import dataclasses
from typing import Any, Collection, Iterable, Dict

from toolbox.param import TBox
from toolbox.utils import logger
from toolbox.utils.cache import PersistCache
from toolbox.utils.events import Timer
from toolbox.engines.label_io import Labeled, DataC, DCType, LabeledType, LT

_log = logger('engines')

__all__ = ['AlgoEngine', 'EngineType', 'Internals']


class EngineMeta(abc.ABCMeta):
    """Meta class for creating Algorithms Engines.

    Main responsibilities:
        - Verification of consistency of the interface and sub-classsing
        - Verification that class name follows convention (check_name=True during subclassing)
        - Re-generation of Inputs and Ouputs classes to subclass DCtype
        - Generation of Task nested class with proper attributes
    """

    @staticmethod
    def parse_class_name(class_name: str, fail=True):
        """
        Extract informative parts from the engine's name, assuming the template:
        ::
            {name}[_]{kind}[[_]{pfm}][_v{ver}][_]Eng[ine]


        Here `{}` indicates parts to extract, `[]` - optional elements.

        Additional constrains:
         - `name` must start from Capital letter and include only letters and digits
         - `kind` must start from Capital letter and include only letters
         - `pfm` (if provided) the platform must be from: ``LT``, ``T`` (or ``Torch``), ``TF``
         - `ver` (if provided) must start from `v` or `V` followed by any string

        Separation between `name` and `kind` may be achieved only by first capital letter:
            `SenseDisp` -> `Sense`, `Disp`
        Multiple capital letters inside either of them leads to ambiguity:
            `EncDecDispPrior` -> `Enc`, `DecDispPrior` | `EncDecDisp`, `Prior`?
        In such cases use `_` as separator. (Not required for if includes sequential caps and digits)
            `Sense24sDispPrior`, `CRESDispPrior` -> are separated correctly.

        (Based on the regex: https://regex101.com/r/7kPfYB/1 )

        **Note!** `kind` field is returned in *snake_case* form.

        **Examples**
        ::

            SenseDispEng
            -> {name: Sense, kind: disp}

            EncDec_DispPrior_LTEng_vSplit_2
            -> {name: EncDec, kind: prior_disp, pfm: LT, version}

        :param class_name: name to parse
        :param fail: if ``True`` on failed parsing raise ``NameError``, otherwise return ``{}``
        :return: dict with extracted fields, or ``{}``
        """
        import re
        from toolbox.utils.strings import camel_to_snake

        # https://regex101.com/r/7kPfYB/1
        name_regex = re.compile(  # name regex digests sequential Caps and digits (Slack, CoPoSS, CoPo34)
            "(?P<name>[A-Z]+[A-Za-z0-9]+?)_?"  # for other multiple CamelCase segments use _ sep (CoPo)
            "(?P<kind>(?:[A-Z][a-z]+)+?)"  # FewParts (not FewPartS)
            "(?:_?(?P<pfm>LT|T|Torch|TF))?"
            "(?:_[vV](?P<ver>.+))?"
            "_?Eng(?:ine)?"
        )

        if m := name_regex.fullmatch(class_name):
            fields = m.groupdict()
            field_transform = lambda f, func: {f: func(fields[f])}
            return (fields
                    | field_transform('kind', camel_to_snake)
                    | field_transform('pfm', lambda pfm: 'T' if pfm == 'Torch' else pfm))
        if fail:
            template = "{name}[_]{kind}[[_]{pfm}][_v{ver}][_]Eng[ine]"
            raise NameError(f"Engine's {class_name=} does not match {template=}."
                            f"`AlgoEngine` supports subclassing flag `check_name`")
        return {}

    @classmethod
    def check_class_name(mcs, name, *, kind=None, pfm=None, check_name=True):
        """Check if class name metches AlgoEngine naming template.

        :raise ``NameError`` if no match and `fail` is True
        :return: dict of extracted fields or {}
        """
        attrs = dict(kind=kind, pfm=pfm)
        if not check_name:
            return {}
        parsed = mcs.parse_class_name(name, fail=False)
        if not parsed:
            return {}

        for k, v in attrs.items():
            if parsed[k] != v:
                raise NameError(f"Invalid engine's {name=}: parsed {k} {parsed[k]} != {v}")
        return parsed

    @property
    def kind(cls):
        return cls.__kind__

    @property
    def is_abstract(cls):
        return bool(cls.__abstractmethods__)

    @staticmethod
    def _func_like(f, name, Inputs):
        from types import FunctionType
        # first make a copy of the function
        nf = FunctionType(f.__code__, f.__globals__, f.__name__,
                          f.__defaults__, f.__closure__)
        nf.__kwdefaults__ = f.__kwdefaults__
        nf.__annotations__ = f.__annotations__

        # now alter it
        nf.__doc__ = f.__doc__.format(
            Algorithm=name.split('Eng')[0],
            Inputs=Inputs.__doc__)
        nf.__annotations__['inputs'] = Inputs
        return nf

    @staticmethod
    def _update_attrs(name, attrs, bases):
        """If new class is derived from AlgoEngine and defines its interface
        then rebuild its attributes to maintain interface standards.
        """
        # find in the bases a sub-class of AlgoEngine - a super class for this one
        super_engines = [b for b in bases if issubclass(b, AlgoEngine)]
        if not super_engines:
            return False  # we are creating the AlgoEngine class itself
        if len(super_engines) > 1:
            raise NotImplementedError("Multiple Inheritance of Engines")

        # construct Inputs and Outputs classes
        ios = {
            field: DCType(io_cls) if hasattr(io_cls, '__annotations__') else io_cls
            for field in ['Inputs', 'Outputs'] if (io_cls := attrs.get(field, None))
        }
        if not ios:  # none of IOs is defined
            return False  # Superclass of some abstact AlgoEngine wo IO
        if len(ios) < 2:
            raise NotImplementedError("Engine without BOTH Inputs AND Outputs")
        Inputs, Outputs = ios.values()

        class Task(AlgoEngine.Task):
            # namespaces passed for postponed annotations evaluations (3.9)
            inputs: Inputs
            outputs: dict = DCType.field(default=None, init=False)
            labels: dict = None

        attrs.update(ios | dict(
            Task=DCType(Task, globalns=globals(), localns=locals()),
            __call__=EngineMeta._func_like(AlgoEngine.__call__, name, Inputs)
        ))
        return True

    def __new__(mcs, name, bases, attrs: Dict[str, Any],
                *, kind: str | None = None, pfm: str | None = None,
                check_name=True, **kwargs):
        io_defined = EngineMeta._update_attrs(name, attrs, bases)
        cls = super().__new__(mcs, name, bases, attrs, **kwargs)

        pfm = cls._set_valid_pfm(pfm)
        EngineMeta._validate_kind(cls, kind, io_defined)
        cls._check_name = check_name
        EngineMeta.check_class_name(name, kind=cls.kind, pfm=pfm, check_name=check_name)
        return cls

    def _set_valid_pfm(cls, pfm):
        if pfm != cls.__pfm__:
            if cls.__pfm__ is None:
                cls.__pfm__ = pfm
            elif pfm is not None:
                raise TypeError(f"Can't override {cls.__name__}.__pfm__={cls.__pfm__} with {pfm}")
        return cls.__pfm__

    @staticmethod
    def _validate_kind(cls, kind: str, io_defined: bool):
        from inspect import isabstract
        old_kind = getattr(cls, '__kind__', None)
        kind_defined = bool(kind) + bool(io_defined)
        complete_cls = None if isabstract(cls) else cls

        if kind_defined == 2:  # new kind is correctly defined in cls
            if old_kind:  # stepping over already defined kind
                raise AttributeError(f"Subclassing {cls} tries to redefine engine kind!")
            cls.__kind__ = kind  # mark class as "interface"
        elif kind_defined == 0:  # a new kind is not being defined, so register new class only
            if complete_cls and old_kind:  # if kind is defined in a super-class look for it
                for c in cls.__mro__:  # in mro until first (from the top) is found
                    if issubclass(c, AlgoEngine):
                        if c.__kind__:
                            kind_cls = c  # kind_cls always has kind
                        else:
                            break  # since we check `if old_kind` it MUST be found!
        elif kind_defined == 1:  # invalid definition of kind
            raise NotImplementedError(
                f"When introducing a new Engine kind class {cls} both must be provided:\n"
                f"  attributes 'Inputs', 'Outputs' defining io types\n"
                f"  and the class 'kind' argument")

    def __repr__(cls):
        return f"{cls.__name__}⋮{cls.kind}"


class AlgoEngine(metaclass=EngineMeta, check_name=False):
    """ See `engines.md`"""

    kind: str  # property define in EngineMeta

    __kind__ = None
    __ver__ = '0.0'
    __pfm__ = None

    # ToDo: Make engine config a Pydantic Model
    Config = dict  # type of config

    # those classes MUST be overridden by the inheritance
    class Inputs(DataC):
        pass  # must be overridden

    class Outputs(DataC):
        pass  # must be overridden

    @dataclasses.dataclass
    class Task:
        inputs: AlgoEngine.Inputs
        labels: dict
        outputs: dict

        def set_outputs(self, main_outputs, internal_outputs):
            self.outputs = dict(main_outputs) | (internal_outputs or {})

    def __del__(self):
        self.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()

    def process_iter(self, iterator: Iterable[Task]) -> Iterable[Task]:
        """Given iterable over tasks creates generator of processed tasks
        producing them in same order, with task.output attribute updated \
        by the with algorithmic results.

        **Note**
        Tasks are not pulled from the iterator just until that is required
        by the processing (meaning on-demand allocation of tasks resources)

        However, take into account that if engine supports batch processing,
        the whole batch of tasks is allocated before processing starts,
        but no more than O(self.batch_size) tasks resources at any time.

        :param iterator: collection or generator of input tasks.
        :param internal_outs: If True create also internal outputs
        :return: generator of processed tasks.
        """

        def process_buffer():
            self._process_batch(buffer)
            for task in buffer:
                # assert isinstance(task.outputs, self.Outputs)
                yield task

        self._activate()
        if batch_size := self.batch_size:  # batch processing mode
            buffer = []
            for task in iterator:
                buffer.append(task)
                if len(buffer) == batch_size:
                    yield from process_buffer()
                    buffer = []
            if buffer:  # remaining tasks < batch_size
                yield from process_buffer()
        else:  # regular task by task processing generator
            for task in iterator:
                task.set_outputs(*self.__call__(task.inputs, internal_outs=True))
                yield task

    @abc.abstractmethod
    def _process(self, inputs: Inputs) -> tuple | dict | Outputs:
        """ Run _existing_ engine on a single set of inputs
        :param inputs: dictionary {input_name: input_data}
        :return: self.Outputs object
        """
        raise NotImplementedError

    def _process_batch(self, tasks: Collection[AlgoEngine.Task] = None) -> int:
        """
        Override this function to support batch processing capabilities
        of the engine.

        AlgoEngine design requires encapsulation of two batch-related
        services in this single function, by implementation two
        distinct modes of its invocation:

        1. **Query batching support** when calling without arguments:

           ``eng._process_batch()`` must return ``batch_size`` supported
           by the engine WITHOUT assuming ``self._state`` is initialized.
           Possible ways to access it:
            - defined inside this function or class attribute if its constant
            - from engine's ``self.config`` if its engine-configurable

           *Default implementation returns 0 to indicate no batch support.*

        2. Process given sequence of tasks as in a single batch
           (beneficial for optimizations in some cases).

           Number of given tasks MUST be less or equal to this number,
           or raise ``BufferError``.

           *Default implementation raises ``NonImplemented`` in this mode.*

        **Note:** *This function is not designed to be called by the class users!*

        :param tasks: Collection of tasks to process in one batch
               MUST BE len(tasks) <= self._process_batch()
        :return: batch_size OR processed tasks_num
        """
        if tasks is None:
            return 0
        raise NotImplementedError

    @property
    def batch_size(self):
        """Batch processing capabilities of the engine 0 - not supported"""
        try:
            return self._process_batch()
        except AttributeError:
            print("_process_batch() without arguments must not access self._state!")

    @classmethod
    def _init_state(cls, cfg: dict = None):
        """Initialize internal engine's state from its configuration parameters,
        (presumably resource-heavy operation requiring control)

        Invoked once (depending on the instance initialization ``delay`` argument):
            - either during the initialization
            - or just before the first processing

        Returned state is stored in ``self._state`` attribute and may be accessed
        by the ``_process()`` or other methods in derived classes.

        Please notice that the **engine state** is different from an *algorithm state*
        - it influence HOW the execution happens, but not its function or results.

        The content of the state is specific to an engine kind, so this method
        must be overridden unless no special engine state management is required.

        :param cfg: dict-like engine configuration parameters
        :return: state

        """
        return None

    def _destroy_state(self):
        """Override this function to properly release allocated resources"""
        self._state = None

    def _activate(self):
        """Initialize the state if its not initialized yet!"""
        if self._state is None:
            cls = self.__class__
            with Timer(f'{cls.__name__} engine initialized in', _log.debug):
                self._state = self._init_state(self.config)

        self.internals.configure(validate=True, keep_state=False)  # already activated

    @staticmethod
    def default_config() -> dict:
        """Return a COPY of default configuration of the algorithm: a
        nested dict-like structure, an instance of `cls`::inu.param.TBox

        Default configuration also defines COMPLETE template of the
        configuration structure, where all the nodes are defined.
        Its necessary for tracking configuration-related differences
        in algorithms results.
        `__init__` validates passed `config` against this template
        to ensure sync between actual and declared structures.
        """
        return {}

    @property
    def ver(self):
        return self.__ver__

    @classmethod
    def _auto_name(cls, name=None):
        if not name:
            import re
            name = cls.__name__
            if m := re.fullmatch(r'(\w*)_?eng(?:ine)?_?(\w*)', name, flags=re.IGNORECASE):
                name = ''.join(m.groups()) or name
        return name

    def __repr__(self):
        return f"<{self.__class__.__qualname__} v{self.ver}> {self.name} [{self.hash}]"

    @classmethod
    def full_config(cls, config: dict = None, **kwargs) -> TBox:
        """Create full config from a partial one.
        Also possible to provide keys as keyword arguments, where
        names may be a parts of valid nodes names unless that ambiguous.
        """
        config = TBox(config or {})
        full_config = TBox(cls.default_config())

        for k, v in kwargs.items():
            if key := full_config.find_key(k, multi=False):
                config.merge_update({key: v})

        wrong = set(config.keys()).difference(full_config.keys())
        if wrong:
            raise KeyError(f"Wrong configuration keys in {cls.__qualname__}:\n{wrong}")

        full_config.merge_update(config)
        return full_config

    def _calc_out_name(self, inputs: DataC, ext='out'):
        """Return unique file path segment for the calculation output"""
        calc_id = '_'.join([self.name, self.ver, self.hash, inputs.hash(6)])
        return f"{self.__kind__}/{calc_id}.{ext}"

    def _internal_traces(self, keep_state=True) -> list[str]:
        """
        Return list of names of supported internal outputs
        (dot-separated strings reflecting their hierarchy).

        Override this methos ONLY for sub-classes supporting internal outputs

        If implementation requires it may self._activate() algo state to retreave the available traces.

        Argument `keep_state` instucts to return the algorithm into its original state
        if it was not activated to have control over resources,
        like GPU memory, consumed by multiple algorithms.

        **Notice**:
          1. If implementation requires algorithm's configuration, its accessible as `self.config`.

        :param keep_state: if `True` requires that after this function call
                engine releases system resources **specifically allocated during this call**.
        :return: list of available traces
        """
        raise NotImplementedError(f"Internal outputs not supported by {type(self)}")

    @classmethod
    def supports_internals(cls) -> bool:
        """Fast check if class supports internal outputs"""
        return (cls is not AlgoEngine and
                cls._internal_traces is not AlgoEngine._internal_traces)

    # ---------------- Basic Functionality:
    def __init__(self, config: dict = None, *, name=None,
                 delay=True, caching=False, internal_outs: dict = None):
        """Creates disparity engine
        :param config: could be configuration object or
                        path to file with engine configuration
        :param name: Application-level label for the implemented algorithm.
                  If not provided deduced from the class engine name.
        :param delay: delay initialization of the engine until first use
                      to avoid expensive creation of unused objects
        :param internal_outs: Internal outputs to return.
                              Must be in dict form: { alias: {output_name: '', labels: {} }}
                              alias is the new name
                              output_name is the original name
                              labels are a dict with labels.
        """
        self._cacher = None
        self._state = None

        if caching:  # Currently disabled, to replace current evals caching
            raise NotImplementedError("Currently insensitive to processor's code change!")
            self._cacher = PersistCache(  # TODO: remove pre and post when DataC is pickle-able
                self._core_processing, mode='rw', namer=self._calc_out_name,
                pre_save=self.Outputs.to_dict, post_load=lambda dct: self.Outputs(**dct))
            self._core_processing = self._cacher._cached_call

        cls = self.__class__
        self.name = cls._auto_name(name)  # FixMe: use single naming mechanism
        self.config = cls.full_config(config)
        self.hash = TBox(self.hashable_config(self.config)).hash_str(4)
        self.internals = Internals(internal_outs or {}, self)

        if not delay:
            self._activate()
        assert isinstance(batch := self.batch_size, int) and batch >= 0

    @classmethod
    def hashable_config(cls, config):
        """
        creates unique config for hash.
        """
        return config

    def _core_processing(self, inputs):
        self._activate()

        outs, internal_outs = self._process(inputs=inputs)

        if not isinstance(outs, Labeled):  # Skip packing for processed Outputs
            if isinstance(outs, tuple):
                outs = self.Outputs(*outs)
            elif isinstance(outs, dict):
                out_keys = set(outs)
                if out_keys.issubset(self.Outputs.defined_fields):
                    outs = self.Outputs(**outs)
                else:  # if its not dict of fields, than it must be labeled first field
                    outs = self.Outputs(outs)
            elif not isinstance(outs, self.Outputs):
                outs = self.Outputs(outs)

        return outs, internal_outs

    def __call__(self, inputs: AlgoEngine.Inputs = None, *,
                 internal_outs=False, **kws) -> AlgoEngine.Outputs | tuple[AlgoEngine.Outputs, dict]:
        """ Run instance of the {Algorithm} engine on a given inputs
            with optional updated parameters.
            :param inputs: of Inputs type or DataTable to query from,
                           or None to pass inputs through kws args
            :param internal_outs: Whether to include also the internal outputs.
            :param kws: pass fields of the inputs as kw args
            :return: dict with named outputs
        """
        # FixMe: remove
        if inputs is None:
            inputs = self.Inputs(**kws)
        elif not isinstance(inputs, self.Inputs):
            # support also simple types: Inputs = float
            inputs = getattr(self.Inputs, 'from_table', self.Inputs)(inputs)

        outs, internals = self._core_processing(inputs=inputs)
        if internal_outs:
            return outs, internals
        else:
            return outs

    def clear(self):
        """Clear allocated resources. Override for engine specific staff"""
        self._state = None
        # self.internals.clear()


class Internals:
    class _OutInfo:
        def __init__(self, alias: str, outp: dict):
            self.alias = alias
            self.labeled_type = LT(**outp['type'], internal=outp['trace'])

        def __repr__(self):
            return f"{self.alias}: {self.labeled_type['internal']}"

    @classmethod
    def _cfg_to_info(cls, config: dict) -> dict[str, _OutInfo]:
        """
        Convert config from its original form:
        ::
            {'alias' : dict(trace='trace', type=type_labels)}

        into internal form: dict of OutInfo objects:
        ::
            {'trace': _OutInfo(
                        alias='alias'
                        labeled_type=LT(type_labels | {'internal': 'dotted_trace'})
                    )}

        (LT - LabeledType)

        :param config: the initial config form
        :return: internal form
        """

        return {
            outp['trace']: cls._OutInfo(alias, outp)
            for alias, outp in config.items()
        } if config else {}

    def __init__(self, config: dict, eng: AlgoEngine):
        """
        Partial initialization of internals. Full validation is postponed
        until the first call to `_complete_init` (not by user!)

        :param config:
        :param eng:
        """

        if config and not eng.supports_internals():
            raise ValueError(f"Class {eng} not supporting internals recieves their config!")
        self._eng = eng  # related Engine instance producing those internal outputs
        self._active = self._cfg_to_info(config)
        self.available: list[str] = None  # list of all available internal outputs provided by the engine

    def __bool__(self):
        """Checks `True` if any non-trivial configuartion is set.
        Does not assumes its validated!
        """
        return bool(self._active)

    @property
    def validated(self):
        """Check if current configuration has been validated."""
        return bool(self.available is not None)

    def _init_available(self, keep_state):
        """Queries engine for available traces and sets self.available to the resulted list or []"""
        if self.available is None:
            err_msg = None
            try:
                self.available = self._eng._internal_traces(keep_state=keep_state)
            except NotImplementedError as ex:
                err_msg = f"Engine {type(self._eng)} does not support internal traces"

            if self.available is None:
                err_msg = f'Invalid implementation {self._eng._internal_traces} returns None!'
                self.available = []

            if not isinstance(self.available, Collection):
                raise TypeError(f"Invalid implementation  {self._eng._internal_traces} "
                                f"returns {type(self.available)}")

            if not all(isinstance(_, str) for _ in self.available):
                raise TypeError(f"Invalid implementation  {self._eng._internal_traces} "
                                f"must return collection of strings!")

    def configure(self, config=None, *, validate=True, keep_state=True, fail=False):
        """
        May be uset either either
            - to set (and validate) a new configuration, or
            - to validate the current one (`config=None`)

        Update active configuration and validates immediately if either:
            - `validate=True`, or
            -  `self.availabe` has been already aquired

        If validation is failed raise if `fail=True` otherwise return false

        In some cases validation of configuration may require activation of the engine,
        and allocate significant system resources.

        Argument `keep_state=True` ensure that this call returns engine to the initial allocation state.
        Notice, that
            - that during the call the allocation still can accoure.
            - such allocation may happen at most once during instance lifetime.

        :param config: new configuration or `None` (to validate the current)
        :param validate: validate new configuration or just set it.
        :param keep_state: request to return the engine into its original allocation state
        :return: `True` if valid config, `False` - invalid, `None` - not validated
        """

        def validate_current() -> bool:
            """Validate current config against engine's capabilities (queries the engine if necessary)."""
            if self._active and not self.validated:
                self._init_available(keep_state=keep_state)
                if diff := set(_map).difference(self.available):
                    msg = f"Unknown traces for {self._eng}:\n{diff}"
                    if not fail:
                        _log.warning(msg)
                        return False
                    raise ValueError(msg)
            return True

        if config is None:  # not a new config - self validation
            if validate or self.validated:
                return validate_current()
            return None  # not validated

        # new config - pass through usual construction - validation process:
        new = type(self)(config, self._eng)
        new.available = self.available  # use available to avoid their re-calculation
        if new.configure(validate=validate, keep_state=keep_state, fail=fail):
            self._active = new._active
            self.available = new.available
            return True

        # if reached this point - either invalid or not validated
        return False if self.validated else None

    def __repr__(self):
        s = f"{self._eng}.internals"
        if not self._eng.supports_internals():
            return s + '(∅)'
        s += '(?)' if self.available is None else f'({len(self.available)})'
        s + f' {len(self._active)} active traces'
        if active := '\n\t'.join(map(str, self._active.values())):
            return f"{s}:\n\t{active}"
        return s

    @property
    def active_traces(self) -> list:
        """List active traces (full names)"""
        return list(self._active)

    @property
    def active_types(self) -> dict[str, LabeledType]:
        """Active outputs: {alias: labeled_type}"""
        return {_.alias: _.labeled_type for _ in self._active.values()}

    def find_available(self, *keys: str) -> list[str]:
        """
        From the available traces find those who contain ALL the given keys
        (wo keys - all the available).

        :return: subset of availabe traces matching the query.
        """
        self._init_available(keep_state=True)
        return list(filter(lambda _: all(s in _ for s in keys), self.available))

    def cast_outputs(self, traced_outputs: dict[str, Any]):
        """
        Casting the internal outputs to the type specified in the config.
        For labeled type, Labeling the outputs
        :param traced_outputs: key - trace of an output,
        :return: Internal outputs in final format ( when labeled type - with labels)
        """
        return {(info := self._active[trace]).alias: info.labeled_type(data)
                for trace, data in (traced_outputs or {}).items()}
