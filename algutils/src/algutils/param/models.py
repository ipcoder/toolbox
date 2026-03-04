from __future__ import annotations

import re
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Union, Type, Callable, List, Literal, Iterable

import pydantic as pyd

from ..io.format import FileFormat, CT, FormatHandler, MetaFormat
from .. import logger, as_list, as_iter
from .fixed_pydantic_yaml import YamlModelMixin
from .tbox import TBox

PathT = Union[Path, str]

Integers = Union[int, list[int]]
Strings = Union[str, list[str]]
Scalar = Union[float, bool, str]
VarName = pyd.constr(regex=r'^[A-Za-z]\w*$')
Regex = pyd.constr(regex=r'.*[\{\}\+\*\?]+.*')
NumArgs = dict[VarName, Union[float, int, bool]]
ScalarArgs = dict[VarName, Scalar]
Desc = lambda s: pyd.Field(default=None, description=s, required=False)

_log = logger('models', level='WARNING')


def model_arguments(*, exclude: Union[str, List[str]] = None, config=None,
                    **pydantic_kwargs):
    """
    Pydantic validate_arguments wrapper.

    For more information about pydantic.validate_arguments check:
    https://docs.pydantic.dev/1.10/usage/validation_decorator/

    Adds exclusion functionality that allows the user to exclude fields from the model.
    The excluded fields are passed to 'exclude' argument.
    In addition to user defined keys, pydantic.validate arguments add a key as well.
    This key is added to a list containing it along 'args' and 'kwargs'.

    Excluded fields types, are generally non-serializable types, such as np.ndarray.
    In addition, excluded keys might be the ones with no validator possibly defined.

    The remaining fields are being casted and validated using their type hint using pydantic.create_model
    For more information about pydantic.create_model check:
    https://docs.pydantic.dev/1.10/usage/models/#dynamic-model-creation

    The function flow can be summarized as ({step:description}):
    1. CAST : Casting user defined keys to a set.
    2. VALIDATE_ARGS : Running pydantic.validate_arguments on ALL keys which results on pydantic object.
    3. ISSUBSET : Check whether the arguments asked are actually the pydantic object model fields.
    4. USER ARGS REMOVAL : User exclude arguments removal from pydantic object fields.
    5. DEFAULTS REMOVAL : Default arguments removal, if so exists.
    6. MODEL CREATION : Pydantic Model creation from the remaining fields.
    7. MODEL ASSIGNMENT : Assignment of the new model to the pydantic object.
    8. RETURN : Returns the pydantic object with the newly created model assigned as 'model' attribute.

    Usage example of the decorator:
    >>> @model_arguments(exclude=['exclude_arg1', 'exclude_arg2'])
    ... def f(exclude_arg1, exclude_arg2, arg3, *args, **kwargs)
    The decorator will create a pydantic model without 'exclude_arg1', 'exclude_arg2'.

    :param exclude: A set-friendly container of the excluded arguments.
    :param pydantic_kwargs: pydantic.validate_arguments acceptable kwargs
    :return: pydantic object with appropriate model attached.
    """
    from ..datatools import issubset_report
    config = config or {}
    if exclude:
        config['extra'] = 'allow'
    pydantic_kwargs['config'] = config

    def dec(func: Callable):
        _exclude = set(exclude or [])
        exclude_defaults = ['v__duplicate_kwargs', 'args', 'kwargs']
        # TODO add type hint here
        pydantic_obj = pyd.validate_arguments(func, **pydantic_kwargs)
        model_cls = pydantic_obj.model
        issubset_report(_exclude, model_cls.__fields__, on_diff=True)

        for excl in _exclude:
            model_cls.__fields__.pop(excl)

        for excl in exclude_defaults:
            model_cls.__fields__.pop(excl, None)

        thin = pyd.create_model(f'{model_cls}', __base__=model_cls)
        pydantic_obj.model = thin
        return pydantic_obj

    return dec


def _template_missing_defaults(model: type[pyd.BaseModel], _rep_models=None):
    from ..codetools import NamedObj
    _fixing = NamedObj('FixInProcess')
    if _rep_models is None:
        _rep_models = {}

    def convert(field):
        t = field.type_
        if field.required:
            if isinstance(t, type) and issubclass(t, pyd.BaseModel):
                fixed = _rep_models.get(t, None)
                if fixed is _fixing: raise RuntimeError("Recursive class definitions not supported")
                if not fixed:  # new model to fix
                    _rep_models[t] = _fixing  # mark we are going to fix it
                    _rep_models[t] = fixed = _template_missing_defaults(t, _rep_models)
                return fixed, fixed()
            else:
                return f"<<{getattr(t, '__name__', '')}>> {getattr(field, 'description', '')}"
        else:
            return t, field.get_default()

    fields = {name: convert(fld) for name, fld in model.__fields__.items()}
    _log.info(f"Creating template for model {model}")
    return pyd.create_model(model.__name__, __module__=model.__module__, __base__=YamlModel, **fields)


_hash_exclude_attr = '__hash_exclude'


class _ExcludeContext:
    _active: bool = False
    _exclude: set | None = None
    _attr: str | None = None

    @classmethod
    def enter(cls, obj: YamlModel, exclude: str | Iterable(str)):
        if not isinstance(obj, YamlModel):
            raise TypeError("Exclude context requires YamlModel instance")
        if cls._active:
            raise NotImplementedError("Nested exclude contexts are not supported")
        cls._active = True

        if exclude == 'hash':
            cls._attr = _hash_exclude_attr
            cls._exclude = None
            return getattr(obj, _hash_exclude_attr, set())
        else:
            cls._attr = None
            cls._exclude = set(as_iter(exclude))
            return cls._exclude

    @classmethod
    def exit(cls):
        assert cls._active
        cls._active = False
        cls._exclude = None
        cls._attr = None

    @classmethod
    def excludes_for(cls, obj) -> None | set:
        """Return excludes only if they are defined for the obj"""
        if isinstance(obj, YamlModel):
            if cls._attr:
                return getattr(obj, cls._attr, None)
            return cls._exclude


class YamlModel(YamlModelMixin, pyd.BaseModel):
    """
    Extends ``pydantic_yaml YamlModel`` to support:

    - ``to_file`` method
    - nicer str representation
    - model's YAML scheme ``.json`` file (for editors to provide error checking)
    - model's default configuration ``.yml`` file
    - case independent (always lower) fields
    - extra fields allowed by default
    - implemented validator to lower str values

    Used as a base class for Yaml based models. To subclass one must provide
    information of the file format.
    The possible FileFormat support is:
    1. Explicitly pass FileFormat object.
    2. Pass patterns or description when:
        -   desc : The description of the
        -   patterns : Regex with file names suffixes

    Subclassing Arguments
    ---------------------

    When sub-classing YamlModels the following **optional** arguments are supported
    ::
        desc: Description of the model
        patterns: Pattern or PathScheme to define file format for the model
        file_format: (alternatively to patterns) pre-defined FileFormat
        finalize: [True] apply recursive update_forward_refs
        hash_exclude: fields to exclude from hash
        extra: 'allow' | 'forbid' | 'ignore'
    """

    #  -------  HASH EXCLUDE --------------
    # This is a temporal workaround for missing mechanism in pydantic 1.10 of
    # restricting specific fields be used foy dumps. Not needed in pydantic 2+
    #
    # This mechanism is built around 'hash_exclude' arguments for __init_subclass__.
    # It allows to exclude such attributes from being emitted by content exporting
    # functions built based on ._iter(), like .dict(), .yaml(), etc.
    #
    # Such functions has (.., exclude,..) arguments and exclusion is achieved
    # with `._hash_exclude_context` context manager:
    #
    # with model._hash_exclude_context() as exclude:
    #     model.dict(exclude=exclude)
    #
    # Notice, that models own attributes are excluded by passing required attributes.
    # The Context manager return them as a context just for convinience,
    # instead one could just use `model._hash_exclude`.
    #
    # Main role of the context manager is to ensure that within it ANY retrival of the
    # sub-models activates their own exclusion mechanism.

    @contextmanager
    def exclude_context(self, exclude: Literal['hash'] | list[str] | str = 'hash'):
        exclude = _ExcludeContext.enter(self, exclude)
        try:
            yield exclude
        finally:
            _ExcludeContext.exit()

    @classmethod
    def _get_value(cls, v: Any, exclude, **kws) -> Any:
        """Override BaseModel._get_value to exclude from nested models hash_exclude attributes
        # during damping"""
        if exclude_set := _ExcludeContext.excludes_for(v):
            exclude = exclude_set.union(exclude) if exclude else exclude_set
        return super()._get_value(v, exclude=exclude, **kws)

    # ------------------------------------------------------------
    class Config:
        extra = pyd.Extra.allow

    _format: FormatHandler | None = None  # an optional format handler associated with the model
    _kind_name = ''  # Essential name of the model. Subclasses override that in __init_subclass__

    def __init_subclass__(cls, *, desc=None, patterns=None,
                          file_format: FormatHandler = None, finalize=True,
                          hash_exclude=None, **kwargs):
        """
        :param desc: Description of the model
        :param patterns: Pattern or PathScheme to define file format for the model
        :param file_format: (alternatively to patterns) pre-defined FileFormat
        :param finalize: apply recursive update_forward_refs
        """
        cls._kind_name = cls._kind_name or cls.__name__.replace('Model', '')
        super().__init_subclass__(**kwargs)

        # bind hash recipe
        hash_exclude = set(as_list(hash_exclude))
        if prev := getattr(cls, _hash_exclude_attr, None):  # append from super class
            hash_exclude |= prev
        setattr(cls, _hash_exclude_attr, hash_exclude)

        # meta reading logic
        if file_format:
            if not issubclass(file_format, FileFormat) or isinstance(file_format, FileFormat):
                raise TypeError(f"Expected type: {FormatHandler} received: {file_format}")
            if desc or patterns:
                raise TypeError(f"passing `file_format' excludes 'desc' and 'patterns' arguments")
            cls._format = file_format  # ToDo: Use this to read YAML files?
        elif desc or patterns:
            cls._format = make_yml_model_format(
                f'{cls._kind_name}Format', desc=desc,
                model=cls, patterns=patterns)
        else:
            cls._format = None

        if finalize:
            cls.update_forward_refs()

    def _str_form(self, width=80, levels=3, indent=3 * ' ', exclude=None):
        # -------------
        from ..strings import repr_nested
        head = f"<{type(self).__name__}>"
        exclude = set(as_iter(exclude))
        dct = {k: getattr(self, k) for k in self.__fields__ if k not in exclude}

        s = repr_nested(dct, width, indent, levels)

        if len(s) < width:
            return f"{head} {s}"

        return f"{head}\n{indent}{s}"

    def __str__(self):
        return self._str_form(width=50, levels=0)

    def __repr__(self):
        # return self._str_form(width=60, levels=4)  # FixMe: @Ilya!task task hierarchy is broken
        from pprint import pformat
        s = pformat(self.dict(), indent=2, width=80, depth=3,
                    compact=True, sort_dicts=False)
        s = re.sub("'([^']+)':", r"\1:", s)

        sep = '\n' if '\n' in s else ':: '
        return f"<{type(self).__name__}>{sep}{s}"


    def _deep_copy_attrs(self, src):
        """Deep copy values of all the attributes, including private.
        """
        if not src.__class__ is self.__class__:
            raise TypeError(f"{src.__class__=} is NOT {self.__class__=}!")
        object.__setattr__(self, '__dict__', deepcopy(src.__dict__))
        object.__setattr__(self, '__fields_set__', set(src.__fields_set__))

        Undefined = pyd.fields.Undefined
        for name in self.__private_attributes__:
            value = getattr(self, name, Undefined)
            if value is not Undefined:
                if deep:
                    value = deepcopy(value)
                object_setattr(self, name, value)

    def hash_str(self, length):
        """
        Creates hash string for the model, while avoiding all exclude_hash fields in the model.
        Works in a recursive manner, every lower level models' exclude_hash fields are also avoided.
        :param length: length of the hash str
        :return: hashed string
        """
        with self.exclude_context('hash') as exclude:
            return TBox(self.dict(exclude=exclude)).hash_str(length)

    @classmethod
    def write_templates(cls, folder, scheme=True, default=True) -> tuple[Path | None, Path | None]:
        """
        Write template file(s) for this model, named by default automatically:
        ::
            <model_snake_case_name>_default.yml
            <model_snake_case_name>_scheme.json

        :param folder: location for the file(s)
        :param scheme: YAML scheme file name, ``False`` - skip, ``True`` - auto name
        :param default: default config file name, ``False`` - skip, ``True`` - auto name
        :return:
        """
        if not (scheme or default): return None, None
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        # remove 'Model' from the class name and convert into snake case
        name = re.sub('model', '', cls.__name__, re.I)
        name = re.sub('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))', r'_\1', name).lower()

        if scheme is True:
            scheme = f"{name}_yaml_scheme.json"
        if scheme:
            if scheme.rsplit('.', 1)[-1].lower() not in ('jsn', 'json'):
                scheme += '.json'
            scheme = folder / scheme
            scheme.write_text(cls.schema_json(indent=2))

        if default is True:
            default = f"{name}_default.yml"
        if default:
            if default.rsplit('.', 1)[-1].lower() not in ('yml', 'yaml'):
                default += '.yml'
            try:
                inst = cls()
            except pyd.ValidationError:
                _log.info("Model can not generate fully defined default configuration!")
                inst = _template_missing_defaults(cls)()
            default = folder / default
            inst.to_yaml(default)

        return scheme, default

    @staticmethod
    def lower_contain_str(v):
        return v.lower() if isinstance(v, str) else \
            type(v)(map(str.lower, v)) if hasattr(v, '__len__') else v

    @classmethod
    def lower_values(cls, *fields: str):
        """Validator to be used in the inheriting classes
        on specific fields to lower case of their string values.

        Example::

            class Model(BaseYaml):
                domain: str
                name: str
                size: int
                _lower_values = BaseYaml.lower_values("domain", "name")
        """
        assert len(fields) > 0
        return pyd.validator(*fields, pre=True, allow_reuse=True, check_fields=False
                         )(cls.lower_contain_str)

    @pyd.root_validator(pre=True)
    def _low_case_fields(cls, values):
        return {k.lower(): v for k, v in values.items()}

    @classmethod
    def field_type(cls, name):
        """"""
        return cls.__fields__[name].type_

    def to_yaml(self, filename=None, **kws) -> str | None:
        """Generate a YAML representation of the model.

            Support ``yaml_dump`` kws:

            include, exclude:
               Fields to include or exclude. See `dict()`.
            by_alias : bool
                   Whether to use aliases instead of declared names. Default is False.
            skip_defaults, exclude_unset, exclude_defaults, exclude_none
                Arguments as per `BaseModel.dict()`.
            sort_keys : bool
                If True, will sort the keys in alphanumeric order for dictionaries.
                Default is False, which will dump in the field definition order.
            default_flow_style : bool or None
                Whether to use the "flow" style in the dumper. By default, this is False,
                which uses the "block" style (probably the most familiar to users).
            default_style : {None, "", "'", '"', "|", ">"}
                This is the default style for quoting strings, used by `ruamel.yaml` dumper.
                Default is None, which varies the style based on line length.
            indent, encoding, kwargs
                Additional arguments for the dumper.

        :param filename: optional name to dump into, otherwise return the text buffer
        """
        buf = self.yaml(**kws)
        if not filename:
            return buf
        Path(filename).expanduser().resolve().write_text(buf)

    def to_box(self, *args, **kws):
        return TBox(self, *args, **kws)

    @classmethod
    def update_forward_refs(cls, **kws):
        def models_dict(c):
            return {name: obj for name, obj in c.__dict__.items()
                    if hasattr(obj, 'update_forward_refs')}

        models = models_dict(cls)
        for m in models.values():
            m.update_forward_refs(**models_dict(m))
        super().update_forward_refs(**(kws | models))

    @staticmethod
    def nested_classes(model_cls: Type[YamlModel]):
        """Decorate class to indicate it contains nested definitions of models classes"""
        model_cls.update_forward_refs()
        return model_cls

    @classmethod
    def suffixes(cls) -> list[str] | None:
        """If the model is associated with a ``FileFormat``,
        rerturn list of its suffixes:
        ::
            [".scm.yml", ".scheme.yml"]
        """
        return cls._format and cls._format.suffixes

    @classmethod
    def format_match_pattern(cls) -> str | None:
        """
        If the model is associated with a ``FileFormat``,
        represents regular expression matching any of the file names defined by the format.

        Otherwise, its None
        """
        return cls._format and cls._format.match_pattern

    def difference(self, other: YamlModel):
        """
        Calculate difference wuith other model of same type.

        Return nested dict tree following the nested structure of the model
        but only with fields where differences are found.

        Leaf nodes of this tree are fields with different values `(self_value, other_value)`

        Parent nodes contain either:
         - dict with differences in sub-models
         - list containing differences in sub-models of its elements

        :param other:
        :return:
        """
        if (t1 := type(self)) != (t2 := type(other)):
            return (self, other)

        def equals(v1, v2):
            try:
                return (v1 == v2)
            except Exception:
                return False

        def compare(pair: Iterable):
            """Compare TWO values of arbitrary type.

            Retrun:
             - ``None | False | 0 | [] | {} | ...`` if equal
             - ``tuple(v1, v2)`` if not equal
             - ``[..., (pos, <compare>), ...]`` if lists of same size containing sub-models
             - ``difference(v1, v2)`` if both are models
            """
            v1, v2 = (pair := tuple(pair))
            if isinstance(v1, YamlModel):
                return v1 - v2
            elif (  # if v1 and v2 are lists of sub-models of same size
                    all(isinstance(_, list) for _ in pair)
                    and len(v1) == len(v2)
                    and len(v1) < 100 and  # models not expected in long lists
                    any(isinstance(_, YamlModel) for lst in pair for _ in lst)
            ):
                return [(i, delta) for i, p in enumerate(zip(pair))
                        if (delta := compare(p))]
            else:
                try:  # try to compare values,
                    if v1 == v2:  # if __eq__ is not supported
                        return None  # catch and return the pair
                except Exception:
                    pass
                return (v1, v2)

        return {k: delta for k, fld in self.__fields__.items() if  # only if delta contains something
                (delta := compare(getattr(_, k, None) for _ in [self, other]))}

    __sub__ = difference

    def __eq__(self, other: YamlModel):
        return not self.difference(other)


class YamlModelFormat(FileFormat, content=CT.CONFIG, patterns=['.yml']):
    _model: Type[pyd.BaseModel] = None

    @property
    def model(self):
        return self._model

    def __init_subclass__(cls, *, model: Type[YamlModel], **kwargs):
        cls._model = model
        super().__init_subclass__(**kwargs)

    @classmethod
    def read(cls, filename, *, content=CT.CONFIG, **kwargs):
        return cls._model.parse_file(filename, **kwargs)

    @classmethod
    def write(cls, filename: PathT, data: YamlModel, **kws):
        data.to_yaml(filename, **kws)


def make_yml_model_format(name, model, patterns, desc=None):
    """Create Format subclass of ``YamlModelFormat`` for given pydantic ``model``."""
    return MetaFormat(name, (YamlModelFormat,), {},
                      content=CT.CONFIG, model=model, specific=1,
                      patterns=patterns, desc=desc or f'{model._kind_name} Model'
                      )
