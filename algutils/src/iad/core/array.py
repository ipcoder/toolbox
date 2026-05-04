from __future__ import annotations

import sys
import warnings
import dataclasses as dcs
from enum import Enum
from typing import (Any, Literal, overload, Collection, Callable, Iterable, Tuple,
                    Union, Iterator, get_type_hints, get_args, Container)
import re
import numpy as np
import numba
from numbers import Number

from .codetools import NamedObj
from . import as_list
from . import nptools as npt
from .datatools import rm_keys

__all__ = ['DForm', 'Array', 'FormArray', 'form_array', 'Color', '_wC2G']

FLOAT = np.float32  # default float for images
_wC2G = np.array([0.2125, 0.7154, 0.0721], FLOAT).reshape(3, 1)  # RGB -> Gray weights


def _cast_int(inp, out, round):
    """
    Simple logic to handle clipping of inf values where casting float to int, where np.clip returns garbage.
    Also for clipping when doing int -> int casting.

    :param inp: input array
    :param out: output array
    :param round: If true, use round function of python, else let astype do the rounding (would be np.fix)
    :return: array after casting
    """
    out_type = out.dtype.type
    inp_type = inp.dtype.type

    lims = np.iinfo(out.dtype)
    mn, mx = out_type(lims.min), out_type(lims.max)
    inp = inp.reshape(inp.size)
    out = out.reshape(inp.size)
    if inp.dtype.kind == 'f':
        inf, ninf = inp_type(np.inf), inp_type(-np.inf)
        for i in range(inp.size):
            if inp[i] > mx or inp[i] == inf:
                out[i] = mx
            elif inp[i] < mn or inp[i] == ninf:
                out[i] = mn
            elif round:
                out[i] = out_type(np.round(inp[i]))
            else:
                out[i] = out_type(inp[i])
    else:
        for i in numba.prange(inp.size):
            if inp[i] > mx:
                out[i] = mx
            elif inp[i] < mn:
                out[i] = mn
            else:
                out[i] = out_type(inp[i])


_cast_parallel = numba.jit(nopython=True, fastmath=True, parallel=True)(_cast_int)
_cast_no_parallel = numba.jit(nopython=True, fastmath=True)(_cast_int)
_cast_parallel_size = 50000  # size to start parallel


def cast(a: np.ndarray, dtype: Union[str, np.dtype],
         *, out=None, copy=True, round=True):
    """
    Casting function of arrays to better cast in problematic use-cases, as a replace for astype()

    Casting could change value for different reasons:
        1. range of the out type is narrower: i2(255:u2) = -1
        2. precision of the output type is lower: f4(i4.max) - i4.max = 1

    The improvement is by:
     1. Clipping better for integers.
     2. Rounding  instead of np.fix for integers. # todo update optional
     3. Allowing to include an output array.

    By default,
     - out of range values for float outputs are *automatically* rendered into +-inf.
     - outputs values in integers and floats are clipped *if needed*.
     - precision of float is decreasing when lowering the size of the float (as astype)

    :param a: input array to be cast to a new type
    :param dtype: dtype to cast
    :param out: If exists, it is the array to be used as the output. It MUST be in the shape of the input, and in
                the same dtype of the dtype requested, else casting would fail.
                If None, a new array would be created.
    :param round: When True, using the round function to round float numbers to integers.
                     If False, it uses the astype default strategy (np.fix)
                     Default is True.
    :param copy: Relevant only to casting to same dtype, and when there isn't an "out" parameter:
                 If True, there is a new copy of the input. The default.
                 If False, the output is a view of the input and not a copy.
    :return: Array that was cast to new dtype
    """

    dtype = np.dtype(dtype)
    if out is not None and out.dtype != dtype:
        raise TypeError(f"Missmatch of types between out type : {out.dtype} and cast to type {dtype}")

    if a.dtype == dtype:  # special cases where there is no changing in dtype
        if out is not None and (a is out or shared_view(a, out)):  # a and out have same view, and no casting
            return out
        elif not copy:  # not copying - only if same type
            if out is None:
                return a
            raise ValueError(f"Can't cast without copying when out array is present")

    if dtype.kind == a.dtype.kind and dtype.itemsize >= a.dtype.itemsize:  # only increasing size of same kind
        if out is None:
            return a.astype(dtype, casting='safe')
        else:
            np.copyto(out, a.astype(dtype, casting='safe'))
            return out
    else:
        if dtype.kind == 'f' or np.can_cast(a, dtype,
                                            casting='safe'):  # inf values created automatically by astype
            if out is None:
                return a.astype(dtype, casting='unsafe')
            else:
                np.copyto(out, a.astype(dtype, casting='unsafe'))
                return out
        else:  # int out or can't safely cast - int16 -> int8, float16 -> int32
            if out is None:  # create out if not included
                out = np.empty_like(a, dtype=dtype)
            cast_int = _cast_parallel if a.size > _cast_parallel_size else _cast_no_parallel
            cast_int(a.astype('f4') if a.dtype == np.dtype('f2') else a,  # numba doesn't support float16
                     out, round)
            return out


def shared_view(a: np.ndarray, b: np.ndarray, shape=True) -> bool:
    """Checks if two arrays share same data in the memory.

    If shape is ``False`` - they may be of different shape, but size must match -
    it does not deal with partial overlaps, and its main purpose to recognize if
    one array is a full views of each over or both are views of some other array.
    """
    if shape:
        if a.shape != b.shape:
            return False
    elif a.size != b.size:
        return False
    return a.__array_interface__['data'][0] == b.__array_interface__['data'][0]


def _name_to_dtype(name: str) -> np.dtype | None:
    """
    Return dtype for given string  or None if not possible
    :param name:
    :return: dtype
    """
    try:
        return np.dtype(name)
    except TypeError:
        return None


def _dtype_str(t: np.dtype):
    return f"{t.kind}{t.itemsize}"


def _collect_result_type(*v: np.ndarray | None):
    """Result type of multiple vector or scalar objects:

    >>> _collect_result_type([1, 0,1, np.array([1,2,3])])
    """
    v = (_ for _ in v if _ is not None)
    return np.result_type(*map(np.min_scalar_type, v))


class LinT:
    """Helper class to implement linear transformation ``k*a+b`` over ndarray ``a``
    with carefully managed type conversions and data copying
     """

    def __init__(self, k=1., b=0.):
        """
        :param k: initial k (a number of vector of ``chn`` size)
        :param b: initial b (a number of vector of ``chn`` size)
        """
        # self.channels = 1 if chn is None else chn
        self.k, self.b = (np.array(v, dtype=float, ndmin=1) for v in (k, b))

    @property
    def identity(self):
        """True if transform does not change the data"""
        return np.allclose(self.k, 1) and np.allclose(self.b, 0)

    def __repr__(self):
        one_or_all = lambda x: x[0] if np.all(x == x[0]) else x
        k, b = map(one_or_all, (self.k, self.b))
        return f"<{type(self).__name__}> [{k = }, {b = }]"
        # f" chn: {self.channels}"

    def _chain_mult(self, k, b):
        return self.k * k, k * self.b + b

    def __mul__(self, kb):
        return type(self)(*self._chain_mult(*kb))

    def __imul__(self, kb):
        self.k, self.b = self._chain_mult(*kb)
        return self

    def apply(self, a: np.ndarray, *, out: np.ndarray) -> np.ndarray:
        """
        To minimize unnecessary operations return input for equivalent transform.
        Also perform multiplication with dtype defined by target dtype and the input.

        :param a: array to apply the transform to
        :param out: array for output results
        """
        inp_type, out_type = a.dtype, out.dtype

        k, b = map(npt.as_min_type, (self.k, self.b))
        k_not_1, b_not_0 = not np.allclose(k, 1), not np.allclose(b, 0)
        float_out = out_type.kind == 'f'

        if not (k_not_1 or b_not_0):  # no transformation - just casting
            return cast(a, out_type, out=out)

        if float_out:  # transformation into float does not require clipping
            tmp = np.multiply(a, self.k) if k_not_1 else a
            tmp = np.add(tmp, self.b) if b_not_0 else tmp
            return cast(tmp, out_type, out=out)

        # determine dtype for operations
        mlt_type = np.result_type(inp_type, k.dtype) if k_not_1 else inp_type
        add_type = np.result_type(mlt_type, b.dtype) if b_not_0 else mlt_type
        # upgrade dtype for possibly range expansion during operations (partially heuristics)
        if add_type.kind != 'f' and add_type.itemsize <= 2:  # todo we don't handle itemsize == 4 and increasing to 8?
            add_type = np.dtype(f'i{add_type.itemsize * 2}')
        tmp = out if add_type is out_type else np.empty(a.shape, add_type)
        np.add(np.multiply(a, k, out=tmp) if k_not_1 else a, b, out=tmp)
        return cast(tmp, out_type, out=out)

    def __bool__(self):
        return not self.identity


class Kind(Enum):
    NA = None
    UND = None
    DISP = 'disp'
    IMG = 'image'
    IMAGE = 'image'
    DEP = 'depth'
    DEPTH = 'depth'
    RGN = 'region'
    REGION = 'region'
    CNF = 'conf'
    CONF = 'conf'

    def __bool__(self):
        return bool(self.value)

    @classmethod
    def from_str(cls, s: str):
        """Return Color instance from name if exists or return None"""
        return cls.__members__.get(s.upper(), None) if isinstance(s, str) else None


class Color(Enum):
    """Enum listing differnt color codings of the channels."""
    BIN = 'BIN'
    GRAY = 'GRAY'
    RGB = 'RGB'
    B = 'BIN'
    G = 'GRAY'
    C = 'RGB'

    def __bool__(self):
        return bool(self.value)

    @classmethod
    def from_str(cls, s: str) -> str | None:
        """Return Color instance from name if exists or return None"""
        return cls.__members__.get(s.upper(), None) if isinstance(s, str) else None

    @property
    def channels(self):
        """Number of color channels"""
        if self is self.RGB: return 3
        if self in (self.GRAY, self.BIN): return 1
        raise RuntimeError("All cases must return channels number")

    def short_code(self):
        """Return the shortest alias to the color"""
        code = self.value
        for name, v in self.__class__.__members__.items():
            if self is v and len(name) < len(code): code = name
        return code

    def __str__(self):
        return self.value


def field(default=None, *, init=True, repr=True,
          hash: bool | Callable[[Any], int] = None,
          compare: bool | Callable[[Any, Any], bool] = True,
          **metadata):
    """
    Wraps and modifies ``dataclass.field`` interface.

    Changes:
        1. ``default`` argument is optional and is None by default
        2. ``metadata`` is provided not as a argument but as a keyword arguments
        2. ``hash`` may additionally be a function defining custom hash calculation
        3. ``compare`` may additionally be a function defining custom comparison

    ``metadata`` will contain all the keyword arguments not defined in ``datclass.field``.

    More, if ``hash`` and ``compare`` are Callables they are passed as ``True``
    into ``dataclass.field``, but additionally stored in ``metadata`` with their
    function values.

    All that allows to simplify definition of DForm fields with extended functionality,
    not only by supporting custom hashing and comparison, but also to
    define entirely new functionality through metadata.

    In particular ``calc`` keyword argument defines function to estimate field value from the data.

    :param default
    :param init:  add as ``__init__`` argument
    :param repr:  include in ``__repr__``
    :param hash: True to include or Callable to calc hash with
    :param compare: True to include in comparison or Callable for special __eq__
    :metadata: additional field attribute as described below:

    :key calc: function to calculate field from data

    :return:
    """
    if isinstance(hash, Callable):
        metadata['hash'], hash = hash, True

    if isinstance(compare, Callable):
        metadata['compare'], compare = compare, True

    if 'default_factory' in metadata:  # move from metadata into locals
        default_factory = metadata.pop('default_factory')

    if 'calc' in metadata and not isinstance(metadata['calc'], Callable):
        raise ValueError("Argument `calc` must be a function")

    return dcs.field(**locals())


FormKind = Union['DForm', np.dtype, str, type]
hash_array = lambda a: None if a is None else hash(tuple(a))


# noinspection PyShadowingBuiltins
@dcs.dataclass
class DForm:
    """
    Provides extended description of *form* and *content* of data arrays.

    In addition to the standard ``dtype`` field, identical to the ``ndarray.dtype``,
    may optionally include information about
     - statistics (``min, max, avr, std``)
     - color information:  (``color``, ``cax``)

    Provides method to automatically ``transform`` array of different forms.

    Also supplies convenience properties derived from this information:
     - ``range`` (``max`` - ``min``)
     - ``ndim`` - dimension of data if its describe a color image
     - ``norm`` - ``(avr, std)`` how data was normalized

    All those return ``None`` if not relevant.

    Initialization
    ==============
    All ``DForm`` arguments are initialized with ``None`` by default:
    ::
        dtype: np.dtype or convertable into dtype
        name: str | DForm,
        min: float, max: float
        avr: float | list[float], std: float | list[float]
        color: DForm.Color, cax: int

    Color
    -----
    Color is described by
        - ``color: DForm.Color`` ``enum`` containing all the supportted color spaces

        - ``cax`` - index of the color axis in the data dimensions

            could be ``None`` also for ``color` like ``BIN``, ``GRAY``, wih dimensions ``[H x W]``,

            unless additional color dimention is used ``[H x W x 1]``

    Empty Form
    ----------
    ``dtype`` must be provided, except of the case of *empty* form:

    >>> bool(DForm()) is False  # empty form

    Name and Registrattion
    ----------------------
    If ``name`` string is provided, a resulting instance is automatiocally
    registered under ``name.lower()``.

    It then may be accessed using either:
     - ``registered()`` query or
     - constructor with name as only argument.

    >>> dform = DForm('Cu1', dtype='uint8', color=DForm.Color.RGB)
    >>> assert DForm.from_name('Cu1') is dform
    >>> assert DForm('Cu1') == dform

    Use ``DForm.list_registered()`` to list of all the registered forms.
    Defining a ``DForm`` with same currently replaces the previous one with a warning.
    Avoid that to prevent having different forms with same name.

    Practical Examples
    ==================
    Not any combination of the forms attributes are meaningfull.
    Below are some typical cases.

    Colored
    ------
    Color requires either ``range``, or ``norm`` to be defined.

    Range of float color is usually ``(0,1)`` or ``(-1,1)``,  but not necessary.

    (Curled brackets means every option is possible independent on the others)
    ::

        color: {RGB, GRAY}
          - dtype: u8               # unsigned 8 bits per channel
            range: (0, 255)         # type based range

          - dtype: u16              # unsigned 16 bits
            range: {(0, 1023),      # specific (10 bits for example)
                    (0, 65535)}     # type based

          - dtype: i16
          - range: {(-32768, 32767), (-32768, 32767), ...}

          - dtype: {f16, f32 f64}
            range: {(0, 1), (-1, 1)}


    Supported basic transformations
    -------------------------------
    As ``DForm`` extends ``dtype`` so this transform extends notion of casting:
     - only obvious and safe data manipulations are applied automatically,
       like range mappings, rescaling, etc,
     - aim to safe type casting keeping data values, except of type range clipping.

    Those basic kind of transformations can be easily defined separately:
        1. range -> range
        2. normalization (avr, std) -> (avr, std)
        3. color -> color
        4. dtype -> dtype

    All the transformations (except of dtype) are performed if the corresponding attribute is
    defined in both source and target DataForms.

    Missing input attribute makes transformation impossible and it fails.

    Missing output attribute keeps data (in its respect) unchanged, however
    it could be changed because of another attributes.

    In this example:
        ``range, color:RGB -> color:GRAY``
    input range is irrelevant, but the color transformation does occur.

    Implementation tries to minimize number of calculations and data copy.
    That can be facilitated even farther by providing the ``out`` argument
    with array to write the data into.

    This array must match both the source dimensions and target form.
    If possible it is used along all the transformation stages to reduce copying.

    Supported Combinations
    ======================
    Only some of the combinations of the attributes in the source and target forms can be allowed.

    DType
    ~~~~~
    ``dtype`` is always defined for both source and target forms, and any conversion is possible
    together with allowed transform of any other attributes.

    Range
    ~~~~~
    Mathematically ``range`` is incompatible with ``normalization``, therefore
     - ``range, norm -> range norm`` is not allowed
     - ``range -> std`` is possible however
     - ``std -> range`` is not allowed since the initial range is not known

    When automatic transform is not possible, a ``TypeError`` is raised.

    That sometimes can be avoided if desired transform is explicitly enabled or disabled using
    ``do_range`` and ``do_norm`` arguments. Then exception is raised if a specific transform(s)
    are explicitly enabled but is not possible.

    Color
    ~~~~~
    ``color`` can be used in different combinations:
      ::

            color, range  ->  color, range
            color, range  ->  color, norm
            color, norm   ->  color, norm

    """

    _reg_names = {}
    _reg_hashes = {}

    _auto_dtype = True  # allow guessing dtype when not provided
    _auto_names = True
    _default_dtype = np.dtype('f4')
    _verify_cache = False or sys.gettrace()  # for debugging. FIXME: remove after freeze

    @classmethod
    def auto_naming(cls, state: bool = None):
        """
        Controls over possibility to auto-generate ``DForms`` using name coding conventions

        If turned ON (default) than accessing a ``DForm`` by name following this convention, instead of
        raising an error, generates and registers it automatically.

        :param state:True - AutoNaming On (default),
                     False - AutoNaming Off.
                     None - for getting the current mode.
        :return: the current mode if `state` is ``None`` or ``None``

        Examples:
        >>> DForm('CNu1')
        Works
        >>> DForm.auto_naming(False)
        >>> DForm('CNu2')
        KeyError: "No registered DForm named 'CNu2'"
        """
        if state is None:
            return cls._auto_names
        cls._auto_names = state
        return None

    @classmethod
    def _field_type(cls, fld, allowed: Container) -> type:
        """Found in the field defintion first type listed in `allow` or raise ``TypeError``.
        """
        if not getattr(cls, '_fields_types', None):
            cls._fields_types = get_type_hints(cls)

        _type = cls._fields_types[fld]
        type_options = [_type] if isinstance(_type, type) else get_args(_type)
        for _type in allowed:  # find first!
            if _type in type_options:
                return _type
        raise TypeError(f"Neither {type_options=} for field {fld} are {allowed=}")

    @classmethod
    def _parse_name(cls, name: str) -> dict | None:
        """
        Parse through the name, validate and return fields as ``dict``"

        :param name: The name of dform for parsing
        :return: dict of all attributes of dform

        **The Pattern**:
        ::
            [C|G|B][R[<min>:]<max>][N[<std>][|<avr>]]((u|i|f)<bytes>|(s|b)<bits>)

        **Conventions**:
        ::
            (x|y|...) - selection group
            []        - optional group
            <name>    - group containing value of variable `name`

            C - var:color = 'RGB'
            G - var:color = 'Grey'
            B - var:color = 'Binary'

        **Group ideyntification symbols starting definition of**:
        ::
            R - Range
            N - Normalization

        **Defaults for missing groups**:
        ::
                R<max>  =   R<0>:<max>
                N       =   N<1>|<0>


        The only *mandatory* group is for **type**, which may be given in two forms:
            - as number of bits prefixed by 'b' or 's' (for signed)  \n
              only for `int` types, contaner ``dtype`` is deduced from that
            - using ``np.dtype`` shortcut notations (u1 = uint8, f8 = float64, ...)

        **Limitations**:
            - Parameters of `range` and `normalization` are represented by ``float``,
              which can be parsed only in this form:
            ::
                [-][<digits>.]<digits>

            Though `DForm` supports defining normalization *per channel*,
            that is **not** supported by the name parser.

        Examples:
            >>> DForm('CNu1') == DForm(color='RGB', std=1, avr=0, dtype=np.uint8)
            True
            >>> DForm('N2.5|-3f2') == DForm(std=2.5, avr=-3, dtype=np.float16)
            True
            >>> DForm('R1N|1.2f4') == DForm(min=0, max=1, avr=1.2, dtype=np.float32)
            True
            >>> DForm('Gb10') == DForm(color=Color.GRAY, min=0, max=1023, dtype=np.uint16)
            True
            >>> DForm('s2') == DForm(min=-1, max=1, dtype='i1')
            True
        """

        _codes_rex = re.compile(r"""
            (?P<color>[GCB])?
                (?:R
                    (?:
                        (?P<min>-?\d*\.?\d+):
                    )?
                    (?P<max>-?\d*\.?\d+)
                )?
                (?P<norm>N
                    (?P<std>\d*\.?\d+)?
                    (?:\|
                        (?P<avr>-?\d*\.?\d+)
                    )?
                )?
                (?:
                    (?P<bs>[bs])(?P<bits>[1-9]\d*) |
                    (?P<dtype>[uif][1248])
                )
           """, re.VERBOSE)

        def set_default(fld, val):
            if attr[fld] is None:
                attr[fld] = val

        def cast_numeric_attr(fld):
            if attr[fld] is not None:
                attr[fld] = cls._field_type(fld, (float, int))(attr[fld])
            return attr[fld]

        if (m := _codes_rex.fullmatch(name)) and (attr := m.groupdict()):
            # shortcuts
            if attr['max'] is not None:  # R10 -> R0:10
                set_default('min', 0)
            if attr['norm']:  # N -> N1 -> N1|0
                set_default('std', 1)
                set_default('avr', 0)

            r_min, r_max, *_ = map(cast_numeric_attr, ['min', 'max', 'std', 'avr', 'bits'])

            if bits_code := attr.pop('bs'):
                if (bits := attr['bits']) > 64:
                    raise ValueError(f"Invalid number of {bits=}")
                if bits_code == 'b':  # unsigned of size bits
                    t_min, t_max = 0, 2 ** bits - 1
                    attr['dtype'] = np.min_scalar_type(t_max)
                else:  # s - signed
                    t_min, t_max = -2 ** (bits - 1), 2 ** (bits - 1) - 1
                    attr['dtype'] = np.min_scalar_type(t_min)
            else:  # if not a bits code, then dtype is defined
                dtype = attr['dtype'] = np.dtype(attr['dtype'])  # ensure valid dtype
                t_info = (np.finfo if dtype.kind == 'f' else np.iinfo)(dtype)
                t_min, t_max = t_info.min, t_info.max

            if not ((r_min is None or r_min >= t_min) and (
                    r_max is None or r_max <= t_max)):
                raise ValueError(f"Range ({r_min},{r_max}) is inconsistent with {bits_code}{bits}")

            attr['color'] = Color.from_str(attr['color'])

            # remove parsed keys which are not valid fields
            return rm_keys(attr, set(attr).difference(cls.fields()))

        return None

    @classmethod
    def _register(cls, dform):
        """ Add to the global registry """
        h = hash(dform)
        if found := cls._reg_hashes.get(h, None):
            if dform.name and found.name and found.name != dform.name:
                warnings.warn(f"DForm {dform} is already known under {found.name}")

        if dform.name:
            key = hash(dform.name.lower())
            if found := cls._reg_names.get(key, None):
                warnings.warn(f"{cls.__name__}('{dform.name}') overrides registered {found.name}")
            cls._reg_names[key] = dform

    @classmethod
    def from_name(cls, name: str, *, auto: bool = None, reg=True) -> DForm:
        """
        Check if the name is registered as a dform in the registered list.
        If the name is not registered, AutoNaming is On, and the name have values other than None,
            it includes it in the registered list and return the dform created
        Else we return the dform if it was found, or None if it wasn't found
        :param name: The dform to check
        :param reg: look into the registry
        :param auto: allow automatic creation from name
        :return: dform or None
        """
        if reg and (found := cls._reg_names.get(hash(name.lower()), None)):
            return found
        if auto is True or auto is None and cls._auto_names:
            if attrs := cls._parse_name(name):
                return DForm(name=name, **attrs)

    @classmethod
    def list_registered(cls, out=False) -> dict[int, DForm] | None:
        """
        Print list of the registered DataForms.

        :param out: if True just silently return registration dict
        :return: None or dict
        """
        if out:
            return cls._reg_names

        for i, (k, v) in enumerate(cls._reg_names.items()):
            print(f"{i:3}. {v.name:20}: {v}")
        return None

    class Shaper:
        def __init__(self, src_shape, src_form: DForm, trg_form: DForm):
            """
            Summary of possible color options and resulting re-shaping:
            ::
              case      src       trg      re-shape
                      chn cax   chn cax
                0      -   -     +          error
                1      -   -     -   -       =
                2      +         -   -       =
                3      1   -     1   -       =
                4      1   -     1   +       + dim
                5      1   -     3   +       + dim
                6      1   +     1   -       - dim
                7      3   +     1   -       - dim
                8      1   +     1   +       ? move
                9      3   +     1   +       ? move
               10      1   +     3   +       ? move

            :param src_shape:
            :param src_form:
            :param trg_form:
            """
            # start with 'no shape changes' configuration:
            self.src_shape = self.trg_shape = src_shape
            self.channels, self.cax = None, None
            # if that is actually the case - exit
            if not (src_form and trg_form and src_form.color and trg_form.color): return
            src_chn, trg_chn = src_form.color.channels, trg_form.color.channels
            if src_chn == trg_chn and src_form.cax == trg_form.cax: return

            self.channels = trg_chn != src_chn and (src_chn, trg_chn)
            self.cax = trg_form.cax != src_form.cax and (src_form.cax, trg_form.cax)

            shape = list(src_shape)  # evaluate target shape
            if src_form.cax is not None: shape.pop(src_form.cax)
            if trg_form.cax is not None:
                shape.insert(trg_form.cax % (len(shape) + 1), trg_chn)
            self.trg_shape = tuple(shape)

        def __bool__(self):
            return self.src_shape != self.trg_shape

        def move_cax(self, a: np.ndarray, src_trg: Literal['src', 'trg'],
                     state: bool):
            """ Move cax (if needed) of the array

            :param a: data array to move axis
            :param src_trg: which data is passed: 'src' or 'trg'
            :param state: ``True`` cax to the end, ``False`` - restore
            :return: array with cax moved
            """
            if self.cax:
                src_trg = src_trg == 'trg'
                chn = self.channels[src_trg]
                cax = self.cax[src_trg] % chn if self.cax[src_trg] else None

                if cax and cax != (end := chn - 1):
                    from_to = (cax, end) if state else (end, cax)
                    return np.moveaxis(a, *from_to)
            return a

    CALC = NamedObj('CALC')

    # =========================================================================
    name: FormKind | None = field(hash=False, compare=False)
    dtype: Any = None
    bits: int = None
    min: float = field(calc=np.nanmin)
    max: float = field(calc=np.nanmax)
    avr: float | np.ndarray | Collection[float] = field(hash=hash_array, calc=np.nanmean)
    std: float | np.ndarray | Collection[float] = field(hash=hash_array, calc=np.nanstd)
    color: Color | str = None
    cax: int = None  # IDEA: turn into axis: ordered list of axis names
    units: float = None  # IDEA: add support for physical units
    kind: Kind = None

    # ==========================================================================

    @classmethod
    def _meta(cls, fld: str, key: str, default=None):
        """
        Return key's value from the metadata of a given field
        :param fld: name of the field
        :param key: meta-key
        :param default: return this if key is not in metadata
        """
        return cls.__dataclass_fields__[fld].metadata.get(key, default)

    @classmethod
    def _filter_apply_fields(cls, fld_proc: Callable[[dcs.Field], Tuple[Any, bool]], *,
                             include: str | Collection[str] | None = None,
                             exclude: str | Collection[str] | None = None):
        """
        Generator yields first item returned by applying given `fld_proc` function
        on all or some of the dataclass fields.

        Function `fld_proc` must accept `Field` object and return a tuple `(result, accept)`:
         - `result` to be yielded ONLY if
         - `accept` is True

        Function is applied either *all* the dataclass fields, unless *one* of `include` or `exclude`
        arguments says differently (``NameError`` raised if invalid field name is given)

        :param fld_proc:
        :param include: fields to include
        :param exclude:
        """
        cls_fields = cls.__dataclass_fields__

        if include:
            if exclude:
                raise ValueError("Both include AND exclude args are provided!")

            include = as_list(include, collect=set)
            if inv := include.difference(cls_fields):
                raise NameError(f"Invalid fields: {inv}")
            fields = include
        elif exclude:
            exclude = as_list(exclude, collect=set)
            if inv := exclude.difference(cls_fields):
                raise NameError(f"Invalid fields: {inv}")
            fields = set(cls_fields) - exclude
        else:
            fields = cls_fields

        for name in fields:
            val, use = fld_proc(cls_fields[name])
            if use:
                yield val

    @classmethod
    def field_calc(cls, fld: str):
        return cls._meta(fld, 'calc')

    @classmethod
    def from_data(cls, a: np.ndarray, /, name: str = None, *,
                  dtype: np.dtype = None, **attrs) -> DForm:
        """
        Create DForm by estimating some attributes from given ``data``, which
        must be either ``ndarray`` or iterable convertible into it.

        In this case its ``dtype`` is first defined in this order:
            ``dform`` argument, ``dform.dtype``, ``DForm._default_dtype``

        Attributes values may be inherited from given dform if provided.
        That can be overriden by providing corresponding argument,
        which could have different meanings:
            - ``None``: not provided
            - <value>: use it for the attribute
            - "calculate": estimate from the data using internally defined function
            - `Callable`: estimate from the data using this function

        If attribute value is not defined by the argument and ``dform`` is ``None``,
        it remains None, except of `dtype` which is then assigned `a.dtype``.

        :param a: ndarray to use for calculations
        :param dform: Provides base to inherit the attributes
        :param dtype:
        :param name: Optional name for new DForm
        :keyword min: min value of the range
        :keyword max: max value of the range
        :keyword std: std of the data distribution
        :keyword avr: avr of the data distribution
        :keyword color: one of the `DForm.Color`
        :keyword cax: ``int`` - color axes location (if color is defined)
        :return: resulting dform
        """
        if not isinstance(a, np.ndarray):
            a = np.array(a, dtype=dtype)
        if not dtype:
            dtype = a.dtype

        def value(fld, v):
            if v == 'calculate':
                if calc := cls._meta(fld, 'calc'):
                    return calc(a)
                raise NotImplementedError(f"Calc function not defined for '{fld}'")
            if isinstance(v, Callable):
                return v(a)
            return v

        attrs = {k: value(k, v) for k, v in attrs.items()}

        return DForm(name=name, dtype=dtype, **attrs)

    @classmethod
    def from_merge(cls, dform: DForm | str | type = None, **fields) -> DForm:
        """Create a new object by replacing one with new fields values.

        - If ``dform`` is omitted, use fields defaults.
        - If ``fields`` are *not provided either*, return ``None``!

        :param dform to merge with in any form acceptable by DForm constructor
        :param fields: any subset of valid DForm fields
        """
        if dform:
            if not isinstance(dform, cls):
                dform = cls(dform)
            if not fields:
                return dform
            fields = dict(dform.items()) | fields
        if fields:
            return DForm(**fields)
        raise ValueError("No arguments in from_merge")

    def type_info(self) -> str:
        base = f"{self.name or ''}{_dtype_str(self.dtype)}"
        if not self.extra_info():
            return f"☉{base}"
        rng = "" if self.min is None else f"⎣{self.min:g}➔{self.max:g}⌉"
        vals = lambda s: ', '.join(f"{_:.3g}" for _ in (s if hasattr(s, '__len__') else [s]))
        avr = "" if self.avr is None else f"<{vals(self.avr)}>"
        std = f"" if self.std is None else f"σ{vals(self.std)}"
        return f"Ⓕ{base}{rng}{avr}{std}"

    def __repr__(self):
        if self.name and self.name == self._code_name():
            return 'Ⓕ' + self.name
        return self.type_info()

    def same(self, other: FormKind, *, name: bool | Literal['case'] = 'case'):
        """
        Return ``True`` if other is the same as self including name.
        Default comparison of names is case-sensitive, use ``True`` otherwise.

        :param other: dform to compare with
        :param name: 'case' - case-sensitive when comparing names. The default.
                     True - compare names. False - don't.
        :return: bool
        """
        # Consider: replace that by hash comparison once DForm is frozen
        cls = type(self)
        if not isinstance(other, cls):
            # noinspection PyBroadException
            try:
                other = cls(other)  # creates default dform (f4) if dform of 'other' is None
            except:
                return False

        if name not in (allowed := (True, False, 'case')):
            raise ValueError(f'name argument {name} is not among {allowed=}')
        if name and not (
                self.name == other.name if (name is True or self.name is None or other.name is None) else
                self.name.lower() == other.name.lower()
        ): return False

        def not_equal(f):
            """Return only yield-able False"""

            def size(v):
                return 0 if v is None else 1 if not isinstance(v, np.ndarray) else len(v)

            if f.compare:
                v1, v2 = getattr(self, f.name), getattr(other, f.name)
                size_v1 = size(v1)
                size_v2 = size(v2)  # Consider: cmp() is not enough?
                if (cmp := f.metadata.get('compare', None)) and cmp(v1, v2) or \
                        (size_v1 == size_v2 and size_v1 > 1 and (v1 == v2).all()) or \
                        (size_v1 == size_v2 and size_v1 <= 1 and v1 == v2):  # 1D
                    return None, False  # don't yield (None not to be used)
                else:
                    return False, True  # yield False
            return None, False  # don't yield

        for _ in self._filter_apply_fields(not_equal): return _
        return True

    def __eq__(self, other: DForm):
        return self.same(other, name=False)

    def _calc_hash(self):
        d = self.__dict__

        def hash_filter(f):
            if f.hash is False:
                return None, False

            v = d[f.name]
            if v is not None and (hash_fnc := f.metadata.get('hash', None)):
                v = hash_fnc(v)
            return v, True

        return hash(tuple(self._filter_apply_fields(fld_proc=hash_filter)))

    def __hash__(self):
        found_hash = getattr(self, '_cached_hash', None)
        if not found_hash or self._verify_cache:  # FIXME: freeze DForm and remove verifications
            now_hash = self._calc_hash()
            if found_hash and now_hash != found_hash:  # can be true only if self.__verify_cache
                raise RuntimeError(f"Inconsistent cache for {self}")
            self._cached_hash = now_hash

        return self._cached_hash

    def extra_info(self) -> dict:  # ToDo: replace with hash based after DForm freeze
        """Return dict with extra type information (besides name and dtype) which is not None."""
        return dict(
            self._filter_apply_fields(
                exclude=('name', 'dtype'),
                fld_proc=lambda f: (
                    (f.name, val := self.__dict__[f.name]),
                    val is not None
                )
            )
        )

    @classmethod
    def fields(cls, hash_only=False, exclude=None, include=None) -> Iterator[str]:
        """Return iterator over dataclass fields' names.
        :param hash_only: if ``True`` - only those used for hash calculation
        :param exclude: optionally exclude one or more of those
        :param include: only those, first validate their existence
        """
        return cls._filter_apply_fields(
            fld_proc=lambda f: (f.name, not (hash_only and f.hash is False)),
            exclude=exclude, include=include
        )

    def items(self, hash_only=False) -> Iterator[tuple[str, Any]]:
        """Return iterator over fields items (name, value)
        :param hash_only: if True return only fields used in hash calculations
        """
        d = self.__dict__
        return ((k, d[k]) for k in self.fields(hash_only=hash_only))

    @property
    def range(self) -> float | None:
        """None or length of the range (max - min)"""
        return None if self.min is None else self.max - self.min

    @property
    def norm_dim(self) -> int:
        """
        Number of normalization channels, also can be used to check if normalized:
          - 0 - no normalization,
          - 1 - normalize without separation by channel
          - 1+ - number of separate channels

        Normalization is positive if either avr or std is defined.
        """
        return self._norm_channels

    def _code_name(self):
        # todo Why is this different name convention from type_info?
        # [C | G | B][R[ < min >:] < max >]][N[ < std >][ |][ < avr >]](u | i) < bytes > | (s | b) < bits >
        # Idea: produce signal if _parse_name(_code_name(x)) != x
        clr = self.color.short_code() if self.color else ''

        if self.range:
            rng = 'R'
            if self.min: rng += f"{self.min:g}:"
            rng += f"{self.max:g}"
        else:
            rng = ''

        if self.norm_dim > 1:
            nrm = 'N...'
        elif self.norm_dim == 1:
            std, avr = (x if x is None else x.item() for x in (self.std, self.avr))
            nrm = 'N'
            if not (std == 1 and avr == 0):
                if std is not None:
                    nrm += f"{std:g}"
                if avr is not None:
                    nrm += f"|{avr:g}"
        else:
            nrm = ''

        typ = ''
        if self.dtype.kind == 'i':
            if self.range and self.min == -self.max and (b := np.log2(self.max + 1)).is_integer():
                rng = ''
                if (b := int(b + 1)) // 8 != self.dtype.itemsize:
                    typ = f's{b}'
        elif self.dtype.kind == 'u':
            if self.range and self.min == 0 and (b := np.log2(self.max + 1)).is_integer():
                rng = ''
                if (b := int(b)) // 8 != self.dtype.itemsize:
                    typ = f'b{int(b)}'
        return f"{clr}{rng}{nrm}{typ or _dtype_str(self.dtype)}"

    def _copy_from(self, src: DForm | dict):
        """Set fields by copying them from other DForm or dict
        (if DForm _norm_channels are also copied).

        :return: number of fields set with not `None` values
        """
        if isinstance(src, DForm):
            fields = src.fields(hash_only=False)
            self._norm_channels = src._norm_channels
            src = src.__dict__
        elif isinstance(src, dict):
            fields = self.fields(hash_only=False, include=src)
        else:
            raise TypeError(f"Invalid {type(src) = }!")

        count_real_values = 0
        for f in fields:
            if (val := src[f]) is not None:
                count_real_values += 1
            setattr(self, f, val)
        return count_real_values

    def __post_init__(self):
        self._norm_channels = 0  # number of normalization channels (0 - not normalized)

        parsed = self._decode_name()  # name parsed and return True only if only argument DForm(name)
        self._init_dtype()
        self._init_range()
        self._init_norm()
        self._init_color()
        self._init_kind()

        # check name consistency for multi-fields init: DFom(name, color, dtype, range, ...)
        if (not parsed and self.name
                and (attrs := self._parse_name(self.name))  # name encodes attributes
                and self.same(DForm(**attrs), name=False)):  # matching provided through arguments
            raise NameError(f"Name is inconsistent with form's attributes")

        DForm._register(self)

    def _decode_name(self) -> bool | int:
        """ Unpack attributes encoded in the name.

        The `name` attribute string is parsed ONLY if it was the only argument.

        :return: if state changed return ``True`` or number of fields set to not ``None``
        """
        if not (name := self.name):
            return False

        name_is_str = isinstance(name, str)
        extra = self.extra_info()  # the current state of extra attributes

        if not (extra or self.dtype):  # name was the only argument:  DForm(name)
            if name_is_str:  # from a single str argument: DForm('Cu1') or DForm('KnownType')
                if dform := self.from_name(name, auto=False):  # query from DataForms registry
                    return self._copy_from(dform)  # copy from registered type
                elif attrs := self._parse_name(name):
                    return self._copy_from(attrs)
                else:
                    raise NameError(f"Unknown DForm '{self.name}'")
            elif isinstance(name, DForm):
                return self._copy_from(name)  # including name
            elif isinstance(name, type):  # first arg is type -> dtype
                self.name = None
                self.dtype = np.dtype(name)
                return True
            else:
                raise ValueError(f"Invalid name argument {self.name}")
        elif not name_is_str:  # must be string if other arguments are given
            raise ValueError(f"Expected string for DForm name {self.name}")
        elif dtype := _name_to_dtype(name):  # name encodes dtype
            if extra:  # this form is more than pure dtype
                raise NameError(f'Assigning dtype {name=} to custom dform')
            if self.dtype and np.dtype(self.dtype) != dtype:
                raise NameError(f"Assigning dform.{name=} for dtype {self.dtype}")

            self.name = None  # name argument was dtype name - not to register
            self.dtype = dtype
            return True
        return False

    def _init_color(self):
        if self.color:
            self.color = Color(self.color)  # convert from possible string
            if self.cax is None and self.color.channels > 1:
                self.cax = - 1  # default for multiple color channels
            # here we assume that if normalization is defined per channels - that color channels
            if self.norm_dim > 1 and self.norm_dim != self.color.channels: raise ValueError(
                f"Number of color & normalization channels ({self.color.channels}) != {self.norm_dim}")

            if self.color is Color.BIN:
                mn, mx = map(self.dtype.type, (0, 1))
                if self.min is None:
                    self.min, self.max = mn, mx
                elif (self.min, self.max) != (mn, mx):
                    raise ValueError(f"DForm {self} with BW color must have range [0,1]")

            if self.color is not Color.BIN and not (self.norm_dim or self.range):
                if self.dtype in (np.uint8, np.uint16):
                    info = np.iinfo(self.dtype)
                    self.min, self.max = info.min, info.max
                else:
                    raise TypeError(f"Color form of type {self.dtype} requires norm or range")
        elif self.cax is not None:
            raise ValueError(f"Color axis is defined {self.cax=} without color")

    def _init_dtype(self) -> bool:
        """Process state to determine and set dtype.
        :return: ``True`` if found and changed
        """
        if self.dtype is None:
            if not self._auto_dtype:  # see if guessing is allowed
                raise TypeError('Missing dtype in DForm definition')
            elif self.color and Color(self.color) is Color.RGB:
                self.dtype = 'uint8'
            else:
                self.dtype = self._default_dtype
            return True

        self.dtype = np.dtype(self.dtype)
        if self.dtype.kind not in 'biuf':
            raise TypeError(f"Not a numeric type for dtype!")
        return False

    def _init_range(self):
        """:return: ``True`` if range been set"""

        if (self.min, self.max) == (None, None):
            return False

        if self.min is None:
            self.min = 0
        elif self.max is None:
            raise ValueError("DForm min is defined without max")

        dtype = _collect_result_type(self.min, self.max)
        if not np.can_cast(dtype, self.dtype):
            TypeError(f"Can't cast range {dtype=} into {self.dtype}")

        self.min, self.max = map(self.dtype.type, (self.min, self.max))
        if self.max <= self.min:
            raise ValueError(f"DForm {self} has invalid range")
        return True

    def _init_norm(self):
        """
        Calculate, update and return `_norm_channels`.
        :return: 0 means neither `std` nor `avr` are defined
        """
        if self.std is not None:
            self.std = np.asarray(self.std).flatten()
            self._norm_channels = max(self._norm_channels, self.std.size)
            if self.std.min() < 10 * np.finfo('float32').eps:
                raise ValueError(f"Expected DForm std = {self.std.min()} > eps")

        if self.avr is not None:
            self.avr = np.asarray(self.avr).flatten()
            self._norm_channels = max(self._norm_channels, self.avr.size)

        return self._norm_channels

    def _init_kind(self):
        """Ensure that `color` consistent with IMAGE `kind`"""
        if not self.kind and self.color:
            self.kind = Kind.IMAGE
        elif self.kind != Kind.IMAGE and self.color:
            raise ValueError(f"{self.color} implies {Kind.IMAGE} found {self.kind}")

    def _verify_color(self, a: np.ndarray) -> str | None:
        """Verify that array matches the color info.

        Return str message describing why not or None if OK
        """
        if self.color:
            # 1 chn: MxN or MxNx1 | 3 chn: MxNx3
            if not (self.color.channels == 1 and a.ndim in (2, 3) or a.ndim == 3):
                return f"Array's dims {a.ndim} incompatible with color {self.color}"

            if not (self.cax is None or  # no color channel
                    a.ndim == 2 and self.color.channels == 1 or  # 2D image with no separate color channel
                    a.shape[
                        self.cax] == self.color.channels):  # color channel location and size
                return f"Wrong color channels # {a.shape}[{self.cax}] != {self.color.channels}"
        return None

    def verify_array(self, a: np.ndarray, exc=TypeError):
        """Verify that array matches the data form

        :param a: array to verify
        :param exc: Exception class to raise on error,
                if None - return error message instead
        :raises: If shape does not match color information
        :return: None if no error, otherwise Error message (if not exc)
        """
        assert isinstance(a, np.ndarray)

        def failure(message):
            if exc: raise exc(message)
            return message

        if self.dtype is not a.dtype: failure(f"Array and form dtypes mismatch {a.dtype}!={self.dtype}")
        if msg := self._verify_color(a):
            return failure(msg)
        return None

    @classmethod
    def guess_from_data(cls, a: np.ndarray, fail=False) -> DForm | None:
        """
        Using arrays shape and dtype try to guess its ``DForm``.
        If fails return ``None`` or raise ``TypeError`` if ``fail is True``
        :param a: np.ndarray
        :param fail: default = False return None
        :return: guessed dform or None
        """
        if a.dtype == np.uint8:  # currently only this case is supported
            return cls(a.dtype, min=0, max=255,
                       color=Color.RGB if a.ndim == 3 and (a.shape[-1] == 3)
                       else Color.GRAY if a.ndim == 1 else None)
        if fail: raise TypeError(f"Can't guess {cls.__name__} from array's properties")

    def _range_trans_params(self, src_form) -> LinT:
        """
        Calculate linear parameters (k, b) in y = kx+b for normalization amd range transforms
        If do_range and do_norm not provided, then searching first for range changing, and
        if there is not, searching for norm changing.
        :param src_form: source data format
        :return: k, b - tuple, (1, 0) if transform is not required
        """
        if not src_form.range:  # RANGE -> RANGE
            raise TypeError(f"Can't convert into range from not range-limited source {src_form}")

        k = (self.max - self.min) / (src_form.max - src_form.min)
        b = self.min - k * src_form.min
        return LinT(k, b)

    def _norm_trans_params(self, src_form: DForm) -> LinT:
        """
        Calculate linear parameters (k, b) in y = kx+b for normalization amd range transforms
        If do_range and do_norm not provided, then search first for range changing, and
        if there is not, search for norm changing.
        :param src: the source array to be transformed
        :param src_form: source data format
        :return: k, b - tuple, (1, 0) if transform is not required
        """
        src_avr = src_form.avr

        # TODO: how to continue if src_avr is None?
        if self.std is not None:  # output std requires src std or given or calculated
            src_std = src_form.std
            if self.avr is None:  # output avr not defined - only scaling
                return LinT(self.std / src_std, 0)
            return LinT(self.std / src_std, self.avr - src_form.avr * self.std / src_std)
        self.avr: float
        return LinT(1, self.avr - src_avr)  # no std conversion means k=1

    def _apply_transform(self, src, src_form: DForm, out) -> np.ndarray:
        if self.range and src_form.range and self.norm_dim:
            # TODO: need to think of all the possibilities of norm and range combinations
            if self.min != src_form.min or self.max != src_form.max:
                raise ValueError("Simultaneous Range mapping AND Normalization is not supported")

        lt = (self.range and self._range_trans_params(src_form) or
              self.norm_dim and self._norm_trans_params(src_form) or
              LinT())

        return lt.apply(src, out=out)

    def transform(self, src: np.ndarray | Array,
                  src_form: DForm | Literal['guess', 'cast', 'ignore'] = None, *,
                  round=True, out: np.ndarray = None, copy: bool = None) -> np.ndarray:
        """
        Convert ``src`` array from its DForm ``src_form`` into``self`` form.

        Argument ``src_form`` must be one of those:
            - a ``DForm`` (or cast-able into) object (overrides with warning possible ``src.dform``)
            - 'guess' str to attempt guessing DForm from ``src`` (or inherit its ``dform`` if available)
            - 'ignore' try to ``self.dform`` to src ignoring its possible ``dform`` with minimal changes.
            - ``None``, ``src`` must have ``.dform`` in this case

        If ``out`` is provided with ``.dform`` it must either match ``self.dform``
        or be compatible: same ``dtype`` and implied shape (color channels).



        :param src: ndarray to transform
        :param src_form: DForm of the source array or 'guess' to request guessing
        :param out: provide array to fill the transform in
        :param copy: if ``True`` enforce copy even if data has not been changed
        :return: ndarray converted into self DForm
        """
        src, src_form = self._prepare_src_form(src, src_form)  # None means src_form ignore is requested!
        if src_form in ('ignore', self):  # semantically equal or ignore src_form
            return self._bypass_transform(src, out, copy)

        shaper = DForm.Shaper(src.shape, src_form, self)
        if out is None:  # allocate output array if not provided
            out = np.empty(shape=shaper.trg_shape, dtype=self.dtype)  # created with cax always last!
        else:  # or verify it if it is provided
            if out.dtype != self.dtype: raise TypeError(f"Mismatch {out.dtype=} != {self.dtype}")
            if out.shape != shaper.trg_shape: raise TypeError(f"Mismatch {out.shape=} != {shaper.trg_shape}")
            out = out.view(np.ndarray)  # strip possible dform

        src = shaper.move_cax(src, 'src', True)
        out = shaper.move_cax(out, 'trg', True)

        src, src_form = self._color_pre_transform(src, src_form, out)
        self._apply_transform(src, src_form, out)

        return shaper.move_cax(out, 'trg', False)

    @staticmethod
    def _bypass_transform(src, out, copy):
        if out is None:
            out = src.copy() if copy else src
        else:
            if out.shape != src.shape or out.dtype != src.dtype:
                raise TypeError(f"Incompatible out: {out.shape}{out.dtype} with {src.shape}{src.dtype}")
            if npt.shared_view(out, src) and copy is True:
                raise ValueError("Can't copy: out shares src data")
            if copy:
                np.copyto(out, src, casting='no')
        return out

    def _color_pre_transform(self, src, src_form: DForm, out):
        """
        Color transformation can be achieved by pre-transforming ``src``, ``src_form``
        before passed to the ``_apply_transform`` method.

        :return: prepared ``src``, ``src_form``
        """
        if self.color and self.color != src_form.color:  # only if target color is defined
            clr = Color
            colors = (src_form.color, self.color)

            if not src_form.color: raise TypeError("Color transform from undefined color")
            #  if src_form.std is not None and self.norm:   # TODO: check this condition
            #  raise TypeError("Can't transform color from std-ized image")
            if clr.BIN in colors: raise TypeError("Can't convert between any color and binary")

            if colors == (clr.RGB, clr.GRAY):  # MxNx3 -> MxN
                src = np.dot(src, _wC2G).squeeze()
                if src_form.norm_dim:  # in this case possible per channel norm should be scaled too
                    weighted = lambda _: None if _ is None else np.dot(np.resize(_, 3), _wC2G).squeeze()
                    src_form = DForm(dtype=src.dtype, color=Color.GRAY,
                                     min=src_form.min, max=src_form.max,
                                     std=weighted(src_form.std),
                                     avr=weighted(src_form.avr))
            elif colors == (clr.GRAY, clr.RGB):
                src = np.dstack([src] * colors[1].channels)
            else:
                raise NotImplementedError(f'Not supported transform {colors=}')
        return src, src_form

    def _prepare_src_form(self, src: Array | np.ndarray,
                          src_form: Literal['ignore', 'guess'] | DForm | type
                          ) -> tuple[np.ndarray, DForm | Literal['ignore']]:
        """
        From various forms of arguments ``src`` and ``src_form`` produce tuple:
            - ``src`` as ``ndarray`` (stripped from dform) and
            - ``src_form`` as ``DForm`` (or 'ignore')

        If both define different DForm ``src_form != src.dform`` use ``src_form``  and issue warning.

        Raise error if:
            1. ``src_form="guess"`` but can't guess ``DForm`` from the data
            2. ``src_form=None`` and ``src`` does not have DForm information
            3. ``src_form`` can not be cast into ``DForm``
        """

        if not_array := not isinstance(src, np.ndarray):
            if src_form in (None, 'ignore', 'guess'):
                raise TypeError(f"Neither src nor {src_form=} contain dform information!")
            dform_in_src = None
        elif dform_in_src := getattr(src, 'dform', None):  # recover dform from src
            src = src.view(np.ndarray)  # before stripping

        if src_form is None:
            return src, dform_in_src or DForm(dtype=src.dtype)
        if src_form == 'ignore': return src, src_form  # dissociate any dform from src
        if src_form == 'guess': return src, dform_in_src or self.guess_from_data(src, fail=True)

        if not isinstance(src_form, DForm): src_form = DForm(src_form)
        if not_array: src = np.array(src, dtype=src_form.dtype)

        if not dform_in_src:  # src_form is provided separately - check validity
            if msg := src_form.verify_array(src): raise TypeError(msg)
        elif src_form != dform_in_src:  # IDEA: allowed case and correct behaviour?
            warnings.warn(f"Overriding {dform_in_src} by {src_form}")

        return src, src_form


_format_options = dict(
    rows=16,
    cols=12,
    precision=3,
    rng=True,
    norm=False,
    nans=True,
    stats=(9, 1e7),
    info=True
)


class FormArray(type):
    """
    Metaclass for creating Array-like classes.
    """

    EnableCond = Union[None, bool, Number, tuple[Number, Number]]
    _UND = type('_U', (), {'__str__': lambda _: 'UND'})()

    _reg: dict[FormArray, FormArray] = {}
    _np_options = np.get_printoptions()
    _def_options = _format_options.copy()

    def __new__(mcs, name: str | DForm | type, bases=(), attrs=None, *, dform: str | DForm = None,
                _base=None) -> Array:
        if _base:  # used to create base classes without dform, not for user!
            if dform: raise RuntimeError("Base Array class can't define dform")
            cls = super().__new__(mcs, name, bases, attrs or {})
            cls.dform = None
            return cls

        if dform is None:  # dfrom may be passed through name argument
            if (  # FormArray('NCu1') - name is a registered dform or follows coding convention
                    isinstance(name, str) and (  # FixMe: same as DForm.from_name(name)?
                    dform := DForm.from_name(name, auto=True)) and DForm.auto_naming()
                    or  # FormArray(dform)   - explicitly provided DForm
                    isinstance(name, DForm) and (dform := name)  # assign to dform
            ):
                name = f'Array{dform.name or "_"}'
            elif not bases and not attrs:  # FormArray('uint8') - from dtype
                dform = DForm(dtype=name)
                name = f'Array_{_dtype_str(dform.dtype)}'

        if not isinstance(dform, DForm):  # 1. FormArray() - default dtype DForm
            dform = DForm(dform)  # 2. FormArray('Name', dform=int)

        if not any(issubclass(b, np.ndarray) for b in bases):
            bases = (Array, *bases)
        cls = super().__new__(mcs, name, bases, attrs or {})
        cls.dform = dform
        cls._format_options = FormArray._def_options.copy()
        return FormArray._reg.setdefault(cls, cls)

    @classmethod  # fixme: make it self method
    def from_array(cls, data: np.ndarray | Iterable | Array, *, dform: FormKind = None,
                   copy=True, **dform_attrs) -> Array:
        """From given array create instance of Array-based class,
        with dform derived from the data or optional ``dform`` argument.

        General compatibility requirements between the data and the dform applies.

        In the minimal case with no additional arguments when only ``dtype`` can be deduced from
        the data, the resulting Array will be based on ``DForm(data.dtype)``

        - If given ``dform`` argument overrides data type in all the cases!
        - In addition or alternatively DForm attributes may be provided as ``kws``.

        If data is an iterable, then first an ``ndarray`` is created and then
        ``dtype`` is determined.

        Even if data comes with its own ``dform``, it is replaced **without transformation**!

        Also, available as a direct function in this module:
        ::
            form_array(data, dform, **dform_attrs)

        :param data: ndarray or Array or iterable
        :param dform: DForm in any of supported form
        :param copy: if False try using view when possible
        :param dform_attrs: DForm attributes as key-value pairs
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        data_form = getattr(data, 'dform', None)
        data_type = getattr(data, 'dtype', None)
        dform = DForm.from_merge(dform, **dform_attrs)

        if dform:
            if not data_type:
                return FormArray(dform)(np.array(data, dtype=dform.dtype, copy=copy))

            if data_form:  # noinspection PyUnresolvedReferences
                return FormArray(dform)(data, copy=copy)
            else:
                return FormArray(dform)(data, copy=copy)
        else:
            if data_form:
                return np.asanyarray(data)
            else:
                return FormArray(data.dtype)(data)

    @classmethod
    def global_print_options(mcs, *, rows=_UND, cols=_UND, precision=_UND, info=_UND,
                             stats: EnableCond = _UND, rng: EnableCond = _UND,
                             norm: EnableCond = _UND, nans: EnableCond = _UND, **options):
        """
        Set or get print options for ALL the FormArray classes.

        Arguments ``stats, rng, norm, nans`` define conditions when to calculate
        specific statistics, and could be:
         - set explicitly (True|False)
         - defined (min, max) limits of array's size
         - only by the maximum limit
         - set to None, in which case it follows `stats` condition

        ``stats`` is a general condition, if it evaluates to False, no calculations will be performed.
         Otherwise, other arguments will define their specific condition, as described above.

        :param cols: maximal columns to show content
        :param rows: maximal rows to show content
        :param info: bool - show info summary, (None - automatic)
        :param precision (one of ``np.set_printoptions``)
        :param stats: condition to calculate all the stats
        :param rng: condition to calculate data range (min, max)
        :param norm: condition to calculate avr and std
        :param nans: condition to count found ``nan``s (only for floating point arrays)
        :param options: other options for ``np.set_printoptions``
        :return:
        """
        loc = locals().copy()
        for k in ['mcs', 'options']: del loc[k]

        if inv := set(options).difference(mcs._np_options):
            raise KeyError(f"Unknown numpy print options: {inv}")

        defined = {k: v for k, v in loc.items() if v is not mcs._UND}
        mcs._def_options.update(**defined)
        mcs._np_options.update(**options)

        if defined or options:
            for cls in mcs._reg:
                cls.print_options(**defined, **options)
        else:
            return mcs._np_options | mcs._def_options

    def print_options(cls, *, rows=_UND, cols=_UND, precision=_UND, info=_UND,
                      stats: EnableCond = _UND, rng: EnableCond = _UND,
                      norm: EnableCond = _UND, nans: EnableCond = _UND, **options):
        """
        Set or get print options for specific class of Arrays.

        When called without arguments returns a copy of the current options.

        Arguments ``stats, rng, norm, nans`` define conditions when to calculate
        specific statistics, and could be:
         - set explicitly (True|False)
         - defined (min, max) limits of array's size
         - only by the maximum limit
         - set to None, in which case it follows `stats` condition

        ``stats`` is a general condition, if it evaluates to False, no calculations will be performed.
         Otherwise, other arguments will define their specific condition, as described above.

        :param cols: maximal columns to show content
        :param rows: maximal rows to show content
        :param info: bool - show info summary, (None - automatic)
        :param precision (one of ``np.set_printoptions``)
        :param stats: condition to calculate all the stats
        :param rng: condition to calculate data range (min, max)
        :param norm: condition to calculate avr and std
        :param nans: condition to count found ``nan``s (only for floating point arrays)
        :param options: other options for ``np.set_printoptions``
        :return:
        """
        loc = locals().copy()
        for k in ['cls', 'options']: del loc[k]

        if inv := set(options).difference(cls._np_options):
            raise KeyError(f"Unknown numpy print options: {inv}")

        defined = {k: v for k, v in loc.items() if v is not cls._UND}

        if options or defined:
            cls._format_options.update(**options, **defined)
        else:
            return cls._format_options.copy()

    @overload
    def __init__(cls, class_name: str, *, dform: str | DForm):
        ...

    @overload
    def __init__(cls, dform: str | DForm | type):
        pass

    def __init__(cls, name: str | DForm | type, bases=(), attrs: dict = None, *,
                 dform: str | DForm = None, _base=None):
        """
        Create a new ``Array`` like class with associated ``DForm``.
        Supported initializations:

        >>> FormArray('MyArray', (Array,), {}, dform=DForm('Cu1') )  # full form
        >>> FormArray('MyArray', dform='Cu1')  # Assuming Cu8 is a registered DForm
        >>> FormArray('Cu1') == FormArray('ArrayCu8', 'Cu1') == FormArray(DForm('Cu1'))

        :param name: class name or dform (registered name or an instance)
        :param bases: base classes include at least one subclass of ``np.ndarray``.
                    Otherwise, ``Array`` is automatically.
        :param attrs: optional additional attributes
        :param dform: ``DForm``- name or instance. ``None`` equivalent to ``DForm()``
        """

    def __init_subclass__(cls, *, dform=None):
        if dform is None:
            dform = DForm()
        elif not isinstance(dform, DForm):
            raise TypeError(f"dform must be None or DForm")
        cls.dform = dform

    def __eq__(self, other: FormArray):
        if isinstance(other, FormArray):
            return self.dform == other.dform
        return False

    def __hash__(self):
        return hash(self.dform) + hash(self.__name__)

    def __repr__(self):
        return self.__qualname__


# noinspection PyShadowingBuiltins
class Array(np.ndarray, metaclass=FormArray, _base=True):
    dform: DForm

    @property
    def ndarray(self):
        return np.asarray(self)

    def __repr__(self):
        return npt.array_info_func(**self._format_options)(self)

    def __str__(self):
        return self.__repr__()

    def __new__(cls, obj, dform=None, *, trans=False, round=False, copy=False):
        """
        Creating a new instance of an Array-like class.
        Notice! This class can create instances of another classes, not only Array.
        Use cases:

        >>> Array([])                   # same as Array([], 'f4').from_array([])
        >>> Array([], dform='Nf2')      # creates instance of ArrayNf2

        >>> ArrayCu1([])                # ok
        >>> ArrayCu1([], dform='Nf2')   # error

        :param obj: data as iterable or ndarray or Array-like with dfrom
        :param trans: allow data transfromation when casting into new form
        :param copy: if False try to avoid memory copy (use view)
        :param dform: only allowed in base class (Array)
        """
        if cls.dform:  # ArrayF(a)
            if dform and dform != cls.dform:  # ArrayF1(af, f1)   !ArrayF1.dform == f1
                raise TypeError(f"argument {dform=} clashes with {cls.dform=}")
        else:  # Array(obj, dform='Nu1')
            return FormArray(dform or FLOAT)(obj, trans=trans, round=round, copy=copy)

        if type(obj) == cls:  # Array(obj:Frm, Frm)  !From -> Frm
            return obj.copy() if copy else obj

        if not isinstance(obj, np.ndarray):  # ArrayFrm([1,2])
            obj = np.array(obj)  # - create ndarray
            copy = False
            if obj.dtype.kind not in 'bifu':
                raise TypeError('Array data must be of numeric type')

        if trans:  # ArrayF(a:ndarray)
            obj = cls.dform.transform(obj, copy=copy, round=round)
        else:
            obj = cast(obj, dtype=cls.dform.dtype, round=round, copy=copy)
        return obj.view(cls)

    # noinspection PyMethodOverriding
    def __array_finalize__(self, obj):
        if obj is None: return
        if self.dform.dtype and self.dform.dtype != self.dtype:
            raise TypeError(f"dtype {self.dtype} incompatible with {self.dform}'s {self.dform.dtype}")

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        # Based on: https://numpy.org/doc/stable/user/basics.subclassing.html
        # Required to make sure ndarray are provided where expected in Universal functions (Ufunc).
        # Also avoids automatic conversion of scalar results into Array

        # Unless out arrays are provided, all the resulting arrays are returned as Array (empty DForm)
        # leaving to the user reassignment to the proper DForm if needed.
        # That because an automatic propagation of the DForm would be too tricky in most the cases.
        # TODO: should we implement those few clear cases?

        args = tuple(input_.view(np.ndarray) if isinstance(type(input_), FormArray) else
                     input_ for input_ in inputs)  # view inputs as regular ndarrays

        if out:  # also view as ndarrays optional outputs
            outputs = out
            kwargs['out'] = tuple(output.view(np.ndarray) if isinstance(type(output), FormArray) else
                                  output for output in outputs)
        else:
            outputs = (None,) * ufunc.nout

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented: return NotImplemented
        if method == 'at': return

        def select(r, o):
            if o is not None: return o
            if not isinstance(r, np.ndarray): return r
            return np.asarray(r).view(FormArray(r.dtype))

        if ufunc.nout == 1: return select(results, outputs[0])
        return tuple(select(result, output) for result, output in zip(results, outputs))

    # def __init__(self, obj, dform=None, *, trans=False, copy=False):
    #     """
    #     Something
    #
    #     :param obj:
    #     :param dform:
    #     :param trans:
    #     :param copy:
    #     """
    #     ...

    def asform(self, dform: str | DForm = None, **kws):
        """
        Return present array as a different `DForm`, based on the:
         1. empty `DForm()` if ``dform = None (and no kws)``
         2. provided as DForm instance
         3. provided as a name to registered DForm instance

        Additional keyword arguments may be provided to override specific ``DForm`` attributes:
            ``name, color, cax, std, avr, min, max``

        Returned array is of a different type (FormArray sub-class).

        Use this method when specificity of the form of an existing data must be updated.
        Initial transform is ignored

        Notice:
            - only attributes are replaced
            - ``dtype`` can not be changed
            - resulting array is another view on same data

        :return: An instance of a new Array subclass with different ``DForm`` but same data.
        """
        if isinstance(dform, str):
            dform = DForm(dform)
        elif dform is None:
            if not kws:  # neither dform nor kws defined - strip dform
                return FormArray(self.dtype)(self.ndarray)
            dform = self.dform  # inherit from self.dform with updates from kws
        elif not isinstance(dform, DForm):
            raise ValueError(f"Invalid argument {dform=}")

        if kws:
            attrs = dcs.fields(DForm)
            if unknown := set(kws).difference(attrs):
                raise KeyError(f"DForm has no attribute: {unknown}")

            # inherit undefined attributes from the dform
            if 'dtype' in kws: raise TypeError("dtype can not be overriden")
            kws = {f.name: kws.get(f.name, getattr(dform, f.name))
                   for f in attrs if f.init}
            return FormArray(DForm(**kws))(self.ndarray)
        elif dform == self.dform:
            return np.asanyarray(self)
        else:
            return dform.transform(self, src_form='ignore')

    def transform(self, dform: DForm, *, guess=True, copy=True, out=None):
        """Creates a NEW object of required form.
        If ``dform`` is same as ``self.dform`` data is not transformed.

        If ``self.dform`` is not defined, attempt to guess it before failing.
        Undefined ``dform`` always fails.

        :param dform: target DForm of resulting array
        :param guess: if ``self.dform`` is not defined, guess from the data
        :return: new array instance with requested ``dform``.
        """
        # todo implement copy and out (with and without dform). If out neglect copy
        if isinstance(dform, str): dform = DForm(dform)
        if dform == self.dform: return type(self)(self)
        src_form = 'guess' if (self.dform is None and guess) else self.dform
        return FormArray(dform)(dform.transform(self, src_form=src_form))


# Consider auto-generation based on the naming convention:
#
# [C|G[<bits>b]][R[<min>[:]]<max>|N[<std>[:<avr>]]
# C - color (RGB)
# G - Gray scale
# R - Range
# N - Normalized (by default std=1, avr=0)
#
# In R<min>:<max> ':' optional if min and max single digit
# That makes Rxy ambiguous: R0:xy or Rx:y. Second is assumed


# Some standard data forms
# DForm('R01f4', FLOAT, min=0, max=1)  # any dim
# DForm('R-11f4', dtype=FLOAT, min=-1, max=1)  # any dim
#
# DForm('CR01f4', dtype=FLOAT, min=0, max=1, color=DForm.Color.RGB)  # HxWx3
# DForm('CNf4', FLOAT, color=DForm.Color.RGB, avr=0, std=1)
# DForm('Nf4', dtype=FLOAT, avr=0, std=1)
#
# DForm('Cu1', dtype='u1', color=DForm.Color.RGB)
# DForm('Gb10', dtype='u2', min=0, max=1023, color=DForm.Color.GRAY)

@overload
def form_array(data: Iterable | np.ndarray, *, dform: FormKind = None, copy=False) -> Array: ...


@overload
def form_array(data: Iterable | np.ndarray, *, copy=False, **dform_attrs) -> Array: ...


def form_array(data: Iterable | np.ndarray | Array, *,
               dform: FormArray | FormKind = None, copy=False,
               **dform_attrs) -> Array:
    """
    From given array create instance of Array-based class,
    with dform derived from the data or optional ``dform`` argument.

    In basic case resulting Array will be based on ``DForm(data.dtype)``

    - If given ``dform`` argument overrides data type in all the cases!
    - In addition or alternatively DForm attributes may be provided as ``kws``.

    If data is an iterable, then first an ``ndarray`` is created and then
    ``dtype`` is determined.

    Equivalent to:
    ::
        FormArray.from_array(data, dform=dform, **dform_attrs)

    :param data: ndarray or Array or iterable
    :param dform: Array-like class or DForm in any of supported form
    :param dform_attrs: DForm attributes as key-value pairs
    """
    return FormArray.from_array(data, dform=dform, copy=copy, **dform_attrs)
