import warnings
from functools import wraps
from numbers import Number
from operator import itemgetter
from typing import Union

import numpy as np
import uncertainties as uncert
from uncertainties import unumpy as unp

from .pdtools import pd, DataTable

UV = Union[pd.DataFrame, pd.Series, np.ndarray, tuple, 'UncT']


class UncT(tuple):
    def __new__(self, n, s=None):
        if s is None:
            n, s = split_unc(n)
        #
        # assert n.shape == s.shape
        # assert not (is_unc_array(n) or is_unc_array(s) or is_unc_series(n) or is_unc_series(s))
        # assert all(isinstance(n, t) and isinstance(s, t)
        #            for t in (pd.DataFrame, pd.Series, np.ndarray))
        return tuple.__new__(UncT, (n, s))

    def __repr__(self):
        v = self.n
        shape = "{}x{}".format(*v.shape) if v.ndim == 2 else f"{len(v)}"
        return f"{self.__class__.__qualname__}<{self.n.__class__.__qualname__}>[{shape}]"

    def __getattr__(self, item):
        return UncT(getattr(self.n, item), getattr(self.s, item))

    def join(self):
        return join_unc(*self)

    def __truediv__(self, other):
        return divide(self, other)

    def __mul__(self, other):
        return multiply(self, other)

    def __add__(self, other):
        return add(self, other)

    def __neg__(self):
        return UncT(-self[0], self[1])

    def __sub__(self, other):
        return add(self, -other)

    def __pow__(self, power, modulo=None):
        return pow(self, power, sep=False)

    def divide(self, other, axis=0, eps=0, sep=True):
        return divide(self, other, axis=axis, eps=eps, sepr=sep)

    def conv(self, other, axis=0, sep=True):
        return conv(self, other, axis=axis, sep=sep)

    def sum(self, axis=0, **kws):
        return sum(self, axis=axis, **kws)


UncT.n = UncT.nominal_value = property(itemgetter(0))
UncT.s = UncT.std_dev = property(itemgetter(1))


def is_unc_array(v):
    return isinstance(v, np.ndarray) and v.dtype == 'O' and isinstance(next(v.flat), uncert.UFloat)


def is_unc_series(s):
    return isinstance(s, pd.Series) and s.dtype == 'O' and isinstance(s[0], uncert.UFloat)


def is_unc_df(s):
    return isinstance(s, pd.DataFrame) and any(map(lambda _, x: is_unc_series(x), s.iteritems()))


def is_unc_cont(v):
    return is_unc_series(v) or is_unc_array(v) or is_unc_df(v)


def join_unc(dfn, dfs, cls=None):
    cls = cls or dfn.__class__
    a = unp.uarray(dfn, dfs)
    if isinstance(cls, np.ndarray):
        return a
    kws = dict(columns=dfn.columns) if isinstance(dfn, pd.DataFrame) else dict(name=dfn.name)
    return cls(a, index=dfn.index, **kws)


def is_sep_array(v1):
    return isinstance(v1, tuple) and isinstance(v1[0], np.ndarray)


def split_unc_df(df: pd.DataFrame):
    recreate_table = lambda x: df.__class__._from_arrays(x, index=df.index, columns=df.columns)
    return tuple(map(recreate_table, zip(*(
        (sr.nominal_value, sr.std_dev) if is_unc_series(sr) else (sr, sr * 0)
        for _, sr in df.iteritems()
    ))))


def split_unc(v: UV):
    return v if isinstance(v, tuple) else (
        (unp.nominal_values(v), unp.std_devs(v)) if is_unc_array(v) else
        (v.nominal_value, v.std_dev) if is_unc_series(v) else
        split_unc_df(v) if isinstance(v, pd.DataFrame) else (v, 0)
    )


def add(v1, v2, *, axis=0, sep=None):
    oper = (lambda x, y: np.add(x, y)) if is_sep_array(v1) else (
        lambda x, y: x.add(y, axis=axis))
    err = lambda n1, s1, _, s2: (s1**2 + s2**2) ** 0.5
    return bin_operation(v1, v2, oper, err=err, sep=sep)


def multiply(v1, v2, axis=0, *, sep=None):
    oper = (lambda x, y: np.multiply(x, y)) if is_sep_array(v1)\
        else (lambda x, y: x.multiply(y, axis=axis))
    err = lambda n1, s1, n2, s2: (oper(s1**2, n2**2) + oper(s2**2, n1**2)) ** 0.5
    return bin_operation(v1, v2, oper, err=err, sep=sep)


def divide(v1: UV, v2: UV, axis=0, *, eps=0, sep=None):
    oper = (lambda x, y: np.true_divide(x+eps, y+eps)) if is_sep_array(v1)\
        else (lambda x, y: (x+eps).divide(y+eps, axis=axis))
    rel_err = lambda n1, s1, n2, s2: (oper(s1, n1) ** 2 + oper(s2, n2) ** 2) ** 0.5
    return bin_operation(v1, v2, oper, rel_err=rel_err, sep=sep)


def pow(v1: UV, p: float, *, sep=None):
    oper = lambda x, y: x**y
    err = lambda n1, s1, n2, _: abs(n2 * n1**(n2-1)) * s1
    return bin_operation(v1, (p, 0), oper, err=err, sep=sep)


def conv(v1: UV, v2: UV, axis=0, *, unc=True, sep=False):
    """
    Calculate multiple 1D convolutions with uncertainty along selected axis
    between two sets of vectors provided as series, arrays or dataframes
    each optionally containing the uncertainty component.

    Uncertainty calculation and inclusion in the result may be disabled.
    Type of the result is DataTable or array (if both inputs are arrays)

    Shape of the result is D1 x D2, where D* are lengths of non-convolved
    dimensions in first and second inputs.

    Inputs must have same alignment of the convolution axis, same
    length and same index (if its a series or a dataframe)

    :param v1: first set of vectors
    :param v2: first set of vectors
    :param axis: axis to sum along - must match actual dimension!
    :param unc: if False disables uncertainty calculations
    :param sep: if True return as a UncT tuple (n, s)
    :return: D1 x D2 DataTable or array
    """
    assert axis in (0, 1)
    assert v1.shape[axis] == v2.shape[axis], f"inputs must have size along {axis} dimension"

    array_inputs = isinstance(v1, np.ndarray) + 2*isinstance(v2, np.ndarray)
    if not array_inputs:  # series and frames, index must match!
        v1, v2 = map(lambda x: x.to_frame() if isinstance(x, pd.Series) else x, (v1, v2))
        v2 = v2.loc[:, v1.columns] if axis else v2.loc[v1.index, :]
    if array_inputs in (1, 2):  # convert to DataTable unless ALL the inputs are arrays
        v1, v2 = map(lambda x: DataTable(x) if isinstance(x, np.ndarray) else x, (v1, v2))

    # convert to numpy arrays
    n1, s1 = split_unc(v1)
    n2, s2 = split_unc(v2)
    n1, n2, s1, s2 = map(lambda x: x.to_numpy(), (n1, n2, s1, s2))

    if axis == 0:   # transpose for matrix multiplication
        n1, s1 = n1.T, s1.T
    else:
        n2, s2 = n2.T, s2.T

    a = n = n1 @ n2
    if unc:
        s = ((s1**2)@(n2**2) + (n1**2)@(s2**2)) ** 0.5
        if not sep:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'invalid value')
                a = unp.uarray(n, s)

    if array_inputs == 3:
        return (UncT(n, s) if unc else n) if sep else a

    iax = 1 - axis  # the other axis
    return (UncT(DataTable(n, index=v1.axes[iax], columns=v2.axes[iax]),
                 DataTable(s, index=v1.axes[iax], columns=v2.axes[iax])
                 ) if unc else
            DataTable(n, index=v1.axes[iax], columns=v2.axes[iax])
            ) if sep else DataTable(a, index=v1.axes[iax], columns=v2.axes[iax])


def sum(v: UV, axis=0, **kws):
    n, s = split_unc(v)
    return join_unc(n.sum(axis=axis, **kws), (s**2).sum(axis=axis, **kws)**0.5)


UncT.add = add
UncT.divide = divide
UncT.multiply = multiply
UncT.conv = conv
UncT.pow = pow


def bin_operation(v1: UV, v2: UV, func, *, err=None, rel_err=None, sep=None):
    assert err is None or rel_err is None
    n1, s1 = split_unc(v1)
    n2, s2 = split_unc(v2)

    n = func(n1, n2)
    s = err(n1, s1, n2, s2) if err else rel_err(n1, s1, n2, s2) * abs(n)

    if sep is None:
        sep = isinstance(v1, tuple) and (isinstance(v2, tuple) or
                                         isinstance(v2, Number) or
                                         not is_unc_cont(v2))
    if sep:
        return UncT(n, s)

    if isinstance(n, Number):
        return uncert.core.ufloat(n, s)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'invalid value')
        res = unp.uarray(n, s)

    if isinstance(n, pd.Series):
        return n.__class__(res, index=n.index, name=n.name)

    if isinstance(n, pd.DataFrame):
        return n.__class__(res, index=n.index, columns=n.columns)

    if isinstance(n, np.ndarray):
        return res

    raise TypeError(f"Unsupported type: {type(n)}")


def strip_unc(v: UV):
    if is_unc_series(v, tuple):
        return v[0]
    if is_unc_array(v):
        return unp.nominal_values(v)
    if isinstance(v, np.ndarray):
        return v
    return v.nominal_value  # series or frame


def unc_func(f):
    @wraps(f)
    def uf(x, *args, **kws):
        if is_unc_series(x):
            return x.__class__(f(x.values, *args, unc=True, **kws), index=x.index, name=x.name)
        if isinstance(x, pd.DataFrame):
            return x.apply(lambda s: f(s.values, *args, unc=is_unc_series(s), **kws))
        if isinstance(x, uncert.UFloat) or is_unc_array(x):
            return f(x, *args, unc=True, **kws)
        return f(x, *args, **kws)

    return uf
