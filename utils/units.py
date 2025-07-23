import warnings
from typing import Union

import numpy as np
import pint

un = pint.UnitRegistry(auto_reduce_dimensions=True)
# print(f'--- init units reg {id(un)}!')
Q_ = Quantity = un.Quantity


def isQ(x):
    return isinstance(x, pint.Quantity)


def is_array(x):
    """ Check if its any kind of array """
    return isinstance(x, pint.Quantity) and isinstance(x.magnitude, np.ndarray) \
        or isinstance(x, np.ndarray)


def magnitude(x):
    """Safely return magnitude even if not a Quantity"""
    return x.m if isinstance(x, pint.Quantity) else x


def __format__(self, spec):
    if isinstance(self.magnitude, np.ndarray) and self.magnitude.size > 32:
        units = '{:~P}'.format(self.units)
        a = self.magnitude
        nans_num = np.isnan(a).sum()
        metrics = "" if a.size == nans_num else \
            "\u2208 [{:{fmt}}, {:{fmt}}] <{:{fmt}}> \u03c3={:{fmt}} ".format(
                np.nanmin(a), np.nanmax(a), np.nanmean(a), np.nanstd(a),
                fmt=".4" if a.dtype.kind == 'f' else '')
        metrics += f"({nans_num / a.size:.1%} nans)" * int(nans_num > 0)
        return f"[{a.shape[0]}x{a.shape[1]}]{units} ({a.dtype}) {metrics}"
    return pint.Quantity.__format__(self, spec)


Quantity.__format__ = __format__
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Quantity([])


def assign_units(q, u: Union[float, str, pint.Quantity]) -> pint.Quantity:
    """Assign units to the value """
    if u != 1:
        if not hasattr(q, 'units'):
            if isinstance(u, str):
                u = un(u)
            elif not hasattr(u, 'units'):
                raise TypeError('Invalid type of units arg. Expected Pint')
            q = q * u
        elif u.units != q.units:
            raise ValueError(f"Can't apply units {u.units} on quantity with {q}")
    return q
