__all__ = ['Pose', 'zero3D', 'Vec2d', 'Range', 'Rect', 'in_region', 'TupleXY']

import numpy as np
from typing import Union, Iterable, Optional
from numbers import Real, Number
import operator
import math
from collections import namedtuple

TupleXY = namedtuple('TupleXY', ['x', 'y'])
Range = namedtuple('Range', ['low', 'high'])


class Vec2d(object):
    """2d vector class, supports vector and scalar operators,
       and also provides a bunch of high level functions
       """
    __slots__ = ['x', 'y']

    def __init__(self, x_or_pair, y=None):
        if y is None:
            self.x = x_or_pair[0]
            self.y = x_or_pair[1]
        else:
            self.x = x_or_pair
            self.y = y

    def __len__(self):
        return 2

    def __getitem__(self, key):
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        else:
            raise IndexError("Invalid subscript " + str(key) + " to Vec2d")

    def __setitem__(self, key, value):
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        else:
            raise IndexError("Invalid subscript " + str(key) + " to Vec2d")

    # String representation (for debugging)
    def __repr__(self):
        return 'Vec2d(%s, %s)' % (self.x, self.y)

    # Comparison
    def __eq__(self, other):
        if hasattr(other, "__getitem__") and len(other) == 2:
            return self.x == other[0] and self.y == other[1]
        else:
            return False

    def __ne__(self, other):
        if hasattr(other, "__getitem__") and len(other) == 2:
            return self.x != other[0] or self.y != other[1]
        else:
            return True

    def __nonzero__(self):
        return bool(self.x or self.y)

    # Generic operator handlers
    def _o2(self, other, f):
        """Any two-operator operation where the left operand is a Vec2d"""
        if isinstance(other, Number):
            return Vec2d(f(self.x, other), f(self.y, other))
        if isinstance(other, Vec2d):
            return Vec2d(f(self.x, other.x), f(self.y, other.y))
        if hasattr(other, "__getitem__") and hasattr(other, "__len__"):
            return Vec2d(f(self.x, other[0]), f(self.y, other[1]))
        return Vec2d(f(self.x, other), f(self.y, other))

    def _r_o2(self, other, f):
        "Any two-operator operation where the right operand is a Vec2d"
        if hasattr(other, "__getitem__") and hasattr(other, "__len__"):
            return Vec2d(f(other[0], self.x),
                         f(other[1], self.y))
        else:
            return Vec2d(f(other, self.x),
                         f(other, self.y))

    def _io(self, other, f):
        "inplace operator"
        if hasattr(other, "__getitem__"):
            self.x = f(self.x, other[0])
            self.y = f(self.y, other[1])
        else:
            self.x = f(self.x, other)
            self.y = f(self.y, other)
        return self

    # Addition
    def __add__(self, other):
        if isinstance(other, Vec2d):
            return Vec2d(self.x + other.x, self.y + other.y)
        elif hasattr(other, "__getitem__"):
            return Vec2d(self.x + other[0], self.y + other[1])
        else:
            return Vec2d(self.x + other, self.y + other)

    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, Vec2d):
            self.x += other.x
            self.y += other.y
        elif hasattr(other, "__getitem__"):
            self.x += other[0]
            self.y += other[1]
        else:
            self.x += other
            self.y += other
        return self

    # Subtraction
    def __sub__(self, other):
        if isinstance(other, Vec2d):
            return Vec2d(self.x - other.x, self.y - other.y)
        elif hasattr(other, "__getitem__"):
            return Vec2d(self.x - other[0], self.y - other[1])
        else:
            return Vec2d(self.x - other, self.y - other)

    def __rsub__(self, other):
        if isinstance(other, Vec2d):
            return Vec2d(other.x - self.x, other.y - self.y)
        if hasattr(other, "__getitem__"):
            return Vec2d(other[0] - self.x, other[1] - self.y)
        else:
            return Vec2d(other - self.x, other - self.y)

    def __isub__(self, other):
        if isinstance(other, Vec2d):
            self.x -= other.x
            self.y -= other.y
        elif hasattr(other, "__getitem__"):
            self.x -= other[0]
            self.y -= other[1]
        else:
            self.x -= other
            self.y -= other
        return self

    # Multiplication
    def __mul__(self, other):
        if isinstance(other, Vec2d):
            return Vec2d(self.x * other.x, self.y * other.y)
        if hasattr(other, "__getitem__"):
            return Vec2d(self.x * other[0], self.y * other[1])
        else:
            return Vec2d(self.x * other, self.y * other)

    __rmul__ = __mul__

    def __imul__(self, other):
        if isinstance(other, Vec2d):
            self.x *= other.x
            self.y *= other.y
        elif hasattr(other, "__getitem__"):
            self.x *= other[0]
            self.y *= other[1]
        else:
            self.x *= other
            self.y *= other
        return self

    # Division
    def __div__(self, other):
        return self._o2(other, operator.div)

    def __rdiv__(self, other):
        return self._r_o2(other, operator.div)

    def __idiv__(self, other):
        return self._io(other, operator.div)

    def __floordiv__(self, other):
        return self._o2(other, operator.floordiv)

    def __rfloordiv__(self, other):
        return self._r_o2(other, operator.floordiv)

    def __ifloordiv__(self, other):
        return self._io(other, operator.floordiv)

    def __truediv__(self, other):
        return self._o2(other, operator.truediv)

    def __rtruediv__(self, other):
        return self._r_o2(other, operator.truediv)

    def __itruediv__(self, other):
        return self._io(other, operator.floordiv)

    # Modulo
    def __mod__(self, other):
        return self._o2(other, operator.mod)

    def __rmod__(self, other):
        return self._r_o2(other, operator.mod)

    def __divmod__(self, other):
        return self._o2(other, operator.divmod)

    def __rdivmod__(self, other):
        return self._r_o2(other, operator.divmod)

    # Exponentation
    def __pow__(self, other):
        return self._o2(other, operator.pow)

    def __rpow__(self, other):
        return self._r_o2(other, operator.pow)

    # Bitwise operators
    def __lshift__(self, other):
        return self._o2(other, operator.lshift)

    def __rlshift__(self, other):
        return self._r_o2(other, operator.lshift)

    def __rshift__(self, other):
        return self._o2(other, operator.rshift)

    def __rrshift__(self, other):
        return self._r_o2(other, operator.rshift)

    def __and__(self, other):
        return self._o2(other, operator.and_)

    __rand__ = __and__

    def __or__(self, other):
        return self._o2(other, operator.or_)

    __ror__ = __or__

    def __xor__(self, other):
        return self._o2(other, operator.xor)

    __rxor__ = __xor__

    # Unary operations
    def __neg__(self):
        return Vec2d(operator.neg(self.x), operator.neg(self.y))

    def __pos__(self):
        return Vec2d(operator.pos(self.x), operator.pos(self.y))

    def __abs__(self):
        return Vec2d(abs(self.x), abs(self.y))

    def __invert__(self):
        return Vec2d(-self.x, -self.y)

    # vectory functions
    def length_sqr(self):
        return self.x ** 2 + self.y ** 2

    def __get_length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def __set_length(self, value):
        k = value / self.__get_length()
        self.x *= k
        self.y *= k

    length = property(__get_length, __set_length, None, "gets or sets the magnitude of the vector")

    def rotate(self, angle_degrees):
        radians = math.radians(angle_degrees)
        cos = math.cos(radians)
        sin = math.sin(radians)
        x = self.x * cos - self.y * sin
        y = self.x * sin + self.y * cos
        self.x = x
        self.y = y

    def rotated(self, angle_degrees):
        radians = math.radians(angle_degrees)
        cos = math.cos(radians)
        sin = math.sin(radians)
        x = self.x * cos - self.y * sin
        y = self.x * sin + self.y * cos
        return Vec2d(x, y)

    def __get_angle(self):
        if self.length_sqr() == 0:
            return 0
        return math.degrees(math.atan2(self.y, self.x))

    def __set_angle(self, angle_degrees):
        self.x = self.length
        self.y = 0
        self.rotate(angle_degrees)

    angle = property(__get_angle, __set_angle, None, "gets or sets the angle of a vector")

    def angle_between(self, other):
        cross = self.x * other[1] - self.y * other[0]
        dot = self.x * other[0] + self.y * other[1]
        return math.degrees(math.atan2(cross, dot))

    def normalized(self):
        length = self.length
        if length != 0:
            return self / length
        return Vec2d(self)

    def normalize_return_length(self):
        length = self.length
        if length != 0:
            self.x /= length
            self.y /= length
        return length

    def perpendicular(self):
        return Vec2d(-self.y, self.x)

    def perpendicular_normal(self):
        length = self.length
        if length != 0:
            return Vec2d(-self.y / length, self.x / length)
        return Vec2d(self)

    def dot(self, other):
        return float(self.x * other[0] + self.y * other[1])

    def distance(self, other):
        return math.sqrt((self.x - other[0]) ** 2 + (self.y - other[1]) ** 2)

    def get_dist_sqrd(self, other):
        return (self.x - other[0]) ** 2 + (self.y - other[1]) ** 2

    def projection(self, other):
        other_length_sqrd = other[0] * other[0] + other[1] * other[1]
        projected_length_times_other_length = self.dot(other)
        return other * (projected_length_times_other_length / other_length_sqrd)

    def cross(self, other):
        return self.x * other[1] - self.y * other[0]

    def interpolate_to(self, other, range):
        return Vec2d(self.x + (other[0] - self.x) * range, self.y + (other[1] - self.y) * range)

    def convert_to_basis(self, x_vector, y_vector):
        return Vec2d(self.dot(x_vector) / x_vector.length_sqr(), self.dot(y_vector) / y_vector.length_sqr())

    def __getstate__(self):
        return [self.x, self.y]

    def __setstate__(self, dict):
        self.x, self.y = dict


class _Meta2D(type):
    @classmethod
    def __prepare__(metacls, name, bases, dtype=float, default=np.nan):
        cls_dict = super().__prepare__(name, bases)
        slots = ['x', 'y']

        def __init__(self,
                     first: Union[dtype, Iterable] = (default, default),
                     second: Optional[dtype] = None):
            """
            Create 2D coordinate container of {dtype}
            Args:
                first:  x or iterable of len=2 for xy (deafults = ({default}, {default}) )
                second: y (if first is x) or not needed if iterable is passed
            """
            self.x, self.y = ((dtype(v) for v in first) if hasattr(first, '__iter__')
                              else (first, second))

        __init__.__doc__ = __init__.__doc__.format(dtype=dtype,
                                                   default=default)  # insert data type infor into the docstring

        cls_dict.update(
            __slots__=slots,
            __annotations__={m: dtype for m in slots},
            xy=property(lambda self: (self.x, self.y)),
            ij=property(lambda self: (int(self.x), int(self.y))),
            array=property(lambda self: np.ndarray(self.xy)),
            dtype=dtype,
            __init__=__init__,
            __iter__=lambda self: iter((self.x, self.y)),
            __eq__=lambda self, other: ((self.x, self.y) == other),
            __getitem__=lambda self, key: getattr(self, 'xy'[key]),
            __setitem__=lambda self, key, val: setattr(self, 'xy'[key], val)
        )
        return cls_dict

    def __new__(mcs, name, bases, namespace, **kargs):
        return super().__new__(mcs, name, bases, namespace)


# TODO: 3D geometry classes?
zero3D = np.zeros(3)
V3D = np.ndarray


class Pose:
    rot: V3D
    trans: V3D

    def __init__(self, rot=0, trans=0):
        def prep(v):
            if v == 0:
                return zero3D
            assert len(v) == 3
            return v if isinstance(v, np.ndarray) else np.array(v)
        self.rot, self.trans = (prep(v) for v in (rot, trans))

    def __repr__(self):
        rot, trans = ('0' if (self.rot == zero3D).all() else '{},{},{}'.format(*v)
                      for v in (self.rot, self.trans))
        return f'R:{rot} T:{trans}'


class Rect:
    __slots__ = ['low', 'high']
    low:  Vec2d
    high: Vec2d

    def __init__(self, low: Union[Vec2d, Iterable],
                 high: Optional[Union[Vec2d, Iterable]] = None,
                 *, dim: Optional[Union[Vec2d, Iterable]] = None):
        """
        Create rectangle from two corner points or from low point and dimension
        :param low: tuple, pix2D
        :param high: tuple, pix2D - optional
        :param dim: tuple, pix2D - if high is not provided
        """
        self.low = Vec2d(low)
        self.high = Vec2d(self.low + dim if high is None else high)
        if self.low.x > self.high.x or self.low.y > self.high.y:
            raise ValueError('Low coordinates may not be larger than the high')

    def __repr__(self):
        return f'Rect({self.low.x}, {self.low.y}; {self.high.x}, {self.high.y})'

    def __iter__(self):
        return iter((self.low, self.high))

    def __eq__(self, other):
        return self.low == other.low and self.high == other.high

    def extend(self, add: Union[Real, Vec2d]):
        self.low -= add
        self.high += add

    def move(self, shift: Vec2d):
        self.low += shift
        self.high += shift

    def __contains__(self, p: Vec2d):
        if isinstance(p, Rect):
            return self.high.x <= p.high.x <= p.low.x <= self.low.x and \
                   self.high.y <= p.high.y <= p.low.y <= self.low.y
        elif not isinstance(p, Vec2d):
            p = Vec2d(p)
        return self.low.x <= p.x <= self.high.x and self.low.y <= p.y <= self.high.y

    @property
    def dim(self):
        return self.high - self.low

    @property
    def ranges(self) -> TupleXY:
        """ representation as pair of two segments """
        return TupleXY(Range(self.low.x, self.high.x),
                       Range(self.low.y, self.high.y))

    def vertices(self):
        """ Generator of all the vertices from low in clock-wise direction"""
        yield Vec2d(self.low.x, self.low.y)
        yield Vec2d(self.low.x, self.high.y)
        yield Vec2d(self.high.x, self.high.y)
        yield Vec2d(self.high.x, self.low.y)


def in_region(coords, region):
    """
    Checks if D-dim coordinates of N points are inside D-dim region
    :param coords: [D x N] - iterable of D 1-d arrays of length N
    :param region: [D x 2] - iterable of D segments of (low, high) for each dimension
                    low <= v < high
    :return: np.bool(N) boolean array of length N  indicating if a point inside the region
    """
    from functools import reduce
    by_dimensions = (((x >= low) & (x < high)) for x, (low, high) in zip(coords, region))
    return reduce(np.logical_and, by_dimensions)
