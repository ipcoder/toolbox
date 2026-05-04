""" Convenience tools for NumPy package  """

__all__ = ['var_filter', 'lindex', 'xy2i', 'Array', 'min_ids',
           'valid_in_range']

from typing import Union, Any

import numpy as np
from scipy import signal


class Array(np.ndarray):
    _default_options = dict(
        rows=16,
        cols=12,
        precision=3,
        stats=1e6,
        info=True
    )

    _print_options = _default_options.copy()

    UNDEF = object()

    @classmethod
    def set_printoptions(self, rows=UNDEF, cols=UNDEF, stats=UNDEF, info=UNDEF,
                         precision=UNDEF, **options):
        """

        :param cols: maximal columns to show content
        :param rows: maximal rows to show content
        :param stats: maximal number of elements to calc stats for info
        :param info: bool - show info summary, (None - automatic)
        :param precision (one of np.setprintoptions)
        :param options: other options for `np.set_printoptions`
        :return:
        """
        options = {k: v for k, v in options.items() if k in np.get_printoptions()}

        loc = locals()
        defined = {k: v for k, v in loc.items() if k in self._default_options and v is not self.UNDEF}

        self._print_options.update(**defined, **options)

    def __repr__(self):
        return array_info_func(**self._print_options)(self)

    def __str__(self):
        return self.__repr__()

    def __new__(cls, obj, *, dtype=None, **kwargs):
        return np.asarray(obj, dtype).view(cls)


def valid_in_range(m, min_v, max_v):
    """Returns boolean mask
    with True where m is not nan and in the given range.
    Avoids Runtime warning on nan comparisons!
    :param m: array
    :param min_v: m >= min_v
    :param max_v: m <= max_v
    :return:
    """
    np.warnings.filterwarnings('ignore')
    valid = m >= min_v
    np.logical_and(valid, m < max_v, out=valid)
    np.warnings.resetwarnings()
    return valid


def var_filter(im, ker_sz):
    """ Fast variance in a window filter (normalized by n*n-1)

    :param im: 2D array
    :param ker_sz: windows 1D size (will create window sz x sz)
    :return: 2D array os same size as im
    """
    if hasattr(ker_sz, '__len__'):
        assert len(ker_sz) == 2
        var_win = ker_sz
    else:
        var_win = (ker_sz, ker_sz)
    assert var_win[0] % 2 == 1 and var_win[1] % 2 == 1

    flt = np.ones(var_win, dtype='int')
    im = im.astype(float)
    sum_x2 = signal.convolve2d(im ** 2, flt, mode='same', boundary='symm')
    sum_x = signal.convolve2d(im, flt, mode='same', boundary='symm')
    num = var_win[0] * var_win[1]
    return (sum_x2 * num - sum_x ** 2) / (num ** 2)


def has_nans(a) -> bool:
    # Checks if an array has any nans (fast implementation)
    a = a.reshape(a.size)
    return np.isnan(np.dot(a, a))


def lindex(coords):
    """
    Create index for n-dimensional array from array of N n-dimensional coordinates.
    Coordinates may not be integer - they are converted to int inside
    :param coords: (N x n) N - number of coordinates, n -dimension of array to index
    :return: numpy nd array compatible index : tuple of tuples
    """
    return tuple(zip(*coords.astype(int)))


def xy2i(xy, y=None):
    """
    Translate into ij 2d-indexes various representation of set of x-y coordinates.

    :param xy: x or array[N x (x,y)]  or obj with attrs or items 'x', 'y'
    :param y: if xy is x, then y  - otherwise None!
    :return: index for 2d arrays addressing tuple(is, js)

    Examples:
        xy2i(x, y)       # numerical arrays of x, y coordinates

        xy2i(xy_array)   # convertible to Nx2 array of pairs [[x0, y0], ..., ]

        xy2i(dict_xy)    # object with items ['x'] and ['y']

        xy2i(obj_xy)     # object with attributes .x and .y

        array [Nx2] of N 2-dimensional coordinates (x, y)

    Create tuple(is, js) index selecting corresponding elements from a 2D array.
        - values casted to int!


    """
    if y is None:
        if hasattr(xy, 'x') and hasattr(xy, 'y'):
            xy = xy.x, xy.y
        elif hasattr(xy, '__getitem__') and 'x' in xy and 'y' in xy:
            xy = xy['x'], xy['y']
        else:
            xy = np.array(xy, dtype=int).T
    else:
        xy = (xy, y)
    return tuple(np.round(np.array(v)).astype(dtype=int) for v in xy[::-1])


def slice_1_axis(ndim, n, sl):
    """ Create n-dim slice with given 1-d slice for given dimension
    and rest of dimensions not sliced
    :param ndim: number of dimensions
    :param n: location of the slice dimension
    :param sl: the slice to insert
    """
    assert 0 <= n < ndim
    if ndim == 1:
        return sl
    ll = [slice(None), ] * (ndim - 1)
    ll.insert(n, sl)
    return tuple(ll)


def min_ids(a, n=1, *, axis=-1):
    """
    Find indices of n smallest elements along given axis.
    Args:
        a - array
        n - number of best elements
        axis - direction to sort by (default -1 to flatten)
    Return:
        For a.shape == N_0 ... N_axis ... N_dim, returns
        array.shape == N_0 ...   n    ... N_dim
    """
    return np.argpartition(a, n - 1, axis=axis).take(range(n), axis=axis)


def abool(value: Union[Any, np.ndarray]) -> bool:
    """ A convenience utility: boolean conversion which works on both scalars and numpy arrays.

    For arrays returns:
    Args:
        value: any type supporting bool() or numpy array

    Returns:
        True | False

    """
    return value.all() if isinstance(value, np.ndarray) else bool(value)


def array_info_str(a, stats=1e7):
    """
    Short description of array, like:
        [4×816×1200×7]f4 ⌊-30.53|454.01⌋ 0.2%NA

    To be used with or formatters:
        np.set_string_function(array_info_str)

    :param a: array-like object
    :param stats: maximal number of elements to add calculate statistics
    """
    sz = 1
    for _ in a.shape: sz *= _  # to make it work for both numpy and torch tensors
    s = '×'.join(map(str, a.shape)) or (f"val={a.item()}" if sz == 1 else '×')

    if isinstance(a, np.ndarray):
        kind = a.dtype.kind
        s = f"[{s}]{kind}{a.dtype.itemsize}"
    else:  # torch! - No stats and fancy type info implemented yet
        dtype = str(a.dtype)[6:].replace('float', 'f')  # remove 'torch.'
        return f"[{s}]{dtype}{(':c' if a.is_cuda else '')}"

    if not 1 < sz < stats: return s

    if kind == 'f':
        _a = a[np.isfinite(a)]
        mn, mx = np.min(_a), np.max(_a)
        s = f"{s} ⌊{mn:.3g}|{mx:.3g}⌉"

        def occurrence_info(name, num, num_lim=1000):
            if not (num := int(num)):
                return ''
            ss = str(num) if num < num_lim else f"{num / a.size}%"
            return f"{ss}{name}"

        if occ := ', '.join(filter(None, [
            occurrence_info('∅', nans := np.isnan(a).sum()),
            occurrence_info('∞', a.size - _a.size - nans)  # _a excluded both inf and nan!
        ])): s += f"({occ})"

    elif kind != 'O':
        s += f" ⌊{a.min():.3g}|{a.max():.3g}⌉"

    return s


def ascii_map(a, levels=None, show=False):
    """
    Unicode representation of array as map of shaded blocks.
    :param a: array
    :param levels: number of levels
    :param show: if True - print debug info
    :return: string (for printing)
    """
    import re
    shades = '·⬚░▒▓█'
    full = lambda smb: re.sub(r'[0\[\]]', ' ', str(np.full_like(a, 1, dtype=int)).replace('1', smb))

    mx = a.max()
    mn = a.min()

    if not (np.isfinite(mn) and np.isfinite(mx)):
        return full('⊗')
    if mn == mx:
        return full(shades[0] if mn == 0 else shades[-1])

    ua = np.unique(a)
    if levels is None:
        levels = min(len(shades), len(ua))

    if levels < 2:
        return full()

    max_level = levels - 1
    assert max_level < len(shades)
    idx = np.linspace(0, len(shades) - 1, max_level + 1, dtype=int)

    levels = np.linspace(0, max_level, levels, dtype=int)
    if len(ua) == max_level + 1:
        la = np.empty_like(a, dtype=int)
        for l, v in zip(levels, ua):
            la[a == v] = l
        # print(la)
        s = str(la)
    else:
        s = str(((max_level / (mx - mn)) * (a - mn)).astype(int))

    if show:
        print('unique:', ua)
        print('idx:   ', idx)
        print('levels:', levels)

    for i, l in zip(idx, levels):
        s = s.replace(str(l), shades[i])
    return re.sub(r'[0\[\]]', ' ', s)


def pprint_arrays(do=True, **kws):
    """
    Set representation function of nd arrays and torch tensors
    to ``array_info_func(**kws)``.

    :param do: set to False to reset
    :param kws: kwargs for array_info_func
    """
    if do:
        func = array_info_func(**kws)
        np.set_string_function(func)
        pprint_tensors(do)
    else:
        np.set_string_function()
        pprint_tensors(False)


def pprint_tensors(do=True):
    """
    Session level hack to replace (or restore)
    ``Tensor.__repr__`` with ``array_info_func()``
    """
    try:
        import torch
    except ModuleNotFoundError:
        from warnings import warn
        do and warn("torch not found  Can't hack Tensor repr")
        return

    if do:
        torch.Tensor._original_repr = torch.Tensor.__repr__
        torch.Tensor.__repr__ = array_info_func()
    else:
        torch.Tensor.__repr__ = torch.Tensor._original_repr


def array_info_func(cols=12, rows=8, *, stats=1e7, info=None, **options):
    """ Return regular numpy array smart compact representation,
    optionally including info summary (`array_info_str`).

    :param cols: maximal columns to show content
    :param rows: maximal rows to show content
    :param stats: maximal number of elements to calc stats for info
    :param info: bool - show info summary, (None - automatic)
    :param options: for `set_printoptions`, like:
        edgeitems=3, threshold=1000, floatmode=maxprec, precision=2,
        suppress=False, linewidth=75
    """

    def content_repr(a):
        if options:
            prev = np.get_printoptions()
            np.set_printoptions(**options)
            s = np.ndarray.__str__(a)
            np.set_printoptions(**prev)
            return s
        return np.ndarray.__str__(a)

    def is_compact(last_dim, size):
        return last_dim == 0 or (
                0 < last_dim <= cols and size / last_dim <= rows)

    def full_repr(a):
        show_content = a.ndim and (info is False or is_compact(
            a.shape[-1], a.size if isinstance(a, np.ndarray) else a.numel()
        ))
        content = show_content and content_repr(a)
        if content and not info:  # return only content unless info is requested
            return content

        inf = array_info_str(a, stats=stats)
        return f"{inf}:\n{content}" if content else inf

    return full_repr


def stats(data, measure, out=str, prec=3, sep=', '):
    """Calculate multiple metrics on data
    :param data: array
    :param measure: list of functions names from `numpy` module
                    first try to prepend `nan` to the name.
    :param out:  if `str` return result as string otherwise dict
    ":param prec: precision of the fixed-point representation
    """
    import numpy as np
    itr = [(msr, getattr(np, 'nan' + msr, getattr(np, msr))(data)) for msr in measure]
    if out is str:
        fmt = f"{{}}: {{:.{prec}f}}"
        return sep.join(map(lambda mv: fmt.format(*mv), itr))
    return dict(itr)


def cast_float(a, dtype=int, *, nan=0, inf='nan'):
    """
    From floating point array create integer typed and set nans and inf to specific values

    :param a: array
    :param nan: value to replace nans
    :param inf: value to replace inf or
                'nan' - same as nan
                'clip' - clip by maximal positive or negative int
    :param dtype: int type to cast into
    :return: int array
    """

    res = a.astype(dtype)

    if inf == 'nan':
        res[~np.isfinite(a)] = nan
        return res

    res[np.isnan(a)] = nan
    if inf == 'clip':
        res[a == np.inf] = np.iinfo(dtype).max
        res[a == -np.inf] = np.iinfo(dtype).min
    else:
        res[np.abs(a) == np.inf] = inf

    return res


def fill_mask(a, mask, *, val=np.nan, inplace=False):
    """
    Fill array with elements where `~mask` set to `nan`
    :param a:
    :param mask: binary array describing (True) area to fill
    :param val: value to fill the mask with (default - nan)
    :param inplace: change and return the input array
    :return:
    """
    na = a if inplace else a.copy()
    na[mask] = val
    return na


def full_indices(shape, *, base=1, dtype=int):
    """
    Create array of given shape full of values composed of its indices.

    Example:

    >>> full_indices([2,3])
    array([[11, 21, 31],
           [12, 22, 32]])

    :param shape: shape to create
    :param base: number to start the indexing from (usually 0 or 1)
    :param dtype: of resulting array.
    :return: array filled with index values
    """
    a = np.empty(shape, dtype=dtype)
    p10 = np.power(10, np.arange(len(shape))).astype(int)
    for idx, _ in np.ndenumerate(a):
        a[idx] = ((np.array(idx) + base).astype(int) * p10).sum()
    return a


def copy_nans(src, dst):
    """Set nans found in source into same locations in dst """
    if src.dtype.kind == 'f':
        nans = np.isnan(src)
        if np.count_nonzero(nans):
            dst = dst.copy()
            dst[nans] = np.nan
    return dst
