from copy import deepcopy
from typing import Literal, Callable, Any, Iterable, overload, Collection, get_args, Mapping

import numba as nb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from algutils import as_list
from algutils.datatools import select_from

__all__ = ['Sampler', 'log_compress', 'bins_edges', 'StatGather', 'StatGather2D']

_cache_nb = True

Number = float | int | np.number
_UNDEF = object()

_NormT = Literal['total', 'range', 'nonan', None, False, True]
_Axis = Literal[0, 1]

HistT = np.uint32  # counters type

SUM_METRICS = Literal['s1', 's2', 'finite', 'total']
BASIC_METRICS = Literal['mean', 'rmse', 'std']
METRICS = Literal[SUM_METRICS, BASIC_METRICS]

_sum_metrics = get_args(SUM_METRICS)
_basic_metrics = get_args(BASIC_METRICS)


def pack_bit_arrays_64(arrays):
    """
    Pack list of binary arrays into single array of same shape,
    with every initial arrays represented by its bit-plane.

    Supports up to 64 arrays, raise ValueError if given more than that.

    Resulting array dtype is u[8,16,32, 64] depending on the number of inputs.

    :param arrays:
    :return: bit-packed array
    """
    from algutils.binary import align_type_bits
    n = len(arrays)
    shapes = [a.shape for a in arrays]
    shape = shapes[0]
    if not all(s == shape for s in shapes[1:]):
        raise ValueError(f"All arrays must be of same shape: {shapes}")

    bits = align_type_bits(n)
    dtype = np.dtype(f'uint{bits}')

    out = np.zeros(shape, dtype=dtype)
    _b = np.zeros(1, dtype=dtype)
    arrays = tuple(np.ascontiguousarray(a) for a in arrays)
    _pack_bit_array(arrays, out, _b)
    return out


@nb.njit
def _pack_bit_array(arrays, out, _b):
    out = out.flat
    for shift, array in enumerate(arrays):
        for i, a in enumerate(array.flat):
            _b[0] = a << shift  # force casting into out type
            out[i] += _b[0]


def init_sum_stats(*, num_regions: int, ndim: Literal[1, 2], dtype=np.float64) -> dict[SUM_METRICS, NDArray]:
    """
    Prepare accumulating arrays data structure for different configurations.

    If ``num_regions`` > 0 than creates 2D arrays to have separate dimension for regions.

     **Notice!** that size of regions dimension is ``num_regions`` + 1 for separate 'full' statistics.

    Return dict {metric: zeros[``num_regions + 1`` × ``ndim``] | zeros[``ndim``]}

    :param num_regions: number of selection masks collecting separate statistics
    :param ndim: number of variables to collect statistics for
    :param dtype: type of values accumulator (use default)
    :return:
    """
    shape = (
        (1 + num_regions, ndim) if num_regions else
        (ndim,) if ndim > 1 else ()
    )  # ToDo: masks_num > 1 and ndim == 1?

    return {'s1': np.zeros(shape, dtype=dtype),
            's2': np.zeros(shape, dtype=dtype),
            'finite': np.zeros(shape, dtype=np.int64),
            'total': np.zeros(shape, dtype=np.int64)}


def _basic_stats(s1, s2, finite, **_) -> dict[BASIC_METRICS, NDArray | Number]:
    """Calculate `mean, msre, std` from the accumulated sums.
    If nothing is accumulated, set them to nan.
    :param s1: sum of finite values
    :param s2: sum of their squares
    :param finite: their number
    :return: dict(mean=..., std=...)
    """
    if not np.asarray(finite).any():
        return select_from({}, _basic_metrics, default=np.nan)
    with np.errstate(all='ignore'):
        m1, m2 = s1 / finite, s2 / finite
    return {'mean': m1,
            'rmse': np.sqrt(m2),
            'std': np.sqrt(m2 - m1 * m1)}  # ToDo: Add test of matching keys to _basic_measures


@nb.jit(nopython=True, cache=_cache_nb)
def _equal_bins_stats(a, low, bin_size, hist: np.ndarray,
                      log_scale: float = 0, sums=False):
    """
    Low level optimized function to collect various typical statistical measurement over array:
      1. Histogram with bins *equidistant* in either the `data` or `log1p(data)` space.
        - *The histogram array must be pre-allocated and is filled by the function.*
      2. Basic statistical measurements over array of numbers:
        - Counts distribution between equal-sized bins distribution
        - Calculates `sum(x)`, `sum(x**2)`, `count(x)` for *finite* `x`.

    Special 3 bin are created for:
    values outside the given range and nan, with indices:
      * 0 for x <= low      (below)
      * -2 for x > high     (inf)
      * -1 for nans         (nans)

    **Note**: for case of `data >= 0` to use the first (below) bin for `x > 0`, set `low=bin_size`.
    ::
                low                            high       inf       nan    <- right edges[]
        _________|_________|____...____|_________|_________|_________|
        low_bin=0                        high_bin  inf_bin   nan_bin       <- bins indices

    :param a: array - could be float
    :param low: right edge of the lowest bin (must be finite)
    :param bin_size: size of every bin
    :param hist: hist array to be updated inside
    :param sums: if `True` also calculate for finite values their number, sum and sum of squares
    :param log_scale: flag to pass the data through log before distributing between the bins
    :return (s1, s2, finite, total)
    """
    bins = len(hist)
    high_bin, inf_bin, nan_bin = range(bins - 3, bins)  # high_bin == number of bins in [low, high]

    finite: int = 0  # count finite values
    s1 = s2 = 0.  # accumulate sum and sum of squares over finite values
    for x in a.flat:
        if np.isnan(x):
            bin_id = nan_bin
        elif np.isinf(x):
            bin_id = inf_bin if x > 0 else 0  # below
        else:
            if sums:  # statistics is calculated on uncompressed data
                s1 += x  # requires attention in parallel mode to avoid race condition!
                s2 += x * x
                finite += 1

            if log_scale != 0:
                x = _log_compress(x, log_scale)

            # here x is finite, but still can fall into [< low] or [> high] bins.
            bin_id = int((x - low) / bin_size) + 1  # +1 since 0 for < low
            if bin_id > high_bin:
                bin_id = high_bin
            elif bin_id < 0:
                bin_id = 0

        hist[bin_id] += 1

    return s1, s2, finite, a.size
    # return dict(
    #     [('s1', s1), ('s2', s2), ('finite', finite), ('total', a.size)])  # this form is eeasy for nb :-


@nb.njit(inline='always', cache=_cache_nb)
def _find_bin(x, low, bin_size, nan_bin, log_scale):
    """find bin id ∈ `[0, bins-1]` for `x` ∈ `[low-bins_size, high]` with 3 special bins:
    ::
        0:              x < low
        nan_bin - 1:    x > high
        nan_bin: for    x == nan

    :param x: value *before compression*
    :param low: right edge of first bin inside the `x` range (bin id = 1)
    :param bin_size: size of the bin *after compression*
    :param nan_bin: id of the bin for `nan`s ( = bins - 1)
    :param log_scale: values compression factor, 0 for no compression
    """
    if np.isnan(x):  # first deal with nan to avoid nan arithmetics
        bin_id = nan_bin
    else:
        if log_scale != 0:
            x = _log_compress(x, log_scale)
        bin_id = int((x - low) / bin_size) + 1  # +1 since 0 for < low

        if bin_id < 0:  # apply range clipping on the calculated bin_id
            bin_id = 0
        elif bin_id >= nan_bin:
            bin_id = nan_bin - 1
    return bin_id


def _equal_bins2d_stats(a1, a2, *,
                        low: tuple[Number, Number],
                        bin_size: tuple[Number, Number],
                        masks: Collection[np.ndarray] | None = None,
                        hist: np.ndarray = None,
                        log_scale: tuple[Number, Number] = (0, 0),
                        stats: dict[SUM_METRICS, NDArray[Number]] | None = None,
                        sums=True) -> dict[SUM_METRICS, NDArray]:
    """
    Wraps low level compiled versions to:
        1. support both with and without ``masks`` cases
        2. bring input data into the required shape
        3. create ``stats`` data structure if not passed in

    :param a1:
    :param a2:
    :param low:
    :param bin_size:
    :param masks:
    :param hist:
    :param log_scale:
    :param stats:
    :param sums:
    :return: stats
    """

    assert all(len(_) == 2 for _ in (low, bin_size, log_scale))
    a1, a2 = map(np.asarray, [a1, a2])
    assert a1.shape == a2.shape

    num_masks = 0 if masks is None else len(masks)
    if not stats:
        stats = init_sum_stats(num_regions=num_masks, ndim=2)

    a1, a2 = (a.reshape(-1) for a in [a1, a2])  # _equal.. requires 1D for a and
    if num_masks:
        assert hist.ndim == 3 and hist.shape[0] == num_masks + 1
        masks = pack_bit_arrays_64(masks).reshape(-1)
        _nb_equal_bins2m_stats(a1, a2, masks, low=low, bin_size=bin_size, hist=hist,
                               **stats, log_scale=log_scale, sums=sums)
    else:
        assert hist.ndim == 2
        a1, a2 = (a.reshape(-1) for a in [a1, a2])  # _equal.. requires 1D for a and
        _nb_equal_bins2_stats(a1, a2, low=low, bin_size=bin_size, hist=hist,
                              **stats, log_scale=log_scale, sums=sums)
    return stats


@nb.jit(nopython=True, cache=_cache_nb)
def _nb_equal_bins2m_stats(a1: np.ndarray, a2: np.ndarray, masks: np.ndarray,
                           low: tuple, bin_size: tuple, hist: np.ndarray,
                           s1: np.ndarray, s2: np.ndarray, finite: np.ndarray,
                           total: np.ndarray, log_scale: tuple = (0, 0), sums=True):
    """
    Compiled low level version with special requirements for some of its inputs.

    Also, there are some additional conventions:
     - results are provided by *incrementing* input arguments: ``hist``, ``s1``, ``s2``, ``finite``
     - ``hist`` array is a stack of 2d arrays (for every mask + 1 for *full*)
     - accumulating arrays (``s1``, ``s2``, ``finite``, ``total``) also have additional row for full
     - number of bins is deduced from the shape of ``hist``
     - to calculate *without* ``masks`` ensure `masks.shape[0] == 0`
     - ``low`` tuple[2] represents right edges of first range bin for 2 dimensions (a1, a2)
     - ``bin_size`` tuple[2] also represents 2 dimensions

    :param a1: ndarray[`size`] 1D
    :param a2: ndarray[`size`] 1D
    :param masks: ndarray[masks_num × `size`] or ndarray[`masks_num = 0` × any]
    :param low: right edges of the first inside the values range (per dimension)
    :param bin_size: sine of bin > 0, per dimension
    :param hist: ndarray[1 + `masks_num` × bins1 × bins2]
    :param s1: ndarray[1 + `masks_num` × 2] sum of finite values per mask per dimension
    :param s2: ndarray[1 + `masks_num` × 2] sum of squares of finite values per mask per dimension
    :param finite: ndarray[1 + `masks_num` × 2] number of the summed finite values per mask per dimension
    :param log_scale: compression scale factor per dimension
    :param sums: calculate stats by updating arguments: ``s1``, ``s2``, ``finite``
    :return:
    """
    assert a1.shape == a2.shape and a1.ndim == 1 and hist.ndim == 3
    # num_masks = masks.shape[0]  # always add first (0) "full" mask
    # assert masks.size == num_masks * a1.size
    num_hist, bins = hist.shape[0], hist.shape[1:]
    # assert num_hist == 1 + num_masks

    one = HistT(1)

    for i in range(a1.size):  # over data elements (pixels)
        v1, v2, m = a1[i], a2[i], masks[i]
        b1 = _find_bin(v1, low[0], bin_size[0], bins[0] - 1, log_scale[0])
        b2 = _find_bin(v2, low[1], bin_size[1], bins[1] - 1, log_scale[1])

        for ii in range(num_hist):  # over 1(full) + len(masks) regions
            if not ii or ((m >> (ii - 1)) & 1):  # first (ii == 0) is the virtual "full" region
                hist[ii, b1, b2] += one
                for d in range(2):
                    v = v2 if d else v1
                    if sums:  # ToDo: move above for d
                        if np.isfinite(v):  # statistics is calculated on uncompressed data
                            s1[ii, d] += v
                            s2[ii, d] += v * v
                            finite[ii, d] += one
                        total[ii, d] += one


@nb.jit(nopython=True, cache=_cache_nb)
def _nb_equal_bins2_stats(a1: np.ndarray, a2: np.ndarray,
                          low: tuple, bin_size: tuple, hist: np.ndarray,
                          s1: np.ndarray, s2: np.ndarray, finite: np.ndarray,
                          total: np.ndarray, log_scale: tuple = (0, 0), sums=True):
    """
    Compiled low level version with special requirements for some of its inputs.

    Also, there are some additional conventions:
     - results are provided by *incrementing* input arguments: ``hist``, ``s1``, ``s2``, ``finite``
     - hist array is a stack of 2d arrays per every masks + 1 for *full*
     - number of bins is deduced from the shape of ``hist``
     - to calculate *without* ``masks`` ensure `masks.shape[0] == 0`
     - ``low`` tuple[2] represents right edges of first range bin for 2 dimensions (a1, a2)
     - ``bin_size`` tuple[2] also represents 2 dimensions

    :param a1: ndarray[`size`] 1D
    :param a2: ndarray[`size`] 1D
    :param low: right edges of the first inside the values range (per dimension)
    :param bin_size: sine of bin > 0, per dimension
    :param hist: ndarray[1 + `masks_num` × bins1 × bins2]
    :param s1: ndarray[1 + `masks_num` × 2] sum of finite values per mask per dimension
    :param s2: ndarray[1 + `masks_num` × 2] sum of squares of finite values per mask per dimension
    :param finite: ndarray[1 + `masks_num` × 2] number of the summed finite values per mask per dimension
    :param total: total number of data elements processed per mask per dimension
    :param log_scale: compression scale factor per dimension
    :param sums: calculate stats by updating arguments: ``s1``, ``s2``, ``finite``
    :return:
    """
    bins = hist.shape
    for v1, v2 in zip(a1, a2):
        b1 = _find_bin(v1, low[0], bin_size[0], bins[0] - 1, log_scale[0])
        b2 = _find_bin(v2, low[1], bin_size[1], bins[1] - 1, log_scale[1])
        hist[b1, b2] += 1
        for d in range(2):
            v = v2 if d else v1
            if sums:
                if np.isfinite(v):  # statistics is calculated on uncompressed data
                    s1[d] += v
                    s2[d] += v * v
                    finite[d] += 1
                total[d] += 1


#
# @nb.jit(nopython=True)
# def _equal_bins_stats(a, low, bin_size, hist: np.ndarray,
#                       log_scale: float = 0, sums=False):
#     """
#     Low level optimized function to collect various typical statistical measurement over array:
#       1. Histogram with bins *equidistant* in either the `data` or `log1p(data)` space.
#         - *The histogram array must be pre-allocated and is filled by the function.*
#       2. Basic statistical measurements over array of numbers:
#         - Counts distribution between equal-sized bins distribution
#         - Calculates `sum(x)`, `sum(x**2)`, `count(x)` for *finite* `x`.
#         *Stats are returned explicitly, as dict, even if empty array is given*.
#
#     Special 3 bin are created for:
#     values outside the given range and nan, with indices:
#       * 0 for x <= low      (below)
#       * -2 for x > high     (inf)
#       * -1 for nans         (nans)
#
#     **Note**: for case of `data >= 0` to use the first (below) bin for `x > 0`, set `low=bin_size`.
#     ::
#                 low                            high       inf       nan    <- right edges[]
#         _________|_________|____...____|_________|_________|_________|
#         low_bin=0                        high_bin  inf_bin   nan_bin       <- bins indices
#
#     :param a: array - could be float
#     :param low: right edge of the lowest bin (must be finite)
#     :param bin_size: size of every bin
#     :param sums: if `True` also calculate for finite values their number, sum and sum of squares
#     :param log_scale: flag to pass the data through log before distributing between the bins
#     :return dict.keys() = (s1, s2, finite, total)
#     """
#     bins = len(hist)
#     high_bin, inf_bin, nan_bin = range(bins - 3, bins)  # high_bin == number of bins in [low, high]
#
#     finite: int = 0  # count finite values
#     s1 = s2 = 0.  # accumulate sum and sum of squares over finite values
#     for x in a.flat:
#         if np.isnan(x):
#             bin_id = nan_bin
#         elif np.isinf(x):
#             bin_id = inf_bin if x > 0 else 0  # below
#         else:
#             if sums:  # statistics is calculated on uncompressed data
#                 s1 += x  # requires attention in parallel mode to avoid race condition!
#                 s2 += x * x
#                 finite += 1
#
#             if log_scale != 0:
#                 x = _log_compress(x, log_scale)
#
#             # here x is finite, but still can fall into [< low] or [> high] bins.
#             bin_id = int((x - low) / bin_size) + 1  # +1 since 0 for < low
#             if bin_id > high_bin:
#                 bin_id = high_bin
#             elif bin_id < 0:
#                 bin_id = 0
#
#         hist[bin_id] += 1
#
#     return dict([('s1', s1), ('s2', s2),
#                  ('finite', finite), ('total', a.size)])  # easy to compile
#


@nb.njit(cache=_cache_nb)
def index_count(indices, num):
    """Count histogram of integer numbers representing indices ranging from 0 to num-1."""
    counts = np.zeros(num, dtype=np.int32)

    for i in indices.flat:
        counts[i] += 1
    return counts


@nb.njit(cache=_cache_nb)
def _log_compress(v, scale) -> NDArray[np.float64] | np.float64:
    """Apply logarithmic scale compression or de-compression to array or number.
    Signed data is processed as well according to:
    ::
        compressed(x) = sign(x) * log(1 + abs(x / scale)) (decompression is inverse of that)

    :param v: array or number to compress decompress
    :param scale: direction and scale of compression:
                        `+` (*positive*): compress,
                        `-` (*negative*): decompress,
                        `0`: not supported for scale == 0

    :return transformed array
    """
    scale = nb.float64(scale)

    direct = scale > 0
    scale = np.fabs(scale)
    if direct:
        out = np.log1p(np.fabs(v / scale))
    else:
        out = np.expm1(np.fabs(v)) * scale
    return np.copysign(out, v)


def log_compress(v: NDArray[Number] | Number, scale: float = 1) -> NDArray[np.float64]:
    """Apply logarithmic scale compression or de-compression to array or number.
    Signed data is processed as well according to:
    ::
        compressed(x) = sign(x) * log(1 + abs(x / scale)) (decompression is inverse of that)

    :param v: array or number to compress decompress
    :param scale: direction and scale of compression:
                        `+` (*positive*): compress,
                        `-` (*negative*): decompress,
                        `0`:  disable and return the input
    :return transformed array
    """
    if not (np.isscalar(v) or isinstance(v, np.ndarray)):
        v = np.fromiter(v, np.float32)
    if not scale:
        return v
    return _log_compress(v, scale)


def bins_edges(low, high, bins: int, below=True) -> NDArray[Number]:
    """
    Creates array of bins *right* edges for histogram, given:
      - range limits (`low, `high`)
      - and number of bins to cover this range.
      That is the *bin size* is always `(high-low)/bins`

    Resulting `edges` array includes two *special* elements at the end: `inf`, `nan`,
    to reserve bins for correspondingly values > high, and 'nan' (requires `high < inf` !)

    Additional bin for `values < low` is controlled by `below` argument:
     - ``below`` is `True`, first bin has its right `edges[0] = low`
     ::

           low                  high    inf   nan           <- right edge value
       <..._|______|_ ... _|______|_...> | ~~~~ |
            0      1             bins  bins+1  bins+2       <- bin/edge id

     - ``below`` is `False`, first bin collects [low, low+step], `edges[0] = low + bin`
     ::

                low+step        high    inf   nan           <- right edge value
            |______|_ ... _|______|_...> | ~~~~ |
                   0            bins-1  bins  bins+1        <- bin/edge id

    That is, the size of the resulting `edges` array: `bins + 2 + int(below)`.
    In all the cases, it *always* represents *right* edges of the bins.

    :param low: **left** edge of the binning range
    :param high: *right* edge of the binning range (must obey `high > low`)
    :param bins: number of *whole* bins between `low` and `high` values
    :param below: expect and count data below the ``low`` value
    :return: edges
    """
    x_range = high - low
    bin_size = x_range / bins

    assert np.isfinite(x_range) and x_range > 0
    assert bins > 0

    edges = np.arange(
        start=low + (0 if below else bin_size),  # create a bin for x < low
        stop=high + (0.01 + 2) * bin_size,  # +0.1 to ensure high is included
        step=bin_size
    )
    edges[-2:] = np.inf, np.nan  # special bins
    return edges


# class SamplerModel(YamlModel, extra='forbid'):
#     low: float
#     high: float
#     bins: Optional[int]
#     step: Optional[float]
#     log_scale: float = 0
#     below: bool = True
#     zero: bool | None
#     relax: Literal['low', 'high', 'step'] = 'relax'
#     name: str = ''
#
#     @root_validator(pre=True)
#     def _params(cls, values):
#         from algutils.datatools import select_from
#         values['high'] = Sampler._val_range(**select_from(values, ['high', 'low', 'bins', 'step']))
#         return values
#

class Sampler:
    @classmethod
    def from_range(cls, low: Any, high: Any, step_or_bins: float | int, name: str = ''):
        """Helper constructor useful when constructing from tuples:
            - (`low`, `high`, `step`: ``float``), or
            - (`low`, `high`, `bins`: ``int``)

        Distinguished by the `step` type!
        """
        return cls(low, high, name=name, **{
            'step' if isinstance(step_or_bins, float) else 'bins': step_or_bins})

    def __init__(self, low, high=None, *, below=True, step=None, bins=None, zero: bool | None = None,
                 log_scale: Number | bool = False, name='', relax: Literal['low', 'high', 'step'] = 'high'):
        """

        Create *equidistant* sampling of given interval for binning, meshes and similar needs.

        Allows defining the sampling using either:
         - min, max and bins
         - min, max and step

        In the second case provided parameters may be inconsistent with even sampling.
        In this case number bins is rounded up and one of the *min, max, step* is relaxed to fit.

        ----------

        **Sampling of 0**

        Argument ``zero`` requires `0` value to be placed:
            - *on* a sample if `True`
            - in the *middle* between two successive samples if `False`
            - `None` for no requirements

        It adjusts bins number and is applicable ONLY for symmetric ranges: `-low=high`.

        **Range Log Compression**

        ``Sampler`` supports *log compression* of the sampled range to increase inter-sampling distance
        as values grow.

        That is achieved by applying ``log_compress`` function on both:
            - the range parameters ``low``, ``high``,  ``step``, and
            - the incoming data being processed

        In this case sampling remains *equidistant* in the compressed (not the actual values) space.
        That is reflected in the `edges` attribute, which will produce exponentially uncompressed samples.

        *Notice* specific meaning of ``step`` argument in this case, such that `log_compress(step)`
        corresponds to the equal samples distance in the compressed space.

        :param low: low edge of the sampling interval
        :param high: high edge of the sampling interval
        :param step: sampling step
        :param bins: number of bins (either that or steps must be given)
        :param name: Optional name of the sampler
        :param below: True to allow values < `low`
        :param zero: ``True``: sample at **0**, ``False``: **0** at half `step`, ``None`` - don't care
        :param relax: when step is given - which one of `min`, `max`, `step` to fit.
        """
        #  --------- validations --------------
        high = self._val_range(low, high, bins, step)

        # ------------ initialization -------------
        self.low, self.high, self.step = low, high, step
        self.log_scale = float(log_scale)
        self.below = below  # collect below the range low limit
        self.bins = bins
        self.name = str(name)
        self._adjust_sampling(zero, relax)

    @staticmethod
    def _val_range(low, high, bins, step):
        if high is None:
            high = low + step * bins
        else:
            assert high > low
            assert (bins and not step) or (step and not bins), "Both bins and step are provided"
            assert not step or step < high - low
        assert not bins or int(bins) == bins and bins > 0

        return high

    def low_high_step(self, compress=False):
        """Return tuple `(low, high, step)` in the original or compressed space."""
        params = (self.low, self.high, self.step)
        if compress and self.log_scale:
            return tuple(log_compress(_, self.log_scale) for _ in params)
        return params

    def _adjust_sampling(self, zero, relax):
        """Adjust sampling to the constraints in the COMPRESSED space!"""
        low, high, step = self.low_high_step(compress=True)

        def bins_zero_correction(_bins):
            # make even or odd depending on the goal
            if zero is None or np.isclose(low, 0):  # we always allow [0, high]
                return 0
            elif zero not in (True, False):
                raise ValueError(f"Invalid value {zero=}")

            if not np.isclose(low, -high):
                raise ValueError(f"Sampling at 0 can't may be adjusted with range {[low, high]}")

            fix_even = _bins %  2  # in symmetric case if 1/2 sample ensure
            return fix_even if zero else 1 - fix_even

        if self.bins:  # high, low, bins -> step
            self.bins += bins_zero_correction(self.bins)
            step = (high - low) / self.bins
        else:
            bins = (high - low) / step
            self.bins = int(bins + .5) + bins_zero_correction(int(bins + .5))
            if not np.isclose(self.bins, bins):
                match relax:
                    case 'low':
                        low = high - self.bins * step
                    case 'high':
                        high = low + self.bins * step
                    case 'step':
                        step = (high - low) / self.bins
                    case False | None | 'none':
                        raise ValueError(f"Can't meet sampling constraints with {relax=}!")
                    case _:
                        raise ValueError(f"Unsupported 'relax' value {relax}")
        # return to the uncompressed space
        self.low, self.high, self.step = log_compress([low, high, step], -self.log_scale)

    def __eq__(self, other):
        return all(np.isclose(getattr(self, attr), getattr(other, attr))
                   for attr in ['low', 'high', 'step', 'bins'])

    def bins_edges(self, compress=True) -> NDArray[np.number]:
        """Create right edges for binning.

        Includes two elements at the end:
            [-2] = `inf` for bin counting `values > high`
            [-1] = `nan` for bin counting `nan`s

        **Note**, `compress` has effect only if `Sampler` is log-compressed.

        By default, (`True`) return equidistant

        :param compress: relevant if ``Sampler`` has `log compression` activated.
               returned `edges` the *compressed* space or the original values space.
        """
        low, high, _ = self.low_high_step(compress=True)  # bins_edges operates in equidistant space!
        edges = bins_edges(low=low, high=high, bins=self.bins, below=self.below)
        if self.log_scale and compress is False:  # if compression is active
            return log_compress(edges, -self.log_scale)
        return edges

    def near_index_below(self, values, compressed: bool = False, fp_index=False):
        """Return indices from the nearest edges below the given values, and the offsets from them.

        If offsets are returned in same compression state as the input values

        :param values: values to search the nearest index for
        :param compressed: indicate if values has been already compressed (relevant if log_scale)
        :param fp_index: return only index with fp precision
        :return: indices (int), offsets (right from edges), index with fp precision
        """
        if not compressed:
            values = log_compress(values, self.log_scale)
        fp_idx = (values - self.low) / self.step
        if fp_index:
            return fp_idx

        idx = np.int32(fp_idx)
        offsets = fp_idx - idx
        if compressed:
            offsets = log_compress(offsets, -self.log_scale)
        return np.int32(fp_idx), offsets, fp_idx

    @property
    def bins_centers(self):
        """
        Return position of the centers of the bins in original data space.

        :param below: if `below` is `False`, center of the first bin > `low`, otherwise < `low`.
        """
        centers = self.bins_edges(compress=True)[:-2] - self.step / 2
        if self.below:
            centers = centers[1:]
        return log_compress(centers, -self.log_scale)

    @property
    def steps(self):
        prepend = {} if self.below else {'prepend': self.low}
        return np.diff(self.bins_edges(compress=False)[:-2], **prepend)

    def __repr__(self):
        name = f"<{self.name}>" if self.name else ''
        tos = lambda _: f"{_:.3g}"
        items = self.low_high_step(compress=True)
        left = '⍇' if self.below else '['
        rng = f"{left}{':'.join(map(tos, items))}]"

        if self.log_scale:
            org = self.bins_edges(compress=False)[:-2]
            if len(org) < 8:
                seq = ','.join(map('{:.3g}'.format, org))
            else:
                seq = (', '.join(map('{:.3g}'.format, org[:3]))
                       + ', ..., ' +
                       ', '.join(map('{:.3g}'.format, org[-3:])))
            rng = f"{rng} → [{seq}]"

        return f"{self.__class__.__name__}{name}[{self.bins}] {rng}"


def _decode_metrics_arg(metrics):
    match metrics:
        case True:
            return _sum_metrics + _basic_metrics
        case False:
            return ()
        case None:
            return _basic_metrics
        case [_] | (_):
            if unknown := set(metrics).difference(_basic_metrics + _sum_metrics):
                raise NameError(f"Requested metrics are {unknown = }")
            return as_list(metrics, collect=tuple)


class StatGather:
    """
    Basic Statistics Calculator.

    Operates in accumulative and stateless modes.

    Supports 3 main types of statistics:
        1. histogram, with bins defined by the `sampler` argument
        2. basic statistical metrics:
            - `mean`, `std`, `rmse`, counters over all and finite values
        3. counters of values under specific named *levels*

    Levels counters are
    """

    # FixMe: @Ilya align names with 2D
    # FixMe: @Ilya reuse with 2D
    # FixMe: stats=None does not work
    def __init__(self, sampler: Sampler,
                 arrays: Iterable[np.ndarray] | np.ndarray = (), *,
                 name='', stats: bool | list[METRICS] = True,
                 levels: Mapping[str, Number] | None = None):
        """
        Initialize data gatherer for histogram and statistical metrics:
        ::

            SUM_METRICS = Literal['s1', 's2', 'finite', 'total']
            BASIC_METRICS = Literal['mean', 'rmse', 'std']

        Allows extension of the basic statistical metrics with accumulative hist sampled at given levels.

        :param sampler: defines binning for the histogram (see ``Sampler`` for details)
        :param arrays: optionally one or `Iterable` over multiple arrays to process.
        :param stats: if ``False`` skip gathering and calculating the basic statistical `measures`
        :param levels: add to the `stats` accumulations by levels
        :param name: useful for logistics of multiple instances of gatherers.
        """
        self._sampler = sampler

        self._equid_edges = sampler.bins_edges()
        self._calc_stats = _decode_metrics_arg(stats)
        self._appends = None

        self.name = name
        self.levels = levels
        self.measures = {}
        self.hist = None

        self.reset()  # initializes all three

        if isinstance(arrays, np.ndarray):
            arrays = [arrays]
        for a in map(np.asarray, arrays):
            if a.ndim and a.size:
                self.process(a, update=True)

    def __repr__(self):
        from algutils.strings import dict_str
        sm = self._sampler
        compress, step = ('', f"{sm.step:.3g}") if not sm.log_scale else (
            "🗜", "{:.3g}→{:.3g}".format(*sm.steps[[0, -1]])
        )
        rng = f"{sm.low:.3g}:{sm.high:.3g}:{step}"
        measures = dict_str(select_from(self.measures, _basic_metrics), prec=3, sep='|', to=':')
        below = "<-|" if sm.below else "["

        name = f"<{self.name}>" if self.name else ''
        return (f"{self.__class__.__name__}{name} "
                f"{below}{rng}]{compress} "
                f"∑{self._appends}{{{measures}}}")

    @property
    def edges(self):
        """Edges in the original values space (decompressed)"""
        return self._sampler.bins_edges(compress=False)

    def _empty_hist(self, hist=None):
        """Prepare empty histogram array, by creating new or zeroing existing"""
        if hist is None:
            hist = np.zeros_like(self._equid_edges, dtype=HistT)
        else:
            hist[:] = 0
        return hist

    def _calc_hist_stats(self, a: np.ndarray | None, hist: np.ndarray, prev_stats: dict):
        """Update aggregated measurements and calculate integrating statistics.

        The core of the statistical computations of the class.

        Relays on external flow providing it with proper data context: `hist` and `prev_stats`.

        If `a` is None or empty - valid structure of stats is returned with values `0` or `nan`

        :param a: data array to measure
        :param hist: histogram array to accumulate into (must be pre-allocated)
        :param prev_stats: previous stats to update
        :return: updated `hist`, `stats`
        """
        stats = (
            select_from({}, _sum_metrics, default=0)
            if a is None or not a.size else
            dict(zip(_sum_metrics,
                     _equal_bins_stats(a, low=self._equid_edges[0],
                                       bin_size=self._sampler.step,
                                       log_scale=self._sampler.log_scale,
                                       sums=bool(self._calc_stats), hist=hist)
                     ))  # stats contain ONLY aggregable metrics!
        )
        if prev_stats:  # aggregate what may be aggregated
            stats = {k: prev_stats[k] + v for k, v in stats.items()}

        # append or overwrite fields which must be calculated anew
        stats |= _basic_stats(**stats)  # averaging over accumulated measures

        if self.levels:
            interp = self.interp_hist(self.levels.values(), cum=True, norm=True)
            stats |= dict(zip(self.levels, interp))

        if self._calc_stats is not True:  # select only requested stats
            stats = select_from(stats, self._calc_stats, strict=False)
        return stats

    def reset(self):
        """Resets accumulated statistics"""
        #  reflects stages in _calc_hist_stats
        self.hist = self._empty_hist(self.hist)
        self.measures = self._calc_hist_stats(None, self.hist, prev_stats={})
        self._appends = 0

    def process(self, a: np.ndarray, *, restart=False):
        """Process data array and update the histogram and integral stats.
        :param a: array to process
        :param restart: if True `reset` the stats
        :return:
        """
        restart and self.reset()
        self.measures = self._calc_hist_stats(a, self.hist, prev_stats=self.measures)
        self._appends += 1
        return self.hist.copy(), self.measures

    # ToDo: @Ilya stat_table, hist_table

    def __getattr__(self, item):
        if (ms := self.measures) and (val := ms.get(item, _UNDEF)) is not _UNDEF:
            return val
        raise KeyError(f"{self.__class__.__name__}.measures has no field '{item}'")

    def norm_hist(self, total=False):
        """Return normalized histogram instead of integer counters"""
        norm = self.measures.get('total' if total else 'finite', None)
        return self.hist / norm if norm else None

    def cum_hist(self, norm=True, total=False):
        """Return cumulative histogram optionally `normalized` over `total` or `finite` counts"""
        hist = self.norm_hist(total=total) if norm else self.hist
        return None if hist is None else np.cumsum(hist)

    def interp_hist(self, values: Collection, cum=False, norm=True, total=False):
        """Interpolate computed histogram in given values.

        If requested normalization is not available (no total stats) return array of `nan`.

        :param values: Iterable of location to interpolate at.
        :param cum: if `True` - interpolate the cumulative histogram
        :param norm: normalize the histogram values or use the original counters
        :param total: use total number of values to normalize or only the finite ones.
        :return: selected form of histogram interpolated for the provided values.
        """
        if norm:
            if not self.measures.get(total and 'total' or 'finite', 0):
                return np.full(len(values), np.nan)
            h = self.cum_hist(norm=norm, total=total) if cum else self.norm_hist(total=total)
        else:
            h = self.cum_hist(norm=False) if cum else self.hist

        values = log_compress(values, self._sampler.log_scale)
        return np.interp(values, self._equid_edges, h, left=0)


# def _tuple(itr: Iterable, func: Callable | str, *args, **kwargs):
#     """Create tuple by applying given function to the given iterable.
#
#     If ``func`` is str, apply the obj method of this name.
#     :param itr: iterable over objects (with method 'func' if func is str)
#     :param func: function or method name to apply
#     :param args: additional func arguments
#     :param kwargs: additional func keyword arguments
#     """
#     gen = (
#         (getattr(_, func)(*args, **kwargs) for _ in itr)
#         if func is str else
#         (func(_, *args, **kwargs) for _ in itr)
#     )
#     return tuple(gen)


class StatGather2D:
    """
    Basic Statistics Calculator.

    Operates in accumulative and stateless modes.

    Supports 3 main types of statistics:
        1. histogram, with bins defined by the `sampler` argument
        2. basic statistical metrics:
            - `mean`, `std`, `rmse`, counters over all and finite values
        3. counters of values under specific named *levels*

    Levels counters are
    """

    def __init__(self, *samplers: Sampler,
                 regions: Iterable[str] | None = None,
                 name='', stats: bool | Collection[METRICS] | Literal['all'] = None,
                 levels: tuple[Mapping[str, Number] | None] = ()):
        """
        Initialize statistics gatherer.

        ``stats`` argument defines which statistical metrics to collect:
         - ``True`` - all possible
         - ``None`` - default, except the internal counters
         - ``False`` - none
         - `collection` of specific *metrics* selected from the supported:
                {'s1', 's2', 'finite', 'total', 'mean', 'rmse', 'std'}

        Allows extension of the basic statistical metrics with accumulative hist sampled at given levels.

        :param sampler1, sampler2: binning for every histogram axis (see ``Sampler`` for details)
        :param stats: if ``False`` skip gathering and calculating the basic statistical `measures`
        :param levels: add to the `stats` accumulations by levels, for every axis
        :param name: useful for logistics of multiple instances of gatherers.
        """
        if len(samplers) > 2:
            raise NotImplemented(f"Provided {len(samplers)} samplers, supported only 2")

        self.samplers = samplers
        self._equid_edges = tuple(_.bins_edges() for _ in samplers)

        self._stats = {}  # actual collection of statistics
        self._metrics = _decode_metrics_arg(stats)
        self.regions = as_list(regions, collect=tuple)
        self.levels: tuple[dict[str, Collection]] = as_list(levels, collect=tuple)
        assert len(self.levels) in (0, 2)

        self.hist = None
        self.name = name
        self._appends = None

        self.reset()  # initializes all three

    def get_stats(self, *metrics: str, copy=True) -> dict[METRICS, NDArray[Number]]:
        """
        Return selected metrics from accumulated stats.

        To override metrics provided in constructor pass:
            - desired list as arguments
            - or *one* `True` for all.
        :param metrics:
        :param copy: if `True` return a copy of the data, to avoid contamination of the internal state
        :return:
        """
        if not metrics:
            metrics = self._metrics
        elif metrics[0] in (True, None):
            assert len(metrics) == 1
            metrics = metrics[0]
            if metrics is False: return None
            metrics = _decode_metrics_arg(metrics)
        else:
            metrics = _decode_metrics_arg(metrics)

        stats = select_from(self._stats, metrics)
        if copy:
            stats = deepcopy(stats)
        return stats

    @property
    def stats(self):
        return self.get_stats()

    def __repr__(self):
        from algutils.strings import dict_str
        samplers = ' × '.join(
            f"{sm.name}[{sm.bins}]{'🗜' if sm.log_scale else ''}"
            for sm in self.samplers
        )
        measures = ''
        if self._metrics:
            measures = select_from(self._stats, _basic_metrics)
            measures = f"{{{dict_str(measures, prec=3, sep='|', to=':')}}}"

        name = f"{self.name}: " if self.name else ''
        return (f"{self.__class__.__name__} <{name}{samplers}> "
                f"∑{self._appends}{measures}")

    @property
    def edges(self):
        """Edges in the original values space (decompressed)"""
        return tuple(s.bins_edges(compress=False) for s in self.samplers)

    def _empty_hist(self, hist=None):
        """Prepare empty histogram array, by creating new or zeroing provided"""

        shape = tuple(_.size for _ in self._equid_edges)
        if self.regions:
            shape = (1 + len(self.regions), *shape)

        if hist is None:
            hist = np.zeros(shape, dtype=HistT)
        else:
            assert hist.shape == shape
            hist[:] = 0
        return hist

    def _calc_hist_stats(self, a12: tuple | None, masks=None, *,
                         hist: NDArray, prev_stats: dict):
        """Update aggregated measurements and calculate integrating statistics.

        The core of the statistical computations of the class.

        Relays on external flow providing it with proper data context: `hist` and `prev_stats`.

        If `a` is None or empty - valid structure of stats is returned with values `0` or `nan`

        :param a12: 2 data arrays to measure
        :param hist: histogram array to accumulate into (must be pre-allocated)
        :param prev_stats: previous stats to update
        :return: updated `hist`, `stats`
        """
        # noinspection PyTypeChecker
        stats = (
            init_sum_stats(num_regions=0, ndim=2) if a12 is None or not a12[0].size else
            _equal_bins2d_stats(
                *a12,
                masks=masks,
                low=tuple(_[0] for _ in self._equid_edges),
                bin_size=tuple(_.step for _ in self.samplers),
                log_scale=tuple(_.log_scale for _ in self.samplers),
                sums=self._metrics,
                hist=hist)  # stats contain ONLY aggregable metrics!
        )  # only summations here

        if prev_stats:  # aggregate with previous
            stats = {k: prev_stats[k] + v for k, v in stats.items()}
        stats |= _basic_stats(**stats)  # add averaging statistics to the summations

        for levels in self.levels:  # for each variable
            interp = self.interp_hist(levels.values(), cum=True, norm=True)
            stats |= dict(zip(levels, interp))

        return stats

    def reset(self):
        """Resets accumulated statistics"""
        #  reflects stages in _calc_hist_stats
        self.hist = self._empty_hist(self.hist)
        self._stats = self._calc_hist_stats(None, hist=self.hist, prev_stats={})
        self._appends = 0

    def process(self, a1: Iterable[Number], a2: Iterable[Number],
                masks: Iterable[Iterable[Number]] = None, *, reset: bool = False):
        """Process 2 data arrays to calculate the histogram and integral stats.

        Masks can be provided in multiple ways:
         - dict `{region_name: region_mask}` with subset of regions defined in constructor,
           the rest are assumed zero-masks
         - iterable over masks in the same order and number as defined regions
         - array with stack of masks with first dimension sweeping the regions

        All the masks must be objects convertable by ``asarray`` into array of

        :param a1: first data collection
        :param a2: second data collection
        :param masks: optional masks matching regions defined in constructor,
        :param reset: reset accumulated statistics
        """
        reset and self.reset()
        a1, a2 = map(np.asarray, (a1, a2))
        assert a1.size == a2.size
        num_regions = len(self.regions)
        assert bool(num_regions) + bool(masks is None) == 1, "Musks are required if regions are defined"

        if masks is not None:
            if isinstance(masks, dict):  # convert into array of masks
                if unknown := set(masks).difference(self.regions):
                    raise KeyError(f"Unknown masks names {unknown}")
                if len(masks) != num_regions:  # missing masks set as zeros
                    zero_mask = np.zeros_like(a1, dtype=bool)
                    masks = (masks.get(k, zero_mask) for k in self.regions)

            masks = tuple(map(np.ascontiguousarray, masks))

            if len(shapes := set(m.shape for m in masks)) != 1:
                raise f"masks have different {shapes = }!"
            shape = shapes.pop()

            if a1.shape != shape:
                raise ValueError(f"Masks {shape=} differs from data array's {a1.shape}!")
            if (num_masks := len(masks)) != num_regions:
                raise ValueError(f"Masks {num_masks=} differs from {num_regions}")

        self._stats = self._calc_hist_stats((a1, a2), masks, hist=self.hist, prev_stats=self._stats)
        self._appends += 1
        return self.hist, self._stats

    def __getattr__(self, item):
        if (ms := self._stats) and (val := ms.get(item, _UNDEF)) is not _UNDEF:
            return val
        raise KeyError(f"{self.__class__.__name__}.measures has no field '{item}'")

    def norm_hist(self, axis: _Axis = None, over: _NormT = True):
        """
        Return 2D histogram normalized over specified domain, and
        optionally collapsed into specified axis by summing along the other.
        ::
          - axis is None: [num_regions × bins1 × bins2]
          - axis in 0, 1: [range × axis_bins]

        Supported domains to normalize `over`:
        ::
            'total' (or True), 'range', 'nonan', None (or False)

        So that
         - `(over=None, axis=None)` returns the original histogram.
         - `(over=None, axis=1)` returns not normalized collapsed into axis 1.

        :param axis: Optional axis along which histogram is calculated
        :param over: domain kind over which to normalize
        """
        match over:
            case 'total' | True:
                slc = np.s_[:]
            case 'range':
                slc = np.s_[self.samplers[0].below:-2, self.samplers[1].below:-2]
                if self.hist.ndim == 3:
                    slc = np.s_[:, *slc]
            case 'nonan':
                slc = np.s_[self.samplers[0].below:-1, self.samplers[1].below:-1]
                if self.hist.ndim == 3:
                    slc = np.s_[:, *slc]
            case None | False:
                pass
            case _:
                raise ValueError(f"Invalid norm method {over}")

        if over:
            hist = self.hist[slc]
            hist = hist / hist.sum()
        if axis is not None:
            hist = hist.sum(1 - axis)  # sum over 1 if axis 0, over 0 if axis 1
        return hist

    def cum_hist(self, axis: _Axis, norm: _NormT = True):
        """Return cumulative histogram optionally `normalized` over `total` or `finite` counts.

        Supported ``norm`` values:
        ::
            'total' (or True), 'range', 'nonan', None (or False)

        :param axis: select along which axis calculate teh histogram (collapse into)
        :param norm: kind of normalization to use (See `norm_hist`)

        """
        hist = self.norm_hist(axis=axis, over=norm)
        return np.cumsum(hist)

    def interp_hist(self, axis: _Axis, values: Collection, *,
                    cum=False, norm: _NormT = True):
        """Interpolate computed histogram in given values.

        :param axis: axis to collapse histogram 2D → 1D into before interpolation.
        :param values: Iterable of location to interpolate at.
        :param cum: if `True` - interpolate the cumulative histogram
        :param norm: normalize the histogram values or use the original counters
        :return: selected form of histogram interpolated for the provided values.
        """
        assert axis in (0, 1), "Must specify valid axis to collapse the histogram into"

        hist = self.cum_hist(axis=axis, norm=norm) if cum else self.norm_hist(axis=axis, over=norm)

        values = log_compress(values, self._sampler.log_scale)
        from scipy.interpolate import interp1d
        return interp1d(self._equid_edges[axis], hist, assume_sorted=True)(values)

    def hist_table(self):
        if not self._appends:
            return None

        axis = dict(zip(['index', 'columns'], (
            pd.Index(edg, name=sm.name) for edg, sm in zip(self.edges, self.samplers))))

        frame = lambda h2d: pd.DataFrame(h2d, **axis)

        if self.regions:
            return pd.concat(map(frame, self.hist), keys=['full', *self.regions],names=['region', 'erg'])
        else:
            return frame(self.hist)

    def stats_table(self, metrics: bool | METRICS | None = ()):
        if not (self._appends and self._metrics):
            return None

        if self.regions:
            #FixMe: Why are we adding 'full'? It is not requested by user.
            frame = lambda s: pd.DataFrame(
                s, index=pd.Index(['full', *self.regions], name='region'),
                columns=pd.Index([s.name for s in self.samplers], name='var'))

            stats = self.get_stats(*as_list(metrics), copy=False)
            df = pd.concat(map(frame, stats.values()), keys=stats, names=['metric'])
            return df.stack('var').unstack(['region', 'var'])

    def plot(self, region=None, title: str = None, ticks: int | tuple[int, int] = 5,
             cmap='terrain', norm: Literal['asinh', 'log', 'logit', 'symlog'] = None, **kws):
        """
        Plot image of 2D histogram
        :param title: to put on the top of the plot
        :param ticks: number of ticks of both or each of the axes
        :param norm: normalization scaling for visualization of z-axis (colors compression)
        :param cmap: color map to use
        :param kws: `imshow` anf `figure` keywords
        :return: fig, axes[2d]
        """
        all_regions = ['full', *self.regions]
        regions = as_list(region) or all_regions
        regions = [all_regions[r] if isinstance(r, int) else r for r in regions]  # indices into names
        if region is not None and (inv := set(regions).issubset(all_regions)):
            raise ValueError(f"Unknown regions {inv}")
        hists = self.hist
        if hists.ndim == 2:
            hists = hists[None, :, :]
        named_hists = {r: hists[all_regions.index(r)] for r in regions}

        from toolbox.vis.insight import hist_grid
        axis_names = [s.name or f'v{i}' for i, s in enumerate(self.samplers)]
        return hist_grid(named_hists, dict(zip(axis_names, self._equid_edges)),
                         title=title, cmap=cmap, norm=norm, **kws)


@overload
def equal_bins_stats(a, sampler: Sampler, stats=False, edges=True,
                     hist: np.ndarray | None = None): ...


@overload
def equal_bins_stats(a, low: float, high, *, bins: int, below=True,
                     log_scale: float | bool = False, stats=False,
                     edges=True, hist: np.ndarray | None = None): ...


@overload
def equal_bins_stats(a, low: float, high, *, bin_size: float, below=True,
                     log_scale: float | bool = False, stats=False,
                     edges=True, hist: np.ndarray | None = None): ...


# ToDo: remove this function after integration of Gatherers
def equal_bins_stats(a, low: float | Sampler, high=None, *, below=True,
                     bins: int = None, bin_size=None,
                     log_scale: float | bool = False, stats=False,
                     edges=True, hist: np.ndarray | None = None):
    """
    Calculates equal bins histogram and basic statistics over given array.

    Wraps in simplified interface usage of more flexible machinery of
        - more flexible ``Sampler`` class
        - fast low level ``_equal_bins_stats`` function

    :param a: nd array to measure
    :param low: right edge of the lower bin (if ``below`` is True, otherwise its left edge).
    :param high: right edge of the high bin
    :param below: controls if ``low`` denotes *left* or *right* edge of the first bin
    :param bins: number of bins in the histogram
    :param bin_size: an alternative to the `bins` argument
    :param log_scale: compress data range by log-scaling bins intervals
    :param stats: request collect additionally some basic statistics
    :param edges: return edges if True
    :return: hist, [stats], [edges] -  return if corresponding arguments are True
    """
    if isinstance(low, Sampler):
        sampler = low
    else:
        assert not (bins and bin_size)
        sampler = Sampler(low, high, log_scale=log_scale,
                          **({'bins': bins} if bins else {'step': bin_size}))

    edges = sampler.bins_edges(compress=True)  # first equidistant for hist params

    if hist is None:
        hist = np.zeros(len(edges), HistT)
    else:
        if hist.ndim != 1 and hist.size != edges.size:
            raise ValueError("Invalid size of hist array")

    sums = stats  # stats request translates to request for sums
    stats = dict(zip(_sum_metrics,
                     _equal_bins_stats(
                         a, low=edges[0], bin_size=np.diff(edges[:2]).item(),
                         hist=hist, sums=sums, log_scale=log_scale)
                     ))
    if sums:
        stats |= _basic_stats(**stats)

    if edges:
        edges = sampler.bins_edges(compress=False)  # edges in the original data space
        return hist, edges, stats if sums else hist, edges
    else:
        return hist, stats if sums else hist


_Range = tuple[float, float, float]  # (start, stop, step)


def hist2d(a1: np.ndarray, a2: np.ndarray, range_1: _Range, range_2: _Range):
    """
    :param a1: array 1
    :param a2: array 2
    :param range_1: (start, stop, step)
    :param range_2: (start, stop, step)
    """
    bins = [int((r[1] - r[0]) / r[2]) for r in (range_1, range_2)]
    ranges = [r[:2] for r in (range_1, range_2)]
    assert all(_ > 1 for _ in bins)
    return np.histogram2d(a1.ravel(), a2.ravel(), bins=bins, range=ranges)


class Hist2D:

    def __repr__(self):
        return f"Hist2D{self.samplers}"

    def clear(self):
        """Clear histogram counts"""
        self.counts, _, _ = np.histogram2d([], [], bins=self.bins)

    def __init__(self, bins_1: _Range, bins_2: _Range,
                 a1: np.ndarray = None, a2: np.ndarray = None):
        """
        `bins_*` can be represented as tuple:
          - `(min, max, step: float | bins: int, [name: str])`
          - or ``Sampler`` instance.

        `arrays` are optional at the initialization, could be added later using ``add`` method.
        """

        self.samplers = [Sampler.from_range(*r[:3], name=str(r[3:]) or f'a{i}')
                         for i, r in enumerate([bins_1, bins_2])]
        self.bins = tuple(sampler.bins_edges() for sampler in self.samplers)

        self.counts = None
        self.clear()

        if a1 is None:
            assert a2 is None
        else:
            self.add(a1, a2)

    def add(self, a1: np.ndarray, a2: np.ndarray):
        """Add to the histogram accumulators pair of arrays (must be of same size)"""
        arrays = list(np.asarray(a).ravel() for a in [a1, a2])
        counts, _, _ = np.histogram2d(*arrays, bins=self.bins)
        self.counts += counts

    def plot(self, title: str = None, ticks: int | tuple[int, int] = 20,
             transform: Callable | None = None, **imgrid_kws):
        """
        Plot image of 2D histogram
        :param title: to put on the top of the plot
        :param ticks: number of ticks of both or each of the axes
        :param transform: optional function to apply on the hist2d counts before forming an image
        :param imgrid_kws: relevant arguments to imgrid used internally to plot the image
        :return:
        """
        from toolbox.vis import imgrid

        title = title or 'Hist2D'
        if transform and (name := transform.__name__) != (lambda _: _).__name__:
            title += f' ( {name}(counts) )'
        counts = transform(self.counts) if transform else self.counts
        imgrid_kws = dict(cmap='terrain') | imgrid_kws
        (ax,) = imgrid(counts, out='axs', titles=[title], **imgrid_kws)
        ax.invert_yaxis()

        ticks = [ticks] * 2 if isinstance(ticks, int) else ticks
        labels = [s.name for s in self.samplers]
        for axis, label, edges, num_ticks in zip(['y', 'x'], labels, self.bins, ticks):
            num_ticks = min((num_edges := len(edges)), num_ticks)
            tick_ixs = np.linspace(0, num_edges - 1, num_ticks, endpoint=True, dtype=int)
            tick_edges = [f"{edges[i]:.2f}" for i in tick_ixs]
            getattr(ax, f'set_{axis}ticks')(tick_ixs, tick_edges)
            getattr(ax, f'set_{axis}label')(label)
        ax.figure.set_tight_layout(True)
        return ax
