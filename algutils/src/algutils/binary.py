import logging
import warnings
from math import *

import numpy as np

_log = logging.getLogger(__name__)


class Bin(int):
    @staticmethod
    def _test_range(val, bits):
        val_bits = int(val).bit_length()
        if bits and bits < val_bits:
            raise ValueError('%d bits of %d exceeds bits limit of %d' % (val_bits, val, bits))

    def __new__(cls, val, bits=None):
        if hasattr(val, '__len__'):
            return np.array([Bin(v, bits) for v in val], dtype='O')
        Bin._test_range(val, bits)
        return super(Bin, cls).__new__(cls, val)

    def __init__(self, _, bits: object = None):
        self.bits = bits  # type: int

    def __str__(self):
        return bstr(int(self), bits=self.bits, dec=True)

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        return Bin(int(self) + int(other),
                   max(bits_num(self), bits_num(other)) + 1
                   if self.bits or isinstance(other, Bin) and other.bits else None)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return Bin(int(self) - int(other),
                   max(bits_num(self), bits_num(other)) + 1
                   if self.bits or isinstance(other, Bin) and other.bits else None)

    def __rsub__(self, other):
        return Bin(int(other) - int(self),
                   max(bits_num(self), bits_num(other)) + 1
                   if self.bits or isinstance(other, Bin) and other.bits else None)

    def __mul__(self, other):
        return Bin(int(self) * int(other),
                   bits_num(self) + bits_num(other)
                   if self.bits or isinstance(other, Bin) and other.bits else None)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __lshift__(self, n: int):
        return Bin(int(self) << n, self.bits + n if self.bits else None)

    def __rshift__(self, n):
        return Bin(int(self) >> n, self.bits - n if self.bits else None)

    def __floordiv__(self, other):
        return Bin(int(self) // int(other),
                   (self.bits if isinstance(other, Bin) else (2**self.bits-1) // int(other))
                   if self.bits else None)

    def __rfloordiv__(self, other):
        return Bin(int(other) // int(self), bits_num(other))

    def __truediv__(self, other):
        raise NotImplementedError('True Division not implemented for Bin')

    def __pow__(self, n: int):
        return Bin(int(self) ** n, self.bits * n if self.bits else None)


def set_bits(reg_val, field_val, start_bit, bit_len):
    """ Set bit field of bit_len (start_bit - included) to the reg_val

    :param reg_val: initial value - a number of integer type
    :param field_val: field value to be set
    :param start_bit: the first bit of field
    :param bit_len: number of field bits
    :return: final reg_val
    """
    # mask = ~(((1 << bit_len) - 1) << start_bit)
    # if field_val < 0:
    #     field_val &= 0xffffffff
    # return (reg_val & mask) | (field_val << start_bit)

    mask_f = (1 << bit_len) - 1
    if field_val < 0:
        field_val &= mask_f
    mask_0 = ~(mask_f << start_bit)
    return (reg_val & mask_0) | (field_val << start_bit)


def get_bits(data, start_bit, bit_len):
    """ Extract bit field of bit_len (start_bit - included) from the data

    :param data: a number of integer type
    :param start_bit: the first bit to extract
    :param bit_len: number of bits to extract
    :return: resulted data
    """
    mask = (1 << bit_len) - 1
    return (data >> start_bit) & mask


def extract_bits(data, lsb, msb):
    """ Eextract specific range of bits (from lsb to msb - included) from the data

    :param data: a number of ndarray (integer types)
    :param lsb: the first bit to extract
    :param msb: the last bit to extract
    :return: resulted data
    """
    return (((1 << (msb - lsb + 1)) - 1) << lsb & data) >> lsb


def split_bits(data, bits):
    """ Extract data channels encoded in bits into separate arrays

    :param data: ndarray or number (integer types)
    :param bits: numbers of bits in each array - from low to high
    :return:
    """
    if isinstance(data, np.ndarray) and data.dtype.kind not in ['i', 'u']:
        raise TypeError('Integer type expected')

    offsets = None
    if type(bits[0]) == tuple:
        offsets = list(list(zip(*bits))[1])
        bits = list(list(zip(*bits))[0])

    res, off = [], 0
    for i, b in enumerate(bits):
        if offsets is not None:
            off = offsets[i]
        res.append((((1 << b) - 1) << off & data) >> off)
        off += b
    return res


def check_valid_offsets(zip_bits):
    """ Check whether the offsets and the bit numbers are valid

    :param zip_bits: list of tuples in the fio [(stream_bits_size, stream_offset), ...]
    :return:
    """
    zip_bits = sorted(zip_bits, key=lambda x: x[1])
    print(zip_bits)
    for i in range(len(zip_bits)-1):
        if sum(zip_bits[i]) > zip_bits[i+1][1]:
            raise AssertionError('You are writing twice on the same place..')


def join_bits(bits, streams, check_clip=False):
    """ Pack together bits of the streams

    :param bits:
    :param streams:
    :param check_clip:
    :param offsets: pre-known offsets of the bits. same fio like bits
    :return:
    """
    assert len(bits) == len(streams)

    offsets = None
    if type(bits[0]) == tuple:
        check_valid_offsets(bits)
        offsets = list(list(zip(*bits))[1])
        bits = list(list(zip(*bits))[0])

    res = np.zeros_like(streams[0], dtype='uint64')
    shift = 0
    for i,b in enumerate(bits):
        a = streams[i]
        masked_a = a & max_val(b)
        if check_clip and (masked_a != a).any():
            raise OverflowError('bitwise and clipped values')
        if offsets is not None:
            shift = offsets[i]
        res |= (masked_a.astype(res.dtype) << shift)
        shift += b

    if offsets is not None:
        shift = max(offset + bit for offset, bit in zip(offsets,bits))
    return res.astype(fit_bits_uint(shift))


def max_val(bits):
    return 2**bits-1


def bits_num(val):
    """ Count the minimal number of bits required to represent the given value

    :param val:
    :return: number of bits
    """
    return val.bits if isinstance(val, Bin) and val.bits else val.bit_length()


def bstr(num: int, bits=0, dec=0, msb=True, template='{dec}<{exc}{cnt}{bin}>'):
    """ String representation of number in binary form.

    Optionally decimal representation may be added.
    If actual bits exceed declared number exclamation mark is added according to the template
    Format template
    :param num:         the number to represent
    :param bits:        number of bits filled by 0 from the left
    :param dec:         positions for decimal representation. If not - will not appear
    :param msb          show position of the most significant bit
    :param template:    template for the output
    :return:            string representation
    """

    return template.format(
        dec='{0:%d}' % dec if dec else '',
        bin='{0:0%db}|%d' % (bits, bits) if bits else '{0:b}',
        cnt='%d|' % int(num).bit_length() if msb and not bits else '',
        exc='' if not bits or len(('{0:0%db}' % bits).format(num)) == bits else '!').format(num)


def encode_array(bits, a, range_weight=0.01, steps=500, show=False):
    """ Encodes array of numbers into array of fixed bits elements
    optimizing the bits utilization and encoding accuracy.

    :param bits:            number of bits to allocate for each element
    :param a:               array of numbers to encode
    :param range_weight:    factor diminishing cost of the range utilization
    :param steps:           number of steps (resolution of the search) range is [0 1.5*max_fitted]
    :param show:            if true displays some internals of the algorithms

    Algorithm scans for a factor minimizing the total cost of the accuracy and range lost
    after the original elements are multiplied by this factors and rounded to the required bits.
    """

    a_double = np.array(a, dtype='double')
    max_rng = 2 ** bits - 1

    # find index of a minimal non zero element
    nzi = a_double.nonzero()[0]
    mi = nzi[np.argmin(a_double[nzi])]
    assert a[mi]

    original_ratios = a_double / a_double.mean()

    def pack(factor):
        new_a = (a_double * factor).astype('uint64')
        new_a[new_a > max_rng] = max_rng
        return new_a

    def rounding_error(factor):
        a_k = pack(factor).astype('double')
        avr = a_k.mean()
        ratio_error = (((a_k / avr - original_ratios) * max_rng) ** 2).sum() if avr else np.inf
        range_error = ((a_k - max_rng) ** 2).sum()
        return np.array([ratio_error, range_error, ratio_error + range_weight * range_error])

    k_range = np.arange(1, steps) * (max_rng * 1.5) / max(a_double) / (steps - 1)
    errors = np.stack(np.vectorize(rounding_error, otypes='O')(k_range))

    if show:
        import matplotlib.pyplot as plt
        plt.figure()
        errors = np.sqrt(errors)
        labels = ['ratio', 'range', 'total']

        plt.plot(k_range, errors, '-')
        plt.hold(True)
        print('costs factor range total  results')
        print('-----------------------------------------------------')
        y_max = 2 * np.nanmean(errors[errors < np.inf])
        for j, label in zip(range(errors.shape[1]), labels):
            i = np.argmin(errors[:, j])
            k = k_range[i]
            astr = ' '.join('{: 4g}'.format(x) for x in pack(k))
            print(
                '{}: {:.3f} {:.3f} {:.3f} [{} ]'.format(label, k, errors[i, 0], errors[i, 2], astr))
            plt.plot(k_range[i], errors[i, j] + y_max * 0.02, 'v' + 'gbr'[j])
        print('-----------------------------------------------------')
        print('original values:         [%s ]' % ' '.join('{: 4g}'.format(x) for x in a))
        plt.ylim(0, y_max)
        plt.xlim(k_range[[0, -1]])
        plt.title('Bits Packing Errors')
        plt.legend(labels)
        plt.xlabel('coefficient (k)')
        plt.ylabel('errors')
        plt.show()

    return Bin(pack(k_range[np.argmin(errors[:, 2])]), bits)


def align_type_bits(num_bits):
    """ Align given bits to the bits of the minimal standard container type (8, 16, 32, 64)"""

    if num_bits < 8:
        return 8
    elif num_bits > 64:
        raise ValueError('%d exceeds maximal supported bits %d' % (num_bits, 64))
    else:
        return 2 ** ceil(log(num_bits, 2))


def fit_bits_uint(x):
    """ Return numpy style type name of the minimal uint container for the given bits numbers."""
    return 'uint%s' % align_type_bits(x)


def parse_fxp_bits(fxp_bits):
    """ Parse fix bits string """
    return [int(d) for d in fxp_bits.split('.')] if isinstance(fxp_bits, str) else fxp_bits


def to_fxp(data: np.ndarray, fxp_bits, inv_flp='max', inv_fxp='max_int'):
    """ Convert data into fixed point fio - that is integer with last bits denoting a fractional part.
    Convert invalid values if inv_fxp is not None.

    :param data: any ndarray
    :param fxp_bits: may be string ('8.6') or tuple-like (8, 6)
    :param inv_flp: invalid code in the source data:
                    'max' - maximal for this data type,
                    None - don't bother
                    or a specific value
    :param inv_fxp: invalid data code in the output array:
                    'max' - maximal integer value
                    'max_int' - maximal integer part of the fixed point
                    None - don't convert invalids
                    or a specific value encoding the invalids
    """
    i_bits, f_bits = parse_fxp_bits(fxp_bits)
    max_fxp = 2 ** (i_bits + f_bits) - 1
    inv_options = {'max_int': 2 ** i_bits - 1, 'max': max_fxp}

    if inv_flp == 'max':
        inv_flp = np.finfo(data.dtype) if issubclass(data.dtype.type, np.float) else np.iinfo(data.dtype)
    if inv_fxp in inv_options:
        inv_fxp = inv_options[inv_fxp]
    elif inv_fxp > max_fxp:
        raise ValueError('Invalidation value %s exceeds the dynamic range %d' % (inv_fxp, max_fxp))

    inv_mask = (data == inv_flp)
    _log.info('Found %d invalids' % inv_mask.sum())

    data *= 2 ** f_bits
    sat_mask = (~inv_mask) * (data > max_fxp )
    sat_num = sat_mask.sum()
    if sat_num:
        sat_range = (lambda sat: (sat.min(), sat.max()))(data[sat_mask])
        warnings.warn('Conversion into %d.%d fio has saturated %d pixels in range %s.' %
                      (i_bits, f_bits, sat_num, sat_range))
        data[sat_mask] = inv_fxp
    if inv_fxp is not None:
        data[inv_mask] = inv_fxp
    return data.astype(fit_bits_uint(i_bits + f_bits))


if __name__ == '__main__':
    pass

    # Consider: to move testing to test_binar or remove
    ba = np.random.randint(0, 4, size=(3, 4))
    bb = np.random.randint(0, 8, size=(3, 4))

    my_bits = [2, 3]

    my_offsets = [3, 5]
    my_bits = list(zip(my_bits, my_offsets))

    joined = join_bits(my_bits, [ba, bb])
    ra, rb = split_bits(joined, my_bits)

    a=5
