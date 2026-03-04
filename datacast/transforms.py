# ---------------------------------------------------------------------
#                 Implementation if specific transforms
# ----------------------------------------------------------------------

from __future__ import annotations
from typing import Iterable, Union, Tuple

# imports below bring into the namespace functions to be available as transforms
# and exported accordingly by this module.

from numpy import squeeze, number
from algutils.image.transforms import gamma, shot_noise, norm, take_ch, alpha_blend
from algutils.image.tools import center_crop

__all__ = ['gamma', 'shot_noise', 'norm', 'center_crop',
           'squeeze', 'regions', 'recode', 'take_ch', 'alpha_blend']

transform_module = globals()

Number = Union[float, int, number]
Pair = Tuple[Number, Number]


def recode(im, from_to: Union[Pair, Iterable[Pair]]):
    """replace values in the im according to the codes map"""
    if not from_to:
        return im
    if not hasattr(from_to[0], '__len__'):
        from_to = [from_to]
    for src, trg in from_to:
        im[im == src] = trg
    return im


def regions(im, **rgn_codes):
    from algutils.image.regions import Regions
    """Create Regions object from the encoded image data."""
    return Regions(im, rgn_codes=rgn_codes)


