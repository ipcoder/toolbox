"""
Non standard image writing formats
"""
import os
from pathlib import PurePath, Path

import numpy as np
from skimage.io import imsave as ski_imsave


def imsave(path, im, **kwargs):
    """
    Save image into file using format defined by its extension.
    Supported formats:
        - usual image formats: ``bmp, png, tif``, ...
        - compressed (floating point data): ``zfp``

    For `tif` supported kwargs as in `tifffile.imwrite`:
        compression: 'none', lzw, deflate, zstd, lzma, packbits, jpeg2000
        predictor: None, 2 (for int dtype), 3 (for float)

    :param path: destination fie absolute path
    :param im: image data (`ndarray`)
    :param kwargs: `tifffile` or `skimage.imsave` supported arguments
    """

    if not (parent := Path(path).expanduser().parent).exists():
        parent.mkdir(mode=0o777, parents=True)

    if isinstance(path, PurePath):
        path = str(path)
    if path.endswith('.zfp'):
        from zfp import save_zfp
        save_zfp(path, im)
    elif path.endswith('.pfm'):
        from pfm import save_pfm
        save_pfm(path, im)
    elif path.rsplit('.', 1)[-1].lower() in ('tif', 'tiff'):
        from tifffile import imwrite
        imwrite(path, im, **kwargs)
    else:
        kwargs.setdefault('check_contrast', False)
        ski_imsave(path, im, **kwargs)


def save_depth_nu4(file, depth, *, units='mm/100', inv=0):
    """
    Save depth data in special nu4000 format

    :param file: to save - must be tif
    :param depth: float (default units: mm)
    :param units: convert to those units when saving depth/units -> file
    """
    if hasattr(depth, 'magnitude'):
        Q = type(depth)
    else:
        from toolbox.utils.units import Quantity
        Q = Quantity
        depth = Q(depth, 'mm')

    if isinstance(units, (int, float)):
        units = Q(units, depth.units)
    else:
        units = Q(str(units))

    assert depth.check(units), "Incompatible units"
    depth = (depth / units).magnitude

    dtype = np.uint16
    max_val = np.iinfo(dtype).max

    with np.errstate(invalid='ignore'):
        depth[np.isnan(depth) | (depth > max_val)] = inv

    imsave(file, depth.astype(dtype), check_contrast=False)


def save_disp_nu4(file, disp, conf=None, inv=255):
    """
    Saves disp in nu4k format, including confidence.
    # unsigned 16 bits (from high to low bits): disp:8, spx:4, conf:4
    :param file:
    :param disp: may be float or int
                - if former - crop to (0, 255) and invalidate outside
                - in the later case leaves as ir is
    :param conf: 4 bits max values packed in any integer container
                if None - sets its bits to 0000
    :param inv: invalidation code for 8 bis disparity part
    """
    assert os.path.splitext(file)[-1].lower() in ('.tif', '.tiff'), 'Must be TIF format'
    dtype = np.uint16

    if disp.dtype.kind == 'f':
        with np.errstate(invalid='ignore'):
            disp[np.isnan(disp) | (disp < 0) | (disp >= 2 ** 8)] = inv
        disp = (disp * 2 ** 4).astype(dtype)

    disp <<= 4
    if conf is not None:
        assert conf.dtype.kind in 'ui' and conf.max() < 2 ** 4 and conf.min >= 0
        conf = conf.astype(np.uint8)  # asserts ensures it has only low 4 bits
        disp += conf

    imsave(file, disp, check_contrast=False)
