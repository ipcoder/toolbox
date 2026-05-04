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
        from .zfp import save_zfp
        save_zfp(path, im)
    elif path.endswith('.pfm'):
        from .pfm import save_pfm
        save_pfm(path, im)
    elif path.rsplit('.', 1)[-1].lower() in ('tif', 'tiff'):
        from tifffile import imwrite
        imwrite(path, im, **kwargs)
    else:
        kwargs.setdefault('check_contrast', False)
        ski_imsave(path, im, **kwargs)


def save_depth_nu4(file, depth, *, units='mm/100', inv=0):
    """Delegate to :mod:`inu.utils.imread`."""
    from iad.io.inu.utils.imread import save_depth_nu4 as _fn
    return _fn(file, depth, units=units, inv=inv)


def save_disp_nu4(file, disp, conf=None, inv=255):
    """Delegate to :mod:`inu.utils.imread`."""
    from iad.io.inu.utils.imread import save_disp_nu4 as _fn
    return _fn(file, disp, conf=conf, inv=inv)
