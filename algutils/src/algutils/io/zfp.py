from __future__ import annotations

from pathlib import Path

import numpy as np
import zfpy


def save_zfp(save_location, image):
    """
    Save ndarray into zfp (compressed floating point) file.
    In order to avoid decompression failures, unsupported types are being cast into float point type.
    :param image: ndarray of the input image
    :param save_location: location to save the file into
    """
    assert type(image) == np.ndarray, "Input image is not numpy.ndarray!"
    try: zfpy.dtype_to_ztype(image.dtype)
    except: image = image.astype(float)
    compressed = zfpy.compress_numpy(image)
    with open(save_location, 'wb') as file:
        file.write(compressed)


def load_zfp(path: Path | str):
    """
    Loads zfp (compressed floating point) file into ndarray
    :param path: absolute path to file
    return - image as np.ndarray
    """
    with open(path if isinstance(path, str) else path.__str__(), 'rb') as file:
        image = zfpy.decompress_numpy(file.read())
    return image

