"""General imread function that support the most common formats in Inuitive """
from __future__ import annotations

from iad.core.events import Timer

with Timer(f" ← {__file__} imports", "timing", min=0.1, pre=f' → importing in {__file__} ...'):

    from pathlib import Path

    import numpy as np
    import yaml

    from collections import namedtuple
    from warnings import warn


def __strings_enumeration(strings):
    """
    If given collection of strings differs only by integer suffix (some4, some10, ...)
    Return those numbers as list of integers, and the prefix - otherwise None, None
    :param strings:
    """
    if len(strings) < 2:
        return None, None
    from os.path import commonprefix
    prefix = commonprefix(strings)
    pfx_len = len(prefix)
    suffixes = [nm[pfx_len:] for nm in strings]
    unique_symbols = set(''.join(suffixes))
    if len(unique_symbols) <= 10 and unique_symbols.issubset(set('0123456789')):
        return [int(i) for i in suffixes], prefix
    return None, None


SUPPORT_READ = {'.png', '.pfm', '.jpg', '.tif', '.tiff', '.bmp', '.pfm', '.flo', '.zfp', '.cif', '.ciif'
                , '.npy', '.exr'}
SUPPORT_WRITE = {'.png', '.pfm', '.jpg', '.tif', '.tiff', '.bmp', '.pfm', '.zfp', '.cif', '.ciif'}
SUPPORTED_EXT = {'.png', '.pfm', '.jpg', '.tif', '.tiff', '.bmp', '.pfm', '.flo', '.zfp', '.npy'}


def imread(file, *, out=np.ndarray, get_meta=False, shape=None, dtype=None
           ) -> np.ndarray | dict[str, np.ndarray] | list[np.ndarray]:
    """ Read the most common file formats with simple uniform interface.
    Auto parsing of stream names according to test point name (if exists),
    when parsing testpoints (npy/ciif) and asking for dict output.

    :param file: path to the file
    :param out: type of the output from: (np.ndarray, list, dict)
    :param get_meta: return additional output with metadata (from tif files)
    :param shape:   (rows, cols) required to load un-formatted data - mainly raw files
    :param dtype:   pixel data type - required to load un-formatted data - mainly raw files
    :return: image, dict, list - depending on the `out` argument.
    """
    file = str(file)  # Path -> str   TODO: use path functions instead
    ext = '.' + file.rsplit('.')[-1].lower()
    if not ext in SUPPORT_READ:
        raise NotImplementedError(f"Image extension of {file=} is not supported!")

    # Consider: Refactor reading logic
    # First read images

    if ext in ('.ciif', '.cif'):
        from utils import io as cif
        if out == dict:
            return cif.load_ciif(file, out=dict)
        images = cif.load_ciif(file, out=list)
    elif ext in ('.npy', '.npz'):
        images = np.load(file)
        if isinstance(images, np.ndarray):
            images = [images]
        elif isinstance(images, np.lib.npyio.NpzFile):
            np_file = images
            if out == dict:
                ret = {name: val for name, val in np_file.items()}
                np_file.close()
                return ret
            else:
                ids, pfx = __strings_enumeration(np_file.files)
                # ensure correct order in the case of numerical ids - otherwise order is undefined!
                names = (f'{pfx}{i}' for i in sorted(ids)) if ids else np_file.files
                images = [np_file[f] for f in names]
                np_file.close()
        else:
            raise ValueError(f'Unsupported np out type {file}')
    elif ext == '.exr':
        from openexr_numpy.openexr_numpy import read_dict
        data = read_dict(file)
        if out == dict:
            return data
        images = [*data.values()]
    elif ext == '.png':
        from PIL import Image
        images = [np.array(Image.open(file))]
    elif ext == '.flo':
        images = [imread_flo(file)]
    elif ext == '.pfm':
        from .pfm import load_pfm
        images = [load_pfm(file, replace_inf=False)]
    elif ext == '.zfp':
        from .zfp import load_zfp
        images = [load_zfp(file)]
    # elif ext in ('.tif', '.tiff'):
    #     from tifffile import tifffile
    #     images = tifffile.imread(file)
    #     if get_meta:
    #         meta = tiff_desc_yaml(file)
    #         return images, meta
    #     else:
    #         images = [images]
    elif ext == '.raw':
        if shape is None or dtype is None:
            raise TypeError('shape and dtype arguments must be provided when reading *.raw files.')
        images = [np.fromfile(file, dtype=dtype).reshape(shape)]
    # TODO check loading jpg time and loading of grayscale images using imread_color
    # elif ext == '.jpg':
    #     images = [cv2.imread(file, cv2.IMREAD_COLOR)]
    else:
        from skimage import io as skio
        images = [skio.imread(file)]

    if out == np.ndarray:
        if len(images) == 1:
            return images[0]
        else:
            return np.stack(images)

    if out == dict:
        return {str(i): v for i, v in enumerate(images)}

    if out == list:
        return images
    else:
        raise ValueError('Unsupported out type')


def imread_flo(path):
    """ Read optical flow format (.flo)

    :param path: path to the file
    :return: 2D vector field WxHx2
    """
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    # Reshape data into 3D array (columns, rows, bands)
    return np.resize(data, (w, h, 2))


def imsave_flo(filename, flow):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    with open(filename, 'wb') as f:
        magic = np.array([202021.25], dtype=np.float32)
        height, width = flow.shape[:2]
        magic.tofile(f)
        np.int32(width).tofile(f)
        np.int32(height).tofile(f)
        np.float32(flow).flatten().tofile(f)

def tiff_desc_yaml(file):
    """ Read metadata from tif file

    :param file:
    :return: meta
    """
    from tifffile.tifffile import TiffFile
    tiff_tmp = TiffFile(file)
    meta = yaml.load(tiff_tmp[0].image_description, Loader=yaml.SafeLoader)
    return meta
def imsave(path, im, **kwargs):
    """
    Save image into file using format defined by its extension.
    Supported formats:
        - usual image formats: ``bmp, png, tif``, ...
        - compressed (floating point data): ``zfp``

    :param path: path in usual forms
    :param im:
    :param kwrgs:

    """
    from pathlib import Path
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(mode=0o777, exist_ok=True)

    ext = path.suffix.lower()

    if ext == '.zfp':
        from .zfp import save_zfp
        save_zfp(path, im, **kwargs)
    elif ext == '.pfm':
        from .pfm import save_pfm
        save_pfm(path, im, **kwargs)
    elif ext == '.flo':
        imsave_flo(path, im, **kwargs)
    elif ext in ('.cif', '.ciif'):
        from .ciif import images_to_ciif
        images_to_ciif(path, im)
    else:
        from skimage import io as skio
        skio.imsave(path, im, **kwargs)


def identical_images(file1: str | Path, file2: str | Path) -> bool:
    """Return True if images in two image files are identical"""
    import pyvips
    im1, im2 = map(pyvips.Image.new_from_file, (file1, file2))
    return bool(np.all(im1.numpy() == im2.numpy()))


def convert_image(src: np.ndarray | str | Path,
                  trg_file: str | Path, *,
                  save_kws: dict | None = None, src_kws: dict | None = None):
    """
    Convert source image into a different format.

    Target file should be specifed as a Path object or str.format form with optional
    parameters `{parent}`, `{stem}`, `{suffix}`:

        trg_file = "{parent}/sub/conv_{stem}_new{suffix}"

    Return information about the operation:

        dict(
            src=im.filename,
            trg=str(trg_file),
            width=im.width,
            height=im.height,
            bands=im.bands,
            bytes = im.width * im.height * im.bands * elm_bytes,
            src_bytes = src_file.stat().st_size,
            trg_bytes = trg_file.stat().st_size,
            time=dt
        )

    :param src: path to the source file or pyvips.Image or numpy array
    :param trg_file: path to the target file, possibly in .format form
    :param src_kws: optional dict with ``new_from_file`` or ``new_from_array`` kwargs
    :param save_kws: optional dict with ``write_to_file`` kwargs

    """
    import pyvips
    from time import time

    src_kws = src_kws or {}
    save_kws = save_kws or {}

    if isinstance(src, (str, Path)):
        src_file = Path(src)
        if isinstance(trg_file, str):
            trg_file = Path(trg_file.format(
                stem=src_file.stem,
                parent=src_file.parent,
                suffix=src_file.suffix,
            ))
        im = pyvips.Image.new_from_file(str(src_file), **src_kws)
    else:
        if '{' in trg_file:
            raise ValueError("Can't reference src file parts in trg_file if src provided as data!")
        src_file = None
        if isinstance(src, pyvips.Image):
            im = src
        else:
            im = pyvips.Image.new_from_array(src, **src_kws)

    trg_file = Path(trg_file)
    trg_file.parent.mkdir(exist_ok=True)

    t0 = time()
    im.write_to_file(str(trg_file), **save_kws)
    dt = time() - t0

    if im.format == 'uchar':
        elm_bytes = 1
    else:
        raise NotImplementedError

    info = dict(
        src=im.filename,
        trg=str(trg_file),
        width=im.width,
        height=im.height,
        bands=im.bands,
        bytes=im.width * im.height * im.bands * elm_bytes,
        src_bytes=src_file and src_file.stat().st_size or None,
        trg_bytes=trg_file.stat().st_size,
        time=dt
    )
    return info
