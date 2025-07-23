"""General imread function that support the most common formats in Inuitive """
from __future__ import annotations

from toolbox.utils.events import Timer

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
        import io.ciif as cif
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
        from pfm import load_pfm
        images = [load_pfm(file, replace_inf=False)]
    elif ext == '.zfp':
        from zfp import load_zfp
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


def imread_disp_nu4(file, dtype=np.float32):
    """
    Read disparity in NU4000 format
    It contains 16bits integer with the following bits codinf (from lower)
        0-3 (4): confidence
        4-7 (4): sub-pixel disparity
        8-15 (8): pixel disparity
    :param file:
    :param dtype: return type, if float - return as float number in pixels
                               otherwise - as integer in 1/16 of pixel
    :return: tuple(disp (float or int), conf: 4b)
    """
    data = imread(file)
    if np.dtype(dtype).kind == 'f':
        disp = (data >> 4).astype(dtype) / 16
        disp[disp > 144] = np.nan
    else:
        disp = data >> 4
    return disp, (data & 0b1111).astype(np.uint8)


def imread_depth_nu4(file, dtype=np.float32, units='mm'):
    """
    Read disparity in NU4000 format
    It contains 16bits integer with the following bits codinf (from lower)
        0-3 (4): confidence
        4-7 (4): sub-pixel disparity
        8-15 (8): pixel disparity
    :param file:
    :param dtype: return type, if float - return as float number in pixels
                               otherwise - as integer in 1/16 of pixel
    :return: tuple(disp (float or int), conf: 4b)
    """
    assert units == 'mm', "Only mm is supported!"
    depth = data = imread(file)
    if np.dtype(dtype).kind == 'f':
        depth = data.astype(dtype) / 100
        depth[data == 0] = np.nan
    return depth


def tiff_desc_yaml(file):
    """ Read metadata from tif file

    :param file:
    :return: meta
    """
    from tifffile.tifffile import TiffFile
    tiff_tmp = TiffFile(file)
    meta = yaml.load(tiff_tmp[0].image_description, Loader=yaml.SafeLoader)
    return meta


_nu4_tiff_tags = {256: 'left.resolution.x', 257: 'left.resolution.y',
                  5016: 'left.a_gain', 5017: 'right.a_gain',
                  5018: 'left.d_gain', 5019: 'right.d_gain',
                  5020: 'left.expos', 5021: 'right.expos',
                  5023: 'left.focal.x', 5024: 'left.focal.y',
                  5025: 'right.focal.x', 5026: 'right.focal.y',
                  5027: 'left.center.x', 5028: 'left.center.y',
                  5029: 'right.center.x', 5030: 'right.center.y',
                  5032: 'baseline'}


def tiff_inu_cam(file_name: str):
    """
    Read Inuitive camera meta-data from the tif image
    :param file_name:
    :return: Hierarchical structure:
        TBox for single camera image
        StereoCam for stereo image
        TBox({}) if unrecoverable error or no data
    """
    from PIL import Image, TiffImagePlugin
    import numbers
    from toolbox.io.camera import StereoCam, TBox

    stack_x_tag = 5005  # 1 or 2: number of images per line

    info = TBox(default_box=True)
    with Image.open(file_name) as img:  # type: TiffImagePlugin.TiffImageFile
        try:
            stack_x = img.tag[stack_x_tag][0]
            stack_x_expected = (1, 2)
            if stack_x not in stack_x_expected:
                raise ValueError(f'Expecting image stacking in {stack_x_expected} not {stack_x}')
        except Exception as e:
            warn(str(e))
            warn('Continue assuming a single image')
            stack_x = 1

        for tag_id in _nu4_tiff_tags:  # look for specific tag numbers in the tif files
            if tag_id in img.tag:
                val = img.tag[tag_id][0]
                if isinstance(val, numbers.Number) and (not np.isnan(val)):
                    info[_nu4_tiff_tags[tag_id]] = val

        if hasattr(info, 'left.resolution') and stack_x > 1:
            if not (info.left.resolution.x / stack_x).is_integer():
                raise ValueError(f'Odd resolution in {stack_x}-folded image.')
            info.left.resolution.x = info.left.resolution.x // 2
            info.right.resolution = info.left.resolution

        if len(info.left) and len(info.right):
            info = StereoCam(info, units='mm')
        return info


def imread_stereo(file, cam_info=None, source=False) -> namedtuple:
    """ Read stereo pair from Inuitive Captured files in different formats:
         - Output_Video_<side>_00000864.*  - either of the sided may be specified
         - Video_00000864.*
         - StereoImage_00000864.tif

    :param file: full path to one of the stereo files
    :param cam_info: None|True|False
                        - if None - try to extract camera information meta-data if possible
                        - if True return also `cam_info` field from the tiff file meta-data or raise exception
                        - if False - don't even attempt
    :param source: if True  - return also information about the source:
                               (dataset, file, fid)
    :return: namedtuple('StereoImages', image_left, image_right,
                        [cam_info - optional], [source - optional])
    """
    from re import sub
    import os
    from toolbox.io.camera import TBox

    filename = os.path.split(file)[-1]
    if filename.startswith('Video_') or filename.startswith('StereoImage_'):
        # print('Loading stereo pair from a single image...')
        im = imread(file)
        w = im.shape[1] // 2
        res = {'image_left': im[:, w:], 'image_right': im[:, :w]}
        if cam_info is not False:
            from toolbox.io.camera import StereoCam
            try:
                res['cam_info'] = StereoCam.from_tiff(file)
            except Exception as e:
                if cam_info is True:
                    raise e

    elif filename.startswith('Output_Video_'):  # Consider: Add cam_info support!
        res = {'image_' + side.lower():
                   imread(sub('(?<=Output_Video_)(Left|Right)', side, file))
               for side in ('Left', 'Right')}
    else:
        raise FileNotFoundError('Unsupported file type')
    try:
        fid = file.rsplit('.', 1)[0].rsplit('_', 1)[-1]
    except:
        fid = 'NOT_SET'
    if source:
        res['source'] = TBox(dataset='Inuitive', file=file, fid=fid)
    return namedtuple('StereoImages', res.keys())(**res)


imread_stereo_nu4 = imread_stereo


def imread_fly3d(fid, scid=0, group='A', *, disp=False, test=True,
                 loc='~/datasets/FlyingThings3D', cam_info=False, show=False) -> namedtuple:
    """
    Get images from "Flying Things 3d" dataset
    :param fid:  5 < fid < 16
    :param scid: 0 <= scid < (150 if testing set else 750)
    :param group: in ('A', 'B', 'C'), default = 'A'
    :param disp: return also dispqrity [False]| True=left|'left'|'right'|'all'
    :param test: use testing set if True - otherwise training
    :param loc: location of the dataset (root path)
        (use 'ln /mnt/e/FlyingThings3D ~/datasets -s' to link the dataset from any location)
    :param show:  print debug info
    :returns:
        namedtuple('Frame_fly3d', img_left, image_right, [disp_left, [disp_right]], [cam_info])

    Notes:
        https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html#information
        "
        The virtual imaging sensor has a size of 32.0mmx18.0mm.
        ... scenes use a virtual focal length of **35.0mm**
        ... the virtual camera intrinsics matrix is given by
            fx=1050.0   0.0         cx=479.5
            0.0	        fy=1050.0   cy=269.5
            0.0	        0.0	        1.0
        where (fx,fy) are focal lengths and (cx,cy) denotes the principal point."
    """
    from toolbox.io.camera import StereoCam, TBox
    import os

    part = 'TEST' if test else 'TRAIN'
    loc = os.path.expanduser(loc)
    fl3d_base = os.path.join(loc, '{kind}/{part}/{group}/{scid:04}')
    cam_pat = '/camera_data.txt'
    img_pat = r'/{side}/{fid:04}.{ext}'

    def file_name(**kw):
        pat = cam_pat if kw['kind'] == 'camera_data' else img_pat
        file = (fl3d_base + pat).format(**kw)
        if show: print(f'image read: {file}')
        return file

    def camera_info(camera_data_file, fid):
        from re import finditer
        par = StereoCam(resolution=(960, 540), focal=1050, baseline=100)  # see link above (100 - empirical!)

        with open(camera_data_file) as f:
            text_data = f.read()
            reg_exp = r'Frame\s(?P<fid>\d+)\s*L\s*(?P<left>(?:-?\d+\.\d+\s)+)\s*R\s(?P<right>(?:-?\d+\.\d+\s)+)'
            for match in finditer(reg_exp, text_data):
                rec = match.groupdict()
                if int(rec.pop('fid')) == fid:
                    for k, v in rec.items():
                        par[k]['mat'] = np.array([float(x) for x in v.strip().split(' ')]).reshape(4, 4)
                    break
        return par

    res = {f'image_{s}': imread(file_name(kind='frames_cleanpass', scid=scid, fid=fid, side=s, group=group,
                                          part=part, cam=False, ext='png')) for s in StereoCam.views}
    if disp:
        if disp is True:
            disp = 'left'
        elif isinstance(disp, str):
            disp = disp.lower()
            assert disp in ('all', *StereoCam.views)
        use_sides = StereoCam.views if disp == 'all' else [disp]

        res.update({'disp_gt_' + s: imread(file_name(kind='disparity', scid=scid, fid=fid, side=s,
                                                     group=group, part=part, ext='pfm')) for s in use_sides})
    if cam_info:
        res.update({'cam_info': camera_info(file_name(kind='camera_data', scid=scid, group=group, part=part),
                                            fid=fid)})
    res['source'] = TBox(dataset='Flying Things 3D',
                         loc=loc, scid=scid, group=group, fid=fid, test=test)
    from collections import namedtuple
    return namedtuple('Frame_fly3d', res.keys())(**res)


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
        from zfp import save_zfp
        save_zfp(path, im, **kwargs)
    elif ext == '.pfm':
        from pfm import save_pfm
        save_pfm(path, im, **kwargs)
    elif ext == '.flo':
        imsave_flo(path, im, **kwargs)
    elif ext in ('.cif', '.ciif'):
        from .ciif import images_to_ciif
        images_to_ciif(path, im)
    else:
        from skimage import io as skio
        skio.imsave(path, im, **kwargs)


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
    import os
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
