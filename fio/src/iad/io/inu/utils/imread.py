"""Inuitive / NU4-specific helpers built on iotools and imgtools."""
from __future__ import annotations

from collections import namedtuple
from warnings import warn

import numpy as np

from iad.io.imread import imread, imsave

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
    from iad.core.param import TBox
    from iad.img.camera import StereoCam

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
    from iad.core.param import TBox

    filename = os.path.split(file)[-1]
    if filename.startswith('Video_') or filename.startswith('StereoImage_'):
        # print('Loading stereo pair from a single image...')
        im = imread(file)
        w = im.shape[1] // 2
        res = {'image_left': im[:, w:], 'image_right': im[:, :w]}
        if cam_info is not False:
            from iad.img.camera import StereoCam
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
    from iad.core.param import TBox
    from iad.img.camera import StereoCam
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
        from iad.core.units import Quantity
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

