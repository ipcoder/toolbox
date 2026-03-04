import numpy as np

from .regions import Regions


def center_crop(im, width, height):
    from algutils.io.camera import StereoCam

    def slice_(sz, lim):
        offs = (sz - lim) // 2
        return slice(offs, offs + lim) if offs > 0 else slice(None)

    if isinstance(im, Regions):
        assert im.shape  # shape must be defined (all regions share same)
        for k in im.keys():
            im[k] = center_crop(im[k], width, height)
        return im
    elif isinstance(im, StereoCam):
        return im.center_crop(width, height)

    if im.ndim == 3:
        h, w, _ = im.shape
        return im[slice_(h, height), slice_(w, width), :]
    else:
        h, w = im.shape
        return im[slice_(h, height), slice_(w, width)]


def _crop_slices(rect, shape):
    """ Return slices to crop a `rect` shape from an image of given `shape`.

    :param rect: [i0, j0, height, width] - may exceed the range of the shape
    :param shape: shape of 2D array to crop
    :return src, trg: 2d indexers to broadcast src image into the trg image of rect[2,3] shape

    :Example:
        - src, trg = crop_crop_slices([-10, 40, 100, 160], im.shape]
        - zeros([100, 160])[trg] = im[src]
    """
    src = np.array(rect)
    src[2:] += src[:2]  # [i0, j0, i_end, j_end
    trg = np.array([0, 0, *rect[2:]])
    for i in [0, 1]:
        if src[i + 2] > shape[i]:
            src[i + 2], trg[i + 2] = shape[i], shape[i] - src[i]
        if src[i] < 0:
            src[i], trg[i] = trg[i], -src[i]
    return tuple(np.s_[x[0]:x[2], x[1]:x[3]] for x in (src, trg))


def crop(im, rect):
    """ Crop rect from the image. rect may exceed the image are (filled wth 0).

    :param im: 2D array
    :param rect: rect: [i0, j0, height, width] - may exceed the range of the shape
    :return cropped_image: negative margins will be filled with 0

    :Example: crop_im = _crop_slices(im, [-10, 40, 100, 160])
    """
    new_im = np.zeros(rect[2:], dtype=im.dtype)
    src, trg = _crop_slices(rect, im.shape)
    new_im[trg] = im[src]
    return new_im


def image_2D(im: np.ndarray):
    """Squeeze dimensions or collapse colors to produce 2D image"""
    if im.ndim == 2:
        return im
    im = im.squeeze()
    if im.ndim != 2:
        if im.ndim == 3 and im.shape[2] == 3:
            from skimage.color import rgb2gray
            im = rgb2gray(im)
        else:
            raise ValueError('Invalid input image shape')
    return im


def to_3d(im: np.ndarray):
    """Add if needed 3-rd dimension to the 2d array"""
    if im.ndim == 3:
        return im
    if im.ndim == 2:
        return im[..., None]
    raise ValueError('Input must be 2 or 3 dimensional!')


def im2col_sliding_broadcasting(A, BSZ, stepsize=1):
    """ # TODO: [DOC]

    :param A:
    :param BSZ:
    :param stepsize:
    :return:
    """
    # Parameters
    M, N = A.shape
    col_extent = N - BSZ[1] + 1
    row_extent = M - BSZ[0] + 1

    # Get Starting block indices
    start_idx = np.arange(BSZ[0])[:, None] * N + np.arange(BSZ[1])

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    return np.take(A, start_idx.ravel()[:, None] + offset_idx.ravel()[::stepsize])


def im2col_sliding_strided(A, BSZ, stepsize=1):
    """ # TODO: [DOC]

    :param A:
    :param BSZ:
    :param stepsize:
    :return:
    """
    # Parameters
    m, n = A.shape
    s0, s1 = A.strides
    nrows = m - BSZ[0] + 1
    ncols = n - BSZ[1] + 1
    shp = BSZ[0], BSZ[1], nrows, ncols
    strd = s0, s1, s0, s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0] * BSZ[1], -1)[:, ::stepsize]


def im2col_sliding_strided_v2(A, BSZ, stepsize=1):
    """  # TODO: [DOC]

    :param A:
    :param BSZ:
    :param stepsize:
    :return:
    """
    from skimage.util import view_as_windows as viewW
    return viewW(A, (BSZ[0], BSZ[1])).reshape(-1, BSZ[0] * BSZ[1]).T[:, ::stepsize]


def wind_center_proc(imc, func, wnd, *, imw=None, shape=None, fill=None, out=None):
    """ Apply custom rolling window processing on the image.

    The processing sums outputs of the provided func(wind_center, pixel) over the window pixels
    to produce a single pixel in the output.

    The shape of the output may be either 'same' as the input or limited only to the 'valid' area.
    The ill-defined boundary of the 'same' shape can be filled by:
    - a provided numeric value (dafault is 0)
    - 'mirror' the valid results along the boundary
    - 'extend' the valid values on the boundary

    :param im:   the input image
    :param oper: function(c,p) to be applied on each pixel p in the window
    :param wnd:  (height, width) of the window - must be even
    :param shape: shape of the result, one of: ['same'] | 'valid'
    :param fill:  [number=0] | 'mirror' | 'extend'
    """
    # process arguments
    hy, hx = tuple(w // 2 for w in wnd)
    assert tuple(w * 2 + 1 for w in (hy, hx)) == wnd
    wsz = wnd[0] * wnd[1]

    valid_area = np.s_[hy:-hy, hx:-hx]
    imc_valid = imc[valid_area]

    if imw is None:
        imw = imc

    if out is None:
        if shape is None:
            shape = 'same'
    else:
        if out.shape == imc.shape:
            assert shape == 'same' or shape is None
            shape = 'same'
        elif out.shape == imc_valid.shape:
            assert shape == 'valid' or shape is None
            same = 'valid'
        else:
            raise ValueError('out.shape %s must be either ''same'' %s or ''valid'' %s!' %
                             (out.shape, imc.shape, imc_valid.shape))

            # calculate
    imc_stride = imc_valid.reshape(-1).repeat(wsz, axis=0).reshape(imc_valid.size, wsz).T
    imw_stride = im2col_sliding_strided_v2(imw, wnd)
    res = func(imc_stride, imw_stride).sum(axis=0).reshape(imc_valid.shape)

    # pack output
    if shape == 'valid':
        if out is None:
            out = res
        else:
            out[:, :] = res
    else:
        if out is None:
            out = np.zeros_like(imc, dtype=res.dtype)
        out[valid_area] = res
        fill_bounds(out, margs=(hx, hy), fill=fill)
    return out


def fill_bounds(im, margs, *, fill='extend', out=True):
    """ Fill (in place) boundary area of the image according to the given rule

    :param im:  the image with boundaries to fill
    :param margs:   (top, left, bot, right) or
                    (top, left) or  # then right == left and bottom = top
                    marg            # then all (top, left, bot, right) == marg
    :param fill: ['extend'] | 'mirror' | None - do nothing | number to fill
    :param out:  [True] | False - return filled image
    """
    import numbers
    if isinstance(margs, numbers.Number):
        margs = (margs,) * 4
    elif len(margs) == 2:
        margs = margs * 2
    top, left, bot, right = margs

    if fill is None:
        pass
    elif fill == 'extend':
        im[:top, :] = im[top, :].reshape(1, im.shape[1])
        im[-bot:, :] = im[-bot, :].reshape(1, im.shape[1])
        im[:, :left] = im[:, left].reshape(im.shape[0], 1)
        im[:, -right:] = im[:, -right].reshape(im.shape[0], 1)
    elif fill == 'mirror':
        im[:top, :] = im[2 * top:top:-1, :]
        im[-bot:, :] = im[-2 * bot:-bot:-1, :]
        im[:, :left] = im[:, 2 * left:left:-1]
        im[:, -right:] = im[:, -2 * right:-right:-1]
    else:
        im[:top, :] = fill
        im[-bot:, :] = fill
        im[:, :left] = fill
        im[:, -right:] = fill

    if out:
        return im
