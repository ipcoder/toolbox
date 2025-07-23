"""
Includes transforms for images
"""
import numpy as np
from xxhash import xxh32_intdigest


def norm(im, scale=1, offset=0, dtype=None, inverse=False):
    res = (im / scale + offset) if inverse else (im - offset) * scale
    return res.astype(dtype) if dtype else res

def take_ch(im, channel):
    mapping = {'red': 0, 'green': 1, 'blue': 2}
    return im[:, :, mapping[channel.lower()]]

def alpha_blend(im, channel):
    from skimage.color.colorconv import rgba2rgb
    return rgba2rgb(im, channel_axis=channel)

def gamma(im: np.array, g=1.0, inp_bits=8, out_bits=8):
    inp_range = 2 ** inp_bits - 1  # 8 -> 255 ; 10 -> 1023
    out_range = 2 ** out_bits - 1

    # build a lookup table mapping the pixel values [0, inp_range+1] to
    # their adjusted gamma values using the range [0, out_range+1]
    inp_range = float(inp_range)
    inv_g = 1.0 / g
    return ((im / inp_range) ** inv_g) * out_range


def shot_noise(im, *, max_electrons, num_bits, norm_pool=None, clip: float | bool = True, stable=True,
               detect_overflow: bool | float = True, dtype=None):
    """
    Add shot noise to image.
    Uses internal normal generator unless pool is provided.

    :param im: input image to add the noise to
    :param norm_pool: 1D array of pre-generated random numbers with normal distribution sigma=1
    :param max_electrons: Sensor parameter - electrons / charge for full well. MUST include
    :param num_bits:      Data parameter - number of bits the images are made of. MUST include.
                          Both max_electrons and num_bits create the gl2el (gray_level->electrons) conversion.
    :param detect_overflow: do not add noise to the pixels with overflow (dtype.max or given number)
    :param clip: clip result into range (0, dtype.max or clip value)
    :param dtype: output must fit into this dtype, if None - use im.dtype
    """
    import torch
    if dtype is None:
        dtype = im.dtype

    rnd = np.random

    if isinstance(im, torch.Tensor):
        if norm_pool or stable or detect_overflow:
            raise ValueError("Normal pool, stable output or overflow not supported for torch tensor input.")
        create_norm = lambda : torch.normal(0,1, im.shape).to(device)
        device = im.device
        sqrt = lambda _ : torch.sqrt(_)
        max_val = torch.iinfo(dtype).max if clip is True else clip
        clip = lambda _ : torch.clamp(_, 0, max_val)

    elif isinstance(im, np.ndarray):
        if dtype.kind == 'f' and True in (clip, detect_overflow):
            raise ValueError("Clipping or overflow are defied only for integer types")
        create_norm = lambda : rnd.normal(0, 1, im.shape)
        sqrt = lambda  _ : np.sqrt(_)
        max_val = np.iinfo(dtype).max if clip is True else clip
        clip = lambda _ : np.clip(_, 0, max_val)
    else:
        raise ValueError(f'Type {type(im)} for input is not supported')

    if stable is True:
        if not im.flags.c_contiguous:
            im = np.ascontiguousarray(im)
        seed = xxh32_intdigest(im)
        rnd = np.random.RandomState(seed)

    if norm_pool is None:
        rn = create_norm()
    else:
        rn = rnd.choice(norm_pool, im.size).reshape(im.shape)

    gl2el = max_electrons / (2 ** num_bits)
    scaler = sqrt(im / gl2el) # noise =  [sqrt(im * gl2el)] / gl2el , returned as gray levels
    noise = scaler * rn

    if detect_overflow:
        max_val = np.iinfo(dtype).max if detect_overflow is True else detect_overflow
        noise[im >= max_val] = 0

    res = im + noise
    if clip:
        res = clip(res)
    return res


def gamma_lut(im: np.array, g=1.0, inp_bits=8, out_bits=8):
    """
    Using gamma transformation to change dynamic range of input image.
    Also enables to switch between number of bits - only if input bits are higher than outputs'.

    Gamma values < 1 will shift the image towards the darker end of the spectrum,
     while gamma values > 1 will make the image appear lighter.
    A gamma value of g == 1 will have no effect on the input image.

    Using a Look-up table to create results in O(1) time.
    The built-in function of LUT in cv2 works with either 1 or more channels,
    and for 10 bit images we create our own LUT.

    :param im: input image. Can either be uint array with 1 channel, or uint8 with 1 or more channels
    :param g: gamma value.
    :param inp_bits: input image bits to calculate range
    :param out_bits: output image bits to calculate range
    :return:
    """
    import cv2
    if im.dtype.kind == 'f' and im.max() <= 1.0:  # image with range 0-1
        raise NotImplementedError("Please use only  integer types as input."
                                  "to not lose information in the transformation")
    inp_range = 2 ** inp_bits - 1  # 8 -> 255 ; 10 -> 1023
    out_range = 2 ** out_bits - 1

    # build a lookup table mapping the pixel values [0, inp_range+1] to
    # their adjusted gamma values using the range [0, out_range+1]
    inp_range = float(inp_range)
    inv_g = 1.0 / g
    table = np.array([((i / inp_range) ** inv_g) * out_range
                      for i in np.arange(0, inp_range + 1)])

    table = np.round(table)
    # special case
    if out_bits > inp_bits:
        raise ValueError("Please pre process your input image to have "
                         "equal or more bits as your output image bits")

    # op1 - 10 bit input image (with 1 channel)
    elif inp_bits > 8:
        table = table.astype("uint8") if out_bits == 8 else table.astype("uint16")
        return table[im]  # apply gamma correction using the lookup table

    # op2 - 8 bit input image - with 1 or more channels - use cv2
    else:
        table = table.astype("uint8")
        return cv2.LUT(im, table)


def conv(a, b, bound='extend'):
    """ Wraps signal.convolve2d and adds bound 'extend' to extend the boundary values into the margins"""
    from scipy.signal import convolve2d
    from .tools import fill_bounds

    if bound == 'extend':
        marg = (b.shape[0] // 2, b.shape[1] // 2)
        am = np.empty([da + db - 1 for da, db in zip(a.shape, b.shape)], a.dtype)
        am[marg[0]:-marg[0], marg[1]:-marg[1]] = a
        fill_bounds(am, marg, fill=bound, out=False)
        return convolve2d(am, b, mode='valid')
    else:
        return convolve2d(a, b, mode='same', boundary=bound)
