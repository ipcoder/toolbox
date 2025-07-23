import re
import sys

import numpy as np


def load_pfm(file_name, get_scale=False, replace_inf=True, reg_mem=True):
    """ Load pfm (floating point) file into ndarray

    :param file_name
    :param get_scale - return scale field of the pfm file as a second tuple item
    :param replace_inf - True|False|Value if not False replace inf by Value or automatic if True
    :param reg_mem: pfm has reversed lines order, True regularize memory layout by making copy!
    """
    with open(file_name, 'rb') as file:
        header = file.readline().decode('cp1252').strip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('cp1252'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().decode('cp1252').rstrip())
        endian = '<' if scale < 0 else '>'  # scale's sign encodes endian and is 1 otherwise
        scale = abs(scale)

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        image = np.reshape(data, shape)[::-1, :]
        if reg_mem:  #
            image = image.copy()
        # replace_inf: True|False|Value. if not False replace inf by Value or automatic if True
        if (type(replace_inf) is not bool) or (replace_inf is True):
            if replace_inf is True:
                bits = int(np.ceil(np.log(np.amax(image[image < np.inf]) + .00001) / np.log(2)))
                replace_inf = max(1, np.ceil(bits) / 8) * 2 ** 8 - 1
            image[image == np.inf] = replace_inf

        return (image, scale) if get_scale else image

    return None


def save_pfm(file, image, scale=1):
    """ Save ndarray into pfm (floating point) file

    :param file: output file path
    :param image: ndarray of float32 type
    :param scale: scale field of the pfm file
    :return:
    """
    if image.dtype.name != 'float32':
        raise TypeError('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:       # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # grey scale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    endian = image.dtype.byteorder
    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    with open(file, 'w', newline="\n") as file:     #, encoding='cp1252'
        file.write("PF\n" if color else "Pf\n")
        file.write("%d %d\n" % (image.shape[1], image.shape[0]))
        file.write("%f\n" % scale)   # scale = float(file.readline().decode('cp1252').rstrip())
        image[::-1,:].tofile(file)   # as in load! -according to the spec:
        # They are grouped by row, with the pixels in each row ordered left to right and the rows ordered bottom to top"
        file.close()
