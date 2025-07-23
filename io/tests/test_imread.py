import numpy as np
import pytest

from toolbox.io import imread, imsave
from toolbox.io.imread import SUPPORT_READ, SUPPORT_WRITE


@pytest.fixture
def image():
    return np.random.randint(0, 200, (64, 64), dtype=np.uint8)


RW = SUPPORT_READ & SUPPORT_WRITE
if '.tif' in RW: RW -= {'.tiff'}
if '.cif' in RW: RW -= {'.cif', '.ciif'}


@pytest.mark.io
@pytest.mark.parametrize("ext", RW)
def test_image_formats(image, ext, tmp_path):
    in_file = tmp_path / f'image{ext}'
    if ext in ('.zfp', '.flo', '.pfm'):
        image = image.astype('float32')
    imsave(in_file, image)
    assert isinstance(imread(in_file), np.ndarray), f'imread failed for {ext}'


def test_imread_stereo(stereo_image_file):
    from toolbox.io.imread import imread_stereo
    stereo = imread_stereo(stereo_image_file)
    assert type(stereo.image_left) is np.ndarray
    assert type(stereo.image_right) is np.ndarray
