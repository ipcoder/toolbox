import numpy as np

from toolbox.io import zfp


def test_zfp_save_and_load(tmp_path):
    """
    Test zfp compression, save and load

    Steps:
        1) create random float array
        2) compress and save using module
        3) load array and decompress
        4) compare to original array
    """
    orig_array = np.random.rand(3, 2, 2)
    save_path = tmp_path / 'test_compress.zfp'
    zfp.save_zfp(save_path, orig_array)
    decompressed_array = zfp.load_zfp(save_path)
    assert type(decompressed_array) is np.ndarray
    assert (decompressed_array == orig_array).all()



