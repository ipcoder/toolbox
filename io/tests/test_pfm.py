import pytest

import toolbox.io.pfm as pfm


@pytest.mark.io
def test_load_pfm(tmp_path):
    """load_pfm(file_name, get_scale=False, replace_inf=True):"""
    import numpy as np
    ref_image = np.random.randn(100, 160).astype('float32') * 10000.
    ref_image[0, :] = np.inf
    ref_image[1, :] = np.nan

    pfm_file = tmp_path / 'test_pfm.pfm'
    pfm.save_pfm(pfm_file, ref_image)

    # ----------------------------------------------
    pfm_image = pfm.load_pfm(pfm_file, replace_inf=False)
    valid = ~np.isnan(ref_image)

    assert (pfm_image[valid] == ref_image[valid]).all()
    pfm_image_rep = pfm.load_pfm(pfm_file, replace_inf=1000)

    inf = np.isinf(pfm_image)
    assert (pfm_image_rep[inf] == 1000).all()
    valid = valid & ~inf
    assert (pfm_image[valid] == ref_image[valid]).all()
