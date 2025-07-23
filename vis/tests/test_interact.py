from skimage.io import imsave
import pytest


# Prepare structure for automatic test

_watch_outputs = False

@pytest.mark.refactor
@pytest.mark.skip(reason="Legacy not maintained")
def test_image_annotator(tmp_path):
    """__init__(self, image_file, annotation_file, *argv, **kwargs)"""

    image_file = os.path.join(tmp_path, 'interact.tif')
    annotation_file = os.path.join(tmp_path, 'annotation.png')

    if _watch_outputs:
        imsave(image_file, (np.random.rand(50, 50) * 1023).astype('uint16'))
        ImageAnnotator(image_file=image_file, annotation_file=annotation_file)


@pytest.mark.refactor
@pytest.mark.skip(reason="Legacy not maintained")
def test_interact(tmp_path):
    """ Interact.py image.file output.file points_num lines_num """
    if _watch_outputs:
        image_file = os.path.join(tmp_path, 'interact.tif')
        annotation_file = os.path.join(tmp_path, 'annotation')

        imsave(image_file, (np.random.rand(50, 50) * 1023).astype('uint16'))

        import subprocess
        call_str = ' '.join(['python ', devise_script_path(__file__), image_file, annotation_file])
        print(call_str)
        subprocess.call(call_str)

