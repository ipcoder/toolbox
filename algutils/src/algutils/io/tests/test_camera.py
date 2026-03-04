import numpy as np

from algutils.io.camera import Camera, Resolution, Vec2d


def test_create_camera():
    resolution = Resolution.HD
    cam = Camera(name='HD', pixels=resolution)
    assert cam.sensor.pixels == Vec2d(resolution)


def test_binning():
    from algutils.io import camera
    a = np.arange(3 * 5 * 2 * 3).reshape(10, 9)
    assert np.all(
        (camera.bin_xy(a, 3, 2) == camera.bin_x(camera.bin_y(a, 2), 3)))
