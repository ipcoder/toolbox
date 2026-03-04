from dataclasses import dataclass, InitVar
from datetime import datetime
from enum import Enum
from math import tan, pi
from typing import Union, NamedTuple, TYPE_CHECKING
from warnings import warn

import numpy as np

from ..param import TBox
from ..math.geom import Vec2d, Pose

if TYPE_CHECKING:
    from ..units import Quantity


class Resolution(Enum):
    HQVGA = (240, 160)
    QVGA = (320, 240)
    VGA = (645, 480)
    WGA = (800, 480)
    SVGA = (800, 600)
    DVGA = (960, 640)
    WXGA = (1280, 800)
    QFD = (960, 540)
    HD = (1280, 720)
    FHD = (1920, 1080)
    QHD = (2560, 1440)
    UHD4K = (3840, 2160)
    UHD8K = (7680, 4320)

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return (self.x, self.y)[item]

    def __repr__(self):
        return f'Resolution {self.name}: {self.x}x{self.y}'

    @property
    def total_pixels(self):
        return self.x * self.y

    @property
    def xy(self):
        return self.x, self.y


@dataclass(repr=False)
class Sensor:
    pixels: Union[Vec2d, Resolution]
    center: Vec2d = 0  # [pix], relative to the sensor center
    colors: str = 'RGB'
    bits: int = 10
    pix_size: Vec2d = 1   # in microns

    def __post_init__(self):
        xy = self.pixels.xy if isinstance(self.pixels, Resolution) else self.pixels
        self.pixels = Vec2d(xy)   # to support initialization with iterable
        if self.center == 0:
            self.center = Vec2d(0, 0)
        if not hasattr(self.pix_size, '__len__'):
            self.pix_size = Vec2d(1, 1)

    @property
    def size(self):
        """ Return sensor size in mm"""
        return Vec2d(*self.pixels / np.array(self.pix_size) / 1000)

    def __repr__(self):
        def dims(v): return f'{v.x}\u2A09{v.y}'

        px_sz = (np.array(self.pix_size) * 10).round() / 10  # with 0.1um accuracy
        px_sz = dims(Vec2d(*px_sz)) if np.diff(px_sz) else f'{px_sz[0]}'

        return f'Sensor {self.bits}b[{self.colors}] {dims(self.pixels)} pix({px_sz}\u03BCm)'


@dataclass(repr=False)
class Shot:
    time: datetime = None
    pose: Pose = None
    exposure: float = 0

    def __repr__(self):
        if self.pose is None and self.time is None and not self.exposure:
            return ''
        time = self.time.strftime('%y/%d %H:%M:%S.%f') if self.time else ''
        exps = f'expos:{self.exposure * 1000:.1}ms' if self.exposure is None else ''
        pose = self.pose if self.pose else ''
        return f'[{time}] {exps} {pose}'


@dataclass(repr=False)
class Camera:
    """
    Camera information class.
    Contains both geometrical and physical aspects of the camera
    """
    pixels: InitVar[Vec2d] = None
    sensor: Sensor = None
    focal: float = np.nan  # in mm, along z-axis from the center
    name: str = ''

    def __post_init__(self, pixels):
        if self.sensor is None:
            self.sensor = Sensor(pixels=pixels)

        if np.isnan(self.focal):
            self.focal = max(*self.sensor.size)  # 90 degrees

    def angles(self, measure='rad'):
        measure = measure.lower()
        assert measure in ('rad', 'deg')
        rad = np.arctan(self.sensor.size / self.focal / 2) * 2
        return rad if measure == 'rad' else rad * 180 / np.pi

    def __repr__(self):
        def dims(v): return f'{v.x}\u2A09{v.y}'

        angles = dims(Vec2d(*np.array(self.angles('deg'), dtype=int)))
        name = f'<{self.name}>' if self.name else ''
        return f'Camera{name}: {self.sensor}, angles: {angles}\u00B0'


def bin_x(im, bin_factor):
    """ Binning along x direction by given (integer) factor

    :param im:
    :param bin_factor:
    :return: binned image
    """
    return im.reshape(im.shape[0], im.shape[1] // bin_factor, bin_factor).sum(2)


def bin_y(im, bin_factor):
    """ Binning along y direction by given (integer) factor

    :param im:
    :param bin_factor:
    :return: binned image
    """
    return im.reshape(im.shape[0] // bin_factor, bin_factor, im.shape[1]).sum(1)


def bin_xy(im, x_factor, y_factor):
    """ Binning along x and y directions by given (integer) factor for each
    Make sure that image doesn't overflow during sum (for example cast to 16 bits)
    :param im: input image
    :param x_factor: binning factor for x axis
    :param y_factor: binning factor for y axis
    :return: binned image
    """
    return im.reshape(im.shape[0] // y_factor, y_factor, im.shape[1] // x_factor, x_factor).sum(3).sum(1)


def to_10bit_gray(img):
    """ Convert possible rgb image to gray; convert 8-bits image to 10 bits

    :param img: gray or rgb image with 8 or 10 bits
    :return: gray image with 10 bits
    """
    if len(img.shape) == 3:
        img = img[:, :, 0]
    if np.max(img) < 256:
        img = img.astype('uint16') << 2
    return img


class StereoCam(TBox):
    class XY(NamedTuple):
        x: float
        y: float

    views = ('left', 'right')

    @staticmethod
    def camera_matrix(cam):
        C = np.identity(4)
        C[0, 0] = cam.focal.x
        C[1, 1] = cam.focal.y
        C[0:2, 2] = cam.center
        return C

    def rescale_resolution(self, scale):
        """
        Apply resolution rescaling to the calibration parameters to change those measured in pixels.
        scale * resolution MUST lead to an integer new resolution - otherwise ValueError
        :param scale: coefficient to calculate new resolution
                Note: If cameras have different resolutions they are still scaled by this same factor.
        """
        for side in self.views:
            if 'resolution' in self[side]:
                new_resolution = self.XY(*(v * scale for v in self[side].resolution))
                if not all(v.is_integer() for v in new_resolution):
                    raise ValueError(f'Rescaling must keep resolution integer! Instead received: {new_resolution}')
                self[side].resolution = self.XY(*(int(v) for v in new_resolution))

        pix_keys = ('left.focal', 'right.focal',    # TODO: use views to define
                    'left.center', 'right.center')
        for key in pix_keys:
            self[key] = self.XY(*(v * scale for v in self[key]))
        self.scale *= scale if self.scale else scale

    def __init__(self, obj=None, *, resolution=(640, 480), baseline='60mm', center=None,
                 focal=None, angle=90, units: Union[int, str, 'Quantity'] = 1, **kwargs):
        """
        Stereo (dual) cameras calibration parameters,
        is accessible as attributes or as dict as following structure:
            Par:
                obj: dict or Box kind of object - if provided all the named arguments are ignored (except of kwargs)
                resolution: (x, y) [pix]
                baseline:    b [mm]
                left:
                    focal:  (fx, fy) [pix]
                    center: (cx, xy) [pix]
                right:
                    focal:  (fx, fy) [pix]
                    center: (cx, xy) [pix]
        Args:
            resolution: (x, y) [pixels] - integer
            center: (cx, cy) [pixels] - center of the lenses from corner (0, 0) (default - sensor center)
            focal: (fx, fy) [pixels] - focal length in pixels
            angle: view angle - used to calculate focal only if its not provided - ignored otherwise
            units: str or pint-units object describing baseline units
        """
        from ..units import assign_units
        # process special initialization cases
        if obj is not None:
            super().__init__(self._convert_xy(obj), **kwargs)
            if 'baseline' in self:
                self.baseline = assign_units(self.baseline, units)
            return

        super().__init__(**kwargs)
        if '__box_heritage' in kwargs:
            return
            #  -------------------------------
        #
        # kwargs.setdefault('box_it_up', True)
        # kwargs.setdefault('default_box', True)

        resolution = self.XY(*resolution)
        self.baseline = assign_units(baseline, units)

        if focal is None:
            fsc = tan(angle / 360 * pi)
            focal = [v / 2 / fsc for v in resolution]
        elif not hasattr(focal, '__len__'):
            focal = [focal, focal]
        self.left = dict(focal=self.XY(*focal),
                         resolution=resolution)

        if center is None:
            center = [(v - 1) / 2 for v in resolution]
        self.left.center = self.XY(*center)

        self.right = dict(self.left)

    def validate(self):
        """
        Validate the configuration structure
        :return: True if OK
        """
        from numbers import Number
        ref_keys = self.__class__().keys(deep=True)

        dif = set(ref_keys).difference(self.keys(deep=True))
        invalids = [k for k, v in TBox(self.to_dict()).items(deep=True) if not isinstance(v, Number) or np.isnan(v)]

        if dif: warn(f'Missing keys: {dif}')
        if invalids: warn(f'Invalid values: {invalids}')

        return not (dif or invalids)

    def to_dict(self, *, xy=True):
        """ Recursively converts to dict
        :param xy: if True all the XY namedtuple fieled are also converted to dict
        :return recursive dictionary
         """
        XY = StereoCam.XY
        return {k: (dict(XY._asdict(v)) if isinstance(v, XY) and xy
                    else (self.__class__.to_dict(v, xy=xy) if hasattr(v, 'keys')
                          else v)) for k, v in self.items()}

    def to_yaml(self, filename=None, **kw_box):
        """
        Convert to YAML format.
        :param filename: optional - to save yaml
        :param kw_box: yaml, box keyword arguments
        :return: str if filename=None or None
        """
        from ..units import un
        d = self.to_dict()
        if isinstance(d['baseline'], un.Quantity):
            d['baseline'] = d['baseline'].to_tuple()
        return TBox(d).to_yaml(filename=filename, **kw_box)

    @classmethod
    def _convert_xy(cls, obj):
        XY = cls.XY
        for k in obj:
            item = obj[k]
            if hasattr(item, 'items'):
                obj[k] = XY(**item) if not set(XY._fields).symmetric_difference(item) else cls._convert_xy(item)
        return obj

    @classmethod
    def from_yaml(cls, yaml_string=None, filename=None, **kwargs):
        from ..units import un
        d = TBox.from_yaml(yaml_string=yaml_string, filename=filename, **kwargs)
        if isinstance(d['baseline'], str):
            d['baseline'] = un(d['baseline'])
        return cls._convert_xy(d)

    @classmethod
    def from_tiff(cls, file_name: str):
        from .imread import tiff_inu_cam
        return tiff_inu_cam(file_name)

    def center_crop(self, width, height):
        """Return camera parameters for sensor symmetrically cropped around center

        :param width: new width in pixels
        :param height: new height in pixels
        """
        if self.left != self.right:
            raise ValueError("Central crop on Camera Params is defined only if left == right")
        par = self.left
        xsz, ysz = par.resolution
        return StereoCam(baseline=self.baseline,
                         focal=par.focal,
                         center=(par.center.x - (xsz - width) // 2,
                                 par.center.y - (ysz - height) // 2),
                         resolution=(width, height))
