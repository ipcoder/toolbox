from __future__ import annotations

from itertools import chain
from typing import Callable, Any

from .format import FileFormat, PathT, read_outputs, Content as CT, sfx_of


class MiddleburyCalib(FileFormat, patterns="calib.txt", content=CT.CONFIG):
    @classmethod
    def read(cls, filename: PathT, *, content=CT.CONFIG, **kws):
        from .special import middlebury_calib
        assert not kws and cls.valid_content(content)
        return middlebury_calib(filename)


_cfg_sfx = [['.yaml', '.yml'], ['.json', '.jso', '.js', '.jsn'], ['.toml', '.tml', '.ini']]


class ConfigFormat(FileFormat, desc="Nested Textual Configuration",
                   content=CT.META | CT.CONFIG, patterns=[*chain(*_cfg_sfx)]):
    from ..param import TBox
    _aliases = {sfx: grp[0].strip('.') for grp in _cfg_sfx for sfx in grp}

    @classmethod
    def read(cls, filename: PathT, *, content=CT.UNDEF, **kws):
        assert cls.valid_content(content)
        name = f'from_{cls._aliases[sfx_of(filename)]}'
        return getattr(cls.TBox, name)(filename=filename, **kws)

    @classmethod
    def write(cls, filename: PathT, data: dict, **kws) -> Callable[[PathT], Any]:
        """Acquire read function by the format"""
        name = f'to_{cls._aliases[sfx_of(filename)]}'
        return getattr(cls.TBox(data), name)(filename=filename, **kws)

    @classmethod
    def supports_write(cls, *, data=None, meta=None):
        return meta is None and (
                data is None or issubclass(data, dict))


class TifFormat(FileFormat, desc="Tiff with metadata in ImageDescription Tag",
                patterns=['.tiff', '.tif'],
                content=CT.META | CT.IMAGE):
    _meta_tag = "ImageDescription"

    @classmethod
    @read_outputs
    def read(cls, filename: PathT, *, content=CT.DATA, **kws):
        """No need for file_form"""
        from tifffile import imread, tiffcomment
        out = []
        if content & CT.DATA:
            out.append(imread(filename))
        if content & CT.META:
            import yaml
            comment = tiffcomment(filename, tagcode=cls._meta_tag)
            meta = yaml.load(comment, Loader=yaml.SafeLoader)
            out.append(meta if isinstance(meta, dict) else None)
        return tuple(out)


class ImageFormat(FileFormat, desc="General image format",
                  patterns=['.png', '.bmp', '.jpg'],
                  content=CT.IMAGE):
    @classmethod
    def read(cls, filename: PathT):
        from .imread import imread
        return imread(filename)

    @classmethod
    def write(cls, filename: PathT, data, *, meta=None, **kws):
        from .imread import imsave
        assert meta is None
        imsave(filename, data, **kws)


class InuStereoFormat(FileFormat, desc="Inuitive Stereo Images with Cameras metadata",
                      patterns=[
                          r"StereoImage_\d+\.tif",
                          r"Output_Video_(Left|Right)_\d+\.(bmp|png|tif|jpg)",
                          r"Video_\d+\.(bmp|png|tif|jpg)"],
                      content=CT.IMAGE | CT.META):
    """Inuitive tif based format for 2 stereo images with custom tags."""

    @classmethod
    def read(cls, filename: PathT, content=CT.DATA, *kws):
        """No need for file_form"""
        assert not kws
        out = []
        if content & CT.META:
            from imread import imread_stereo
            out.append(imread_stereo(str(filename), cam_info=False))
        if content & CT.DATA:
            from imread import tiff_inu_cam
            out.append(tiff_inu_cam(str(filename)))
        return tuple(out) if len(out) > 1 else out if len(out) == 1 else 0
