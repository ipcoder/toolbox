from enum import Enum
from typing import Union, Callable, Tuple
from warnings import warn

import cv2 as cv
import numpy as np

Image = np.ndarray
ImageGen = Callable[[int, Tuple[int, int], type], Image ]


# pre-defined generators (for testing amd other purposes)

def gen_const(val) -> ImageGen:
    """ Image generator all filled by a single value """
    return lambda _, size, form: val * np.ones(size[::-1], dtype=form)


def gen_rand_norm(avr: float, sigma: float) -> ImageGen:
    """ Image generator with given average and standard deviation """
    def generator(fid, size, form):
        form = np.dtype(form)
        max_val = np.finfo(form).max if form.kind[0] == 'f' else np.iinfo(form).max
        image = avr + sigma * np.random.randn(*size[::-1])
        return np.clip(image, 0, max_val).astype(form)
    return generator


def gen_randint(min_val: int, max_val: int) -> ImageGen:
    """ Image generator with integer values in given interval """
    return lambda _, size, form: np.random.randint(min_val, max_val, size[::-1], dtype=form)


class VideoStreamer:
    """ Class to help process image streams. Three types of possible inputs:"
        1.) USB Webcam.
        2.) A directory of images (files in directory matching 'img_glob').
        3.) A video file, such as an .mp4 or .avi file.
    """

    class Source(Enum):
        UNDEFINED = 0
        VIDEO_FILE = 1
        IMAGE_FILES = 2
        CAMERA = 3
        GENERATOR = 4

    def __len__(self):
        lim_num = self._lim_num if self._lim_num else int(9e9)
        return min(len(self._index), lim_num) if hasattr(self, '_index') else lim_num

    @property
    def fid(self):
        return self._fid

    @property
    def skip(self):
        return self._skip

    def __init__(self, source: Union[int, str, Callable], *,
                 form=None, size=None,  # image transformation
                 first=0, lim_num=None, skip=1):        # frames iteration
        """
        Create video streamer from the given source.
        :param source:  camera_id | path to video file | glob filter for separate files
        :param form:    format of the output images. None - if leave as it
                            if a type - cast into it
                            if a float type - normalize by dividing by 255
                            if 'grey' (or 'gray') - convert to gray scale (uint8)
        :param size:    resize to (width, height)  !!! (x,y)
                                or scale factor (< 1) or None
        :param first:   0   # skip this number of frames before starting
        :param lim_num: maximal number of frames to produce (if not None)
        :param skip:    frames to skip
        """

        self._form = np.uint8 if isinstance(form, str) and form.lower() in ('gray', 'grey') \
            else np.dtype(form) if form else None
        self._size = size

        self._first = first
        self._skip = skip
        self._lim_num = lim_num

        self._fid = 0
        self._count = 0

        self._capture_dev = []
        self._type = VideoStreamer.Source.UNDEFINED

        if callable(source):  # TODO: check using generators arguments type annotation?
            self._generator = source
            if self._size is None:
                raise ValueError('Argument `size` must be specified (x, y) if `source` is a generator!')
            assert len(self._size) == 2
            for v in self._size:
                assert round(v) == v

            if self._form is None:
                self._form = np.uint8

            if self._lim_num is None:
                raise ValueError('Argument `lim_num` must be specified if `source` is a generator!')
            assert 0 < self._lim_num

            self._type = VideoStreamer.Source.GENERATOR
        elif isinstance(source, int):
            self._capture_dev = cv.VideoCapture(source)
            self._type = VideoStreamer.Source.CAMERA
        elif isinstance(source, str):
            import os
            from glob import glob
            ext = os.path.splitext(source)[1][1:].lower()
            if ext in ('jpg jpeg jp2 pbm pgm ppm png bmp tif tiff gif'.split()):
                if os.path.isdir(source):
                    warn('Files search pattern not found. Considering all the files in the folder as a source!')
                    source = os.path.join(source, '*')
                self._index = [*enumerate(sorted(glob(source)))][self._first::self._skip]
                found = len(self._index)
                if found < 2:
                    warn('Number of expected video frames: {found}!')
                self._type = VideoStreamer.Source.IMAGE_FILES
            else:
                video_file = glob(source)
                if len(video_file) > 1:
                    FileExistsError('More than one file matches search pattern!')
                elif not video_file:
                    FileExistsError('File not found!')
                self._capture_dev = cv.VideoCapture(video_file[0])
                self._type = VideoStreamer.Source.VIDEO_FILE
                num_frames = int(self._capture_dev.get(cv.CAP_PROP_FRAME_COUNT))
                if num_frames < self._first:
                    warn(f'First frame requested {self._first} is beyond the total number {num_frames}!')
                self._index = range(self._first, num_frames, self._skip)

    def _next_video_frame(self):
        """
        iterates over video frames of video file or capturing device.
        Raises StopIteration at the end of the file or if capture device is
        :return: frame image
        """
        if self._type in (VideoStreamer.Source.VIDEO_FILE, VideoStreamer.Source.CAMERA):
            self._fid = self._index[self._count]
            self._capture_dev.set(cv.CAP_PROP_POS_FRAMES, self._fid)
            success, image = self._capture_dev.read()
            if success is False:
                print('VideoStreamer: Cannot get image from camera (maybe bad --camid?)')
                raise IOError
        elif self._type == VideoStreamer.Source.IMAGE_FILES:
            self._fid, file_name = self._index[self._count]
            image = cv.imread(file_name, cv.IMREAD_UNCHANGED)
            if image is None:
                raise FileExistsError(file_name)
        elif self._type == VideoStreamer.Source.GENERATOR:
            self._fid += self._skip
            image = self._generator(self._fid, self._size, self._form)
        else:
            raise TypeError('Unknown source type!')
        return image

    def __iter__(self):
        self._fid = 0
        return self

    def _transform(self, image):
        if self._form:
            if image.ndim > 2:
                image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            image = image.astype(self._form)
            if np.dtype(self._form).kind[0] == 'f':
                image = image / 255

        if self._size is not None:
            new_size = self._size if hasattr(self._size, '__len__') else \
                       (int(x * self._size) for x in image.shape[::-1])
            image = cv.resize(image, tuple(new_size), interpolation=cv.INTER_AREA)
        return image

    def __next__(self):
        if self._lim_num and self._count >= self._lim_num:
            raise StopIteration
        try:
            image = self._next_video_frame()
            if self._form is not None:
                image = self._transform(image)
        except IndexError:
            raise StopIteration
        self._count += 1
        return self._fid, image
