"""
MatPlotLib Utilities Visualization
"""
from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import Callable


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombuffer("RGBA", (w, h), buf.tobytes(), 'raw', "RGBA", 0, 1)


class FigureCatcher:
    """
    Create figure in controlled environment and redirect result into
    different possible ways:

     - Capture as an image
     - Display without interactivity
     - Place into IPython Widget

    Implements a Context Manager protocol.
    """
    images = {}

    def __init__(self, *, out_widget=None, name=None, image_proc=None, mode='stealth'):
        """
        Configure how to process the figure once its created:
        :param out_widget: IPython widget to host the image
        :param name: key to access resulting image in the internal repository
        :param image_proc: function to apply to the image
        :param mode: controls possible behaviour
            - stealth: figure's image is captured and not shown
            - keep:    keeps figure after the capture
            - bypass:  does not captures the image
        """
        assert mode in ('bypass', 'stealth', 'keep')
        self.mode = mode
        self.out_widget = out_widget
        self.image_proc = image_proc
        self.name = name

    def __enter__(self):
        if self.mode in ('stealth', 'bypass'):
            self.initial_mode = mpl.is_interactive()
            mpl.interactive(False)
            return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.mode == 'bypass':
            plt.show()
            mpl.interactive(self.initial_mode)
            return
        fig = mpl.pyplot.gcf()
        img = fig2img(fig)
        if self.mode == 'stealth':
            mpl.pyplot.close()
            mpl.interactive(self.initial_mode)

        if self.out_widget:
            from IPython.display import display
            with self.out_widget:
                display(img)
        if self.name:
            self.images[self.name] = img
        if self.image_proc:
            self.image_proc(img)

    @staticmethod
    def snap(func: Callable):
        """
        Wrapper (can be used as decorator) around a drawing function.

        It will keep its inputs arguments API, but alters the results:
        If function originally returns None - it will return the Image,
        otherwise - tuple: (Image, result)
        """
        from functools import wraps

        @wraps(func)
        def wrapped(*args, **kwargs):
            name = '_last_snap'
            with FigureCatcher(name=name):
                res = func(*args, **kwargs)
                img = FigureCatcher.images.pop(name)
            return img if res is None else (res, img)

        return wrapped
