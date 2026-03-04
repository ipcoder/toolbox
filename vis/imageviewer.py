"""Amazing tool to view images.

http://confl.il.inuitive-tech.com/display/UTM/Test+Points+Viewer
http://confl.il.inuitive-tech.com/display/UTM/Default+viewer+for+image+files
"""

import argparse
import math
import os.path
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from matplotlib.backends.qt_compat import QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QDialog, QApplication, QVBoxLayout

from toolbox.vis.insight import imgrid
from algutils.io.ciif import images_to_ciif
from algutils.short import unless
from algutils.io.imread import imread


def split_by_bits(arr, bits):
    start_bit, arrays = 0, []
    for i in bits:
        parsed_array = (arr >> start_bit) & (2 ** i - 1)
        arrays.append(parsed_array)
        start_bit += i
    return arrays


def titles_from_filename(file_name):
    file_name = os.path.splitext(os.path.basename(file_name))[0]
    # split by '_' and exclude fields which are numbers only or 'left', 'right' ignoring case
    return [s for s in re.split('(?<![a-z])\d+(?![a-z])|_+|left|right', file_name, flags=re.I) if s]


def command_args():
    parser = argparse.ArgumentParser(prog='ImageViewer')
    parser.add_argument('path', type=str, nargs='+',
                        help='Path to the image file')
    parser.add_argument('-b', '--bits', type=int, nargs='+', default=[],
                        help='Number of bits in each signal. (Only one image supported)')
    parser.add_argument('-t', '--titles', type=str, nargs='+', default=[],
                        help='Title for each signal')
    parser.add_argument('-c', '--colormap', type=str, default='wide', help='Color Map')
    parser.add_argument('-o', '--output', type=str, default='',
                        help='Output file')
    parser.add_argument('-r', '--reverse', action='store_true',
                        help='Should Reverse')

    class Args:
        def __init__(self):  # just to please PyCharm ;-)
            self.path = ''
            self.colormap = 'wide'
            self.bits = []
            self.titles = []
            self.output = ''
            self.reverse = False
            self.shape = (None, None)

    arguments = Args()
    parser.parse_args(namespace=arguments)

    return arguments


def _load_images(file_path, titles):
    """ Read data according to the file type
    """

    unless(os.path.isfile(file_path), 'File does not exist ' + file_path)
    res = imread(file_path, out=dict)

    internal_names, arrays = zip(*res.items())
    if len(titles) != len(arrays):
        if internal_names != tuple(map(str, range(len(internal_names)))):
            return internal_names, arrays

        if len(arrays) > len(titles):
            titles = titles_from_filename(file_path)[-len(res):]

        if len(arrays) != len(titles):
            file_name = os.path.splitext(os.path.split(file_path)[1])[0]
            titles = [f'{file_name}_{ii}' for ii in range(len(res))]

    return titles, arrays


class ImgridWindow(QDialog):
    def __init__(self, *arrays, parent=None, **imgrid_pars):
        # self.trayIcon = QSystemTrayIcon(self)

        super().__init__(parent, flags=
        QtCore.Qt.WindowMinMaxButtonsHint | QtCore.Qt.WindowCloseButtonHint)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

        # self.setWindowFlag(self.windowFlags() | )
        # a figure instance to plot on

        self.figure = plt.Figure(dpi=85)

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.canvas.setFocus()

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        # self.button = QPushButton('Plot')
        # self.button.clicked.connect(self.plot_images)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        # layout.addWidget(self.button)
        self.setLayout(layout)

        self.figure.sub_axis = imgrid(*arrays, **imgrid_pars, fig=self.figure, out=True, toolbar=self.toolbar)
        self.canvas.draw()


def _show(arrays, path, colormap):
    def rem_common(strings):
        """ remove common prefix and suffix """

        def rem_pfx_reverse(strings):
            """ remove prefix and reverse strings in iterable """
            match = re.search('[A-Za-z]*$', os.path.commonprefix(strings))
            n = match.span()[0] if match else 0
            return [s[n:][::-1] for s in strings]

        return rem_pfx_reverse(rem_pfx_reverse(strings))

    def window_title(paths):
        common_path = os.path.commonpath(paths)
        n = len(common_path) + len(os.path.sep)
        return ', '.join(s[n:] for s in paths) + f' in {common_path}'

    pars = dict(window_title=window_title(path),
                titles=rem_common(titles),
                cmap=colormap, adj_clim=True)

    app = QApplication(sys.argv)
    main = ImgridWindow(*arrays, **pars)
    main.show()
    sys.exit(app.exec_())
    pass


def _write_out(output, arrays, bits, reverse):
    if output.endswith('.ciif'):
        images_to_ciif(args.output, arrays)
    else:
        if len(bits) == 0:
            bits = [0]
        unless(len(bits) == len(arrays), 'Specifying bit for each stream is required for output operation (use -b)')

        results = np.zeros(arrays[0].shape, dtype=np.dtype('uint64'))
        total_bits = sum(bits)
        unless(total_bits <= 64, 'Total number of bits cannot exceed 64')

        arrs_bits = zip(arrays, bits) if reverse else zip(reversed(arrays), reversed(bits))
        for data, bit in arrs_bits:
            results <<= bit
            results += (data & (2 ** bit - 1)).astype('uint64')
        results = results.astype('uint{}'.format(2 ** math.ceil(math.log(max(total_bits, 8), 2))))
        io.imsave(output, results)
    pass


if __name__ == '__main__':
    args = command_args()

    if type(args.path) is not list:
        args.path = [args.path]

    titles, arrays, assigned_titles = [], [], 0
    for file in args.path:
        file_titles, file_arrays = _load_images(file, args.titles[assigned_titles:])
        assigned_titles += len(file_titles)
        titles += file_titles
        arrays += file_arrays

    if args.bits:
        if len(arrays) > 1:
            raise ValueError(f'Split by bits planes is supported only for a single channel data (found {len(arrays)})')
        arrays = split_by_bits(arrays[0], args.bits)

    if args.output:
        _write_out(output=args.output, arrays=arrays, bits=args.bits, reverse=args.reverse)
    else:
        _show(arrays=arrays, path=args.path, colormap=args.colormap)
