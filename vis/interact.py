import numpy as np

# from toolbox.vis.insight import *
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.widgets import Widget
from collections import OrderedDict

from matplotlib import _api
# from PyQt5.QtWidgets import QLineEdit, QLabel, QComboBox, QPushButton, QFileDialog, QMessageBox


def connect(ax, events):
    for event, func in events:
        ax.figure.canvas.mpl_connect(event, func)


class LinesDrawer(Widget):
    """
    Draw lines in the axis

    Left click - start new line or add point to current line
    Double click - finish line
    Right click - delete last point
    escape - delete current line (if not finished)

    """

    @_api.make_keyword_only("3.6", "useblit")
    def __init__(self, canvas, axes, useblit=True, max_points=0xffff, max_lines=0xffff, regions=None,
                 **lineprops):
        """

        :param ax:
        :param max_points:
        :param max_lines:
        :param regions: Add regions from init as a list of names, or, add regions that were
        already annotated as a dictionary of {name: [list of points]}
        TODO - add connection to Regions object from measure module
        """
        # canvas is stored only to provide the deprecated .canvas attribute;
        # once it goes away the unused argument won't need to be stored at all.
        self._canvas = canvas

        self.axes = axes
        self._canvas_infos = {
            ax.figure.canvas: {"cids": [], "background": None} for ax in axes}
        self.visible = True
        if regions is None:
            regions = {}

        self.useblit = (
                useblit
                and all(canvas.supports_blit for canvas in self._canvas_infos))

        if self.useblit:
            lineprops['animated'] = True
        if not isinstance(axes, list):
            axes = list(axes)
        self.ax = axes
        self.max_points = max_points
        self.max_lines = max_lines

        self.current_line = None
        self.current_shape = None
        regions = {name: [] for name in regions} if not isinstance(regions, dict) else regions
        self.regions = OrderedDict(regions)

        print('max_points = {max_points}, max_lines={max_lines}'.format(**self.__dict__))

        canvas_events = (('button_press_event', self.on_click),
                         # ('button_release_event', self.on_release),
                         ('motion_notify_event', self.on_move),
                         ('key_release_event', self.on_key),
                         ('draw_event', self.clear)
                         )
        self.connect(canvas_events)

    canvas = _api.deprecate_privatize_attribute("3.6")
    background = _api.deprecated("3.6")(lambda self: (
        self._backgrounds[self.axes[0].figure.canvas] if self.axes else None))
    needclear = _api.deprecated("3.7")(lambda self: False)

    def connect(self, events):
        """Connect events."""
        for canvas, info in self._canvas_infos.items():
            info["cids"] = [
                canvas.mpl_connect(name, func) for name, func in events
            ]

    def disconnect(self):
        """Disconnect events."""
        for canvas, info in self._canvas_infos.items():
            for cid in info["cids"]:
                canvas.mpl_disconnect(cid)
            info["cids"].clear()

    def clear(self, event):
        """Clear the cursor."""
        if self.ignore(event):
            return
        if self.useblit:
            for canvas, info in self._canvas_infos.items():
                # someone has switched the canvas on us!  This happens if
                # `savefig` needs to save to a format the previous backend did
                # not support (e.g. saving a figure using an Agg based backend
                # saved to a vector format).
                if canvas is not canvas.figure.canvas:
                    continue
                info["background"] = canvas.copy_from_bbox(canvas.figure.bbox)

    def on_key(self, event):
        if self.current_line:
            if event.key == 'escape':
                self.exit_line()
        elif event.key == 'a':
            pass
        elif event.key == 'i':
            for shape in self.regions:
                print(shape, ':\n', self.regions[shape])

    def select_shape(self, name) -> int:
        if name not in self.regions:
            shape_size = 0
            self.regions[name] = []
        else:
            shape_size = len(self.regions[name])

        self.current_shape = name
        return shape_size

    def complete_line(self, line):
        return len(line.get_xdata()) >= self.max_points

    def add_point(self, x, y):
        xs, ys = self.current_line[0].get_data()
        if self.complete_line(self.current_line[0]):
            self.end_line()
        else:
            [current_line.set_data([*xs, x], [*ys, y]) for current_line in self.current_line]

    def start_line(self, ax, x, y):
        if isinstance(ax, list):
            return [self.start_line(ax_, x, y) for ax_ in ax]
        else:
            return ax.plot([x] * 2, [y] * 2, '.-')[0] if len(ax.lines) < self.max_lines else None

    def end_line(self):
        # close polygon
        [current_line.set_data([*current_line.get_xdata(), current_line.get_xdata()[0]],
                               [*current_line.get_ydata(), current_line.get_ydata()[0]])
         for current_line in self.current_line]
        [current_line.set_visible(True) for current_line in self.current_line]
        self._update()
        self.regions[self.current_shape].append(self.current_line[0].get_xydata())
        self.current_line = None
        for canvas, info in self._canvas_infos.items():
            info["background"] = canvas.copy_from_bbox(canvas.figure.bbox)

    def exit_line(self):
        for idx in range(len(self.current_line)):
            self.ax[idx].lines.remove(self.current_line[idx])
        self.current_line = None
        [ax.figure.canvas.draw() for ax in self.ax]

    def remove_last_point(self):
        xydata = self.current_line[0].get_xydata()
        n = xydata.shape[0]
        if n < 3:
            self.exit_line()
        else:
            [current_line.set_data(xydata[np.arange(n) != n - 2, :].T)
             for current_line in self.current_line]

    def on_click(self, event):
        # print(event)
        if not self.current_shape:
            return
        if self.current_line:
            if event.button == 1:
                if event.dblclick:
                    self.end_line()
                else:
                    self.add_point(event.xdata, event.ydata)
            elif event.button == 3:
                self.remove_last_point()
        elif event.button == 1:
            self.current_line = self.start_line(self.ax, event.xdata, event.ydata)

        if self.current_line:
            for ax in self.ax:
                ax.set_visible(True)
            self._update()

    def on_move(self, event):
        if not self.current_line: return
        self._update()
        xs, ys = self.current_line[0].get_data()
        xs[-1], ys[-1] = event.xdata, event.ydata
        [current_line.set_data(xs, ys) for current_line in self.current_line]
        [current_line.set_visible(True) for current_line in self.current_line]
        # [current_line.figure.canvas.draw() for current_line in self.current_line]

    def _update(self):
        if self.useblit:
            for canvas, info in self._canvas_infos.items():
                if info["background"]:
                    canvas.restore_region(info["background"])
            for ax, line in zip(self.ax, self.current_line):
                ax.draw_artist(line)
            for canvas in self._canvas_infos:
                canvas.blit()
        else:
            for canvas in self._canvas_infos:
                canvas.draw_idle()


from skimage.io import imread
import pickle
# from PyQt5.QtCore import pyqtRemoveInputHook
import os

#
# class ImageAnnotator:
#
#     def __init__(self, images, annotation_file, *argv, **kwargs):
#         self.annotations = annotation_file
#         if isinstance(images, str):
#             images = imread(images)
#         # axs = imgrid(images, cmap='pink', out=True)
#         self.drawer = LinesDrawer(axs, *argv, **kwargs)
#
#         # build gui
#         pyqtRemoveInputHook()
#
#         toolbar = axs[0].figure.canvas.toolbar
#
#         self.shape_comment = QLabel('Current shape name')
#         toolbar.addWidget(self.shape_comment)
#
#         self.shape_edit = QLineEdit()
#         toolbar.addWidget(self.shape_edit)
#         self.shape_edit.editingFinished.connect(self.on_change_shape)
#
#         self.deleteButton = QPushButton('Delete')
#         toolbar.addWidget(self.deleteButton)
#         self.deleteButton.clicked.connect(self.on_delete)
#
#         toolbar.addWidget(QLabel('Shapes Label'))
#         self.shapes_list = QComboBox()
#         toolbar.addWidget(self.shapes_list)
#
#         self.saveButton = QPushButton('Save')
#         toolbar.addWidget(self.saveButton)
#         self.saveButton.clicked.connect(self.on_save)
#
#         plt.show()
#
#     def on_change_shape(self, *args):
#         name = self.shape_edit.text()
#         if not name.strip():
#             return
#         shape_size = self.drawer.select_shape(name)
#
#         if shape_size:
#             message = 'Adding to shape "%s" with %d lines' % (name, shape_size)
#         else:
#             message = 'Building new shape: %s' % name
#             self.shapes_list.addItem(name)
#         self.shape_comment.setText(message)
#
#     def on_delete(self, event):
#         name = self.shape_edit.text()
#         if name in self.drawer.regions:
#             del self.drawer.regions[name]
#
#         self.shapes_list.removeItem(self.shapes_list.find(name))
#
#     def on_save(self, event):
#         if not self.drawer.regions:
#             QMessageBox.information(self.saveButton.parent(), "Nothing to Save", "No shapes to save yet")
#             return
#
#         options = QFileDialog.Options()
#         options |= QFileDialog.DontUseNativeDialog
#         fileName, _ = QFileDialog.getSaveFileName(
#             self.saveButton.parent(),
#             "Annotation file to save", self.annotations,
#             "All Files (*);;Annotation Files (*.ann)", options=options)
#         if fileName:
#             self.saveButton.setToolTip(fileName)
#
#         with open(self.annotations, 'wb') as f:
#             pickle.dump(self.drawer.regions, f, pickle.HIGHEST_PROTOCOL)
#

################################

# axs = imgrid(np.random.rand(10, 10), cmap='gray', out=True, show=False, figsize=(5,5))
# fig = plt.gcf()
# drawer = LinesDrawer(axs[0], max_points=2)
# drawer.ax.figure.show()
# plt.show()


#
# if __name__ == "__main__":
#     import sys
#
#     app = QtWidgets.QApplication(sys.argv)
#     main_window = QtWidgets.QDialog()
#     ui = Ui_Dialog()
#     ui.setupUi(main_window)
#     main_window.show()
#     sys.exit(app.exec_())


#
if __name__ == "__main__":
    import sys

    print(*sys.argv)

    sys.argv[3:] = [int(v) for v in sys.argv[3:]]

    # annotator = ImageAnnotator(*sys.argv[1:])

    # from skimage.io import imread
    #
    # im = imread(sys.argv[0])
    #
    # axs = imgrid(np.random.rand(10, 10), cmap='gray', out=True, show=False)
    # drawer = LinesDrawer(axs[0], max_points=2)
    # drawer.ax.figure.show()
