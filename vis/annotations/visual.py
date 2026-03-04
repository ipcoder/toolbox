from __future__ import annotations
import time
from datetime import datetime
from typing import Union, Sequence
import ipywidgets as ipw
import matplotlib as mpl
import mplcursors
from IPython.display import display
from matplotlib.backend_bases import MouseButton
from matplotlib.pyplot import Axes
from matplotlib.widgets import PolygonSelector
from toolbox.datacast import DataCollection
from algutils.label import Labels
from .collect import IssueCollection

__all__ = ['VisIssue', 'show_issues_on_scenes']

def_format_params = dict(
    poly_fmt_params=dict(edgecolor='r', linewidth=0.8, fill=False),
    txt_fmt_params=dict(color='black', fontsize=10, bbox=dict(facecolor='yellow', alpha=0.8)))

SceneLabels = Union[dict, Labels]

# ToDo Definition of minimum labels make sense?
min_issue_labels = {'scene', 'dataset'}


class VisIssue:
    """
    Visualizing issues on a figure.
    Enables interactive annotating and editing of an issue collection.
    Main function is VisIssue().show.

    This class works on a predefined figure (usually created from a visualizer)
    The figure is required to depict a SPECIFIC scene ( for all the axes).

    axes_queries enables choosing on which axes to show issues, from the axes in the figure.
    Each axes (image) need to be accompanied by its scene labels (scene, dataset...)
    Use the helper class method attach_labels() to attach labels to the axes directly.
    See more on how to define axes_queries in method _show_issues_on_axes()
    """

    SCENE_LABELS = '_scene_labels'

    def __init__(self, issues: IssueCollection,
                 fig: mpl.pyplot.Figure,
                 axes_queries: Sequence[int] | Sequence[tuple[int, SceneLabels]] |
                               Sequence[dict[int, SceneLabels]] = None,
                 scene_labels: SceneLabels = None, **format_params):
        """
        :param issues: Issue collection to visualise and edit.
        :param fig: Figure to show issues on and interact.
        :param scene_labels: Labels to add for each new issue by default (shared scene labels)
        :param axes_queries: list of either tuple of axes **indexes** and labels, or only axes indexes.
                              Used to choose which axes to show on for that figure.
                              If None, all axes would be used ( but all need to have attr ._scene_labels)
        :param format_params: parameters for the polygon and the annotation. poly_fmt_params / txt_fmt_params
        """
        self.issues = issues
        self.categories = issues.categorical_values('issue_type')
        self.algorithms = issues.prop_values('alg')
        self.fig = fig
        for ax in self.fig.axes:
            ax.polygons = []
            ax.my_cursor = None
        if scene_labels and min_issue_labels.difference(set(scene_labels.keys())):
            raise ValueError(f"Scene Labels for adding issues must have at least labels {min_issue_labels} ")
        self.add_labels = scene_labels
        self.axes_queries = axes_queries
        self.fmt_params = def_format_params | (format_params or {})
        self.add_mode = False
        self.active_hover = {}
        self.cursors = {}

        self.out = None
        self.box = None
        self.hbox = None
        self.sel_patch = None
        self.issue = None
        self.txt = None
        self.new_iss_btn = None
        self.sel = None
        self.edit_box = None
        self.new_iss_box = None

    # High Level function for Issue Visualization
    def show(self, **kwargs):
        """
        Show the issues on the requested axes.
        Include also interactive issue editing (jupyter only) which also updates the issue collection:
        1. Add mode for adding new issues using a polygon selector.
        2. Edit mode for including labels for a new issue, editing existing issues or deleting them.


        :param kwargs: updating format parameters for the current show function.
        """

        self._show_issues_on_axes(self.issues,
                                  axes=self._idx_to_axes(self.fig.axes),
                                  format_params=self.fmt_params | kwargs)
        self.new_iss_btn = ipw.ToggleButton(value=False, layout={"width": "80px"}, description='Issue',
                                            disabled=False, button_style='', tooltip='Add Issue',
                                            icon='plus')
        self.new_iss_btn.observe(self.on_press_new_issue, names='value', type='change')
        self.out = ipw.Output(layout={'border': '1px solid black'})
        self.hbox = ipw.HBox([self.new_iss_btn])

        self.box = ipw.VBox([self.fig.canvas.toolbar, self.out,
                             ipw.Label('Ctrl + Left Click to edit:'), self.hbox])

        self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        if self.add_labels is None:
            with self.out:
                self.new_iss_btn.disabled = True
                self.new_iss_btn.tooltip = 'Adding issues is allowed only when including Scene Labels'

        display(self.box)

    # Interactive matplotlib behaviour functions:

    def on_mouse_move(self, event):
        """
        Callback for mouse location change.
        Removing issue annotations when the mouse hovers away
        """
        with self.out:

            def _clear_annotation(ax, sel):
                self.cursors[ax].remove_selection(sel)
                self.active_hover[ax] = None

            if (cur_ax := event.inaxes) not in self.fig.axes:
                for ax in self.fig.axes:
                    if (sel := self.active_hover.get(ax)) is not None:
                        _clear_annotation(ax, sel)
                return

            if (sel := self.active_hover.get(cur_ax)) is not None and not sel.artist.contains(event)[0]:
                _clear_annotation(cur_ax, sel)

    def on_button_press(self, event):
        """
        Callback for mouse button pressing.
        Left mouse + CTRL : Changing to edit mode.
        Double Click : Initiating Polygon Selector when in add mode.
        """
        with self.out:
            if event.button is MouseButton.LEFT and event.key == 'control' and not self.add_mode:
                # ToDo what if click is inside more than one polygon?
                for patch in event.inaxes.patches:
                    if isinstance(patch, mpl.patches.Polygon) and patch.contains(event)[0]:
                        if self.sel_patch is not None:
                            self.exit_edit_mode()
                        self.issue = self.issues.qix(id=patch.labels['id'], as_ic=False
                                                     ).reset_index().to_dict(orient='records')[0]
                        self.sel_patch = patch
                        self.sel_patch.set_edgecolor('blue')
                        self.start_edit_mode()

            elif event.dblclick and self.add_mode:
                self._remove_selector()
                self.sel = PolygonSelector(event.inaxes, onselect=self.on_polygon_select,
                                           useblit=True)

    def on_polygon_select(self, vertices: tuple):
        """
        Callback for finishing the selection of the polygon selector (first closing / after editing it)
        Initiating editing mode with the selected polygon.
        Won't work if the polygon is too small.
        """
        def poly_bbox(verts: tuple):
            """
            Calculate bounding box for vertices.
            :param verts:
            :return: 2x2 array [ x_min, y_min, dx, dy]
            """
            import numpy as np
            verts = np.array(verts)
            return np.array([mins := verts.min(axis=0),
                             verts.max(axis=0) - mins])

        bbox = poly_bbox(vertices)
        bbox_area = bbox[1,0] * bbox[1,1]
        with self.out:
            if len(vertices) <= 2 or bbox_area < 500 or bbox[1, :].min() < 5:
                print('Polygon is too small, try again')
            else:
                self.issue = {'polygon': tuple([(round(vert[0]), round(vert[1])) for vert in vertices])}
                not self.edit_box and self.start_edit_mode()

    def on_key_press(self, event):
        """
        Callback for keyboard pressing.
        ESC : exiting edit mode.
        DEL : If a polygon is selected, delete it ( similar to self.edit_box.del_btn)
        """
        with self.out:
            if event.key == 'escape' and (self.sel_patch is not None or self.add_mode):
                self.exit_edit_mode()
            elif event.key == 'delete':
                self.delete_polygon()

    def on_press_new_issue(self, change):
        """
        Callback for pressing the add_issue ( + ) button.
        Button pressed: Add mode is initiated. Double Click for starting polygon selection
        Button unpressed: Exiting Add mode.
        """
        with self.out:
            if change['new']:
                self.add_mode = True
                self.new_iss_box = ipw.VBox(
                    [ipw.Label('Double Click on an Axes for Initiate Polygon Selector')])
                self.box.children = list(self.box.children) + [self.new_iss_box]
            else:
                self.add_mode = False
                self.box = self._remove_children_from_box(self.box, self.new_iss_box)
                self.exit_edit_mode()

    # Editing functions

    def start_edit_mode(self):
        """
        Initiate a new toolbar with buttons for saving and deleting issues, and dropdown with labels options.
        """
        with self.out:
            self.del_button = ipw.Button(icon='trash', layout={"width": "34px"})
            self.del_button.on_click(self.delete_polygon)
            self.save_button = ipw.Button(icon='save', layout={"width": "34px"})
            self.save_button.on_click(self.save_polygon)
            self.cat_drpdwn = ipw.Dropdown(options=self.categories, layout={"width": "50px"},
                                           value=self.issue.setdefault('issue_type', None))
            self.alg_drpdwn = ipw.Dropdown(options=self.algorithms, layout={"width": "200px"},
                                           value=self.issue.setdefault('alg', None))

            self.edit_box = ipw.HBox([self.del_button, self.save_button,
                                      self.cat_drpdwn, self.alg_drpdwn])

            self.hbox.children = list(self.hbox.children) + [self.edit_box]

            self.new_iss_btn.disabled = True

    def exit_edit_mode(self):
        """
        Removing the edit toolbar
        """
        with self.out:

            if self.sel_patch is not None:
                self.sel_patch.set_edgecolor('red')
                self.sel_patch = None

            self._remove_selector()
            self.hbox = self._remove_children_from_box(self.hbox, self.edit_box)
            self.edit_box = None
            self.issue = None
            if self.add_labels:
                self.new_iss_btn.disabled = False

    def delete_polygon(self, *args, **kwargs):
        """
        Callback for pressing the delete button of the edit toolbar.
        Deleting the selected polygon from the issue collection and from the axes.
        """
        with self.out:
            if self.sel_patch is None:
                print('Select a polygon first')
            else:
                axes = self._idx_to_axes(self.fig.axes)
                self._remove_issue_from_axes(self.issue, axes)
                self.issues.remove(issue={'id': self.issue['id']})
                self.exit_edit_mode()

    def save_polygon(self, *args, **kwargs):
        """
        Callback for pressing the save button of the save toolbar.
        Edited issue - the original issue is deleted and a new issue with the same id (and verts) is created
        New issue - a new ID is created. ID is the time of creation in seconds.
        """
        axes = self._idx_to_axes(self.fig.axes)
        with self.out:
            if not self.cat_drpdwn.value or not self.alg_drpdwn.value:
                print("Choose labels before saving")
            else:
                # editing existing issue
                if self.issue is not None and 'id' in self.issue.keys():
                    iss_id = self.issue['id']
                    self.issues.remove(issue={'id': self.issue['id']})
                    self._remove_issue_from_axes(self.issue, axes)
                else:  # new one
                    iss_id = time.time()
                    self.issue |= self.add_labels

                self.issue |= {'issue_type': self.cat_drpdwn.value, 'alg': self.alg_drpdwn.value, 'id': iss_id}
                self._show_issues_on_axes(self.issue, axes,
                                          self.fmt_params)
                self.issues.add(self.issue)
                self.exit_edit_mode()

                if self.new_iss_btn.value:
                    self.new_iss_btn.value = False

    def _show_issues_on_axes(self, issues,
                             axes: Sequence[Axes] | Sequence[tuple[Axes, SceneLabels]] |
                                   Sequence[dict[Axes, SceneLabels]],
                             format_params=None):
        """
        Show issues on the axes provided.
        Each axes (image) need to be accompanied by its scene labels (scene, dataset...)

        If axes are an iterable of (Axes, SceneLabels), the SceneLabels instance is a dict of labels for each
         axes.
        If axes are an iterable of Axes, there is an assumption the scene_labels are located inside each axes
         in the attribute axes._scene_labels

        :param axes: Iterables of either Axes of matplotlib, or tuples of (Axes, SceneLabels)
        :param format_params: dict of Format parameters for the drawn polygon and the text beside it.
        """
        axes = self.axes_to_standard_format(axes)

        if isinstance(issues, dict):
            issues = IssueCollection([issues])

        for ax, labels in axes:

            ax_issues = issues.qix(**labels,
                                   as_ic=False).squeeze_levels(levels=['scene', 'dataset'])['polygon']
            for poly_labels, polygon in ax_issues.items():
                poly_labels = dict(zip(ax_issues.index.names, poly_labels))
                poly = self._draw_one_issue(ax, poly_labels, polygon, format_params)
                ax.polygons.append(poly)
            self._refresh_cursor(ax)

    def _remove_issue_from_axes(self, issue: dict,
                                axes: Sequence[Axes] | Sequence[tuple[Axes, SceneLabels]] |
                                      Sequence[dict[Axes, SceneLabels]]):
        """
        Removing issue from axes
        :param issue: issue to remove, as dict with labels.
        :param axes: Sequence of some axes, to remove the issue from.
        :return:
        """
        axes = self.axes_to_standard_format(axes)
        for ax, _ in axes:
            for patch in ax.patches:
                if patch.labels['id'] == issue['id']:
                    patch.remove()
                    ax.polygons.remove(patch)
            self._refresh_cursor(ax)

    # Helper functions

    @staticmethod
    def _draw_one_issue(ax: mpl.pyplot.Axes, poly_lbls: dict, polygon, format_params=None):
        """
        Draw one issue on an axes.
        :param ax: Axes to draw the issue on.
        :param poly_lbls: dict of Labels of the polygon ( issue_type, algorithm...)
        :param polygon: vertices of the polygon
        :param format_params: dict of Format parameters for the drawn polygon and the text beside it.
        """
        from matplotlib.patches import Polygon
        poly = ax.add_patch(Polygon(polygon, **format_params['poly_fmt_params']))
        poly.labels = poly_lbls
        return poly

    def _refresh_cursor(self, ax):
        """
        Creating a new cursor for the specified Axes after editing issues,

        Annotation for issues is created using mplcursor.cursor, and each cursor is saved inside its axes.
        When adding new issue or deleting / editing, a new cursor needs to be created to update annotation.
        :param ax:
        :return:
        """
        if ax.my_cursor:
            ax.my_cursor.remove()
        ax.my_cursor = mplcursors.cursor(ax.polygons, hover=True)
        self.cursors[ax] = ax.my_cursor
        self.active_hover[ax] = None

        @ax.my_cursor.connect("add")
        def on_add(sel):
            """
            Callback for a creation of selection (hovering over polygon of an issue).
            :param sel: the selection of the cursor
            :return:
            """
            labels = sel.artist.labels.copy()
            id = labels.pop('id')
            date = datetime.fromtimestamp(id).strftime("%d.%m.%y,%H:%M")
            # sel.annotation.set_text(
            #     f"{', '.join(labels.values())},{date}")  # FixMe - supports only string labels

            iss_typ = labels.pop('issue_type')
            iss_desc = self.issues.categorical_table('issue_type').loc[iss_typ]['Short Description']

            sel.annotation.set_text(
                f"{iss_desc},{', '.join(labels.values())},{date}")
            sel.annotation.update(self.fmt_params['txt_fmt_params'])
            self.active_hover[ax] = sel

    def _idx_to_axes(self, axes: list[mpl.pyplot.Axes]):
        """
        Picks axes from the  indexes specified in self.axes_queries
        :return:
        """
        if self.axes_queries is None:
            return axes
        if isinstance(self.axes_queries[0], int):  # list of indexes
            return [axes[ax_num] for ax_num in self.axes_queries]
        else:
            return [(axes[ax_num], labels) for (ax_num, labels) in self.axes_queries]

    @classmethod
    def attach_labels(cls, objects: object | list[object], labels: dict) -> object | list[object]:
        """
        Helper method for attaching labels to an object as an attribute _scene_labels
        :return: objects with attached labels
        """

        def _attach_lbl(obj):
            if not hasattr(obj, cls.SCENE_LABELS):
                obj._scene_labels = labels
            else:
                obj._scene_labels |= labels
            return obj

        if not isinstance(objects, list):
            objects = [objects]
        return [_attach_lbl(obj) for obj in objects]

    @staticmethod
    def _remove_children_from_box(box, child_to_rmv):
        """
        Helper function to remove children from an ipywidgets box.
        :param box:
        :param child_to_rmv:
        :return:
        """
        children = []
        for child in box.children:
            if child is child_to_rmv:
                child.close()
            else:
                children.append(child)
        box.children = children
        return box

    @staticmethod
    def axes_to_standard_format(axes):
        """
        Make sure axes are a list of  tuple of (axes, labels)
        :param axes:
        :return:
        """
        axes = list(axes)
        if isinstance(axes[0], Axes):  # change to tuples of (Axes, SceneLabels)
            axes = [(ax, ax._scene_labels) for ax in axes]
        elif isinstance(axes[0], dict):
            axes = list(axes.items())
        return axes

    def _remove_selector(self):
        """Remove Polygon selector if exists"""
        if self.sel is not None:
            self.sel.clear()
            self.sel.disconnect_events()
            del self.sel
            self.sel = None


# High Level Visualization functions

def show_issues_on_scenes(issues: IssueCollection,
                          dc: DataCollection, vis, axes_queries=None):
    scn_lbl_names = ['dataset', 'scene']
    for scene_labels, scn_issues in issues.db.groupby(scn_lbl_names):
        vis_outs = vis(dc.qix(*scene_labels, trans=True))
        VisIssue.attach_labels(vis_outs.fig.axes,
                               {name: label for name, label in zip(scn_lbl_names, scene_labels)})
        scn_issues = IssueCollection.from_db(scn_issues, like=issues)
        vis_issue = VisIssue(scn_issues, vis_outs.fig, axes_queries=axes_queries)
        vis_issue.show()
