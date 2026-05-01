import os


from skimage.transform import resize
import numpy as np


def depth2pt(depth, cx, cy, flx, fly):
    z = depth
    i, j = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    x = (j - cx) * (z / flx)
    y = (i - cy) * (z / fly)
    xyz = np.stack([x.flat, y.flat, z.flat], axis=1)
    return xyz


flip_transform = [[1, 0, 0, 0],
                  [0, -1, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, 1]]


class UpdateToggler:
    def __init__(self):
        self.should_update = True

    def __call__(self, vis):
        self.should_update = not self.should_update


def triangulate(h, w):
    # old slow version - use triangulate_grid instead
    print('using slow triangulation')

    l = []
    for i in range(0, h - 2, 2):
        for j in range(0, w - 2, 2):
            idx = i * w + j
            l.append([idx + 1, idx, idx + w])
            l.append([idx + 2, idx + 1, idx + 2 + w])
            l.append([idx + w, idx + w + 1, idx + 1])
            l.append([idx + 1, idx + w + 1, idx + w + 2])

            l.append([idx + w, idx + 2 * w, idx + 2 * w + 1])
            l.append([idx + w + 1, idx + w, idx + 2 * w + 1])
            l.append([idx + w + 1, idx + 2 * w + 1, idx + w + 2])
            l.append([idx + w + 2, idx + 2 * w + 1, idx + 2 * w + 2])
    return np.array(l)


def triangulate_grid(rows, cols, face_front=False):
    """ Triangulate the grid defined by the arguments by splitting each grid cell rectangle
    into two triangles along one of its diagonals.
    The diagonal is selected by the position of the cell in the grid (even-odd).

    Args:
        h -  height (must be even)
        w -  width (must be even)
        face_front - directions of the triangles face - frontal or not
    Return:
        array [Triangles_num x 3] with position of triangles vertices as flat indices in the initial grid.
    """
    def correct_size(dim, sz):
        from warnings import warn
        even_size = (sz // 2) * 2
        if sz != even_size:
            warn(f'Dimensions must be even! {dim} clipped from {sz} to {even_size}.')
        return even_size

    rows, cols = [correct_size(*dim) for dim in (('height', rows), ('width', cols))]

    from numpy import moveaxis as mvax

    # Construct an elementary triangulation cell on the area of 2x2 grid squares
    # even part cell consists of 4 triangles: 2 up for even j + 2 down for odd
    even_part = np.array([
        [[[0, -1,  0],
          [0,  0,  1]],
         [[0,  0,  1],
          [0,  1,  0]]],

        [[[0, -1,  0],
          [0,  1,  1]],
         [[0,  0,  1],
          [0,  1,  1]]]])

    if not face_front:
        even_part = mvax(mvax(even_part, 3, 0)[[0, 2, 1]], 0, 3)

    # add the odd (mirrored) part and move tr_coord to first axis for easy split
    cell = mvax(np.stack([even_part, even_part[[1, 0]]]), 3, 0)
    # axis: [img_i, img_j, tr_id, vertex]
    # dims:      2,     2,     2,      3)

    # Now repeat 2x2 triangulation cell to cover the entire grid
    def vertex_coords(ids, c):
        return (np.repeat(ids, 6) + np.tile(c.reshape(2, 2, 6), (rows // 2, cols // 2, 1)).flat).reshape(
            rows * cols * 2, 3)

    ii, jj = (vertex_coords(gid, c) for gid, c in zip(np.mgrid[0:rows, 0:cols], cell))
    # crop points outside the grid:
    valid = np.all(ii >= 0, axis=1) & np.all(ii < rows, axis=1) & \
            np.all(jj >= 0, axis=1) & np.all(jj < cols, axis=1)
    return (ii * cols + jj)[valid]


class MESHUpdater:
    def __init__(self, disp_file_gen, rgb_file_gen, bl, cx, cy, flx, fly):
        from open3d.geometry import TrinagleMesh
        self.pt = TriangleMesh()
        self.disp_file_gen = disp_file_gen
        self.rgb_file_gen = rgb_file_gen
        self.cx = cx
        self.cy = cy
        self.flx = flx
        self.fly = fly
        self.bl = bl
        # self.pth = pth
        # self.file_gen = fglob(os.path.join(self.pth,"depth_*.tif"))
        self.triangles = None
        self.update()

    def update(self):
        from algutils.io.imread import imread
        import open3d as o3d

        try:
            disp_file, i = next(self.disp_file_gen)
            rgb_file, _ = next(self.rgb_file_gen)
        except Exception as e:
            return False
        i = int(i)
        print(i)
        if isinstance(disp_file, str):
            disp = imread(disp_file)
        else:
            disp = disp_file
        depth = self.flx * self.bl / disp
        # max_depth = 30000000
        # depth[depth > max_depth] = max_depth
        im = imread(rgb_file) / 255
        im = resize(im, (*disp.shape, 3), order=3)
        pta = depth2pt(depth, self.cx, self.cy, self.flx, self.fly)  # TODO: dynamic camera intrinsic
        self.pt.vertices = o3d.Vector3dVector(pta)
        self.pt.vertex_colors = o3d.Vector3dVector(im.reshape((im.shape[0] * im.shape[1], im.shape[2])))
        if self.triangles is None:
            self.triangles = triangulate_grid(*depth.shape)

        tri3d = pta[self.triangles]
        a, b, c = tri3d[:, 0, :], tri3d[:, 1, :], tri3d[:, 2, :]
        ab = b - a
        ac = c - a
        norm = np.cross(ab, ac)
        cos_angle = (a * norm).sum(axis=-1) / (np.linalg.norm(norm, axis=-1) * np.linalg.norm(a, axis=-1))
        self.pt.triangles = o3d.Vector3iVector(self.triangles[abs(cos_angle) > 0.01])
        # self.pt.triangles = o3d.Vector3iVector(self.triangles)

        # self.pt.compute_vertex_normals()
        # vnormal = np.asarray(self.pt.vertex_normals)
        # vnormal = vnormal / 2 + 0.5
        # self.pt.vertex_colors = o3d.Vector3dVector(vnormal)
        self.pt.transform(flip_transform)
        return True

    def skip(self, vis):
        for i in range(50):
            try:
                next(self.disp_file_gen)
                next(self.rgb_file_gen)
            except StopIteration:
                break


def view(disp_file_gen, rgb_file_gen, bl, cx, cy, flx, fly):
    vis = o3d.VisualizerWithKeyCallback()
    updateToggler = UpdateToggler()
    vis.register_key_callback(ord(' '), updateToggler)
    vis.create_window()
    pt = MESHUpdater(disp_file_gen, rgb_file_gen, bl, cx, cy, flx, fly)
    vis.add_geometry(pt.pt)
    keep_update = True
    vis.register_key_callback(ord(' '), updateToggler)

    def close(vis):
        nonlocal keep_update
        keep_update = False

    vis.register_key_callback(ord('X'), close)
    vis.register_key_callback(ord('S'), pt.skip)
    updateToggler.should_update = False
    while keep_update:
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        if updateToggler.should_update:
            if not pt.update():
                keep_update = False
            updateToggler.should_update = False
    vis.destroy_window()


if __name__ == '__main__':
    from glob import glob

    datapath = '/mnt/e/FlyingThings3D'
    left_img = sorted(glob(os.path.join(datapath, 'frames_cleanpass/TEST/*/*/left/*.png')))
    left_disp = sorted(glob(os.path.join(datapath, 'disparity/TEST/*/*/left/*.pfm')))

    flx = 7.183351e+02
    cx = 6.003891e+02
    fly = 7.183351e+02
    cy = 1.815122e+02
    bl = 4.450382e+01 - -3.363147e+02  # 0.5326#

    view(((disp, i) for i, disp in enumerate(left_disp)),
         ((im, i) for i, im in enumerate(left_img)),
         bl, cx, cy, flx, fly)
