from collections import namedtuple
from typing import Collection, Literal

import numpy as np

VT = list | tuple | np.ndarray


def vec_len(v, axis=1):
    """Return length of vectors with coordinates along the given axis"""
    v = np.asarray(v)
    if v.ndim == 1:
        axis = 0
    return np.sum(v ** 2, axis=axis) ** .5


def plot_vector(vec, org=(0, 0, 0), *, labels=None, ax=None, **kws):
    """
    Plot one or more 3D vectors

    :param vec: 1+ of triplets (coordinates)
    :param org: 1+ (of same len as vec) of triplets - origins of the vectors
    :param ax: axis to draw in
    :param kws: Line3DCollection arguments
    """

    def prep(x):
        x = np.asarray(x)
        x = x[None, :] if x.ndim == 1 else x
        assert x.ndim == 2 and x.shape[1] == 3
        return x

    org, vec = map(prep, [org, vec])
    assert org.shape[0] in (1, vec.shape[0])

    if ax is None:
        from matplotlib.pyplot import gca

        ax = gca()

    hs = ax.quiver3D(*org.T, *vec.T, arrow_length_ratio=0.1, **kws)

    if labels:
        assert len(labels) == len(vec)
        for s, xyz in zip(labels, org + vec):
            ax.text(*xyz, s)
    return hs


def plane_fit(points: Collection[VT], normalized=True):
    """
    SVD based pain fit given points and return array plane equation parameters
    `ax + by + cz + d = 0` (in normalized form |a,b,c| = 1 if requested)

    :param points: collections of 3-vectors
    :param normalized: if `True` ensures that the *normal* part is a unit vector
    :return: array of 4 plane equation parameters: [a,b,c,d]
    """
    points = np.column_stack([points, np.ones(points.shape[0])])  # type: ignore
    # SVD best-fitting plane for the test points after subtracting the centroid
    U, S, Vt = np.linalg.svd(points, full_matrices=False)  # type: ignore
    pn = Vt[np.argmin(S), :]
    return pn / (pn[:3] ** 2).sum() ** 0.5 if normalized else pn


AxisIDT = Literal[0, 1, 2, 3, 'x', 'y', 'z']


class AxisID:

    def __init__(self, ax: 'AxisID' | AxisIDT, *, base: Literal[0, 1] = None,
                 axes: Collection[str] | str = None):
        """
        Axis identificator can be initialized either by its name or axis (0 or 1 based) index
         - default `base=0`
         - default `axes` names: `(x, y, z)`

        Also, may be initialized by another `AxisID` instance with optional reset of `base`.

        :param ax: name | idx | axis_id
        :param base: 0 | 1
        :param axes: collection of names to change from `xyz`
        """

        def valid_base(b):
            if b not in (0, 1):
                raise ValueError("Axis enumeration base must be 0 or 1")
            return b

        # ------ copy constructor ------
        if isinstance(ax, AxisID):
            if axes is not None:
                raise SyntaxError("Argument `axes` not allowed when initialized by `AxisID`")
            self.axes = ax.axes
            if base is None:
                self.id = ax.id
                self.base = ax.base
            else:
                self.base = valid_base(base)
                self.id = ax.id - ax.base + self.base
            return

        self.base = 0 if base is None else valid_base(base)  # default 0
        self.axes = tuple(_.lower() for _ in (axes or 'xyz'))  # default 'xyz'
        if len(self.axes) > len(set(self.axes)):
            raise ValueError(f"Repeating axis names: {self.axes}")

        if isinstance(ax, str):
            self.name = ax.lower()
            self.id = self.axes.index(self.name) + self.base
        else:
            self.name = self.axes[ax - self.base]
            self.id = ax

    def __repr__(self):
        return f"Axis('{self.name}'({self.id})"

    def __int__(self):
        return self.id

    def __str__(self):
        return self.name

    @property
    def id0(self):
        """Zero based index of the axis"""
        return self.id - self.base

    @property
    def id1(self):
        """One based index of the axis"""
        return self.id - self.base + 1

    def __eq__(self, other: 'AxisID' | AxisIDT):
        if not isinstance(other, AxisID):
            other = AxisID(other)
        return other.name == self.name

    def __ieq__(self, other: 'AxisID' | AxisIDT):
        return self.__eq__(other)

    @property
    def vector(self):
        """Base vector along the axis"""
        base = np.zeros(len(self.axes))
        base[self.id0] = 1
        return base

    @classmethod
    def basis_vectors(cls, axes='xyz'):
        """Return basis vectors as rows of in 2d array"""
        return np.eye(len(axes))


class Plane:
    """
    Plane can be constructed from different initial types of information:
      - parameters of general equation (ax + by + cz + d = 0)
      - a point and normal at this point
      - 3 points, if not on the same line
      - N > 3 points, from fitting, keep std errors of estimated parameters

    Independently of how it was created, plane internally is represented by its
    *Hessian Normal Form* (https://mathworld.wolfram.com/HessianNormalForm.html).

    Plane provides a convenience access to some standard attributes:
    - normal - normalized vector orthogonal to the plane
    - closest - shortest vector from the coordinates origin to the plane
    - distance - distance from the coordinates origin to plane (len(position))
    """

    def __init__(self, *abcd, stderr=None) -> None:
        """
        Initialize plane with general equation parameters and optionally their std errors
            ax + by + cz + d = 0

        :param abcd: parameters of the plane
        :param stderr: optional std errors for those parameters
        """
        abcd = np.asarray(abcd[0] if len(abcd) == 1 else abcd, dtype=float).flatten()
        if abcd.size != 4:
            raise ValueError("plane must be defined by 4 parameters")

        n = abcd[:3]
        n_len = (n @ n) ** .5
        if np.isclose(n_len, 0):
            raise ValueError(f"Normal length may not be 0!")
        self._n = n / n_len
        self._p = abcd[-1] / n_len

        if stderr is not None:
            if len(stderr) != 4: raise ValueError(f"Invalid {len(stderr)=} != 4!")
            stderr = np.asarray(stderr, dtype=float)
        self._stderr = stderr

    @property
    def abcd(self):
        """Parameters (a,b,c,d) in form of plane equation  ax + by + cz + d == 0"""
        return namedtuple('PlaneParams', ['a', 'b', 'c', 'd'])(*self._n, self._p)

    @property
    def normal(self):
        """Normal of the plane (normalized)"""
        return self._n

    def distance(self, points: Collection[VT]):
        """Return array of signed distances FROM the given points TO the plane
        (negative if point on the opposite side relative to the normal)

        :param points: collection of points vectors, or a single vector
        :return: array of distances, for a single point - a single number
        """
        points = np.asarray(points)
        assert points.ndim == 1 and points.size == 3 or points.ndim == 2 and points.shape[1] == 3
        return points @ self._n + self._p

    @property
    def closest(self):
        """Shortest vector from the coordinates' origin to the plane"""
        return -self._n * self._p

    def closest_to(self, point):
        """Find point on the plane closest to the given `point`"""

        # point on plane is defined as:
        # q' = q + (d - q.n)n
        # where:
        # q' is the point on the plane
        # q is the point we are checking
        # d is the value of normal dot position
        # n is the plane normal
        return point - self.normal * self.distance(point)

    def intersect_axes(self, *, vector=False) -> tuple[float] | tuple[np.ndarray]:
        """Return intersection points with each xyz axes.

        If `vector` - in form of their 3d coordinates, otherwise as distances from the origin:

        :param vector: if True rerun vectors to the intersection points
        :return: 3-tuple of intersection coordinates for every axes
        """
        if self._p:
            cs = tuple(np.nan if np.isclose(c, 0) else -self._p / c for c in self._n)
            if vector:
                return tuple(c * b.vector for c, b in zip(cs, map(AxisID, 'xyz')))  # type: ignore
            return cs
        return tuple([np.zeros(3) if vector else 0.] * 3)  # origin

    @property
    def kxy_params(self):
        """Parameters (kx, ky, b) of the form z = (kx * x + ky * y + b)"""
        a, b, c, d = self.abcd
        return namedtuple('PlaneXYParams', ['kx', 'ky', 'b'])(-a / c, -b / c, -d / c)

    @property
    def xy_coef(self):
        """Linear model coefficients array: `y = <xy_coef> * <x> + xy_intercept`"""
        return -np.array(self.normal[:2]) / self.normal[2]  # - <a,b> / c

    @property
    def xy_intercept(self):
        """Linear model intercept: `y = <coef> * <x> + xy_intercept`"""
        return -self._p / self._n[2]  # -d/c

    @property
    def xy_params(self):
        """Linear model params array `[p0..p3]` (`y = p0 + p1 x1 + p2 x2`)"""
        return -np.array([self._p, *self._n[:2]]) / self._n[2]  # -[d,a,b]/c

    def z(self, xy):
        """
        Calculate `z` for given coordinates (x,y) or collection of coordinates.
        """
        return np.asarray(xy) @ self.xy_coef + self.xy_intercept

    @classmethod
    def from_point_normal(cls, point, normal):
        point, normal = (np.asarray(_, dtype=float) for _ in [point, normal])
        length = (normal ** 2).sum() ** 0.5
        if np.isclose(length, 0):
            raise ValueError("Plane's normal must be non-zero vector")
        normal = normal / length
        distance = -normal.dot(point)
        return cls([normal, distance])

    @classmethod
    def from_points(cls, points: Collection[VT], *, robust=True, **kws):
        """
        Create plane from 3 or more points.

        For >3 points supports robust regression using ransac.

        :param points: N × 3
        :param robust: allow robust regression removing outliers
        :param kws: robust RANSAC arguments
        :return:
        """
        points = np.asarray(points, dtype=float)
        assert points.ndim == 2 and points.shape[1] == 3
        from .. import regress as reg

        if (N := len(points)) < 3:
            raise ValueError(f"at least 3 points are required to define a plane (given {N})")

        if np.linalg.matrix_rank(points) < 3:
            raise ValueError("Plane can't be passed through the given {N} points")

        if N == 3:
            return cls(plane_fit(points))
        else:
            X = np.c_[points, np.ones(N)]
            params, stderr, _, _ = reg.robust_linear_regression(
                X, np.zeros(N), robust=robust, fit_intercept=False, **kws)
            return cls(params, stderr=stderr)

    def __repr__(self):
        if self._stderr:
            _str = lambda v: f"{0:.3g}±{1:.3g}".format(*v)
            params = zip(self.abcd, self._stderr)
        else:
            _str = lambda v: f"{v:.3g}"
            params = self.abcd

        params = list(map(_str, params))
        params = ", ".join(params[:3]) + f'; {params[3]}'
        return f"Plane({params})"

    @property
    def is_axes_plane(self):
        """Check if plane is one of the axes planes: 'xy', 'yz' 'zx'"""
        return np.isclose(self._p, 0) and np.isclose(self._n[abs(np.argmax(self._n))], 1)

    def belong(self, points: VT | Collection[VT], atol=None, **kws):
        """For given collection of points check if each of them belong to the plane within given tolerance
        :param points: point(s) to check
        :param atol: `isclose`(,..., atol=tol)`
        """
        distances = self.distance(points)
        if atol is not None:
            kws[atol] = atol
        return np.isclose(distances, 0, **kws)  # type: ignore

    def project(self, vec):
        """
        Project vector onto the plane, that it subtract its component
        perpendicular to the plane.
        :param vec:
        :return: vector parallel to the plane
        """
        vec = np.atleast_2d(vec)
        return (vec - (self._n[:, None] @ self._n.dot(vec.T)[None, :]).T).squeeze()

    def intersect_line(self, point, direction: AxisIDT | AxisID | np.ndarray):
        """
        Find where line(s) defined by a point(s) (origin) and direction vector
        intersects the plane.

        Return vector of `np.nan` if they don't intersect

        Basic case requires both `point` and `direction` to be 3d vectors.
        Supported also shortcuts, when
          - `point=0` stands for [0,0,0],
          - `direction` can be axis vector from 'xyz` or 0,1,2 or `AxisID`

        Can be used to find intersections of multiple lines.

        Then in the basic form `point` and `direction` are sequences of N 3d vectors.
        If either of them is a single 3d vector, other is broadcasted accordingly.

        :param point: a vector or 0 for (0,0,0)
        :param direction: a vector or axis (name or zero-based index)
        """
        if isinstance(direction, (str, int)):
            direction = AxisID(direction)
        if isinstance(direction, AxisID):
            direction = direction.vector

        if np.isscalar(point) and point == 0:
            point = np.zeros(3)

        # print(f"{point.shape=}, {direction.shape=}")
        point, direction = np.broadcast_arrays(np.atleast_2d(point), direction)
        assert point.ndim == 2, point.shape[1] == 3

        # --- main part
        p, n = self.closest, self.normal
        pd = p.dot(n)

        rd_n = n.dot(direction.T)
        # if np.isclose(rd_n, 0.0):
        #     return np.full((3,), np.nan)
        #
        p0_n = n.dot(point.T)
        t = np.atleast_2d((pd - p0_n) / rd_n).T

        # print(f"{(point + (direction * t)).shape=}")
        return (point + (direction * t)).squeeze()

    def _plotting_patch(self, xlim=None, ylim=None, zlim=None):
        """Calculate 3d-coordinates of rectangular patch representing plane in the plot.
        Return the patch and list of points of intersection with axis (could be 1,2,3)
        """
        def ort(_ax):
            return np.roll(np.array([0, 1, 1], bool), _ax)  # mask for other axes

        def to_base_vec(v, _ax):
            return np.roll(np.array([v, 0, 0], dtype=float), _ax)

        def rect_around_segment(s1, s2, offset1, offset2=None):
            """Build 3D rectangle from a segment (two 3D points) by offsetting it
            into 2 directions: along and inversed to the offsets"""
            if offset2 is None:
                offset2 = -offset1
            return np.array([s1 + offset1, s2 + offset1, s2 + offset2, s1 + offset2])

        basis = AxisID.basis_vectors()

        # special case - not generic version!
        if xlim and ylim and self.normal @ basis[2] > 0.01:
            inters = [to_base_vec(_, d) for d, lims in enumerate([xlim, ylim]) for _ in lims]
            inters = rect_around_segment(*inters)
            patch = inters = self.intersect_line(inters, basis[2])
        else:
            # First find intersections points with every axis
            inters = np.asarray(self.intersect_axes(vector=True))  # axes x points
            inters_mask = np.all(~np.isnan(inters), axis=1)  # binary mask of axis with intersections
            inters = inters[inters_mask]

            sel_axis = np.argmax(abs(self.normal))
            inters_num = len(inters)
            if inters_num == 1 or inters_num == 3 and self.is_axes_plane:
                b1, b2 = basis[ort(sel_axis)]  # 2 basis vectors orthogonal to max_axis
                patch = self.intersect_line(rect_around_segment(-b1, b1, b2), basis[sel_axis])
            elif inters_num == 2:
                patch = rect_around_segment(*inters, *basis[~inters_mask])
            elif inters_num == 3:
                basis2 = (basis if self.belong([0, 0, 0]) else inters)[ort(sel_axis)]
                (p1, l1), (p2, l2) = sorted(zip(basis2, map(vec_len, basis2)), key=lambda _: _[1])
                # complete rectangle from two points p1, p2 laying on the axes
                p4 = -p2 * (l1 / l2)
                p3 = p2 + p4 - p1
                patch = self.intersect_line((p1, p2, p3, p4), basis[sel_axis])
            else:
                raise RuntimeError

        return patch, inters

    def plot(self, *, ax=None, color="lightblue", alpha=0.3,
             aspect='equal', origin=True, normal=True, **opts):
        """
        Create figure and plot 3D the plane and optionally additional points

        Axis set options, like `xlim` can be provided as kw arguments.

        :param ax: if provided, draw on this axis
        :param color: color of the plane
        :param alpha: transparency (0,1)
        :param aspect: argument for `ax.set_aspect()`
        :param opts: options to set: ax.set(**opts)
        """
        from matplotlib import colors, pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        lim_ops_names = {'xlim', 'ylim', 'zlim'}
        lim_ops = {k: opts.get(k, None) for k in lim_ops_names}

        if ax is None:
            ax = plt.figure().add_subplot(projection="3d")
            ax.set_title(str(self))
            ax.set(xlabel="x", ylabel="y")

        patch, inters = self._plotting_patch(**lim_ops)
        ax.add_collection3d(Poly3DCollection(
            [patch], alpha=alpha, shade=True, facecolors=[color],
            edgecolors=0.8 * colors.to_rgba_array(color, alpha=alpha),
        ))  # type: ignore
        ax.plot(*inters.T, '.:', color=color)

        if normal:
            plot_vector(self.normal, self.closest, color='m', lw=1)
        if origin:
            ax.scatter(0, 0, 0, color='red')

        # if aspect == 'equal':  # make sure the minimal axis limit is not too small

        if not lim_ops_names.intersection(opts):
            limits = np.asarray(ax.axis()).reshape(3, 2)
            spans = np.diff(limits).squeeze()
            if spans.min() / spans.max() < 0.2:
                i_min = np.argmin(spans)
                limits[i_min] *= (spans.max() / spans.min()) / 4
                ax.axis(limits.flatten())
                # ax.set_adjustable('box')
        ax.set(aspect=aspect, **opts)
        return ax


def random_points_between_vectors(v1, v2, num: int, sigma: float = 0):
    """
    Generate `num` random points in the parallelogram region of the plane
    (A, B, C, D) where `D` = `B` + (`A` - `B`) + (`C` - `B`).

    Points in ABC not on the plane, are replaced by the closest on the plane.

    :param ABC: list of 3 points in the plane.
    :param num: number of points to generate
    :param sigma: allow normal deviation from the plane with given `sigma`
    """
    v1, v2 = map(np.asarray, (v1, v2))
    points = np.random.rand(num, 2) @ [v1, v2]
    if not sigma:
        return points
    n = np.cross(v1, v2)  # normal direction
    n = (n / vec_len(n))[None, :]
    return points + n * sigma * np.random.randn(num, 1)


if __name__ == "__main__":
    from matplotlib import pyplot as plt, use

    use("QTAgg")

    ex, ey, ez = (AxisID(_).vector for _ in 'xyz')

    pn = Plane([0.1, 0.2, .8, -10])

    v0 = pn.intersect_line(-(ex+ey)/2, ez)
    v1, v2 = pn.project([ex, ey])

    points = v0 + random_points_between_vectors(v1, v2, 125, .1)
    # points = pn.intersect_line(points, ez)

    ax = pn.plot(xlim=(-2,2), ylim=(-2,2))
    ax.scatter(*points.T)

    pnf = Plane.from_points(points)
    pnf.plot(color='yellow', ax=ax)

    # ax.scatter(*np.asarray(v0, v1, ))

    ax.figure.canvas.manager.window.move(100, 100)
    ax.figure.set_size_inches(8, 8)

    # Plane([1, 3, 4, 0]).plot()
    # Plane([0, 0, 1, 0]).plot()
    # Plane([1, 3, 4, 6]).plot()
    # Plane([0, 1, 0, 0]).plot()

    π = np.pi
    rnd = np.random

    # p = Plane([1, 1, 1], [1, 1, 1])
    # print(f"{p = }\n{p.closest = }\n{p.normal = }\n{p.distance = }")
    # p.belong(p.closest)
    # ax = p.plot()
    # ax.scatter(*p.closest[:, None])
    # # %%
    # print(p)
    # p.plot(points=True).scatter(*p.points.T, c="g")
    # # %%
    # p.belong([0, -1, -1])
    plt.show()
    # %%
    # plt.close('all')

    # %%
    # A, B, C = i, j, k
    # normal = ((B - A) ^ (C - A)).normalized
    #
    # # pyrr.plane.create_from_position(B, normal)
    # pyrr.plane.create(normal, np.sum(normal * B))
    # # %%
    # pyrr.plane.normal(pyrr.plane.create_from_points(i, j, k))
    # # %%
    # pp = PlanePoints(1 * i, 2 * j, 3 * k)
    # pp.gen_region_points(60, 0.1)
    # _ax = pp.plot()
    # # %%
    # # SVD best-fitting plane for the test points after subtracting the centroid
    # points = pp.points
    # I = np.ones(len(points))
    # U, S, Vt = np.linalg.svd(np.column_stack([points, I]))
    # nv = Vt[np.argmin(S), :]
    #
    # pn = pyrr.plane.create(nv[:-1], nv[-1])
    # pn
    # # %%
    # pp.base_points
    # # %%
    # plane.create_from_points(i, j, k)
    # # %%
    # plane.position(pn)
    # # %%
    # pn0 = plane.create_from_points(*pp.base_points)
    # plane.position(pn0)
    # # %%
    # pp.base_points[0]
    # # %%
    # ABC_ = [
    #     pyrr.geometric_tests.point_closest_point_on_plane(p, pn) for p in pp.base_points
    # ]
    # ABC_
    #
    # plot_vector(ABC_, color="g", ax=_ax)
    # # %%
    # import pandas as pd
    # from sklearn.model_selection import train_test_split
    #
    # data_url = "http://lib.stat.cmu.edu/datasets/boston"
    # raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    # data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    # target = raw_df.values[1::2, 2]
    # A, b = data, target
    # # %%
    # # append a columns of 1s (these are the biases)
    # A = np.column_stack([np.ones(A.shape[0]), A])
    #
    # # split the data into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(
    #     A, b, test_size=0.50, random_state=42
    # )
    #
    # # calculate the economy SVD for the data matrix A
    # U, S, Vt = np.linalg.svd(X_train, full_matrices=False)
    #
    # # solve Ax = b for the best possible approximate solution in terms of least squares
    # x_hat = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ y_train
    #
    # # perform train and test inference
    # train_predictions = X_train @ x_hat
    # test_predictions = X_test @ x_hat
    #
    # # compute train and test MSE
    # train_mse = np.mean((train_predictions - y_train) ** 2)
    # test_mse = np.mean((test_predictions - y_test) ** 2)
    #
    # print("Train Mean Squared Error:", train_mse)
    # print("Test Mean Squared Error:", test_mse)
    # # %%
    # (Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ y_train).shape
    # # %%
    # Vt.shape, S.shape, U.shape, y_train.shape, x_hat.shape
    # # %%
    # (U[:, -1] @ A.normalized)
    # # %%
    # (1 / 3) ** 0.5
    # # %%
    # 3 * U[:, -1] ** 2
    # # %%
    # U[:, -1] @ ((C - A) ^ (B - A))
    # # %%
    # # plot raw data
    # plt.figure()
    # ax = plt.subplot(111, projection="3d")
    # xs, ys, zs = points
    # ax.scatter(xs, ys, zs, color="b")
