import numpy as np
from collections import namedtuple


def linear_regression(X, y, *, show=False) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate values and std errors of n+1 parameters (p0, ..., pn)
    of the model with n variables (x1..xn):

        y = p0 + p1*x1 +... p*xn

    *Notice*! intercept (the constant parameter) is returned on the `0`-th

    :param X: array [L×n] (L samples of n variables)
    :param y: array [L]  of the corresponding values
    :param show: if `True` print the results
    :return: tuple of 2 arrays: (params, std-errs)
    """
    X = np.c_[np.ones((X.shape[0], 1)), X]
    XtX = np.linalg.inv(X.T @ X)

    pₑ = XtX @ X.T @ y  # estimated parameters

    res = X @ pₑ - y  # residuals
    σₑ = (res.T @ res) / (len(X) - len(pₑ))  # estimated noise (sigma)
    σₚ = σₑ * XtX.diagonal() ** .5  # estimated param variances

    if show:
        with np.printoptions(precision=5, ):
            print(f'Estimating {X.shape[1]} parameters from {len(y)} data points')
            print(f"  z-noise: {σₑ = :.4f}")
            print(f"  parameters:\n\t{pₑ = }\n\t{σₚ = }")
    return pₑ, σₚ


RobustRes = namedtuple('RobustRes', ['params', 'stderr', 'outliers_num', 'inliers_mask'])


def robust_linear_regression(X, y, *, robust=True, fit_intercept=True,
                             show=False, **ransac_args) -> RobustRes:
    """
    Robustly estimate parameters (end their estimation std errors)
    for a linear model `y = p0 + p1*x1 +... p*xn`

    and values and std of `n+1` parameters `(p0, ..., pn)`
    of the model with `n` variables `(x1..xn)`:

    :param X: array `[L × n]` (L samples of n variables)
    :param y: array `[L]`  of the corresponding values
    :param robust: if False - skip the RANSAC robust part
    :param show: if `True` print the results
    :return: (array `[n+1 × 2]` of estimated parameters and their std errors,
              number of outliers, mask of inliers)

    Robustness achieved using RANSAC - an iterative algorithm selecting
    a subset of inliers from the complete data set.

    RANSAC Parameters
    ----------

    residual_threshold : float, default=None
        Maximum residual for a data sample to be classified as an inlier.
        By default, the threshold is chosen as the MAD (median absolute
        deviation) of the target values `y`. Points whose residuals are
        strictly equal to the threshold are considered as inliers.

    is_data_valid : callable, default=None
        This function is called with the randomly selected data before the
        model is fitted to it: `is_data_valid(X, y)`. If its return value is
        False the current randomly chosen sub-sample is skipped.

    is_model_valid : callable, default=None
        This function is called with the estimated model and the randomly
        selected data: `is_model_valid(model, X, y)`. If its return value is
        False the current randomly chosen sub-sample is skipped.
        Rejecting samples with this function is computationally costlier than
        with `is_data_valid`. `is_model_valid` should therefore only be used if
        the estimated model is needed for making the rejection decision.

    max_trials : int, default=100
        Maximum number of iterations for random sample selection.

    max_skips : int, default=np.inf
        Maximum number of iterations that can be skipped due to finding zero
        inliers or invalid data defined by ``is_data_valid`` or invalid models
        defined by ``is_model_valid``.

    stop_n_inliers : int, default=np.inf
        Stop iteration if at least this number of inliers are found.

    stop_score : float, default=np.inf
        Stop iteration if score is greater equal than this threshold.

    stop_probability : float in range [0, 1], default=0.99
        RANSAC iteration stops if at least one outlier-free set of the training
        data is sampled in RANSAC. This requires to generate at least N
        samples (iterations)::

            N >= log(1 - probability) / log(1 - e**m)

        where the probability (confidence) is typically set to high value such
        as 0.99 (the default) and e is the current fraction of inliers w.r.t.
        the total number of samples.

    loss : str, callable, default='absolute_error'
        String inputs, 'absolute_error' and 'squared_error' are supported which
        find the absolute error and squared error per sample respectively.

        If ``loss`` is a callable, then it should be a function that takes
        two arrays as inputs, the true and predicted value and returns a 1-D
        array with the i-th value of the array corresponding to the loss
        on ``X[i]``.

        If the loss on a sample is greater than the ``residual_threshold``,
        then this sample is classified as an outlier.

        .. versionadded:: 0.18

    random_state : int, RandomState instance, default=None
        The generator used to initialize the centers.
        Pass an int for reproducible output across multiple function calls.
    """

    if robust:
        from sklearn.linear_model import RANSACRegressor, LinearRegression
        ransac = RANSACRegressor(
            estimator=LinearRegression(fit_intercept=fit_intercept),
            **ransac_args)
        ransac.fit(X, y)

        in_mask: np.ndarray = ransac.inlier_mask_
        if num_ouliers := in_mask.size - in_mask.sum():
            X, y = X[in_mask, :], y[in_mask]

    return RobustRes(*linear_regression(X, y, show=show), num_ouliers, in_mask)


if __name__ == '__main__':
    import numpy as np
    from collections import namedtuple

    PlaneParams = namedtuple('PlaneParams', ['d', 'kx', 'ky'])

    p = PlaneParams(d=4, kx=.2, ky=-0.5)
    σ = 1
    outliers = 0.2  # potion of data
    n = 30
    np.random.seed(42)


    def random_points_in_rect(num, xlim, ylim):
        # x-y locations
        r = np.array([xlim, ylim]).reshape(2, 2)
        xy = np.random.rand(num, 2)
        return xy * (r[:, 1] - r[:, 0]) + r[:, 0]


    xy = random_points_in_rect(n, (0, 10), (20, 40))
    zₘ = np.c_[np.ones(n), xy] @ p
    z = zₘ + σ * np.random.randn(n)

    # zₒ
    outliers_mask = np.random.rand(n) < outliers
    zₒ = z.copy()
    zₒ[:2] += 6

    from toolbox.math.geom.plane import Plane
    from matplotlib import pyplot as plt
    from matplotlib import use

    use('QTAgg')

    pn = Plane.from_points(np.c_[xy, z])
    ax = pn.plot()
    ax.scatter(*pn.points.T)

    pn = Plane.from_points(np.c_[xy, zₒ])
    ax = pn.plot()
    ax.scatter(*pn.points.T, c='r')

    print(f'Gausian noise {σ=}\n{"-" * 30}')
    linear_regression(xy, z, show=True)
    robust_linear_regression(xy, z, show=True);

    print(f'Gausian noise {σ=} + {outliers_mask.sum()} outliers (+10 σ)\n{"-" * 30}')
    linear_regression(xy, zₒ, show=True)
    robust_linear_regression(xy, zₒ, show=True);
    plt.show()

import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class SVDPlaneEstimator(BaseEstimator, RegressorMixin):
    """
    Custom estimator to fit a plane (ax + by + cz + d = 0) using Singular Value Decomposition (SVD),
    adapted for use with sklearn's `RANSACRegressor`.

    Fitted parameters can be accessed as attributes `a_`, `b_`, `c_`, `d_`
    or as array `coef_` in the order of axis enumeration (1,2,3) with intercept at 0:
        p0 +  ∑ p_i x_i = 0
    """

    def fit(self, X, y, norm=True):
        """
        Fit a plane to the given 3D points using SVD.
        Normalize coefficients |a,b,c| = 1

        Args:
            X (numpy.ndarray): Nx2 array of (x, y) coordinates.
            y (numpy.ndarray): N array of z values.
            norm: if True - normalize to |a,b,c| = 1

        Returns:
            self: Fitted estimator.
        """
        # Construct the augmented matrix [1 X Y Z ]
        A = np.c_[np.ones(len(y)), X, y]
        _, _, Vt = np.linalg.svd(A)

        # Extract plane parameters (smallest singular value)
        c = Vt[-1, :]
        self.coef_ = c / (c[1:] @ c[1:])**.5 if norm else c
        self.d_, self.a_, self.b_, self.c_ = self.coef_

        return self

    def __repr__(self):
        if not hasattr(self, 'coef_'):
            return super().__repr__()

        a, b, c, d = self.coef_[[-1, 0, 1, 2]]
        return f"{type(self).__name__}({a=:.3g}, {b=:.3g}, {c=:.3g}, {d=:.3g})"

    def predict(self, X):
        """
        Predict the z-values given x and y using the estimated plane equation.

        Args:
            X (numpy.ndarray): Nx2 array of (x, y) values.

        Returns:
            z_pred (numpy.ndarray): Predicted z values.
        """
        # Solve for z using the plane equation: ax + by + cz + d = 0
        # Rearrange: z = (-ax - by - d) / c
        if self.c_ == 0:  # Prevent division by zero (degenerated case)
            raise ValueError(
                "Fitted plane has c = 0, which is not valid for regression.")

        z_pred = (-self.a_ * X[:, 0] - self.b_ * X[:, 1] - self.d_) / self.c_
        return z_pred


# class SVDPlaneEstimator(BaseEstimator, RegressorMixin):
#     """
#     Custom estimator to fit a plane (ax + by + cz + d = 0) using Singular Value Decomposition (SVD),
#     adapted for use with sklearn's RANSACRegressor.
#     """
#
#     def fit(self, X, y):
#         """
#         Fit a plane to the given 3D points using SVD.
#
#         Args:
#             X (numpy.ndarray): Nx2 array of (x, y) coordinates.
#             y (numpy.ndarray): N array of z values.
#
#         Returns:
#             self: Fitted estimator.
#         """
#         # Convert to full 3D points
#         points = np.c_[X, y]  # Combine (x, y) with target z
#
#         # Construct the augmented matrix [X Y Z 1]
#         A = np.c_[points, np.ones(points.shape[0])]  # Add column of ones for d
#         # Compute SVD
#         _, _, Vt = np.linalg.svd(A)
#
#         # Extract plane parameters (smallest singular value)
#         self.a_, self.b_, self.c_, self.d_ = Vt[-1, :]
#
#         return self
#
#     def predict(self, X):
#         """
#         Predict the z-values given x and y using the estimated plane equation.
#
#         Args:
#             X (numpy.ndarray): Nx2 array of (x, y) values.
#
#         Returns:
#             z_pred (numpy.ndarray): Predicted z values.
#         """
#         # Solve for z using the plane equation: ax + by + cz + d = 0
#         # Rearrange: z = (-ax - by - d) / c
#         if self.c_ == 0:  # Prevent division by zero (degenerated case)
#             raise ValueError("Fitted plane has c = 0, which is not valid for regression.")
#
#         z_pred = (-self.a_ * X[:, 0] - self.b_ * X[:, 1] - self.d_) / self.c_
#         return z_pred


# Generate synthetic dataset
np.random.seed(42)
num_points = 200

# True plane equation: x - 2y + z - 5 = 0 → z = 5 - x + 2y
X = np.random.rand(num_points, 2) * 10  # (x, y) values
z_true = 5 - X[:, 0] + 2 * X[:, 1]  # Compute true z values
noise = np.random.randn(num_points) * 0.2  # Add small noise
y = z_true + noise

# Add outliers
num_outliers = 40
outliers = np.random.rand(num_outliers, 2) * 10
outlier_z = np.random.rand(num_outliers) * 20  # Outliers far from the plane

# Combine inliers and outliers
X_all = np.vstack((X, outliers))
y_all = np.hstack((y, outlier_z))

# Fit RANSAC using the custom SVD estimator
ransac = RANSACRegressor(SVDPlaneEstimator(),
                         residual_threshold=0.5,
                         min_samples=5,
                         max_trials=100)
ransac.fit(X_all, y_all)

# Get the best estimated plane parameters
a, b, c, d = ransac.estimator_.a_, ransac.estimator_.b_, ransac.estimator_.c_, ransac.estimator_.d_

print(f"Estimated Plane Equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
print(f"Number of inliers found: {ransac.inlier_mask_.sum()} / {len(X_all)}")

