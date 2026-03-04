import numpy as np

from ..nptools import *


def test_xy2idx():
    a = np.arange(100).reshape((10, 10))
    xy = [[1, 2, 3], [4, 5, 6]]
    correct = np.array([j + i * 10 for j, i in zip(*xy)])

    assert (a[xy2i(*xy)] == correct).all()
    assert (a[xy2i([*zip(*xy)])] == correct).all()

    class XY:
        x = xy[0]
        y = xy[1]

    assert (a[xy2i(XY)] == correct).all()
    assert (a[xy2i(XY.__dict__)] == correct).all()


def test_array():
    n = 20
    a = Array(np.random.rand(n, n))
    assert isinstance(a, Array)
    assert isinstance(a, np.ndarray)
    assert isinstance(a[:1, :1], Array)
    assert isinstance(a.view(int), Array)


def test_array_repr():
    n = 20
    assert str(Array(np.zeros((n, n)))) == f"[{n}×{n}]f8 ⌊0|0⌉"
    a = Array(np.random.rand(n, n))

    assert str(a).count(str(n)) == 2 and str(a).count('\n') == 0

    a.set_printoptions(rows=n + 1, cols=n + 1)
    assert str(a).count('\n') >= n

    Array.set_printoptions(info=False)
    assert str(a[:1, :3]).count('\n') == 0

    Array.set_printoptions(info=True)
    assert str(a[:1, :3]).count('\n') == 1

    a[0, :2] = np.inf
    a[1, :5] = np.nan
    a.set_printoptions(cols=1)
    assert str(a).endswith("(5∅, 2∞)")


def test_sampler():
    pytest.importorskip("algutils.math.hist")
    from algutils.math.hist import Sampler

    p = (mn, mx, step, bins) = (0, 10, 2, 5)
    s = Sampler(mn, mx, step=step, below=True)
    assert (s.low, s.high, s.step, s.bins) == p
    assert np.array_equal(s.bins_edges()[:-2], np.arange(mn, mx + step, step))
    assert np.array_equal(s.bins_centers, np.arange(mn + step / 2, mx, step))

    p = (mn, mx, step, bins) = (0, 10, 2, 5)
    s = Sampler(mn, mx, bins=bins, below=False)
    assert (s.low, s.high, s.step, s.bins) == p
    assert np.array_equal(s.bins_edges()[:-2], np.arange(mn + step, mx + step, step))
    assert np.array_equal(s.bins_centers, np.arange(mn + step / 2, mx, step))

    p = (mn, mx, step, bins) = (0, 10 + 1, 2, 6)
    s = Sampler(mn, mx, step=step, below=False)
    assert (s.low, s.high - 1, s.step, s.bins) == p  # s.max enlarged to fit
    assert np.array_equal(s.bins_edges()[:-2], np.arange(mn + 2, mx + step, step))
    assert np.array_equal(s.bins_centers, np.arange(mn + step / 2, mx + 1, step))
