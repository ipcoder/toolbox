import numpy as np

def test_hist2d():
    from algutils.math.hist import Hist2D
    from algutils.math.hist import Sampler

    h = Hist2D((0, 10, 1.), (0, 1, 0.2))
    assert h.samplers[0] == Sampler(0, 10, step=1.)
    assert h.samplers[1] == Sampler(0, 1, step=.2)
    assert np.array_equal(h.bins[0], h.samplers[0].bins_edges(), equal_nan=True)

    h.add([.5, 2], [.02, 0.17])
    h.plot()
    pass


def test_sampler():
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
