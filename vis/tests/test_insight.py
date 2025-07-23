import pytest
from numpy.random import rand

_interact = False


def test_imhist():
    """ imhist(im, *args, out=False, **kw)
    """
    from toolbox.vis.insight import imhist, plt
    im = rand(10, 10)
    imhist(im, im)
    imhist(im, bins=10)
    plt.close('all')


@pytest.mark.slow
def test_imgrid_out():
    """ out:  [False]|True|'axs'|'fig'|'ims'|'all' - what should function return:
                    - 'axs' | True  - list of axes with images
                    - 'fig' - container figure
                    - 'ims' - image objects
                    - 'all' - all above in this order
    """
    from toolbox.vis.insight import imgrid, plt
    inputs = rand(10, 10), rand(10, 10)
    axs = imgrid(*inputs, out=False, window_title="out=False")
    assert axs == None

    axs = imgrid(*inputs, out=True, window_title="out=True")
    assert len(axs) == 2

    axs = imgrid(*inputs, out='axs', window_title="out=axs")
    assert len(axs) == 2

    fig = imgrid(*inputs, out='fig', window_title="out=fig")
    assert isinstance(fig, plt.Figure)

    ims = imgrid(*inputs, out='ims', window_title="out=ims")
    assert len(ims) == 2

    all = imgrid(*inputs, out='all', window_title="out=all")
    assert set(all._asdict()) == {'axs', 'img_axs', 'plt_axs', 'data', 'fig', 'ims'}
    if _interact: plt.show(block=True)

    plt.close('all')


@pytest.mark.slow
def test_imgrid_clim():
    """ adj_clim: [False]|True - Should add clim sliders or not
    """
    from toolbox.vis.insight import imgrid, plt
    im1 = rand(10, 10)
    im2 = rand(10, 10)

    imgrid(rand(10, 10), rand(10, 10), (im1 - im2), adj_clim=True, window_title="adj_clim=True")
    if _interact: plt.show()

    # with warnings.catch_warnings():
    #     warnings.simplefilter('ignore')

    imgrid(im1, im2, clim=[(0, 7), 0, 'auto', 2], adj_clim=False, window_title="adj_clim=False")
    if _interact: plt.show()

    imgrid(im1, im2, (im1 - im2), im1, clim=[(0, 7), 0], window_title="clim=[(0, 7), 0]")
    if _interact: plt.show()

    imgrid(im1, im2, (im1 - im2), im1, clim=[(0, 7), 0, 'auto', 3], window_title="clim=[(0, 7), 0, 'auto', 3]")
    if _interact: plt.show()

    imgrid(im1, im2, (im1 - im2), im1, clim=[(0, 7), 0, 'auto', 2], window_title="clim=[(0, 7), 0, 'auto', 2]")
    if _interact: plt.show()

    imgrid(im1, im2, (im1 - im2), clim=[(0, 60), (0, 60), (-150, 150)], window_title="clim=[(0, 60), (0, 60), (-150, 150)]")
    if _interact: plt.show()

    imgrid(im1, im2, (im1 - im2), clim=['auto', (0, 60), (-150, 150)], window_title="clim=['auto', (0, 60), (-150, 150)]")
    if _interact: plt.show()

    imgrid(im1, im2, (im1 - im2), clim=['auto', 'auto', 'auto'], window_title="clim=['auto', 'auto', 'auto']")
    if _interact: plt.show()

    imgrid(im1, im2, (im1 - im2), clim=[1, 2, (-150, 150)], window_title="clim=[1, 2, (-150, 150)]")
    if _interact: plt.show()
    pass
    plt.close('all')


@pytest.mark.slow
def test_imgrid_window_title():
    """ window_title: window title
    """
    image = rand(10, 10)

    from toolbox.vis.insight import imgrid, plt
    imgrid(image, window_title="WINDOW TITLE")
    if _interact: plt.show()
    pass
    plt.close('all')


@pytest.mark.slow
def test_imgrid_hist():
    """ hist: Show histogram. [None]|True|False|bins.
                 bins - argument to hist(..., bins=bins)
                 None - (default) automatic decision based on number of unique values (<8 is False)
                 True - like None, for backward compatibility
                 False - don't show histogram
    """

    from toolbox.vis.insight import imgrid, plt
    image = rand(10, 10)

    imgrid(image, hist=False, titles=["hist=False: don't show histogram"])
    if _interact: plt.show()

    imgrid(image, hist=None, titles=["hist=None - automatic decision based on number of unique values (<8 is False)"])
    if _interact: plt.show()

    imgrid(image, hist=True, titles=["hist=True - like None, for backward compatibility. \
                                     This figure includes Axes that are not compatible with tight_layout"])
    if _interact: plt.show()

    imgrid(image, hist=20, titles=["hist=range(20) \
                                    This figure includes Axes that are not compatible with tight_layout"])
    if _interact: plt.show()

    plt.close('all')


@pytest.mark.slow
def test_imgrid_titles():
    im1 = rand(10, 10)
    im2 = rand(10, 10)
    im3 = rand(10, 10)
    im4 = rand(10, 10)
    two_images = [im1, im2]

    from toolbox.vis.insight import imgrid, plt
    imgrid(*two_images, im3, im4, titles='1 2'.split(), hist=False, out=True)
    if _interact: plt.show()

    imgrid(im3, im4, *two_images, titles='1 2'.split(), hist=False, out=True)
    if _interact: plt.show()
    plt.close('all')


@pytest.mark.slow
def test_imgrid_auto_clim():
    im1 = rand(10, 10)
    im2 = rand(10, 10)
    from toolbox.vis.insight import imgrid, plt

    axs = imgrid(im1, titles=['1'], hist=False, out=True, clim='auto')
    assert len(axs) == 1
    if _interact: plt.show()

    axs = imgrid(rand(10, 10), rand(10, 10), titles=['1'], hist=False, out=True, clim='auto')
    assert len(axs) == 2
    if _interact: plt.show()
    plt.close('all')


@pytest.mark.slow
def test_imgrid_simple():

    from toolbox.vis.insight import imgrid, plt
    imgrid(rand(10, 10))
    if _interact: plt.show()

    axs = imgrid(rand(10, 10), titles=['1'], hist=False, out=True)
    assert len(axs) == 1
    if _interact: plt.show()

    axs = imgrid(rand(10, 10), rand(10, 10), titles=['1'], hist=False, out=True)
    assert len(axs) == 2
    if _interact: plt.show()
    plt.close('all')


@pytest.mark.slow
def test_imgrid_dict():
    images = dict(one=rand(10, 10), two=rand(10, 10))

    from toolbox.vis.insight import imgrid, plt
    imgrid(images)
    if _interact: plt.show()

    imgrid(rand(10, 10), images, (rand(10, 10), 'four'), titles=['one', ])
    if _interact: plt.show()
    plt.close('all')


@pytest.mark.slow
def test_imgrid_grid():
    images = rand(6, 10, 10)

    from toolbox.vis.insight import imgrid, plt
    imgrid(*images, grid=(1, 4))
    if _interact: plt.show()

    imgrid(*images, grid='vert')
    if _interact: plt.show()

    imgrid(*images, grid='h')
    if _interact: plt.show()

    imgrid(*images, grid='auto')
    if _interact: plt.show()
    plt.close('all')


def test_mosaic_parser():
    codes = [
        'ABCa;ADFd',
        'xs.s;AAAA;xdfd;B.cK;.DfF+dfgh',
        '!A..d+dfgh',
        '''
        .B.
        cBd
        .B.
        '''
    ]
    from toolbox.vis.insight import MosaicParser
    show = False

    for c in codes:
        res = (p := MosaicParser(c)).accommodate(9)
        if '!' in c:
            assert p.transpose and p.extended
        if '+' in c:
            assert p.extended
        assert isinstance(res, list)

        if show:
            print('=' * 30, c, '-' * 30, sep='\n')
            print(p)
            print(*(res.split(';') if isinstance(res, str) else res), sep='\n')

    c = 'AB.CD;abcdf;EF..g+'
    p = MosaicParser(c)
    assert not p.transpose
    assert len(p.fixed) == 2
    assert p.fix_ims == 4
    assert p.rep_ims == 2
    assert p.columns == 5
    assert len(p.accommodate(6)) == 3
    assert len(p.accommodate(10)) == 5

    c = '!' + c
    p = MosaicParser(c)
    assert p.transpose
    assert len(p.fixed) == 2
    assert p.fix_ims == 4
    assert p.rep_ims == 2
    assert p.columns == 5
    assert len(p.accommodate(6)) == 5
    assert len(p.accommodate(8)[0]) == 4

    with pytest.raises(ValueError):
        MosaicParser('AB.C;DEcdf')


if __name__ == '__main__':
    from toolbox.vis import insight

    insight.max_figsize = (12, 5)
    insight.max_figsize = (7, 7)
    insight.imgrid(*rand(4, 10, 10), cmap=['jet', 'wide', 'gray', 1])

    insight.plt.show()
    insight.plt.close('all')
