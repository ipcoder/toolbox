import ipyvolume as ipv
from ipyvolume import *


def show_surface(surf, *, cmap=None, clim=None, zlim=None, angles_xyz=(3, 0, 0), width=500, height=500, show=True):
    """
    Produce 3D graph of the surface described by 2D image or (X,Y,Z) tuple of coordinates
    :param surf: 2D image or (X,Y,Z) tuple of coordinates
    :param cmap: color map to use or a single color name (default is 'lightgreen')
    :param clim: color limits for the data - if provided forces colormap coloring - not a single color!
    :param width: figure's width (pixels)
    :param height: figure's height (pixels)
    :return:
    """

    default_color = 'lightgreen'
    X, Y, Z = surf if type(surf) is tuple else depth2XYZ(surf)

    def is_valid_range(clim):
        return type(clim) is tuple and len(clim) == 2 and np.diff(clim) > 0

    def valid_cmap(cmap):
        try:
            return plt.cm.get_cmap(cmap)
        except Exception:
            return None

    if cmap is None and clim is None:
        colors = default_color
    elif valid_cmap(cmap):
        if not clim:
            clim = hist_range(Z)
        if not is_valid_range(clim):
            raise ValueError('Invalid clim argument')
        colors = valid_cmap(cmap)((Z - clim[0]) / np.diff(clim))[..., 0:3]
    elif clim:
        raise ValueError('Invalid cmap argument')
    else:
        colors = cmap

    fig = ipv.figure(width=width, height=height)
    srf = ipv.plot_surface(X, Y, Z, color=colors)

    if zlim:
        ipv.zlim(*zlim)

    if angles_xyz:
        fig.anglex, fig.angley, fig.anglez = angles_xyz

    if show:
        ipv.show()
    return fig, srf


def fig_remove(fig_key: ipv.Figure):
    """Remove figure object from the ipv contexts.
    :param fig_key: figure or figure key
    :return:
    """
    def remove(key):
        fig = ipv.current.figures.pop(key)
        cnt = ipv.current.containers.pop(key)
        if ipv.current.figure == fig:
            ipv.clear()
        fig.close()
        cnt.close()

    if isinstance(fig_key, ipv.Figure):
        for k, fg in ipv.current.figures.items():
            if fg == fig_key:
                return remove(k)
        return print('Figure not found!')

    if fig_key in ipv.current.figures:
        return remove(fig_key)
    return print(f'Figure {fig_key} not found!')


def fig_clear(fig, attr='all'):
    """Clear content of given figure"""
    fig = ipv.current.figures.get(fig, fig)
    if isinstance(fig, ipv.Figure):
        if attr == 'all':
            attr = ['meshes', 'scatters', 'volumes']
        for at in attr:
            setattr(fig, at, [])


def fig_get(fig_key, **kws):
    """Get existing or new figure without making it current"""
    fig = ipv.current.figures.get(fig_key)
    if fig is None:
        prev_fig = ipv.current.figure
        prev_cnt = ipv.current.container
        fig = ipv.figure(fig_key, **kws)
        if prev_fig is not None:
            ipv.current.figure = prev_fig
            ipv.current.container = prev_cnt
    return fig


def clear_all():
    """Close all the figures and clear ipv contexts"""
    ipv.Figure.close_all()
    ipv.current.figure = None
    ipv.current.container = None
    ipv.current.figures = {}
    ipv.current.containers = {}
