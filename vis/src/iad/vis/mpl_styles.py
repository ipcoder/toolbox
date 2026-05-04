def fixed_qbstyles():
    """Fixed QB style - requires package qbstyles installed."""
    from qbstyles import mpl_style
    from matplotlib import pyplot

    figure = pyplot.figure

    mpl_style(dark=True)
    pyplot.figure = figure
