""" ------------------- """
"""                     """
""" Just some tools  :D """
"""                     """
""" ------------------- """

# Packages
import torch
import numpy as np
from IPython import display
import matplotlib.pyplot as plt


""" Matplotlib Snippet """
# for sharper images
def use_svg_display():
    # use the svg format to display a plot in the notebook
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    # set the figure size for matplotlib
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    # set the axes for matplotlib
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None,
         legend=None, xlim=None, ylim=None,
         xscale='linear', yscale='linear',
         fmts=('-', '--m', '-.g', ':r'),
         figsize=(3.5, 2.5), axes=None):
    """ plot data points """
    # if legend is None:
    #  legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # return true if X (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))  # prendo singolo elemento di lista

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)