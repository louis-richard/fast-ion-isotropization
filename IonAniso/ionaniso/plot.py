#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap

# Local imports
from .utils import thresh

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2022"
__license__ = "Apache 2.0"

__all__ = ["add_threshold", "create_cmap"]


def add_threshold(ax, legend: bool = True):
    r"""Add the thresholds for the electromagnetic ion temperature anisotropy driven
    instabilities from linear Vlasov theory [1]_ [2]_ . The thresholds are given at
    :math:`\gamma = 10^{-2} \omega_{ci}` . Also changes the scale and limits of the
    axis to make it in form of a Brazil plot.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis where to plot the thresholds in a Brazil plot format.


    References
    ----------
    .. [1]  Daniel Verscharen et al 2016 ApJ 831 128
    .. [2]  Bennett A. Maruca et al 2012 ApJ 748 137

    """

    # Proton - electron plasma thresholds [1]_
    # Proton cyclotron instability
    (l0,) = ax.loglog(
        np.logspace(-2, 5),
        thresh(np.logspace(-2, 5), 0.69, 0.4, 0.0),
        color="deeppink",
        linestyle="--",
    )

    # Mirror-mode instability
    (l1,) = ax.loglog(
        np.logspace(-2, 5),
        thresh(np.logspace(-2, 5), 1.040, 0.633, -0.012),
        color="m",
        linestyle="--",
    )

    # Parallel firehose
    (l2,) = ax.loglog(
        np.logspace(-2, 5),
        thresh(np.logspace(-2, 5), -0.647, 0.583, 0.713),
        color="b",
        linestyle="--",
    )

    # Oblique firehose
    (l3,) = ax.loglog(
        np.logspace(-2, 5),
        thresh(np.logspace(-2, 5), -1.447, 1, -0.148),
        color="c",
        linestyle="--",
    )
    ax.set_xlabel("$\\beta_{i \\parallel}$")
    ax.set_ylabel("$T_{i \\perp} / T_{i \\parallel}$")
    ax.set_xlim([1e-1, 3e3])
    ax.set_ylim([1e-1, 4e0])

    f = plt.gcf()
    if legend:
        f.legend(
            handles=[l0, l1, l2, l3],  # The line objects
            labels=[
                "Proton cyclotron",
                "Mirror Mode",
                "Parallel Firehose",
                "Oblique Firehose",
            ],
            loc="upper center",
            borderaxespad=0.1,
            ncol=4,
            frameon=False,
            handlelength=1.0,
        )

    # Plot CPS boundary (\beta_i > 0.5)
    beta_para_mat, r_p_mat = np.meshgrid(
        np.logspace(np.log10(7e-2), np.log10(3e3)), np.logspace(-1, 1)
    )
    bx_b0 = 1 / np.sqrt(1 + beta_para_mat * (1 + 2 * r_p_mat) / 3)

    ax.contour(
        beta_para_mat,
        r_p_mat,
        bx_b0,
        linestyles="dashed",
        colors="k",
        levels=[1 / np.sqrt(1.5)],
    )


def create_cmap(n_max, cmap):
    r"""Create  a colormap from 0 to `n_max` where the 25 first values are plotted as
    grey. Usefull for 2d histograms in counts units to discard counts below 5 \sigma
    assuming Poisson statistics.

    Parameters
    ----------
    n_max : int
        Maximum number of counts i.e., upper limit of the colorscale.
    cmap : str
        Colormap

    Returns
    -------
    newcmp : numpy.ndarray
        New colormap

    """

    # Get original colormap with n_max samples
    spectralr = cm.get_cmap(cmap, n_max)
    newcolors = spectralr(np.linspace(0, 1, n_max))

    # Set values below 25 to grey
    grey = [*list(colors.to_rgb("lightgrey")), 1]
    newcolors[:25, :] = grey

    # Convert to colormap type
    newcmp = ListedColormap(newcolors)
    return newcmp
