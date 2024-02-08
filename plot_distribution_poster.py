#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import argparse
import os

# 3rd party imports
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from ionaniso.plot import add_threshold, create_cmap

# Local imports
from ionaniso.utils import conditional_avg, histogram2d_linlog, percentiles
from pyrfu import pyrf
from pyrfu.plot import make_labels, plot_spectr, use_pyrfu_style
from scipy import constants

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2022"
__license__ = "Apache 2.0"

use_pyrfu_style("aps", usetex=True)


def main(args):
    n_avg = args.average
    brazil = xr.load_dataset(
        os.path.join(
            os.getcwd(),
            "data",
            f"mms_bbfsdb_brst_2017-2021_t-anisotropy_avg_{int(n_avg * 150):04}.nc",
        )
    )

    braqps = xr.load_dataset(
        os.path.join(os.pardir, "data", "mms_quietpsheet_2017-2021_t-anisotropy.nc")
    )

    # BBFS
    # Compute ion plasma beta (parallel, perpendicular and total)
    b_mag = pyrf.norm(brazil.b_gsm)
    p_mag = 1e-18 * b_mag**2 / (2 * constants.mu_0)  # old
    p_i_para = 1e6 * brazil.n_i.data * constants.electron_volt * brazil.t_para_i
    p_i_perp = 1e6 * brazil.n_i.data * constants.electron_volt * brazil.t_perp_i
    beta_para = p_i_para / p_mag
    beta_perp = p_i_perp / p_mag
    beta_tota = (beta_para + 2 * beta_perp) / 3

    # Compute temperature anisotropy
    t_aniso = brazil.t_perp_i / brazil.t_para_i.data

    # Create 2D histogram of the (\beta_{\\parallel i}, R_i) space
    _, h_x, h_y = np.histogram2d(
        np.log10(beta_para),
        np.log10(t_aniso),
        bins=[99, 95],
        density=True,
    )
    x_bins = 10 ** (h_x[:-1] + np.median(np.diff(h_x)) / 2)
    y_bins = 10 ** (h_y[:-1] + np.median(np.diff(h_y)) / 2)

    # Counts
    n = pyrf.histogram2d(
        beta_para,
        t_aniso,
        bins=[10**h_x, 10**h_y],
        density=False,
    )
    n = n.assign_coords({"x_bins": x_bins, "y_bins": y_bins})
    n.data[n.data == 0] = np.nan
    p_t_an_bbfs = percentiles(beta_para, t_aniso, n)

    # Probability Density Function
    h = pyrf.histogram2d(
        beta_para,
        t_aniso,
        bins=[10**h_x, 10**h_y],
        density=True,
    )
    h = h.assign_coords({"x_bins": x_bins, "y_bins": y_bins})
    h.data[n.data < 25] = np.nan

    # Quiet plasma sheet PDF
    beta_para_qps = braqps.beta_para
    beta_perp_qps = braqps.beta_perp
    beta_tota_qps = (beta_para_qps + 2 * beta_perp_qps) / 3

    # Keep only points in CPS
    cond_cps = np.logical_and(
        np.abs(braqps.r_gsm.data[:, 1]) < 12 * 6371, beta_tota_qps > 0.5
    )
    beta_para_qps = beta_para_qps[cond_cps]

    # Compute temperature anisotropy
    t_aniso_qps = braqps.t_perp / braqps.t_para.data
    t_aniso_qps = t_aniso_qps[cond_cps]

    # Counts
    n_qps = pyrf.histogram2d(
        beta_para_qps,
        t_aniso_qps,
        bins=[10**h_x, 10**h_y],
        density=False,
    )
    n_qps = n_qps.assign_coords({"x_bins": x_bins, "y_bins": y_bins})
    n_qps.data[n_qps.data == 0] = np.nan
    p_t_an_qps = percentiles(beta_para_qps, t_aniso_qps, n_qps)

    # Probability Density Function
    h_qps = pyrf.histogram2d(
        beta_para_qps,
        t_aniso_qps,
        bins=[10**h_x, 10**h_y],
        density=True,
    )
    h_qps = h_qps.assign_coords({"x_bins": x_bins, "y_bins": y_bins})
    h_qps.data[n_qps.data < 25] = np.nan

    # Conditional average non-Maxwellianity
    h_eps_i_m, _ = conditional_avg(brazil.eps_i, t_aniso, beta_para, n)

    # Distance to NS
    b_lobe = b_mag * np.sqrt(1 + beta_tota)
    # b_xy = np.sqrt(brazil.b_x.data ** 2 + brazil.b_y.data ** 2) / b_lobe.data
    b_xy = np.sqrt(np.sum(brazil.b_gsm.data[:, :2] ** 2, axis=1)) / b_lobe.data
    b_xy = pyrf.ts_scalar(brazil.time.data, b_xy)

    n_b_xy_eps_i = histogram2d_linlog(b_xy, brazil.eps_i, False, bins=[60, 62])
    p_b_xy_eps_i = percentiles(b_xy, brazil.eps_i, n_b_xy_eps_i)

    f, axs = plt.subplots(2, 2, figsize=(5.2, 4.1))
    f.subplots_adjust(hspace=0.3, wspace=0.78, left=0.102, right=0.88, top=0.92)
    axs[0, 0], caxs00 = plot_spectr(
        axs[0, 0], h, cscale="log", clim=[1e-5, 1e0], cmap="Spectral_r"
    )
    axs[0, 0].semilogy(p_t_an_bbfs[0].x_bins, p_t_an_bbfs[0].data, color="r")
    axs[0, 0].semilogy(p_t_an_bbfs[1].x_bins, p_t_an_bbfs[1].data, color="k")
    axs[0, 0].semilogy(p_t_an_bbfs[2].x_bins, p_t_an_bbfs[2].data, color="r")
    add_threshold(axs[0, 0])
    caxs00.set_ylabel("$p(\\beta_{i \\parallel}, T_{i\\perp}/T_{i\\parallel})$")

    axs[0, 1], caxs01 = plot_spectr(
        axs[0, 1], h_qps, cscale="log", clim=[1e-5, 1e0], cmap="Spectral_r"
    )
    axs[0, 1].semilogy(p_t_an_qps[0].x_bins, p_t_an_qps[0].data, color="r")
    axs[0, 1].semilogy(p_t_an_qps[1].x_bins, p_t_an_qps[1].data, color="k")
    axs[0, 1].semilogy(p_t_an_qps[2].x_bins, p_t_an_qps[2].data, color="r")
    add_threshold(axs[0, 1])
    caxs01.set_ylabel("$p(\\beta_{i \\parallel}, T_{i\\perp}/T_{i\\parallel})$")

    axs[1, 0], caxs10 = plot_spectr(axs[1, 0], h_eps_i_m, cmap="Spectral_r")
    add_threshold(axs[1, 0])
    # caxs10.set_ylabel("$\\langle \\varepsilon_i | (T_{i\\perp}/T_{i\\parallel}, "
    #                  "\\beta_{i\\parallel})\\rangle$")
    caxs10.set_ylabel("$\\varepsilon_i$")

    n_max = int(np.floor(0.95 * np.max(n_b_xy_eps_i)))
    axs[1, 1], caxs11 = plot_spectr(
        axs[1, 1], n_b_xy_eps_i, clim=[0, n_max], cmap=create_cmap(n_max, "Spectral_r")
    )
    axs[1, 1].semilogy(p_b_xy_eps_i[0].x_bins, p_b_xy_eps_i[0].data, color="r")
    axs[1, 1].semilogy(p_b_xy_eps_i[1].x_bins, p_b_xy_eps_i[1].data, color="k")
    axs[1, 1].semilogy(p_b_xy_eps_i[2].x_bins, p_b_xy_eps_i[2].data, color="r")
    axs[1, 1].set_yticks([0.2, 0.4, 0.6, 0.8])
    axs[1, 1].set_yticklabels([str(t) for t in [0.2, 0.4, 0.6, 0.8]])
    axs[1, 1].set_xlabel("$|B_{xy}| / B_{0}$")
    axs[1, 1].set_ylabel("$\\varepsilon_i$")
    caxs11.set_ylabel("counts")

    make_labels(axs[0, :], (0.035, 0.92), pad=0, zorder=0)
    make_labels(axs[1, :], (0.035, 0.92), pad=2, zorder=0)
    f.savefig(
        os.path.join(
            os.pardir, "figures", "draft", f"figure_1_{int(n_avg * 150):04}_poster.pdf"
        )
    )
    f.savefig(
        os.path.join(
            os.pardir, "figures", "draft", f"figure_1_{int(n_avg * 150):04}_poster.png"
        ),
        dpi=300,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--average", type=int, default=0)
    parser.set_defaults(feature=False)
    main(parser.parse_args())
