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
from ionaniso.utils import (
    conditional_avg,
    histogram2d_linlog,
    histogram2d_loglog,
    percentiles,
)
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
    p_mag = 1e-18 * b_mag**2 / (2 * constants.mu_0)
    p_i_para = 1e6 * brazil.n_i.data * constants.electron_volt * brazil.t_para_i
    p_i_perp = 1e6 * brazil.n_i.data * constants.electron_volt * brazil.t_perp_i
    beta_para = p_i_para / p_mag
    beta_perp = p_i_perp / p_mag
    beta_tota = (beta_para + 2 * beta_perp) / 3

    # Compute temperature anisotropy
    # t_aniso = brazil.t_perp / brazil.t_para.data  # old
    t_aniso = brazil.t_perp_i / brazil.t_para_i.data
    t_agyro = brazil.lambda_t[:, 1] - brazil.lambda_t[:, 2]
    t_agyro.data /= brazil.lambda_t.data[:, 1] + brazil.lambda_t.data[:, 2]

    # Create 2D histogram of the (\beta_{\parallel i}, R_i) space
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

    idx_qps_beta = np.logical_and(beta_tota_qps > 0.5, beta_tota_qps < 1e5)
    idx_qps_anis = np.logical_and(t_aniso_qps > 0.1, t_aniso_qps < 1e1)
    idx_qps = np.logical_and(idx_qps_anis, idx_qps_beta)

    beta_para_qps = beta_para_qps[idx_qps]
    t_aniso_qps = t_aniso_qps[idx_qps]

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

    h_agy_i_m, _ = conditional_avg(t_agyro, t_aniso, beta_para, n)

    # Distance to NS
    b_lobe = b_mag * np.sqrt(1 + beta_tota)
    b_xy = np.sqrt(np.sum(brazil.b_gsm.data[:, :2] ** 2, axis=1)) / b_lobe.data
    b_xy = pyrf.ts_scalar(brazil.time.data, b_xy)

    n_b_xy_eps_i = histogram2d_linlog(b_xy, brazil.eps_i, False, bins=[60, 62])
    p_b_xy_eps_i = percentiles(b_xy, brazil.eps_i, n_b_xy_eps_i)
    n_t_a_eps_i = histogram2d_loglog(t_aniso, brazil.eps_i, False, bins=[68, 75])
    p_t_a_eps_i = percentiles(t_aniso, brazil.eps_i, n_t_a_eps_i)

    f, axs = plt.subplots(3, 2, figsize=(4.2, 4.88))
    f.subplots_adjust(
        hspace=0.4, wspace=0.95, left=0.122, right=0.875, bottom=0.075, top=0.93
    )
    n_max = int(np.floor(0.95 * np.max(n)))
    axs[0, 0], caxs00 = plot_spectr(
        axs[0, 0], n, clim=[0, n_max], cmap=create_cmap(n_max, "Spectral_r")
    )

    axs[0, 0].semilogy(p_t_an_bbfs[0].x_bins, p_t_an_bbfs[0].data, color="r")
    axs[0, 0].semilogy(p_t_an_bbfs[1].x_bins, p_t_an_bbfs[1].data, color="k")
    axs[0, 0].semilogy(p_t_an_bbfs[2].x_bins, p_t_an_bbfs[2].data, color="r")
    add_threshold(axs[0, 0])
    caxs00.set_ylabel("counts")
    caxs00.ticklabel_format(axis="y", style="sci", scilimits=(2, 2))
    caxs00.tick_params(which="major", length=3)
    caxs00.tick_params(which="minor", length=2)

    n_max = int(np.floor(0.95 * np.max(n_qps)))
    axs[0, 1], caxs01 = plot_spectr(
        axs[0, 1], n_qps, clim=[0, n_max], cmap=create_cmap(n_max, "Spectral_r")
    )
    axs[0, 1].semilogy(p_t_an_qps[0].x_bins, p_t_an_qps[0].data, color="r")
    axs[0, 1].semilogy(p_t_an_qps[1].x_bins, p_t_an_qps[1].data, color="k")
    axs[0, 1].semilogy(p_t_an_qps[2].x_bins, p_t_an_qps[2].data, color="r")
    add_threshold(axs[0, 1])
    caxs01.set_ylabel("counts")
    caxs01.ticklabel_format(axis="y", style="sci", scilimits=(2, 2))
    caxs01.tick_params(which="major", length=3)
    caxs01.tick_params(which="minor", length=2)

    axs[1, 0], caxs10 = plot_spectr(
        axs[1, 0], h_agy_i_m, clim=[0, 1e0], cmap="Spectral_r"
    )
    add_threshold(axs[1, 0])
    caxs10.set_ylabel(r"$A^{ng}$")
    caxs10.tick_params(which="major", length=3)
    caxs10.tick_params(which="minor", length=2)

    eps_range = [0.35, 0.35 + 3 * 0.09]
    axs[1, 1], caxs11 = plot_spectr(
        axs[1, 1], h_eps_i_m, cmap="Spectral_r", clim=eps_range
    )
    add_threshold(axs[1, 1])
    caxs11.set_ylabel(r"$\varepsilon_i$")
    caxs11.set_yticks(np.linspace(eps_range[0], eps_range[1], 4, endpoint=True))
    caxs11.tick_params(which="major", length=3)
    caxs11.tick_params(which="minor", length=2)

    n_max = int(np.floor(0.95 * np.max(n_b_xy_eps_i)))
    eps_range = [0.35 - 2 * 0.09, 0.35 + 4 * 0.09]

    axs[2, 0], caxs20 = plot_spectr(
        axs[2, 0], n_b_xy_eps_i, clim=[0, n_max], cmap=create_cmap(n_max, "Spectral_r")
    )
    axs[2, 0].plot(p_b_xy_eps_i[0].x_bins, p_b_xy_eps_i[0].data, color="r")
    axs[2, 0].plot(p_b_xy_eps_i[1].x_bins, p_b_xy_eps_i[1].data, color="k")
    axs[2, 0].plot(p_b_xy_eps_i[2].x_bins, p_b_xy_eps_i[2].data, color="r")
    axs[2, 0].set_xlim([0.0, 1.0])
    axs[2, 0].set_ylim(eps_range)
    axs[2, 0].set_yticks(np.linspace(eps_range[0], eps_range[1], 7))
    axs[2, 0].set_xlabel(r"$|B_{xy}| / B_{ext}$")
    axs[2, 0].set_ylabel(r"$\varepsilon_i$")
    caxs20.set_ylabel("counts")
    caxs20.ticklabel_format(axis="y", style="sci", scilimits=(2, 2))
    caxs20.tick_params(which="major", length=3)
    caxs20.tick_params(which="minor", length=2)

    n_max = int(np.floor(0.95 * np.max(n_t_a_eps_i)))
    axs[2, 1], caxs21 = plot_spectr(
        axs[2, 1], n_t_a_eps_i, clim=[0, n_max], cmap=create_cmap(n_max, "Spectral_r")
    )
    axs[2, 1].semilogx(p_t_a_eps_i[0].x_bins, p_t_a_eps_i[0].data, color="r")
    axs[2, 1].semilogx(p_t_a_eps_i[1].x_bins, p_t_a_eps_i[1].data, color="k")
    axs[2, 1].semilogx(p_t_a_eps_i[2].x_bins, p_t_a_eps_i[2].data, color="r")
    axs[2, 1].set_xlim([1e-1, 1e1])
    axs[2, 1].set_ylim(eps_range)
    axs[2, 1].set_yticks(np.linspace(eps_range[0], eps_range[1], 7))

    axs[2, 1].set_xlabel(r"$T_{i\perp}/T_{i\parallel}$")
    axs[2, 1].set_ylabel(r"$\varepsilon_i$")
    caxs21.set_ylabel("counts")
    caxs21.ticklabel_format(axis="y", style="sci", scilimits=(2, 2))
    caxs21.tick_params(which="major", length=3)
    caxs21.tick_params(which="minor", length=2)

    bbox = {
        "boxstyle": "square",
        "ec": (1.0, 1.0, 1.0),
        "fc": (1.0, 1.0, 1.0),
        "alpha": 0.2,
    }
    make_labels(axs[0, :], (0.035, 0.88), pad=0, zorder=2, bbox=bbox)
    make_labels(axs[1, :], (0.035, 0.88), pad=2, zorder=2, bbox=bbox)
    make_labels(axs[2, :], (0.035, 0.88), pad=4, zorder=2, bbox=bbox)

    f.align_ylabels([*axs[:, 0]])

    f.savefig(
        os.path.join(
            os.pardir, "figures", "draft", f"figure_1_{int(n_avg * 150):04}.pdf"
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plot distribution of measurements [Fig. 1].")
    parser.add_argument(
        "-a",
        "--average",
        type=int,
        default=0,
        help="Number of ion VDFs to use in averaging (see also compile.py)",
    )
    parser.set_defaults(feature=False)
    main(parser.parse_args())
