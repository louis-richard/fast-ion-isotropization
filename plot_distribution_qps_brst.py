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
from ionaniso.utils import conditional_avg, percentiles
from pyrfu import pyrf
from pyrfu.plot import make_labels, plot_spectr, use_pyrfu_style
from scipy import constants

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2023"
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

    data = xr.load_dataset(
        os.path.join(os.pardir, "data", "mms_fpi_brst_2017-2021_tail.nc")
    )

    # BBFS
    # Compute ion plasma beta (parallel, perpendicular and total)
    b_mag = pyrf.norm(brazil.b_gsm)
    p_mag = 1e-18 * b_mag**2 / (2 * constants.mu_0)  # old
    p_i_para = 1e6 * brazil.n_i.data * constants.electron_volt * brazil.t_para_i
    beta_para = p_i_para / p_mag

    # Compute temperature anisotropy
    # t_aniso = brazil.t_perp / brazil.t_para.data  # old
    t_aniso = brazil.t_perp_i / brazil.t_para_i.data
    t_agyro = brazil.lambda_t[:, 1] - brazil.lambda_t[:, 2]
    t_agyro.data /= brazil.lambda_t.data[:, 1] + brazil.lambda_t.data[:, 2]

    eps_i_bbf = brazil.eps_i

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

    # Identify quiet plasma sheet
    beta_tot = (data.beta_para + 2 * data.beta_perp) / 3
    idx_ps = np.logical_and(beta_tot > 0.5, np.abs(data.r_gsm.data[:, 1]) / 6371 < 12)
    idx_qps = np.logical_and(idx_ps, np.linalg.norm(data.v_gse_i.data, axis=1) < 100)
    print(np.sum(idx_qps))

    # Quiet plasma sheet PDF
    beta_para_qps = data.beta_para[idx_qps]
    beta_perp_qps = data.beta_perp[idx_qps]
    beta_tota_qps = (beta_para_qps + 2 * beta_perp_qps) / 3

    # Compute temperature anisotropy
    t_aniso_qps = data.t_perp[idx_qps] / data.t_para.data[idx_qps]

    eps_i_qps = data.eps_i[idx_qps]

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
    h_eps_i_bbf_m, _ = conditional_avg(eps_i_bbf, t_aniso, beta_para, n)
    h_eps_i_qps_m, _ = conditional_avg(eps_i_qps, t_aniso_qps, beta_para_qps, n_qps)

    f, axs = plt.subplots(2, 2, figsize=(4, 3.2))
    f.subplots_adjust(hspace=0.3, wspace=0.87, left=0.12, right=0.88, top=0.88)
    n_max = int(np.floor(0.95 * np.max(n)))
    axs[0, 0], caxs00 = plot_spectr(
        axs[0, 0], n, clim=[0, n_max], cmap=create_cmap(n_max, "Spectral_r")
    )
    axs[0, 0].semilogy(p_t_an_bbfs[0].x_bins, p_t_an_bbfs[0].data, color="r")
    axs[0, 0].semilogy(p_t_an_bbfs[1].x_bins, p_t_an_bbfs[1].data, color="k")
    axs[0, 0].semilogy(p_t_an_bbfs[2].x_bins, p_t_an_bbfs[2].data, color="r")
    add_threshold(axs[0, 0])
    caxs00.set_ylabel("counts")

    n_max = int(np.floor(0.95 * np.max(n_qps)))
    axs[0, 1], caxs01 = plot_spectr(
        axs[0, 1], n_qps, clim=[0, n_max], cmap=create_cmap(n_max, "Spectral_r")
    )
    axs[0, 1].semilogy(p_t_an_qps[0].x_bins, p_t_an_qps[0].data, color="r")
    axs[0, 1].semilogy(p_t_an_qps[1].x_bins, p_t_an_qps[1].data, color="k")
    axs[0, 1].semilogy(p_t_an_qps[2].x_bins, p_t_an_qps[2].data, color="r")
    add_threshold(axs[0, 1])
    caxs01.set_ylabel("counts")

    eps_range = [
        np.round(np.mean(eps_i_qps), 2) - np.round(np.std(eps_i_qps), 2),
        np.round(np.mean(eps_i_qps), 2) + 3 * np.round(np.std(eps_i_qps), 2),
    ]
    print(np.round(np.mean(eps_i_qps), 2), np.round(np.std(eps_i_qps), 2))

    axs[1, 0], caxs10 = plot_spectr(
        axs[1, 0], h_eps_i_bbf_m, cmap="Spectral_r", clim=eps_range
    )
    add_threshold(axs[1, 0])
    caxs10.set_yticks(np.linspace(eps_range[0], eps_range[1], 5, endpoint=True))
    caxs10.set_ylabel("$\\varepsilon_i$")

    axs[1, 1], caxs11 = plot_spectr(
        axs[1, 1], h_eps_i_qps_m, cmap="Spectral_r", clim=eps_range
    )
    add_threshold(axs[1, 1])
    caxs11.set_yticks(np.linspace(eps_range[0], eps_range[1], 5, endpoint=True))
    caxs11.set_ylabel("$\\varepsilon_i$")

    axs[0, 0].set_title("reconnection jets")
    axs[0, 1].set_title("quiet plasma sheet")

    make_labels(axs[0, :], (0.035, 0.92), pad=0, zorder=0)
    make_labels(axs[1, :], (0.035, 0.92), pad=2, zorder=0)
    f.savefig(os.path.join(os.pardir, "figures", "figure_1_bbfs_vs_qps.pdf"))
    f.savefig(os.path.join(os.pardir, "figures", "figure_1_bbfs_vs_qps.png"), dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--average", type=int, default=0)
    parser.set_defaults(feature=False)
    main(parser.parse_args())
