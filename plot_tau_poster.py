#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import argparse
import itertools
import os
import pdb

# 3rd party imports
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import ListedColormap
from pyrfu import pyrf
from pyrfu.plot import plot_spectr, make_labels
from scipy import constants

# Local imports
from ionaniso.utils import thresh, conditional_avg, histogram2d_loglog, percentiles
from ionaniso.plot import add_threshold, create_cmap

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2023"
__license__ = "Apache 2.0"

plt.style.use("./aps.mplstyle")


def main(args):
    n_avg = args.average
    brazil = xr.load_dataset(
        os.path.join(
            os.getcwd(),
            "data",
            f"mms_bbfsdb_brst_2017-2021_t-anisotropy_avg_{int(n_avg * 150):04}.nc",
        )
    )

    # Compute ion plasma beta (parallel, perpendicular and total)
    b_mag = pyrf.norm(brazil.b_gsm)
    p_mag = 1e-18 * b_mag.data**2 / (2 * constants.mu_0)
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

    n = pyrf.histogram2d(
        beta_para,
        t_aniso,
        bins=[10**h_x, 10**h_y],
        density=False,
    )
    n = n.assign_coords({"x_bins": x_bins, "y_bins": y_bins})

    # Compute lobe field from pressure balance
    b_lobe = b_mag * np.sqrt(1 + beta_tota)

    # Compute Alfven speed and cyclotron frequency using the lobe field
    coeff = constants.elementary_charge * 1e-9 / constants.proton_mass
    v_a = 1e-9 * 0.5 * b_lobe  # m/s
    v_a /= np.sqrt(constants.mu_0 * 1e6 * brazil.n_i * constants.proton_mass)
    w_ci0 = coeff * 0.5 * b_lobe
    f_ci0 = w_ci0 / (2.0 * np.pi)

    w_cin = coeff * brazil.b_gsm[:, 2]
    f_cin = w_cin / (2.0 * np.pi)

    w_ci = coeff * b_mag
    f_ci = w_ci / (2.0 * np.pi)
    t_ci = 1.0 / f_ci0

    # Magnitude of the magnetic field fluctuations and pitch-angle scattering time
    db_tota = np.sqrt(brazil.db_perp + brazil.db_para)
    db_b = db_tota / (0.5 * b_lobe)
    dt_s = 1 / db_b**2
    tau_s = dt_s / f_ci0

    print(np.nanmedian(dt_s))
    # Travel time from reconnection line (assumed to be at -25 R_E)
    dx_c = np.abs(brazil.r_gsm[:, 0] + 25.0 * 6371.0) * 1e3  # m
    dt_c = (dx_c / v_a) / t_ci
    tau_c = dt_c / f_ci0

    # Compute the bouncing frequency
    # Total temperature (eV)
    t_i_tota = (brazil.t_para_i + 2 * brazil.t_perp_i) / 3.0  # eV
    # Ion thermal speed (m/s)
    v_ti = np.sqrt(2 * constants.electron_volt * t_i_tota / constants.proton_mass)
    # Ion gyrofrequency in lobe field (rad/s)
    w_cil = constants.elementary_charge * 1e-9 * b_lobe / constants.proton_mass
    # Ion gyroradius in lobe field(m)
    r_lil = v_ti / w_cil
    # Current sheet thickness (m)
    l_r_lil = 2 * 10 * r_lil
    w_bi = np.sqrt(2 * v_a * w_ci0 / l_r_lil)
    dt_b = w_ci / w_bi
    tau_b = dt_b / f_ci
    tau_b_eff = (1 + 1 / (np.sqrt(2) * dt_b)) * tau_b

    # Diffusion time scale
    dt_d = np.sqrt(v_ti / v_a) / dt_b**2
    tau_d = dt_d / f_ci

    f, ax = plt.subplots(1, figsize=(4, 3))
    f.subplots_adjust(bottom=0.15)
    ratio = tau_b_eff / tau_c
    bins_ = np.histogram_bin_edges(np.log10(ratio), bins="fd")
    h_t = pyrf.histogram(ratio, bins=10**bins_, density=True)
    ax.plot(h_t.bins.data, h_t.data, label="PDF")
    ax.plot(np.sort(ratio), np.linspace(0, 1, len(ratio)), color="g", label="CDF")
    ax.set_xscale("log")
    ax.set_xlabel("$n_b\\tau_b/\\tau_c$")
    ax.legend(loc="upper right", frameon=True)

    # Simplified formulation for cross-check
    # dt_b = 2 * np.sqrt(7.5 / (1 + beta_tota))
    h_db_b_m, h_db_b_s = conditional_avg(db_b, t_aniso, beta_para, n)
    h_dt_c_m, h_dt_c_s = conditional_avg(dt_c, t_aniso, beta_para, n)

    # Compute the joint PDF of convection time and wave-particle (pitch-angle
    # scattering) interaction time.
    h_tau_cs = histogram2d_loglog(tau_c, tau_s, density=False, bins=[97, 48])

    # Compute the joint PDF of convection time and ion bouncing time.
    h_tau_cb = histogram2d_loglog(tau_c, tau_b, density=False, bins=[98, 99])
    h_tau_cb_eff = histogram2d_loglog(tau_c, tau_b_eff, density=False, bins=[98, 99])

    f, ax = plt.subplots(1, figsize=(4, 3))
    f.subplots_adjust(bottom=0.15, left=0.2, right=0.8)
    n_max = int(np.floor(0.95 * np.max(h_tau_cb_eff)))
    ax, cax = plot_spectr(
        ax, h_tau_cb_eff, clim=[0, n_max], cmap=create_cmap(n_max, "Spectral_r")
    )
    ax.loglog(np.logspace(-5, 7), np.logspace(-5, 7), color="k", linestyle="--")
    ax.set_xlim([1e-1, 1e3])
    ax.set_ylim([1e-1, 1e3])
    ax.set_xlabel("$\\tau_c~[\\mathrm{s}]$")
    ax.set_ylabel("$\\tau_{beff}=n_b\\tau_b~[\\mathrm{s}]$")
    cax.set_ylabel("counts")

    h_tau_bd_eff = histogram2d_loglog(tau_b_eff, tau_d, density=False, bins=[99, 99])

    f, ax = plt.subplots(1, figsize=(4, 3))
    f.subplots_adjust(bottom=0.15, left=0.2, right=0.8)
    n_max = int(np.floor(0.95 * np.max(h_tau_bd_eff)))
    ax, cax = plot_spectr(
        ax, h_tau_bd_eff, clim=[0, n_max], cmap=create_cmap(n_max, "Spectral_r")
    )
    ax.loglog(np.logspace(-5, 7), np.logspace(-5, 7), color="k", linestyle="--")
    ax.set_xlim([1e-1, 1e3])
    ax.set_ylim([1e-1, 1e3])
    ax.set_ylabel("$\\tau_D~[\\mathrm{s}]$")
    ax.set_xlabel("$\\tau_{beff}=n_b\\tau_b~[\\mathrm{s}]$")
    cax.set_ylabel("counts")
    plt.show()

    pdb.set_trace()
    # Compute the joint PDF of convection time and diffusion time.
    h_tau_cd = histogram2d_loglog(tau_c, tau_d, density=False, bins=[99, 99])

    h_dt_b_bins = np.histogram_bin_edges(np.log10(dt_b), "fd")
    h_dt_b_bins = 10**h_dt_b_bins
    h_dt_b = pyrf.histogram(dt_b, bins=h_dt_b_bins, density=True)
    n_dt_b = pyrf.histogram(dt_b, bins=h_dt_b_bins, density=False)

    f, axs = plt.subplots(2, 2, figsize=(5.2, 4.1))
    f.subplots_adjust(hspace=0.3, wspace=0.78, left=0.102, right=0.88, top=0.92)

    axs[0, 0], caxs00 = plot_spectr(axs[0, 0], np.log10(h_db_b_m), cmap="Spectral_r")
    add_threshold(axs[0, 0])
    # caxs00.set_ylabel("$\\mathrm{log}_{10}( \\delta B / B_0)$")
    caxs00.set_ylabel(
        "$\\mathrm{log}_{10}(\\langle \\delta B / B_0 | "
        "(R_i, \\beta_{i\\parallel}) \\rangle)$"
    )

    n_max = int(np.floor(0.95 * np.max(h_tau_cs)))
    axs[0, 1], caxs01 = plot_spectr(
        axs[0, 1], h_tau_cs, clim=[0, n_max], cmap=create_cmap(n_max, "Spectral_r")
    )
    axs[0, 1].loglog(np.logspace(-5, 7), np.logspace(-5, 7), color="k", linestyle="--")
    axs[0, 1].axhline(np.max(tau_c), linestyle="-.", color="k")
    axs[0, 1].set_xlim([1e-2, 1e6])
    axs[0, 1].set_ylim([1e-2, 1e6])
    axs[0, 1].set_xticks(np.logspace(-2, 6, 5))
    axs[0, 1].set_yticks(np.logspace(-2, 6, 5))
    axs[0, 1].set_xlabel("$\\tau_{c}~[\\mathrm{s}]$")
    axs[0, 1].set_ylabel("$\\tau_{s}~[\\mathrm{s}]$")
    caxs01.set_ylabel("counts")

    axs[1, 0].errorbar(
        h_dt_b.bins.data,
        h_dt_b.data,
        h_dt_b.data / np.sqrt(n_dt_b.data),
        label="$p(\\kappa)$",
    )
    x = np.sort(h_dt_b.bins.data)
    y = np.linspace(0, 1, len(h_dt_b.data))
    f_low = y[np.where(x > 0.2 * (1 + (np.sqrt(1 + 1 / np.sqrt(10)) - 1)))[0][0]]
    f_upp = y[np.where(x > 2.0 * (1 + (np.sqrt(1 + 1 / np.sqrt(10)) - 1)))[0][0]]
    print(f_upp - f_low)
    axs[1, 0].loglog(
        np.sort(h_dt_b.bins.data),
        np.linspace(0, 1, len(h_dt_b.data)),
        label="$F(\\kappa)$",
    )
    axs[1, 0].set_xlim([1e-2, 1e1])
    axs[1, 0].set_ylim([1e-2, 1e0])
    axs[1, 0].set_xlabel("$\\kappa=\\tau_{b}f_{ci}$")
    axs[1, 0].set_ylabel("$p(\\kappa),~F(\\kappa)$")
    axs[1, 0].axvline(1, linestyle="--", color="k")
    axs[1, 0].axvline(1 - (np.sqrt(1 + 1 / np.sqrt(10)) - 1), linestyle="-.", color="k")
    axs[1, 0].axvline(1 + (np.sqrt(1 + 1 / np.sqrt(10)) - 1), linestyle="-.", color="k")
    axs[1, 0].legend(loc="lower right", frameon=True)

    n_max = int(np.floor(0.95 * np.max(h_tau_cd)))
    axs[1, 1], caxs11 = plot_spectr(
        axs[1, 1], h_tau_cd, clim=[0, n_max], cmap=create_cmap(n_max, "Spectral_r")
    )
    axs[1, 1].loglog(np.logspace(-5, 7), np.logspace(-5, 7), color="k", linestyle="--")
    axs[1, 1].set_xlim([1e-2, 1e6])
    axs[1, 1].set_ylim([1e-2, 1e6])
    axs[1, 1].set_xticks(np.logspace(-2, 6, 5))
    axs[1, 1].set_yticks(np.logspace(-2, 6, 5))
    axs[1, 1].set_xlabel("$\\tau_{c}~[\\mathrm{s}]$")
    axs[1, 1].set_ylabel("$\\tau_{D}~[\\mathrm{s}]$")
    caxs11.set_ylabel("counts")

    make_labels(axs[0, :], (0.035, 0.92), pad=0, zorder=0)
    make_labels(axs[1, :], (0.035, 0.92), pad=2, zorder=0)
    f.savefig(
        os.path.join(
            os.pardir, "figures", "draft", f"figure_3_{int(n_avg * 150):04}_poster.pdf"
        )
    )
    f.savefig(
        os.path.join(
            os.pardir, "figures", "draft", f"figure_3_{int(n_avg * 150):04}_poster.png"
        ),
        dpi=300,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--average", type=int, default=0)
    parser.set_defaults(feature=False)
    main(parser.parse_args())
