#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import argparse
import os

# 3rd party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from ionaniso.load import load_efield_mmsx, load_fpi_dis_mmsx, load_rsc_bfield_mmsx
from ionaniso.utils import average_mom, average_vdf, histogram2d_loglog
from pyrfu import mms, pyrf
from pyrfu.plot import make_labels, plot_line, use_pyrfu_style
from scipy import constants

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2023"
__license__ = "Apache 2.0"

mms.db_init("/Volumes/mms")
use_pyrfu_style("aps", usetex=True)


def _epsilon_from_n(n_i, n_avg):
    brazil = xr.load_dataset(
        os.path.join(
            os.pardir,
            "data",
            f"mms_bbfsdb_brst_2017-2021_t-anisotropy_avg_{n_avg}_all.nc",
        )
    )

    beta_para = 1e24 * brazil.n_i.data * constants.electron_volt * brazil.t_para
    beta_para /= brazil.b_mag.data**2 / (2 * constants.mu_0)

    n_2d = histogram2d_loglog(brazil.n_i, brazil.eps_i, False, bins=[86, 77])

    idx_n1 = np.digitize(n_i.data, n_2d.x_bins)
    idx_n = np.digitize(brazil.n_i.data, n_2d.x_bins)
    ten_eps, fif_eps, nth_eps = [np.zeros(len(n_i.data)) for _ in range(3)]

    for it in range(len(n_i)):
        perc_ = np.percentile(brazil.eps_i[idx_n == idx_n1[it]], [10, 50, 90])
        ten_eps[it], fif_eps[it], nth_eps[it] = perc_

    ten_eps = pyrf.ts_scalar(n_i.time.data, ten_eps)
    fif_eps = pyrf.ts_scalar(n_i.time.data, fif_eps)
    nth_eps = pyrf.ts_scalar(n_i.time.data, nth_eps)

    return ten_eps, fif_eps, nth_eps


def main(args):
    n_avg = args.average
    bbfs = pd.read_csv(
        os.path.join(os.pardir, "data", "mms_bbfsdb_brst_2017-2021.csv"), header=None
    )
    data = xr.load_dataset(
        os.path.join(os.pardir, "data", "mms_bbfsdb_brst_2017-2021.nc")
    )
    cond_cps = np.logical_and(np.abs(data.y.data / 6371) < 12, data.beta.data > 0.5)
    tint = bbfs.values[cond_cps][172]
    print(tint)

    # Load magnetic field averaged across the spacecraft
    _, b_dmpa, b_gsm = load_rsc_bfield_mmsx(tint)

    # Load electric field averaged across the spacecraft
    e_gsm, sc_pot = load_efield_mmsx(tint)

    # Load ion moments and distributions averaged across the spacecraft
    n_i, v_dbcs_i, v_gse_i, t_dbcs_i, t_gse_i, vdf_i_new = load_fpi_dis_mmsx(tint)
    v_gsm_i = pyrf.cotrans(v_gse_i, "gse>gsm")

    # Temperature anisotropy and linear thresholds
    t_fac_i = mms.rotate_tensor(t_dbcs_i, "fac", b_dmpa, "pp")
    t_para_i = t_fac_i[:, 0, 0]
    t_perp_i = (t_fac_i[:, 1, 1] + t_fac_i[:, 2, 2]) / 2
    p_para_i = n_i * 1e6 * constants.electron_volt * t_para_i
    p_perp_i = n_i * 1e6 * constants.electron_volt * t_perp_i
    p_mag = 1e-18 * pyrf.resample(pyrf.norm(b_gsm), n_i) ** 2 / (2 * constants.mu_0)
    beta_para_i = p_para_i / p_mag.data
    beta_perp_i = p_perp_i / p_mag.data
    beta_tota_i = (beta_para_i + 2 * beta_perp_i) / 3
    b_lobe = np.sqrt(1 + beta_tota_i) * pyrf.resample(pyrf.norm(b_gsm), beta_tota_i)

    t_aniso = t_perp_i / t_para_i

    # Compute the bouncing frequency
    # Total temperature (eV)
    t_i_tota = (t_para_i + 2 * t_perp_i) / 3.0  # eV

    coeff = constants.elementary_charge * 1e-9 / constants.proton_mass
    v_a = 1e-9 * 0.5 * b_lobe  # m/s
    v_a /= np.sqrt(constants.mu_0 * 1e6 * n_i * constants.proton_mass)
    w_ci0 = coeff * 0.5 * b_lobe
    w_ci = coeff * pyrf.resample(pyrf.norm(b_gsm), b_lobe)

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

    # Create a model bi-Maxwellian based on the ion moments and compute its DEF
    model_vdf_i = mms.make_model_vdf(vdf_i_new, b_dmpa, sc_pot, n_i, v_dbcs_i, t_dbcs_i)
    model_vdf_i.data.data *= 1e-30

    _, v_gse_i_avg, t_gse_i_avg = average_mom(n_i, v_gse_i, t_gse_i, n_avg)
    n_i_avg, v_dbcs_i_avg, t_dbcs_i_avg = average_mom(n_i, v_dbcs_i, t_dbcs_i, n_avg)
    vdf_i_avg = average_vdf(vdf_i_new, n_avg)

    # Compute model distribution using averaged data
    model_vdf_i_avg = mms.make_model_vdf(
        vdf_i_avg, b_dmpa, sc_pot, n_i_avg, v_dbcs_i_avg, t_dbcs_i_avg
    )
    # Threshold energy at 0.3 v_th
    e_thresh = np.mean(pyrf.trace(t_dbcs_i_avg).data / 3) / 9
    i_thresh = np.where(vdf_i_avg.energy.data[0, :] > e_thresh)[0][0]

    # Compute non-Maxwellianity
    eps_i = mms.calculate_epsilon(
        vdf_i_avg, model_vdf_i_avg, n_i_avg, sc_pot, en_channels=[i_thresh, 32]
    )

    model_vdf_i_avg.data.data *= 1e-30

    l_, eh_gse = np.linalg.eigh(t_gse_i_avg.data)
    eh_1_gse = pyrf.ts_vec_xyz(t_gse_i_avg.time.data, eh_gse[:, :, 2])  # e1 (l1)
    eh_1_gsm = pyrf.cotrans(eh_1_gse, "gse>gsm")
    eh_2_gse = pyrf.ts_vec_xyz(t_gse_i_avg.time.data, eh_gse[:, :, 1])  # e2 (l2)
    eh_2_gsm = pyrf.cotrans(eh_2_gse, "gse>gsm")
    eh_3_gse = pyrf.ts_vec_xyz(t_gse_i_avg.time.data, eh_gse[:, :, 0])  # e3 (l3)
    eh_3_gsm = pyrf.cotrans(eh_3_gse, "gse>gsm")
    l_ = pyrf.ts_vec_xyz(t_gse_i_avg.time.data, np.fliplr(l_))  # l1 > l2 > l3
    id_ne = np.where(eh_3_gsm.data[:, 2] < 0)[0]
    eh_3_gsm.data[id_ne, :] *= -1
    eh_2_gsm.data[id_ne, :] = np.cross(eh_3_gsm.data[id_ne, :], eh_1_gsm.data[id_ne])
    eh_gsm = np.transpose(
        np.stack([eh_1_gsm.data, eh_2_gsm.data, eh_3_gsm.data]), [1, 2, 0]
    )
    eh_gsm = pyrf.ts_tensor_xyz(t_gse_i_avg.time.data, eh_gsm)

    r_mms = [mms.get_data("r_gsm_mec_srvy_l2", tint, i) for i in range(1, 5)]
    b_mms = [mms.get_data("b_gsm_fgm_brst_l2", tint, i) for i in range(1, 5)]
    r_c = 1 / pyrf.norm(pyrf.c_4_grad(r_mms, b_mms, "curv"))
    rho_i = (
        144 * np.sqrt(pyrf.trace(t_fac_i) / 3) / pyrf.resample(pyrf.norm(b_gsm), n_i)
    )
    k = np.sqrt(pyrf.resample(r_c, rho_i) / rho_i)

    # Compute the scalar product betwenn the eigenvectors and the magnetic field
    b_hat = b_gsm / pyrf.norm(b_gsm)
    b_hat = pyrf.resample(b_hat, eh_1_gsm)
    e1_dot_bh = pyrf.ts_scalar(
        t_gse_i_avg.time.data, np.sum(eh_gsm.data[..., 0] * b_hat.data, axis=1)
    )
    e3_dot_bh = pyrf.ts_scalar(
        t_gse_i_avg.time.data, np.sum(eh_gsm.data[..., 2] * b_hat.data, axis=1)
    )

    t_ls = t_aniso.copy()
    t_ls.data[t_ls.data < 1] = 1 / t_ls.data[t_ls.data < 1]

    b_xy = np.sqrt(b_gsm[:, 0] ** 2 + b_gsm[:, 1] ** 2)

    zeros = [None] * 5
    b_tmp = pyrf.time_clip(b_xy, ["2017-08-21T11:33:45", "2017-08-21T11:34:10"])
    zeros[0] = b_tmp.time.data[np.argmin(b_tmp.data)]
    b_tmp = pyrf.time_clip(b_xy, ["2017-08-21T11:34:10", "2017-08-21T11:34:22"])
    zeros[1] = b_tmp.time.data[np.argmin(b_tmp.data)]
    b_tmp = pyrf.time_clip(b_xy, ["2017-08-21T11:34:22", "2017-08-21T11:34:45"])
    zeros[2] = b_tmp.time.data[np.argmin(b_tmp.data)]
    b_tmp = pyrf.time_clip(b_xy, ["2017-08-21T11:34:45", "2017-08-21T11:35:10"])
    zeros[3] = b_tmp.time.data[np.argmin(b_tmp.data)]
    b_tmp = pyrf.time_clip(b_xy, ["2017-08-21T11:35:10", "2017-08-21T11:35:35"])
    zeros[4] = b_tmp.time.data[np.argmin(b_tmp.data)]

    fig, axs = plt.subplots(7, sharex="all", figsize=(4.2, 4.935))
    fig.subplots_adjust(hspace=0, left=0.127, right=0.86, top=0.995, bottom=0.063)
    plot_line(axs[0], b_gsm)
    plot_line(axs[0], pyrf.norm(b_gsm), color="k")
    axs[0].set_ylim([-12.8, 12.8])
    axs[0].set_ylabel(r"$B~[\unit{\nano\tesla}]$")
    axs[0].legend(
        [r"$B_{x}$", r"$B_{y}$", r"$B_{z}$", r"$|B|$"],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.99, 1.15),
        handlelength=1.0,
    )

    plot_line(axs[1], v_gsm_i)
    axs[1].set_ylim([-198, 550])
    axs[1].set_ylabel(r"$V_i~[\unit{\kilo\meter\per\second}]$")
    axs[1].legend(
        [r"$V_{ix}$", r"$V_{iy}$", r"$V_{iz}$"],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.99, 1.15),
        handlelength=1.0,
    )

    plot_line(axs[2], t_aniso, color="k")
    y_lim = list(axs[2].get_ylim())
    y_lim[0] = np.max([y_lim[0], 0.0])
    y_lim[1] = np.min([y_lim[1], 10.0])

    axs[2].set_ylim([0.47, 1.53])
    axs[2].yaxis.set_ticks([0.5, 0.75, 1, 1.25, 1.5])
    axs[2].set_ylabel(r"$T_{i\perp} / T_{i\parallel}$")

    plot_line(
        axs[3],
        l_[:, 0] / l_[:, 1],
        label=r"$\lambda_1 / \lambda_2$",
        color="pyrfu:blue",
    )
    plot_line(
        axs[3], l_[:, 1] / l_[:, 2], label=r"$\lambda_2 / \lambda_3$", color="pyrfu:red"
    )
    axs[3].set_ylabel(r"$\lambda_i/\lambda_j$")
    axs[3].legend(
        frameon=False, loc="upper left", bbox_to_anchor=(0.99, 1.05), handlelength=1.0
    )
    axs[3].set_ylim([1.0, 1.82])

    plot_line(
        axs[4],
        np.abs(e1_dot_bh),
        color="pyrfu:blue",
        label=r"$|\mathbf{\hat{e}_1}.\mathbf{\hat{B}}|$",
    )
    plot_line(
        axs[4],
        np.abs(e3_dot_bh),
        color="pyrfu:red",
        label=r"$|\mathbf{\hat{e}_3}.\mathbf{\hat{B}}|$",
    )
    axs[4].set_ylabel(r"$|\mathbf{\hat{e}}.\mathbf{\hat{B}}|$")
    axs[4].legend(
        frameon=False, loc="upper left", bbox_to_anchor=(0.99, 1.15), handlelength=1.0
    )
    axs[4].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    plot_line(axs[5], eps_i[1:], color="pyrfu:blue", label=r"$\varepsilon_i$")
    axs[5].set_ylabel(r"$\varepsilon_i$")
    axs[5].set_ylim([0.35, 0.5])
    axs[5].yaxis.set_ticks([0.35, 0.4, 0.45])

    plot_line(axs[6], k, color="pyrfu:green", label=r"$\sqrt{\frac{r_c}{\rho_i}}$")
    plot_line(axs[6], dt_b, color="pyrfu:blue", label=r"$\tau_bf_{ci}$")
    axs[6].legend(
        frameon=False, loc="upper left", bbox_to_anchor=(0.99, 1.05), handlelength=1.0
    )
    axs[6].set_ylabel(r"$\kappa$")
    axs[6].set_yscale("log")
    axs[6].set_ylim([3e-2, 3e1])

    for ax in axs:
        for z_ in zeros:
            ax.axvline(z_, color="k", linestyle="--")

        for i in range(4):
            b_tmp = pyrf.time_clip(pyrf.norm(b_gsm), [zeros[i], zeros[i + 1]])
            maxm = b_tmp.time.data[np.argmax(b_tmp.data)]
            ax.axvline(maxm, color="pyrfu:red", linestyle="--")

    make_labels(axs, (0.018, 0.84), pad=0, color="k")
    fig.align_ylabels(axs)
    fig.savefig(os.path.join(os.pardir, "figures", "draft", "figure_2.pdf"))
    fig.savefig(os.path.join(os.pardir, "figures", "draft", "figure_2.png"), dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Plot example reconnection jet with current sheet flapping motion [Fig. 2]."
    )
    parser.add_argument(
        "-a",
        "--average",
        type=int,
        default=0,
        help="Number of ion VDFs to use in averaging",
    )
    parser.set_defaults(feature=False)
    main(parser.parse_args())
