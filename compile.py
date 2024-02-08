#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import argparse
import os

# 3rd party imports
import numpy as np
import pandas as pd
import xarray as xr
from ionaniso.load import load_efield_mmsx, load_fpi_dis_mmsx, load_rsc_bfield_mmsx
from ionaniso.utils import average_mom, average_vdf
from pyrfu import mms, pyrf
from scipy import constants

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2022"
__license__ = "Apache 2.0"

mms.db_init("/Volumes/PHD/bbfs/data")


def main(args):
    bbfs = pd.read_csv(
        os.path.join(os.pardir, "data", "mms_bbfsdb_brst_2017-2021.csv"), header=None
    )
    data = xr.load_dataset(
        os.path.join(os.pardir, "data", "mms_bbfsdb_brst_2017-2021.nc")
    )
    cond_cps = np.logical_and(np.abs(data.y.data / 6371) < 12, data.beta.data > 0.5)
    tints = bbfs.values[cond_cps]

    n_avg = args.average
    f_name = f"mms_bbfsdb_brst_2017-2021_t-anisotropy_avg_{int(n_avg * 150):04}.nc"
    brazil = xr.load_dataset(os.path.join(os.getcwd(), "data", f_name))

    for i, tint in zip(range(args.start, len(tints)), tints[args.start :]):
        print(i, len(tints), tint)

        # Load magnetic field averaged across the spacecraft
        r_gsm, b_dmpa, b_gsm = load_rsc_bfield_mmsx(tint)

        # Load electric field averaged across the spacecraft
        _, sc_pot = load_efield_mmsx(tint)

        # Load ion moments and distributions averaged across the spacecraft
        n_i, v_dbcs_i, v_gse_i, t_dbcs_i, t_gse_i, vdf_i = load_fpi_dis_mmsx(tint)

        if n_i is None or any(pyrf.norm(v_dbcs_i).data > 2500):
            continue

        # Average 7 distributions and moments
        _, v_gse_i, t_gse_i = average_mom(n_i, v_gse_i, t_gse_i, n_avg)
        n_i, v_dbcs_i, t_dbcs_i = average_mom(n_i, v_dbcs_i, t_dbcs_i, n_avg)
        vdf_i = average_vdf(vdf_i, n_avg)

        # Compute model distribution using averaged data
        model_vdf_i = mms.make_model_vdf(vdf_i, b_dmpa, sc_pot, n_i, v_dbcs_i, t_dbcs_i)

        try:
            # Threshold energy at 0.3 v_th
            e_thresh = np.mean(pyrf.trace(t_dbcs_i).data / 3) / 9
            i_thresh = np.where(vdf_i.energy.data[0, :] > e_thresh)[0][0]

            # Compute non-Maxwellianity
            eps_i = mms.calculate_epsilon(
                vdf_i, model_vdf_i, n_i, sc_pot, en_channels=[i_thresh, 32]
            )
        except:
            continue

        # Transform ion bulk veocty to GSM coordinates
        v_gsm_i = pyrf.cotrans(v_gse_i, "gse>gsm")

        # Resample spcecraft location and magnetic field to ion moments sampling
        r_gsm = pyrf.resample(r_gsm, n_i)
        b_gsm = pyrf.resample(b_gsm, n_i)
        b_mag = pyrf.norm(b_gsm)

        # Transform the ion temperature tensor field aligned coordinates to get the
        # parallel, perpandicular and total ion temperatures.
        t_fac_i = mms.rotate_tensor(t_dbcs_i, "fac", b_dmpa, "pp")
        t_perp_i = (t_fac_i.data[:, 1, 1] + t_fac_i.data[:, 2, 2]) / 2
        t_perp_i = pyrf.ts_scalar(t_fac_i.time.data, t_perp_i)
        t_para_i = t_fac_i.data[:, 0, 0]
        t_para_i = pyrf.ts_scalar(t_fac_i.time.data, t_para_i)
        t_i = pyrf.trace(t_fac_i) / 3

        # Compute the ion temperature tensor eigenvectors and eigenvalues.
        l_, eh_gse = np.linalg.eigh(t_gse_i.data)
        eh_1_gse = pyrf.ts_vec_xyz(t_gse_i.time.data, eh_gse[:, :, 2])  # e1 (l1)
        eh_1_gsm = pyrf.cotrans(eh_1_gse, "gse>gsm")
        eh_2_gse = pyrf.ts_vec_xyz(t_gse_i.time.data, eh_gse[:, :, 1])  # e2 (l2)
        eh_2_gsm = pyrf.cotrans(eh_2_gse, "gse>gsm")
        eh_3_gse = pyrf.ts_vec_xyz(t_gse_i.time.data, eh_gse[:, :, 0])  # e3 (l3)
        eh_3_gsm = pyrf.cotrans(eh_3_gse, "gse>gsm")

        l_ = pyrf.ts_vec_xyz(t_gse_i.time.data, np.fliplr(l_))  # l1 > l2 > l3
        eh_gsm = np.transpose(
            np.stack([eh_1_gsm.data, eh_2_gsm.data, eh_3_gsm.data]), [1, 2, 0]
        )
        eh_gsm = pyrf.ts_tensor_xyz(t_gse_i.time.data, eh_gsm)

        # Compute ion plasma beta and lobe field from pressure balance.
        beta_i = 1e6 * n_i * constants.electron_volt * t_i
        beta_i /= 1e-18 * b_mag**2 / (2 * constants.mu_0)

        # Compute the parallel and perpandicular fluctuating magnetic field
        mb_gsm = pyrf.filt(b_gsm, f_min=0.0, f_max=0.1, order=3)
        db_gsm = pyrf.filt(b_gsm, f_min=0.1, f_max=0.0, order=3)
        mb_gsm = mb_gsm.rename({"represent_vec_tot": "comp"})
        db_gsm = db_gsm.rename({"represent_vec_tot": "comp"})
        db_fac = pyrf.convert_fac(db_gsm, mb_gsm)
        db_para = pyrf.resample(db_fac[:, 2] ** 2, n_i)
        db_perp = pyrf.resample(np.sum(db_fac[:, :2] ** 2, axis=1), n_i)

        # Prepare things for Dataset
        r_gsm = r_gsm.assign_coords({"comp": ["x", "y", "z"]})
        b_gsm = b_gsm.rename({"represent_vec_tot": "comp"})
        b_gsm = b_gsm.assign_coords({"comp": ["x", "y", "z"]})
        v_gsm_i = v_gsm_i.assign_coords({"comp": ["x", "y", "z"]})

        r_gsm.time.attrs = {}
        b_gsm.time.attrs = {}
        db_para.time.attrs = {}
        db_perp.time.attrs = {}
        n_i.time.attrs = {}
        v_gsm_i.time.attrs = {}
        t_para_i.time.attrs = {}
        t_perp_i.time.attrs = {}
        l_.time.attrs = {}
        eh_gsm.time.attrs = {}
        eps_i.time.attrs = {}

        t_aniso = t_perp_i.data / t_para_i.data
        beta_ip = 1e6 * n_i.data * t_para_i.data / (1e-18 * pyrf.norm(b_gsm).data ** 2)
        beta_ip *= constants.electron_volt * 2 * constants.mu_0
        idx_ok = np.logical_not(
            np.logical_or(np.isnan(np.log10(beta_ip)), np.isnan(np.log10(t_aniso)))
        )

        out_dict = {
            "r_gsm": r_gsm[idx_ok],
            "b_gsm": b_gsm[idx_ok],
            "db_para": db_para[idx_ok],
            "db_perp": db_perp[idx_ok],
            "n_i": n_i[idx_ok],
            "v_gsm_i": v_gsm_i[idx_ok],
            "t_para_i": t_para_i[idx_ok],
            "t_perp_i": t_perp_i[idx_ok],
            "lambda_t": l_[idx_ok],
            "eh_gsm": eh_gsm[idx_ok],
            "eps_i": eps_i[idx_ok],
        }

        if i > 0:
            for k in out_dict:
                out_dict[k] = pyrf.ts_append(brazil[k], out_dict[k])

        _, idxs = np.unique(out_dict["n_i"].time.data, return_index=True)
        for k in out_dict:
            out_dict[k] = out_dict[k][idxs]

        brazil = xr.Dataset(out_dict)
        brazil.to_netcdf(os.path.join(os.getcwd(), "data", f_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Calculate the temperature anisotropy, non-Maxwellianity, etc. in the "
        "dataset of reconnection jets."
    )
    parser.add_argument(
        "--start",
        "-s",
        help="Index of the first time interval to compute",
        default=0,
        type=int,
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
