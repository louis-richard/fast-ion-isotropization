#!/usr/bin/env python
# -*- coding: utf-8 -*-


# 3rd party imports
from pyrfu import mms, pyrf
from scipy import constants

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2022"
__license__ = "Apache 2.0"

__all__ = ["load_fpi", "load_fpi_dis_mmsx", "load_efield_mmsx", "load_rsc_bfield_mmsx"]


def load_fpi(tint, mms_id, cs: str = "dbcs"):
    r"""Load FPI-DIS moments and correct the moments removing the penetrating
    radiations.

    Parameters
    ----------
    tint : list
        Time interval.
    mms_id : int
        Spacecraft index.

    Returns
    -------
    n_i : xarray.DataArray
        Time series of the ion number density.
    v_dbcs_i : xarray.DataArray
        Time series of the ion bulk velocity in spacecraft coordinates system.
    t_dbcs_i : xarray.DataArray
        Time series of the ion temperature tensor in spacecraft coordinates system.

    """

    n_i = mms.get_data("ni_fpi_brst_l2", tint, mms_id)
    v_dbcs_i = mms.get_data(f"vi_{cs}_fpi_brst_l2", tint, mms_id)
    t_dbcs_i = mms.get_data(f"ti_{cs}_fpi_brst_l2", tint, mms_id)

    # Split partial momemnts

    p_dbcs_i = n_i.data[:, None, None] * t_dbcs_i
    p_dbcs_i.data *= 1e15 * constants.elementary_charge

    # Background radiation
    nbg_i = mms.get_data("nbgi_fpi_brst_l2", tint, mms_id)
    pbg_i = mms.get_data("pbgi_fpi_brst_l2", tint, mms_id)

    # Remove penetrating radiations
    moms_clean = mms.remove_imoms_background(n_i, v_dbcs_i, p_dbcs_i, nbg_i, pbg_i)
    n_i_clean, v_dbcs_i_clean, p_dbcs_i_clean = moms_clean
    t_dbcs_i_clean = p_dbcs_i_clean / (1e15 * constants.elementary_charge)
    t_dbcs_i_clean.data /= n_i_clean.data[:, None, None]

    # Remove extremely low density points
    v_dbcs_i = v_dbcs_i_clean[n_i_clean > 0.005, ...]
    t_dbcs_i = t_dbcs_i_clean[n_i_clean > 0.005]
    n_i = n_i_clean[n_i_clean > 0.005]

    return n_i, v_dbcs_i, t_dbcs_i


def load_fpi_dis_mmsx(tint):
    r"""Load FPI-DIS moments and velocity distribution functions and correct
    them removing the penetrating radiations.

    Parameters
    ----------
    tint : list
        Time interval.

    Returns
    -------
    n_i : xarray.DataArray
        Time series of the ion number density.
    v_dbcs_i : xarray.DataArray
        Time series of the ion bulk velocity in spacecraft coordinates system.
    v_gsm_i : xarray.DataArray
        Time series of the ion bulk velocity in GSM coordinates system.
    t_dbcs_i : xarray.DataArray
        Time series of the ion temperature tensor in spacecraft coordinates system.
    vdf_i_new : xarray.DataArray
        Time series of the ion distribution function.

    """

    n_i_mms, v_dbcs_i_mms, t_dbcs_i_mms, v_gse_i_mms, t_gse_i_mms = [], [], [], [], []
    vdf_i_new_mms = []

    for mms_id in range(1, 5):
        try:
            n_i, v_dbcs_i, t_dbcs_i = load_fpi(tint, mms_id, "dbcs")
            _, v_gse_i, t_gse_i = load_fpi(tint, mms_id, "gse")
            vdf_i = mms.get_data("pdi_fpi_brst_l2", tint, mms_id)
            def_i_bg = mms.get_data("defbgi_fpi_brst_l2", tint, mms_id)
            vdf_i_new = mms.remove_idist_background(vdf_i, def_i_bg)
            n_i_mms.append(n_i)
            v_dbcs_i_mms.append(v_dbcs_i)
            v_gse_i_mms.append(v_gse_i)
            t_dbcs_i_mms.append(t_dbcs_i)
            t_gse_i_mms.append(t_gse_i)
            vdf_i_new_mms.append(vdf_i_new)
        except (AttributeError, AssertionError):
            continue

    if len(n_i_mms) == 0:
        return None, None, None, None, None, None

    n_i = pyrf.avg_4sc(n_i_mms)
    v_dbcs_i = pyrf.avg_4sc(v_dbcs_i_mms)
    v_gse_i = pyrf.avg_4sc(v_gse_i_mms)
    t_dbcs_i = pyrf.avg_4sc(t_dbcs_i_mms)
    t_gse_i = pyrf.avg_4sc(t_gse_i_mms)

    glob_attrs = vdf_i.attrs
    vdf_attrs = vdf_i.data.attrs
    coords_attrs = {k: vdf_i[k].attrs for k in ["time", "energy", "phi", "theta"]}

    vdf_i_new = pyrf.avg_4sc(vdf_i_new_mms)
    vdf_i_new = pyrf.ts_skymap(
        vdf_i_new.time.data,
        vdf_i_new.data.data,
        vdf_i_new_mms[0].energy.data,
        vdf_i_new_mms[0].phi.data,
        vdf_i_new_mms[0].theta.data,
        esteptable=vdf_i_new_mms[0].attrs["esteptable"],
        energy0=vdf_i_new_mms[0].attrs["energy0"],
        energy1=vdf_i_new_mms[0].attrs["energy1"],
        attrs=vdf_attrs,
        glob_attrs=glob_attrs,
        coords_attrs=coords_attrs,
    )

    vdf_i_new.attrs = vdf_i_new_mms[0].attrs

    n_i = pyrf.resample(n_i, vdf_i_new.time)
    v_dbcs_i = pyrf.resample(v_dbcs_i, vdf_i_new.time)
    v_gse_i = pyrf.resample(v_gse_i, vdf_i_new.time)
    t_dbcs_i = pyrf.resample(t_dbcs_i, vdf_i_new.time)
    t_gse_i = pyrf.resample(t_gse_i, vdf_i_new.time)

    return n_i, v_dbcs_i, v_gse_i, t_dbcs_i, t_gse_i, vdf_i_new


def load_efield_mmsx(tint):
    r"""Load the electric field and the spacecraft potential for all
    spacecraft.

    Parameters
    ----------
    tint : list
        Time interval.

    Returns
    -------
    e_gsm : xarray.DataArray
        Time series of the electric field in GSM coordinates system.
    scpot : xarray.DataArray
        Time series of the spacecraft potential.

    """

    e_gsm_mms, scpot_mms = [], []

    for mms_id in range(1, 5):
        try:
            e_gse = mms.get_data("e_gse_edp_brst_l2", tint, mms_id)
            e_gsm = pyrf.cotrans(e_gse, "gse>gsm")
            scpot = mms.get_data("v_edp_brst_l2", tint, mms_id)
            e_gsm_mms.append(e_gsm)
            scpot_mms.append(scpot)
        except (AttributeError, AssertionError):
            continue

    # Average across the spacecraft
    e_gsm = pyrf.avg_4sc(e_gsm_mms)
    scpot = pyrf.avg_4sc(scpot_mms)

    return e_gsm, scpot


def load_rsc_bfield_mmsx(tint):
    r"""Load the spacraft location and the magnetic field for all spacecraft and
    average across the spacecraft.

    Parameters
    ----------
    tint : list
        Time interval.

    Returns
    -------
    r_gsm : xarray.DataArray
        Time series of the spacecraft location in GSM coordinates system.
    b_dmpa : xarray.DataArray
        Time series of the magnetic field in spacecraft coordinates system.
    b_gsm : xarray.DataArray
        Time series of the magnetic field in GSM coordinates system.

    """

    tint_long = pyrf.extend_tint(tint, [-60, 60])

    r_gsm_mms, b_dmpa_mms, b_gsm_mms = [], [], []

    for mms_id in range(1, 5):
        try:
            r_gsm = mms.get_data("r_gsm_mec_srvy_l2", tint_long, mms_id)
            b_dmpa = mms.get_data("b_dmpa_fgm_brst_l2", tint, mms_id)
            b_gsm = mms.get_data("b_gsm_fgm_brst_l2", tint, mms_id)
            b_dmpa_mms.append(b_dmpa)
            b_gsm_mms.append(b_gsm)
            r_gsm_mms.append(r_gsm)
        except (AttributeError, AssertionError):
            continue

    b_dmpa = pyrf.avg_4sc(b_dmpa_mms)
    b_gsm = pyrf.avg_4sc(b_gsm_mms)
    r_gsm = pyrf.avg_4sc(r_gsm_mms)
    r_gsm = pyrf.time_clip(pyrf.resample(r_gsm, b_gsm), list(tint))

    return r_gsm, b_dmpa, b_gsm
