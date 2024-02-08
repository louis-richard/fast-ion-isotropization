#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import itertools

# 3rd party imports
import numpy as np
import xarray as xr
from pyrfu.pyrf import (histogram2d, optimize_nbins_2d, ts_scalar, ts_skymap,
                        ts_tensor_xyz, ts_vec_xyz)
from scipy import constants

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2022"
__license__ = "Apache 2.0"

__all__ = [
    "thresh",
    "conditional_avg",
    "percentiles",
    "histogram2d_linlog",
    "histogram2d_loglog",
    "remove_idist_background_vdf",
    "average_mom",
    "average_vdf",
]


def thresh(beta, a, b, beta0):
    r"""Compute the threshold using the empirical model defined in [1]_.

    .. math::

        R_i = T_{i\perp} / T_{i\parallel} = 1 + a / (\beta_{i\parallel} -\beta_0)^b


    Parameters
    ----------
    beta : xarray.DataArray or numpy.ndarray
        Time series or array of parallel ion plasma beta.
    a : float
        Coefficient from fit.
    b : float
        Coefficient from fit.
    beta0 : float
        Coefficient from fit.


    Returns
    -------
    r_i_thresh : xarray.DataArray
        Time series or array of threshold temperature anisotropy at the given beta.


    References
    ----------
    .. [1]  Hellinger, P., P. Travnicek, J. C. Kasper, and A. J. Lazarus (2006),
            Solar wind proton temperature anisotropy: Linear theory and WIND/SWE
            observations, Geophys. Res. Lett., 33, L09101, doi:10.1029/2006GL025925.

    """

    # Discard negative values from the denominator
    beta[beta < beta0] = np.nan

    # Compute threshold
    r_i_thresh = 1 + a / (beta - beta0) ** b

    return r_i_thresh


def conditional_avg(inp, t_aniso, beta_para, n):
    r"""Compute the conditional average of the variable `inp` in the (`beta_para`,
    `t_aniso`) space given the grid defined by the histogram `n`.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of values of the parameter to conditional average.
    t_aniso : xarray.DataArray
        Time series of the temperature anisotropy at the same times as the `inp`
    beta_para : xarray.DataArray
        Time series of the parallel ion plasma beta at the same times as the `inp`

    Returns
    -------
    h_m_darr : xarray.DataArray
        Conditional average of the `inp` onto the grid defined in `n`.
    h_s_darr : xarray.DataArray
        Conditional standard deviation of the `inp` onto the grid defined in `n`.

    """

    t_aniso_data_log = np.log10(t_aniso.data)
    beta_para_data_log = np.log10(beta_para.data)

    inp_data = inp.data

    # Define the bins edges from the 2d histogram bin centers defined in n
    x_edges = np.hstack(
        [
            np.log10(n.x_bins.data),
            np.log10(n.x_bins.data[-1]) + np.diff(np.log10(n.x_bins.data))[-1],
        ]
    )
    y_edges = np.hstack(
        [
            np.log10(n.y_bins.data),
            np.log10(n.y_bins.data[-1]) + np.diff(np.log10(n.y_bins.data))[-1],
        ]
    )

    h_m, h_s = [np.zeros(n.shape) for _ in range(2)]

    for j, i in itertools.product(range(len(n.x_bins)), range(len(n.y_bins))):
        if n.data[j, i] >= 25:
            # Indices of measurements with temperature anisotropy and parallel ion
            # plasma beta in the range of the (j, i) bin.
            idx_y = np.logical_and(
                t_aniso_data_log > y_edges[i],
                t_aniso_data_log < y_edges[i + 1],
            )
            idx_x = np.logical_and(
                beta_para_data_log > x_edges[j],
                beta_para_data_log < x_edges[j + 1],
            )
            idx_a = np.logical_and(idx_x, idx_y)

            # Compute the conditional average and standard deviation
            h_m[j, i] = np.nanmean(inp_data[idx_a])
            h_s[j, i] = np.nanstd(inp_data[idx_a])

        else:
            # Eliminate bins with counts below 5 \sigma (fill with nan)
            h_m[j, i] = np.nan
            h_s[j, i] = np.nan

    h_m_darr, h_s_darr = [n.copy() for _ in range(2)]
    h_s_darr.data = h_s
    h_m_darr.data = h_m

    return h_m_darr, h_s_darr


def percentiles(x_data, y_data, hist):
    r"""Compute the 10th, 50th and 90th percentiles of the `y_data` according to
    `x_data` binned on the grid defined in `hist`.

    Parameters
    ----------
    x_data : xarray.DataArray
        Time series of variable to bin against.
    y_data : xarray.DataArray
        Times series of the variable to compute the percentiles

    Returns
    -------
    ten_ : xarray.DataArray


    """

    ten_, fif_, nth_ = [np.zeros(len(hist.x_bins.data)) for _ in range(3)]
    idx_ = np.digitize(x_data, hist.x_bins.data)
    for i in range(len(hist.x_bins.data)):
        if np.sum(idx_ == i) != 0:
            perc_ = np.percentile(y_data[idx_ == i], [10, 50, 90])
            ten_[i], fif_[i], nth_[i] = perc_

    ten_[ten_ == 0] = np.nan
    fif_[fif_ == 0] = np.nan
    nth_[nth_ == 0] = np.nan

    ten_ = xr.DataArray(ten_, coords=[hist.x_bins.data], dims=["x_bins"])
    fif_ = xr.DataArray(fif_, coords=[hist.x_bins.data], dims=["x_bins"])
    nth_ = xr.DataArray(nth_, coords=[hist.x_bins.data], dims=["x_bins"])

    return ten_, fif_, nth_


def histogram2d_linlog(x, y, density, bins=None):
    r"""Creates a 2d linear-logarithmic histogram.

    :param x:
    :param y:
    :param density:
    :return:
    """

    x_lin = x
    y_log = np.log10(y)
    idx_n = np.logical_or(np.isnan(x_lin), np.isnan(y_log))
    idx_i = np.logical_or(np.isinf(x_lin), np.isinf(y_log))
    idx_n = np.logical_or(idx_n, idx_i)
    x_lin = x_lin[~idx_n]
    y_log = y_log[~idx_n]

    if bins is None:
        bins = optimize_nbins_2d(x_lin, y_log)
        print(bins)

    h = histogram2d(x_lin, y_log, bins=bins, density=density)
    h = h.assign_coords({"x_bins": h.x_bins.data, "y_bins": 10**h.y_bins.data})
    h.data[h.data == 0] = np.nan

    return h


def histogram2d_loglog(x, y, density, bins=None):
    r"""Creates a 2d logarithmic-logarithmic histogram.

    :param x:
    :param y:
    :param density:
    :return:
    """

    x_log = np.log10(x)
    y_log = np.log10(y)
    idx_n = np.logical_or(np.isnan(x_log), np.isnan(y_log))
    idx_i = np.logical_or(np.isinf(x_log), np.isinf(y_log))
    idx_n = np.logical_or(idx_n, idx_i)
    x_log = x_log[~idx_n]
    y_log = y_log[~idx_n]

    if bins is None:
        bins = optimize_nbins_2d(x_log, y_log)
        print(bins)

    h = histogram2d(x_log, y_log, bins=bins, density=density)
    h = h.assign_coords({"x_bins": 10**h.x_bins.data, "y_bins": 10**h.y_bins.data})
    h.data[h.data == 0] = np.nan

    return h


def remove_idist_background_vdf(vdf, def_bg):
    r"""Remove the estimated background `def_bg` from the ion velocity distribution
    function `vdf`.

    Parameters
    ----------
    vdf : xarray.Dataset
        Ion velocity distribution function.
    def_bg : xarray.DataArray
        Omni-directional ion differential energy flux.

    Returns
    -------
    vdf_new : xarray.Dataset
        Ion velocity distribution function cleaned.

    """

    def_bg_tmp = np.tile(def_bg, (vdf.energy.shape[1], 1))
    def_bg_tmp = np.transpose(def_bg_tmp, (1, 0))

    coeff = constants.proton_mass / (constants.elementary_charge * vdf.energy.data)
    vdf_bg = def_bg_tmp.copy() * 1e4 / 2
    vdf_bg *= coeff**2
    vdf_bg /= 1e12
    vdf_bg = np.tile(vdf_bg, (vdf.phi.shape[1], vdf.theta.shape[0], 1, 1))
    vdf_bg = np.transpose(vdf_bg, (2, 3, 0, 1))

    vdf_new = vdf.copy()
    vdf_new.data.data -= vdf_bg.data

    return vdf_new


def average_mom(n, v_xyz, t_xyz, n_pts):
    r"""Time averages the plasma moments over `n_pts` in time.

    Parameters
    ----------
    n : xarray.DataArray
        Time series of the number density.
    v_xyz : xarray.DataArray
        Time series of the bulk velocity.
    t_xyz : xarray.DataArray
        Time series of the temperature tensor.
    n_pts : int
        Number of points (samples) of the averaging window.

    Returns
    -------
    n_avg : xarray.DataArray
        Time series of the time averaged number density
    v_xyz_avg : xarray.DataArray
        Time series of the time averaged bulk velocity.
    t_xyz_avg : xarray.DataArray
        Time series of the time averaged temperature tensor

    """

    assert n_pts % 2 != 0, "The number of distributions to be averaged must be an odd"

    n_mom = len(n.time.data)
    times = n.time.data

    pad_value = np.floor(n_pts / 2)
    avg_inds = np.arange(pad_value, n_mom - pad_value, n_pts, dtype=int)
    time_avg = times[avg_inds]

    n_avg = np.zeros((len(avg_inds),))
    v_xyz_avg = np.zeros((len(avg_inds), 3))
    t_xyz_avg = np.zeros((len(avg_inds), 3, 3))

    for i, avg_ind in enumerate(avg_inds):
        l_bound = int(avg_ind - pad_value)
        r_bound = int(avg_ind + pad_value)
        n_avg[i, ...] = np.mean(n.data[l_bound:r_bound], axis=0)
        v_xyz_avg[i, ...] = np.mean(v_xyz.data[l_bound:r_bound, :], axis=0)
        t_xyz_avg[i, ...] = np.mean(t_xyz.data[l_bound:r_bound, ...], axis=0)

    n_avg = ts_scalar(time_avg, n_avg)
    v_xyz_avg = ts_vec_xyz(time_avg, v_xyz_avg)
    t_xyz_avg = ts_tensor_xyz(time_avg, t_xyz_avg)

    return n_avg, v_xyz_avg, t_xyz_avg


def average_vdf(vdf, n_pts):
    r"""Time averages the velocity distribution functions over `n_pts` in time.

    Parameters
    ----------
    vdf : xarray.DataArray
        Time series of the velocity distribution function.
    n_pts : int
        Number of points (samples) of the averaging window.

    Returns
    -------
    vdf_avg : xarray.DataArray
        Time series of the time averaged velocity distribution function.

    """

    assert n_pts % 2 != 0, "The number of distributions to be averaged must be an odd"

    assert np.median(vdf.energy.data[0, :] - vdf.energy.data[0, :]) == 0

    n_vdf = len(vdf.time.data)
    times = vdf.time.data

    pad_value = np.floor(n_pts / 2)
    avg_inds = np.arange(pad_value, n_vdf - pad_value, n_pts, dtype=int)
    time_avg = times[avg_inds]

    energy_avg = np.zeros((len(avg_inds), vdf.data.shape[1]))
    phi_avg = np.zeros((len(avg_inds), vdf.data.shape[2]))
    vdf_avg = np.zeros((len(avg_inds), *vdf.data.shape[1:]))

    for i, avg_ind in enumerate(avg_inds):
        l_bound = int(avg_ind - pad_value)
        r_bound = int(avg_ind + pad_value)
        vdf_avg[i, ...] = np.nanmean(vdf.data.data[l_bound:r_bound, ...], axis=0)
        energy_avg[i, ...] = np.nanmean(vdf.energy.data[l_bound:r_bound, ...], axis=0)
        phi_avg[i, ...] = np.nanmean(vdf.phi.data[l_bound:r_bound, ...], axis=0)

    coords_attrs = {k: vdf[k].attrs for k in ["time", "energy", "phi", "theta"]}
    vdf_avg = ts_skymap(
        time_avg,
        vdf_avg,
        energy_avg,
        phi_avg,
        vdf.theta.data,
        attrs=vdf.data.attrs,
        glob_attrs=vdf.attrs,
        coords_attrs=coords_attrs,
    )
    vdf_avg.attrs["energy0"] = vdf.attrs["energy0"]
    vdf_avg.attrs["energy1"] = vdf.attrs["energy1"]
    vdf_avg.attrs["esteptable"] = vdf.attrs["esteptable"][: len(avg_inds)]
    vdf_avg.attrs["delta_energy_minus"] = vdf.attrs["delta_energy_minus"][avg_inds]
    vdf_avg.attrs["delta_energy_plus"] = vdf.attrs["delta_energy_plus"][avg_inds]

    return vdf_avg
