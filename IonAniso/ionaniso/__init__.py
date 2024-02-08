#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .load import load_efield_mmsx, load_fpi, load_fpi_dis_mmsx, load_rsc_bfield_mmsx
from .plot import add_threshold, create_cmap
from .utils import (
    average_mom,
    average_vdf,
    conditional_avg,
    histogram2d_linlog,
    histogram2d_loglog,
    percentiles,
    remove_idist_background_vdf,
    thresh,
)

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2022"
__license__ = "Apache 2.0"

__all__ = [
    "load_fpi",
    "load_fpi_dis_mmsx",
    "load_efield_mmsx",
    "load_rsc_bfield_mmsx",
    "add_threshold",
    "create_cmap",
    "thresh",
    "conditional_avg",
    "percentiles",
    "histogram2d_linlog",
    "histogram2d_loglog",
    "remove_idist_background_vdf",
    "average_mom",
    "average_vdf",
]
