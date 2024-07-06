#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import sys

from input import L, D, V, B, d, energy_log_flag, fit_flag, save_fig_ext
import numpy as np


# ----- function ----- #
def try_import_flags(variable):
    try:
        exec("from input import {}".format(variable))
        temp_ = eval(variable)
    except ImportError:
        temp_ = False
    return temp_

def try_import_params(params):
    try:
        exec("from input import {}".format(params))
        _param = eval(params)
    except ImportError:
        _param = []
    return _param


# ----- prefix ----- #
pn_term = "neg"
save_dir_prefix = "./spectra_"

# ----- constants ----- #
c_0 = 3e8
try:
    from input import unit_mass, unit_charge
    mass_u = unit_mass
    coulomb_u = unit_charge
except ImportError:
    mass_u = 1.661e-27
    coulomb_u = 1.602e-19

# ----- input data ----- #
# ----- ion parameters to analyse ----- #
# ----- TP parameters
init_rot_for_fitting = 0 * np.pi / 180  # the rotation angle measured by seeing the image by eyes (maybe used for more precise fitting)
try:
    from input import separate_tp
    if separate_tp:
        from input import Le, De
    else:
        Le = L
        De = D
except ImportError:
    Le = L
    De = D

E = V/d
alpha_b = L * (L/2 + D)
alpha_e = Le * (Le/2 + De)
alpha = np.power(alpha_b, 2) / alpha_e

if not fit_flag:
    try:
        from input import coefficient
    except ImportError:
        coefficient = mass_u * E / coulomb_u / B**2 / alpha
    # coefficient = 2.64

# ----- analysis parameters ----- #
if energy_log_flag:
    from input import ek_min, ek_max
    ek_min_j = ek_min * coulomb_u * 1e6
    ek_max_j = ek_max * coulomb_u * 1e6

# ----- background analysis ----- #
# ----- fitting ----- #
cache_flag = try_import_flags("refer_log_flag")
try:
    from input import fit_ion as fit_ion_nmqc
except ImportError:
    fit_ion_nmqc = ["proton", unit_mass, unit_charge, "scan"]

# ----- scanning Q/M ----- #
scan_QM = try_import_flags("scan_qm_flag")
if scan_QM:
    scan_ion_list = []
    for n in range(1, 201):
        scan_ion_list.append(["Q{}_M200".format(n), mass_u * 200, coulomb_u * n, "scan"])

# ----- output ----- #
# save_spectra_ = try_import_flags("save_spectra")
save_raw_parabola_ = try_import_flags("save_raw_parabola")
no_spectra_flag = try_import_flags("no_spectra_flag")
img_scale_params = try_import_params("img_scale")

if isinstance(save_fig_ext, str):
    parabola_ext = ["{}".format(save_fig_ext)]
elif isinstance(save_fig_ext, list):
    parabola_ext = save_fig_ext
else:
    print("Error!! Check the variale 'save_fig_ext' in input. Its type should be 'str' or 'list'.")
    sys.exit()

from input import flip_parabola_flag
flip_flag = not flip_parabola_flag

# ----- drawing on parabola ----- #
scale_bar_flag_ = try_import_flags("scale_bar_flag")
if scale_bar_flag_:
    from input import scale_bar
    if not isinstance(scale_bar, dict):
        print("Input 'scale_bar' as 'dict'!!")
        sys.exit()







