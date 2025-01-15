#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import numpy as np

from main import input_fname
exec("from {} import L, D, V, B, d, energy_log_flag, fit_flag, save_fig_ext, origin".format(input_fname))


# ----- function ----- #
def try_import_flags(variable):
    try:
        exec("from {} import {}".format(input_fname, variable))
        temp_ = eval(variable)
    except ImportError:
        temp_ = False
    return temp_


def try_import_params(params, _default=[]):
    try:
        exec("from {} import {}".format(input_fname, params))
        _param = eval(params)
    except ImportError:
        _param = _default
    return _param


# ----- prefix ----- #
pn_term = "neg"
save_dir_prefix = "./spectra_"

# ----- constants ----- #
c_0 = 3e8
mass_u = try_import_params("unit_mass", 1.661e-27)
coulomb_u = try_import_params("unit_charge", 1.602e-19)


# ----- input data ----- #
# ----- ion parameters to analyse ----- #
# ----- TP parameters
init_rot_for_fitting = 0 * np.pi / 180  # the rotation angle measured by seeing the image by eyes (maybe used for more precise fitting)
separate_tp_ = try_import_flags("separate_tp")
if separate_tp_:
    Le = try_import_params("Le", L)
    De = try_import_params("De", D)
else:
    Le = L
    De = D

E = V/d
alpha_b = L * (L/2 + D)
alpha_e = Le * (Le/2 + De)
alpha = np.power(alpha_b, 2) / alpha_e

if not fit_flag:
    coefficient = try_import_params("foefficient", mass_u * E / coulomb_u / B**2 / alpha)


# ----- analysis parameters ----- #
if energy_log_flag:
    exec("from {} import ek_min, ek_max".format(input_fname))
    ek_min_j = ek_min * coulomb_u * 1e6
    ek_max_j = ek_max * coulomb_u * 1e6

# ----- background analysis ----- #
# ----- fitting ----- #
cache_flag = try_import_flags("refer_log_flag")
fit_ion_nmqc = try_import_params("fit_ion", ["proton", mass_u, coulomb_u, "scan"])


# ----- scanning Q/M ----- #
scan_QM = try_import_flags("scan_qm_flag")
if scan_QM:
    scan_ion_list = []
    for n in range(1, 201):
        scan_ion_list.append(["Q{}_M200".format(n), mass_u * 200, coulomb_u * n, "scan"])

    indication_QM_flag = try_import_flags("qm_indication_flag")
    if indication_QM_flag:
        indication_QM = try_import_params("qm_indication")
        if len(indication_QM) == 0:
            indication_QM = [
                {"mass":12, "charge":6}, 
                {"mass":12, "charge":5}, 
                {"mass":12, "charge":4}, 
                {"mass":12, "charge":3}, 
                {"mass":12, "charge":2}, 
                {"mass":12, "charge":1}, 
            ]


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

# exec("from {} import flip_parabola_flag".format(input_fname))
# flip_flag = not flip_parabola_flag
flip_flag = not try_import_flags("flip_parabola_flag")

# ----- drawing on parabola ----- #
scale_bar_flag_ = try_import_flags("scale_bar_flag")
if scale_bar_flag_:
#    from input import scale_bar
    scale_bar_default = {"length": 10e-3,
                         "position": [0, 0],
                         "color": "black",
                         "fontsize": 30,
                         "width": 2,
                         "rotation": "horizontal"}
    scale_bar_conf = try_import_params("scale_bar", scale_bar_default)
    if not isinstance(scale_bar_conf, dict):
        print("Input 'scale_bar' as 'dict'!!")
        sys.exit()

ek_axis_on_parabola_flag = try_import_flags("ek_axis_on_parabola")
if ek_axis_on_parabola_flag:
    ek_indicate_flag = try_import_flags("ek_indicate")
    ek_indicate_offset_XY = try_import_params("ek_indicate_offset", [10, 10])
    ek_indicate_fontsize_ = try_import_params("ek_indicate_fontsize", 9)
    ek_indicate_origin_offset_XY = try_import_params("ek_indicate_origin_offset", [0, 30])


