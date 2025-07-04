#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np

# ----- constants ----- #
unit_mass = 1.661e-27
unit_charge = 1.602e-19

# ----- input data ----- #
file_names = [
    "image-data.tiff"
]

origin = [700, 708]  
res = 25e-6  # This value is for imaging plate
flip_parabola_flag = True
exclude_r = 100  # [pixel]

# ----- ion parameters to analyse ----- #
ion_list = [
            ["proton", unit_mass * 1.0073, unit_charge, "blue"],
            ["C6+(O8+)", unit_mass * 12.011, unit_charge * 6, "red"],
            ["alpha", unit_mass * 4, unit_charge * 2, "green"],
            ["C5+",      unit_mass * 12.011, unit_charge * 5, "red"],
            ["C4+",      unit_mass * 12.011, unit_charge * 4, "red"],
            ["C3+(O4+)", unit_mass * 12.011, unit_charge * 3, "red"],
            ["C2+",      unit_mass * 12.011, unit_charge * 2, "red"],
            ["C1+",      unit_mass * 12.011, unit_charge * 1, "red"],
            ["O7+", unit_mass * 15.999, unit_charge * 7, "red"],
            ["O6+", unit_mass * 15.999, unit_charge * 6, "red"],
            ["O5+", unit_mass * 15.999, unit_charge * 5, "red"],
            ["O3+", unit_mass * 15.999, unit_charge * 3, "red"],
            ["O2+", unit_mass * 15.999, unit_charge * 2, "red"],
            ["O1+", unit_mass * 15.999, unit_charge * 1, "red"],
            ]

#for n in range(25, 38):
#    ion_list.append(["Au{}+".format(n), unit_mass * 197, unit_charge * n, "green"])

# ----- TP parameters ----- #
# use IS unit (m, V, T)
L = 0.1
D = 0.100
B = 0.2
pinhole_d = 300e-6

V = 2000
d = 4.5e-3

separate_tp = False
if separate_tp:
    Le = 300e-3
    De = 300e-3

# ----- analysis parameters ----- #
analysis_width = 0
energy_log_flag = False
if energy_log_flag:
    # input in MeV
    ek_min = 10
    ek_max = 50
    ek_bin_num = 100

# ----- background analysis ----- #
bg_flag = True
if bg_flag:
    bg_analysis_width = 10
    bg_color = "scan"

# ----- fitting ----- #
refer_log_flag = False
fit_flag = True
if fit_flag:
    fit_cache_flag = True
    img_threshold = 50000 

    top = [100, 100]
    bottom = [150, 100]
    fit_ion = ["proton", unit_mass * 1, unit_charge * 1, "black"]
else:
    wofit_cache_flag = False
    coefficient = 2.89   # [1/m]
    angle = 1.7e-3 * np.pi / 180   # [rad]

# ----- scanning Q/M ----- #
scan_qm_flag = False
use_cache_qm_flag = False
qm_log_flag = False

qm_indication_flag = True
if qm_indication_flag:
    qm_indication = [
#        {"mass":12, "charge":6}, 
#        {"mass":16, "charge":7}, 
    ]

# ----- output ----- #
dpi = 100
raw_color = "viridis"
save_fig_ext = "png"
draw_para_width = 10
origin_color = "green"
origin_size = 3
save_raw_parabola = False
no_spectra_flag = False
img_scale = ["linear", 1, 1e4]

decimate_image = False
if decimate_image:
    cut_left_top = [50, 50]  
    cut_right_down = [950, 950]  

# ----- drawing on parabola ----- #
scale_bar_flag = True
if scale_bar_flag:
    scale_bar = {"length": 10e-3,
                 "position": [600, 600],  
                 "color": "black",
                 "fontsize": 30,
                 "width": 2,
                 "rotation": "horizontal",
                 }

put_dot_flag = False
if put_dot_flag:
    dot_position = [[125, 100],
                    [300, 300],
                    ]

ek_axis_on_parabola = True
if ek_axis_on_parabola:
    number_of_axis = 1
    ek_indicate = True
    ek_indicate_offset = [30, 0]
    ek_indicate_fontsize = 9
    ek_indicate_origin_offset = [0, 20]
    ek_axis_1 = {"ek_mev": [0.4, 1, 5, 20],  # MeV (not MeV/u)
                 "mass": unit_mass * 1,
                 "charge": unit_charge * 1,
                 "color": "red",
                 "origin": [500, origin[1]],
                 "linewidth":2,
                 "name":"proton",
                 }
    ek_axis_2 = {"ek_mev": [1/2, 5/2, 10/2],  # MeV (not MeV/u)
                 "mass": unit_mass * 2,
                 "charge": unit_charge * 1,
                 "color": "purple",
                 "origin": [400, origin[1]],
                 "linewidth":2,
                 }
    ek_axis_3 = {"ek_mev": [1, 5, 10],  # MeV (not MeV/u)
                 "mass": unit_mass * 4,
                 "charge": unit_charge * 2,
                 "color": "green",
                 "origin": [300, origin[1]],
                 "linewidth":2,
                 }
    ek_axis_4 = {"ek_mev": [1, 5, 10],  # MeV (not MeV/u)
                 "mass": unit_mass * 12,
                 "charge": unit_charge * 6,
                 "color": "blue",
                 "origin": [200, origin[1]],
                 "linewidth":2,
                 }
