#!/usr/bin/env python3
# -*-coding:utf-8-*-

###########################
###    version 1.0.1    ###
###########################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import platform, re

if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["pcolor.shading"] = "auto"
plt.rcParams["font.size"] = 20
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.major.size"] = 7
plt.rcParams["ytick.major.size"] = 7
plt.rcParams["legend.frameon"] = False
plt.rcParams["legend.fontsize"] = 15

lines = list(matplotlib.lines.lineStyles.keys())   # linestyles
markers = matplotlib.markers.MarkerStyle.filled_markers   # marker-styles
#colors = ['#e41a1c', '#377eb8', '#f781bf', '#4daf4a', '#984ea3', '#a65628', '#999999', '#ff7f00', '#dede00']
#          blue, yellow, red, pink, green, purple, brown, gray, orange   ### for G-R blindness ###
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
#          matplotlib-default color cycle

# ----- parameters ----- #
data_name_list = ["sp_proton.txt",
                  "sp_C6+(O8+).txt",
                 ]


# ----- initialize ----- #

def define_labels(_data_name):
    if re.findall("C6", _data_name):
        _eklabel_ion = "C"
        _fname_out = "carbon6"
    elif re.findall("proton", _data_name):
        _eklabel_ion = "p"
        _fname_out = "proton"
    else:
        _eklabel_ion = "k"
        _fname_out = ""

    return _eklabel_ion, _fname_out


# ----- main ----- #
for n, data_name in enumerate(data_name_list):
    data = np.genfromtxt(data_name, skip_header=1)
    eklabel_ion, fname_out = define_labels(data_name)

    ek = data[:, 0] * 1e-6
    dek = data[:, 1] * 1e-6
    x = data[:, 9] * 1e2
    y = data[:, 10] * 1e2
    r = np.sqrt(np.power(x, 2) + np.power(y, 2))
    dr = (r[1:len(r)] - r[0:len(r)-1])
#dr = np.average(r[1:len(r)] - r[0:len(r)-1])
    count = data[:, 2] * 1e-4
    bg = data[:, 3] * 1e-4
    count_err = data[:, 7] * 1e-4
    bg_err = data[:, 8] * 1e-4
# print("len(ek)={}".format(len(ek)))
# print("len(dr)={}".format(len(dr)))
# print("len(r[:-1])={}".format(len(r[:-1])))

    fig, ax = plt.subplots(figsize=(7.4, 4.8))   # default size (6.4, 4,8)
    ax.bar(r[:-1], count[:-1], width=dr, alpha=0.5, label=fname_out, color=colors[0])
    ax.bar(r[:-1], bg[:-1], width=dr, alpha=0.5, label="background", color=colors[1])
    ax.errorbar(r[:-1], count[:-1], yerr=count_err[:-1], linewidth=0, markersize=2, capsize=1, elinewidth=1, color=colors[0], marker=markers[0])
    ax.errorbar(r[:-1], bg[:-1], yerr=bg_err[:-1], linewidth=0, markersize=2, capsize=1, elinewidth=1, color=colors[1], marker=markers[0])
    ax.set_xlabel(r"$r$ [cm]")
    ax.set_ylabel(r"count [$\times$10$^4$]")
    ax.set_xlim(0, 8)
    ax.set_yscale("log")
    ax2 = ax.twinx()
    ax2.errorbar(r, ek, linewidth=1, markersize=0, color=colors[2], label="$E_{" + eklabel_ion + "}$")
    ax2.set_yscale("log")
    ax2.set_ylabel("$E_{" + eklabel_ion + "}$ [MeV]")
#ax2.hlines(30, 0, 8, linestyle="--", linewidth=1, color="black")
    fig.legend(bbox_to_anchor=(0.9,1), loc="upper right", borderaxespad=4)
    fig.tight_layout()
    fig.savefig("r-{}.png".format(fname_out))
    ax.cla()
    ax2.cla()
    fig.clf()
    plt.close()

