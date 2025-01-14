#!/usr/bin/env python3
# -*-coding:utf-8-*-

import matplotlib.pyplot as plt
import numpy as np
import glob, os, re, sys
import cv2

sys.path.append("../")
import platform

if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["pcolor.shading"] = "auto"
plt.rcParams["font.size"] = 20
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.major.size"] = 7
plt.rcParams["ytick.major.size"] = 7
plt.rcParams["legend.frameon"] = False


strdn = 1  # 3.14e-8
# ~1.5m, ~300um

ion_names = [
             "proton",
             "C6+(O8+)",
             ]

original_file_extension = "gel"

# ----- initialization ----- #
from main import input_fname
exec("from {} import decimate_image".format(input_fname))
if decimate_image:
    exec("from {} import cut_left_top, cut_right_down".format(input_fname))

cur_dir = os.getcwd()
file_list = ["sp_" + ions + ".txt" for ions in ion_names]

ekmax = open("./max_ek.log", mode="w")


# ----- functions ----- #
def find_original():
    original_fname = cur_dir.split("/")[-1][8:]
#    if "]" == original_fname[-1]:
#        original_fname = original_fname[:-1]

#    original = glob.glob("../{}*".format(original_fname))
#    if (1 < len(original)):
#        for fn in original:
#            if fn[-4:] == "." + original_file_extension:
#                original = fn
#                break
    _original = "../" + original_fname + "." + original_file_extension

    return _original


# ----- main ----- #

original = find_original()

for n, f in enumerate(file_list):
    f_name = os.path.splitext(os.path.basename(f))[0]
    ion_name = f_name[3:]
    data = np.genfromtxt(f, skip_header=1)

    if ion_name == "proton":
        ek_max = 30   # maximum but slightly below the cutoff
        # limit = 30
        ek_min = 0.8   # this should be exactly minimum energy to be normalized 
    elif ion_name == "C6+(O8+)":
        ek_max = 60
        # limit = 30
        ek_min = 12

    ek = data[:, 0] * 1e-6
    dek = data[:, 1] * 1e-6
    count = data[:, 2]  # []
    bg = data[:, 3]  # []
    ek_err = data[:, 6] * 1e-6
    fn_err = np.sqrt(np.power(data[:, 7], 2) + np.power(data[:, 8], 2))

    ek_range = (ek > ek_min) & ((count - bg - fn_err) > 0)
    signal = count - bg
    for place in range(len(ek)):
        k = place + 1
        # if (signal[-k] < fn_err[-k]) & (place > limit):
        if (signal[-k] < fn_err[-k]) & (ek[-k] > ek_max):
            break

    p = len(ek) - place
    print(f_name)
    print("max {:.3f} [MeV], err {:.6f} [MeV]".format(ek[p + 1], ek_err[p + 1]))
    stop = p + 1
    ek_range[:stop] = False

    ekmax.write(f_name + "\n")
    ekmax.write("{:.3f}, {:.5f}".format(ek[p+1], ek_err[p+1]) + "\n")

    norm_value = np.sum(signal[ek_range])  # normalized value
    fn_norm = signal / norm_value / dek / strdn  # [/MeV/sr]
    fn_err_norm = fn_err / norm_value / dek / strdn  # [/MeV/sr]

    save_data = np.empty((len(ek[ek_range]), 4))
    save_data[:, 0] = ek[ek_range]
    save_data[:, 1] = fn_norm[ek_range]
    save_data[:, 2] = ek_err[ek_range]
    save_data[:, 3] = fn_err_norm[ek_range]

    #    save_data = np.array((ek[stop:], fn_norm, ek_err[stop:], fn_err_norm[stop:])).T
    np.savetxt("fn_{}.txt".format(ion_name), save_data, header="Ek [MeV]  fn [/MeV]  ek_err [MeV]  fn_err [/MeV]")

    fig, ax = plt.subplots()
#    ax.errorbar(save_data[:, 0], save_data[:, 1], xerr=save_data[:, 2], yerr=save_data[:, 3], capsize=1.8, elinewidth=0.5)
    ax.errorbar(save_data[:, 0], save_data[:, 1], yerr=save_data[:, 3], capsize=1.8, elinewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("$E_k$ [MeV]")
    if strdn == 1:
        ax.set_ylabel("$f\ (E_k)$ [/MeV]")
    else:
        ax.set_ylabel("$f\ (E_k)$ [/MeV/sr]")
    fig.tight_layout()
    fig.savefig("fn_{}.png".format(ion_name))
    ax.cla()

    if not f_name[0] == "3":
        x_ = data[:, 4][ek_range]
        y_ = data[:, 5][ek_range]
        x_not = data[:, 4][~ek_range]
        y_not = data[:, 5][~ek_range]
        image = cv2.imread(original, cv2.IMREAD_UNCHANGED)
        fig, ax = plt.subplots()
        ax.imshow(image, cmap="Reds")
        ax.scatter(x_ - 10, y_, s=1, c="green", marker=".", label="normalized area")
        ax.scatter(x_not - 5, y_not, s=1, c="blue", marker=".", label="eliminated area")
        ax.tick_params(bottom=False, left=False, right=False, top=False)
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        if decimate_image:
            ax.set_xlim(cut_left_top[0], cut_right_down[0])
            ax.set_ylim(cut_right_down[1], cut_left_top[1])
        ax.legend(frameon=True, fontsize=5)
        fig.savefig("chk_{}.pdf".format(f_name), bbox_inches="tight", pad_inches=0)
        ax.cla()
        fig.clf()

ekmax.close()

