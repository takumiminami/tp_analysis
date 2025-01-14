#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from numpy import log10
import tqdm, datetime, sys, os, platform

from configurations import coulomb_u, mass_u, flip_flag #, save_spectra_
from functions import ek_to_x, x_to_ek, calc_ek_stddev
from image import Image
from main import input_fname
exec("from {} import *".format(input_fname))

if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["pcolor.shading"] = "auto"
plt.rcParams["font.size"] = 15
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.major.size"] = 7
plt.rcParams["ytick.major.size"] = 7
plt.rcParams["legend.frameon"] = False


class Spectra:
    """
    1st axis is horizontal axis
    2nd axis is vertical axis
    origin is at left-bottom
    y = ax^2
    [x, y]
    """

    def __init__(self, name_, mass_, charge_, coef_, color_):
        self.name = name_
        self.mass = mass_
        self.charge = charge_
        self.color = color_
        self.energy, self.xy = self.calc_parabola(coef_)
        self.ek_mid_error = np.zeros(len(self.xy) - 1)
        self.value = np.zeros(len(self.xy) - 1)
        self.bg_value = np.zeros(len(self.xy) - 1)
        self.value_error = np.zeros(len(self.xy) - 1)
        self.bg_value_error = np.zeros(len(self.xy) - 1)
        self.XY = np.zeros(len(self.xy))
        self.len = len(self.energy)

    def calc_parabola(self, coef__):
        if energy_log_flag:
            max_ek = x_to_ek([pinhole_d], self.mass, self.charge)
            from configurations import ek_max_j, ek_min_j
            ek = 10 ** np.linspace(log10(ek_max_j), log10(ek_min_j), ek_bin_num)
            if max_ek < ek[0]:
                print("Error")
                print("Maximum energy to analyze should be smaller than {} [MeV]".format(max_ek / coulomb_u * 1e-6))
                sys.exit()
            x = ek_to_x(ek, self.mass, self.charge)
        else:
            x = np.arange(pinhole_d, origin[1] * res, pinhole_d)
            ek = x_to_ek(x, self.mass, self.charge)

        if flip_flag:
            coef_ = -coef__ * (self.mass / mass_u) * (coulomb_u / self.charge)
        else:
            coef_ = coef__ * (self.mass / mass_u) * (coulomb_u / self.charge)

        xy = np.array((x, coef_ * np.power(x, 2))).T

        return ek, xy

    def save_result(self, save_dir):
        ek_mid_ev = (self.energy[1:self.len] + self.energy[0:self.len - 1]) / 2 / coulomb_u
        dek_ev = (self.energy[0:self.len - 1] - self.energy[1:self.len]) / coulomb_u
        ek_mid_error_ev = self.ek_mid_error / coulomb_u

        X_mid = (self.XY[1:self.len, 0] + self.XY[0:self.len-1, 0]) / 2
        Y_mid = (self.XY[1:self.len, 1] + self.XY[0:self.len-1, 1]) / 2

        x_mid = (self.xy[1:self.len, 0] + self.xy[0:self.len-1, 0]) / 2
        y_mid = (self.xy[1:self.len, 1] + self.xy[0:self.len-1, 1]) / 2

        print("saving fn of {}".format(self.name))
        if bg_flag:
            np.savetxt("{}/sp_{}.txt".format(save_dir, self.name),
                       np.array((ek_mid_ev, dek_ev, self.value, self.bg_value, Y_mid, X_mid, ek_mid_error_ev,
                                 self.value_error, self.bg_value_error, x_mid, y_mid)).T,
                       header="Ek [eV]  dEk [eV]  count []  bg_count []  X [pxl]  Y [pxl]  Ek_err [eV]  cnt_err []  bg_cnt_err []  x [m]  y [m]",
                       fmt="%.10e",
                       )
        else:
            np.savetxt("{}/sp_{}.txt".format(save_dir, self.name),
                       np.array((ek_mid_ev, dek_ev, self.value, Y_mid, X_mid, ek_mid_error_ev, self.value_error, x_mid, y_mid)).T,
                       header="Ek [eV]  dEk [eV]  count []  x [pxl]  y [pxl]  Ek_err [eV]  cnt_err []  x [m]  y [m]",
                       fmt="%.10e",
                       )
        self.plot_result(save_dir, ek_mid_ev, dek_ev, ek_mid_error_ev)

    def plot_result(self, save_dir_, ek_, dek_, ek_err_):
        fig, ax = plt.subplots()
        if bg_flag:
            yerr_ = np.sqrt(np.power(self.value_error, 2) + np.power(self.bg_value_error, 2))
            index_ = np.where((self.value - self.bg_value - yerr_) < 0)

            # bar plot for ion signals
#            ax.bar(ek_, self.value - self.bg_value, label="{}".format(self.name), alpha=0.5, width=dek_)
            ax.bar(ek_, self.value, label="{}".format(self.name), alpha=0.5, width=dek_)
#            ax.errorbar(ek_, self.value - self.bg_value, xerr=ek_err_, yerr=yerr_, capsize=2, ls="None")
#             ax.errorbar(ek_, self.value, xerr=ek_err_, yerr=self.value_error, capsize=2, ls="None")
            ax.errorbar(ek_, self.value, yerr=self.value_error, capsize=2, ls="None")

            # coloring of maximum position
#            ax.bar(ek_[index_], self.value[index_] - self.bg_value[index_], color="orange", width=dek_[index_])
            ax.bar(ek_[index_], self.value[index_], color="red", width=dek_[index_])

            # bar plot for background signals 
            ax.bar(ek_, self.bg_value, label="background", alpha=0.5, width=dek_, color="orange")  #label="{}".format(self.name), 
            # ax.errorbar(ek_, self.bg_value, xerr=ek_err_, yerr=self.bg_value_error, capsize=2, ls="None", color="orange")
            ax.errorbar(ek_, self.bg_value, yerr=self.bg_value_error, capsize=2, ls="None", color="orange")
        else:
            ax.bar(ek_, self.value, label="{}".format(self.name), alpha=0.5, width=dek_)
            # ax.errorbar(ek_, self.value, xerr=ek_err_, yerr=self.value_error, capsize=2, ls="None")
            ax.errorbar(ek_, self.value, yerr=self.value_error, capsize=2, ls="None")
        ax.set_xlabel(r"$E_k$ [eV]")
        ax.set_ylabel(r"count []")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(self.charge / unit_charge * 1e5, self.charge / unit_charge * 1e8)
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_dir_ + "/sp_{}.png".format(self.name))
        plt.close()

    def calc_fn(self, save_dir_root):
        """
        Not be used from version 3.2.0
        """
        print("This method is not available from version 3.2.0.")
        print("Refer verison 3.2.6 or older to read the code of this method.")


class Histogram:
    def __init__(self, length):
        self.length = length
        self.qm = np.empty(self.length)
        self.number = np.empty(self.length)
        self.bg_number = np.empty(self.length)

    def calc_hist(self, n_: int, spectra_: Spectra):
        self.qm[n_] = spectra_.charge * mass_u / spectra_.mass / coulomb_u
        self.number[n_] = np.sum(spectra_.value)
        if bg_flag:
            self.bg_number[n_] = np.sum(spectra_.bg_value)

    def save_hist(self, save_dir_root):
        save_datas = np.array((self.qm, self.number, self.bg_number)).T
        # save_datas = np.sort(save_datas_, axis=0)
        np.savetxt(save_dir_root + "/ion_number.txt", save_datas, header="q/m []  number []  bg_number []")

        fig, ax = plt.subplots()
        ax.bar(self.qm, self.number, label="ion", alpha=0.7, align="center", width=1/self.length)
        if bg_flag:
            ax.bar(self.qm, self.bg_number, label="background", alpha=0.7, align="center", width=1/self.length)
        ax.set_xlabel(r"$Q/M$ []")
        ax.set_ylabel(r"$N$ [a.u.]")

        from configurations import indication_QM_flag
        if indication_QM_flag:
            from configurations import indication_QM
            for _n, _qm in enumerate(indication_QM):
                ax.axvline(_qm["charge"]/_qm["mass"], ls="-.", c="black", lw=1)
                try:
                    _text = _qm["name"]
                    _fontsizeqm = 9
                except KeyError:
                    _text = r"$\frac{" + "{}".format(_qm["charge"]) + "}{" + "{}".format(_qm["mass"]) + "}$"
                    _fontsizeqm = 14
                ax.annotate(_text, xy=(_qm["charge"]/_qm["mass"] + 0.005, np.max(self.number)*0.98), xycoords="data", fontsize=_fontsizeqm)

        if qm_log_flag:
            ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_dir_root + "/histogram.".format(save_fig_ext))
        plt.close()
        print("saving histogram...")


def get_value(spectra_: Spectra, image_: Image):
    xy = spectra_.xy

    r_ = np.sqrt(np.power(xy[:, 0], 2) + np.power(xy[:, 1], 2))
    calc_flag = (r_ / res) > exclude_r

    returns = image_.get_count(calc_flag, xy, analysis_width, spectra_.color)
    spectra_.value = returns[0] / (2 * analysis_width + 1)
    spectra_.value_error = returns[1]
    spectra_.ek_mid_error = calc_ek_stddev(returns[2], returns[3], spectra_.mass, spectra_.charge)
    stop_n = returns[4]
    spectra_.XY = returns[5]

    if bg_flag:
        xy_bg = np.array((xy[:, 0], -xy[:, 1])).T
        if spectra_.color == "scan":
            color_ = "scan"
        else:
            color_ = bg_color
        bg_returns = image_.get_count(calc_flag, xy_bg, bg_analysis_width, color_=color_)
        spectra_.bg_value = bg_returns[0] / (2 * bg_analysis_width + 1)
        spectra_.bg_value_error = bg_returns[1]
        bg_stop_n = bg_returns[4]

        spectra_.value[min(stop_n, bg_stop_n):] = 0
        spectra_.value_error[min(stop_n, bg_stop_n):] = 0
        spectra_.bg_value[min(stop_n, bg_stop_n):] = 0
        spectra_.bg_value_error[min(stop_n, bg_stop_n):] = 0


def get_qm_hist(image_, coef_):
    print("-----")
    ion_numbers_file = image_.save_dir + "/ion_number.txt"
    if use_cache_qm_flag & os.path.exists(ion_numbers_file):
        date = datetime.datetime.fromtimestamp(os.path.getmtime(ion_numbers_file))
        print("Calculating histogram had already done before.")
        print("Last done date : {}".format(date))
        ion_numbers = np.genfromtxt(ion_numbers_file, skip_header=1)
        hist = Histogram(len(ion_numbers[:, 0]))
        hist.qm = ion_numbers[:, 0]
        hist.number = ion_numbers[:, 1]
        if len(ion_numbers[0, :] == 3):
            hist.bg_number = ion_numbers[:, 2]
    else:
        print("calculating histogram...")
        from configurations import scan_ion_list
        hist = Histogram(len(scan_ion_list))
        for n, ions in tqdm.tqdm(enumerate(scan_ion_list), total=len(scan_ion_list)):
            spectra = Spectra(ions[0], ions[1], ions[2], coef_, ions[3])
            get_value(spectra, image_)
            hist.calc_hist(n, spectra)

    hist.save_hist(image_.save_dir)

