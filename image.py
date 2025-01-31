#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import platform, os, re, sys, glob, cv2, datetime

from copy import deepcopy as dc
from datetime import datetime as dt
from scipy.optimize import curve_fit
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.colors import LogNorm, Normalize

from main import input_fname
exec("from {} import *".format(input_fname))
from functions import parabola, rot_parabola, rotation_matrix, calc_stddev, ek_to_x
from configurations import save_dir_prefix, mass_u, coulomb_u, alpha, init_rot_for_fitting, cache_flag, scale_bar_flag_,\
    parabola_ext, flip_flag, save_raw_parabola_, fit_ion_nmqc, img_scale_params

if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["pcolor.shading"] = "auto"
plt.rcParams["font.size"] = 15
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"


class Image:
    """
    1st axis is horizontal axis
    2nd axis is vertical axis
    origin is at left-top
    Y = aX^2
    [Y, X]
    """

    def __init__(self, file_name):
        self.name, self.ext = os.path.splitext(os.path.basename(file_name))
        self.data = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        self.image = dc(self.data)
        self.max_count = np.max(self.data)
        self.origin_YX = origin
        self.angle = 0

        self.save_dir = save_dir_prefix + self.name
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.fig = plt.figure(figsize=(self.image.shape[1] * 3 / dpi, self.image.shape[0] * 3 / dpi))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.init_parabola()

        self.exist_normpy = os.path.exists(self.save_dir + "/normalize.py")
        self.exist_rcountpy = os.path.exists(self.save_dir + "/plt_r-count.py")

        if save_raw_parabola_:
            for ext in parabola_ext:
                save_name = "/raw_" + self.name + "." + ext
                self.fig.savefig(self.save_dir + save_name, bbox_inches='tight', pad_inches=0)

    def get_count(self, calc_flag, xy__, width_, color_):
        XY_ = self.conv_xy_input(xy__)
        # np.savetxt("{}/XY_in_image.txt".format(self.save_dir), XY_)
        length = len(XY_[:, 0])
        count = np.zeros(length - 1)
        count_stddev = np.zeros(length - 1)
        x_average = np.zeros(length - 1)
        x_stddev = np.zeros(length - 1)
        draw_pos = np.full((length - 1, 2), np.nan)
        draw_neg = np.full((length - 1, 2), np.nan)

        diff_XY_ = XY_[1:length, :] - XY_[0:length - 1, :]

        for n in range(length - 1):
            if calc_flag[n]:
                try:
                    returns = core_get_count(self.data, diff_XY_[n, :], XY_[n, :], width_)
#                    count[n] = calc_stddev(returns[0])[0]
                    count[n] = sum(returns[0])
                    count_stddev[n] = calc_stddev(returns[0])[1]
                    x_average[n], x_stddev[n] = self.calc_position_stddev(returns[1], returns[2])
                    draw_pos[n, :] = returns[3]
                    draw_neg[n, :] = returns[4]
                except IndexError:
                    if not color_ == "scan":
                        print("Reached to the end of image at {}th count.".format(n))
                    break
            else:
                count[n] = 0

        for n, err in enumerate(count_stddev):
            if not err == 0:
                count_stddev[n] = np.sqrt(err**2 + self.max_count)

        # np.savetxt(self.save_dir + "/x_ave_stdev2.txt", np.array((x_average, x_stddev)).T)

        if not color_ == "scan":
            # self.ax.plot(draw_pos[:, 0], draw_pos[:, 1], color=color_, linewidth=draw_para_width) #, linestyle=":")
            # self.ax.plot(draw_neg[:, 0], draw_neg[:, 1], color=color_, linewidth=draw_para_width) #, linestyle=":")
            self.ax.scatter(draw_pos[:, 0], draw_pos[:, 1], color=color_, s=draw_para_width)
            self.ax.scatter(draw_neg[:, 0], draw_neg[:, 1], color=color_, s=draw_para_width)
            # np.savetxt("{}/XY_to_plot_pos.txt".format(self.save_dir), draw_pos)
            # np.savetxt("{}/XY_to_plot_neg.txt".format(self.save_dir), draw_neg)

        return count, count_stddev, x_average, x_stddev, n, XY_

    def calc_position_stddev(self, X__, Y__):
        XY__ = np.array((X__, Y__)).T
        xy__ = self.conv_xy_output(XY__)
        x_average__, x_stddev__ = calc_stddev(xy__[:, 0])
        return x_average__, x_stddev__

    def conv_xy_input(self, xy__):
        """
        function to convert coordinates from in m to in pixels
        :param xy__:
        :return:
        """
        x_rot_, y_rot_ = rotation_matrix(x_=xy__[:, 0], y_=xy__[:, 1], angle_=self.angle - init_rot_for_fitting)
        X_ = -x_rot_ / res + self.origin_YX[1]
        Y_ = -y_rot_ / res + self.origin_YX[0]
        return np.array((X_, Y_)).T

    """
    def conv_xy_input__(self, xy__):
        Y_ = -xy__[:, 0] / res  # + self.origin_YX[1]
        X_ = -xy__[:, 1] / res  # + self.origin_YX[0]
        X_rot_, Y_rot_ = rotation_matrix(x_=X_, y_=Y_, angle_=-self.angle)
        # X_rot_, Y_rot_ = rotation_matrix(x__=X_, y__=Y_, angle_=self.angle, origin_=self.origin_YX)
        return np.array((X_rot_ + self.origin_YX[0], Y_rot_ + self.origin_YX[1])).T
    """

    def conv_xy_output(self, XY__):
        """
        not used
        """
        x_rot__ = (-XY__[:, 0] + self.origin_YX[1]) * res
        y_rot__ = (-XY__[:, 1] + self.origin_YX[0]) * res
        x__, y__ = rotation_matrix(x_=x_rot__, y_=y_rot__, angle_=-self.angle)
        return np.array((x__, y__)).T

    def get_local_value(self, x_, y_):
        """
        not used
        """
        x_rot_, y_rot_ = rotation_matrix(x_=x_, y_=y_, angle_=self.angle)
        X_ = -res * x_rot_ + self.origin_YX[0]
        Y_ = -res * y_rot_ + self.origin_YX[1]
        return self.data.item[X_, Y_]

    def add_scale(self):
        from configurations import scale_bar_conf
        fontprops = fm.FontProperties(size=scale_bar_conf["fontsize"])
        bar_length = int(scale_bar_conf["length"] / res)
        position = scale_bar_conf["position"]
        length_digit = int("{:e}".format(scale_bar_conf["length"])[-3:])/3
        if (length_digit < 1) & (length_digit >= 0):
            length_order = 1
            length_prefix = "m"
        elif (length_digit < 0) & (length_digit >= -1):
            length_order = 1e-3
            length_prefix = "mm"
        elif (length_digit < -1) & (length_digit >= -2):
            length_order = 1e-6
            length_prefix = r"$\mu$m"
        else:
            print("Undefined range of scale bar.")
            sys.exit()
        label = "{}".format(int(scale_bar_conf["length"] / length_order)) + " " + length_prefix
        """
        scalebar = AnchoredSizeBar(self.ax.transData,
                                   bar_length, 
                                   label, 
                                   loc=10, 
                                   frameon=False,
                                   size_vertical=scale_bar_conf["width"], 
                                   color=scale_bar_conf["color"],
                                   fontproperties=fontprops, 
                                   pad=0.1,
                                   bbox_transform=self.ax.transData, 
                                   bbox_to_anchor=position,
                                   )
        self.ax.add_artist(scalebar)
        """
        if scale_bar_conf["rotation"] == "horizontal":
            self.ax.hlines(y=position[1], 
                           xmin=position[0], 
                           xmax=position[0] + bar_length, 
                           lw=scale_bar_conf["width"], 
                           color=scale_bar_conf["color"],
                           )
            self.ax.text(x=position[0] + int(bar_length/2), 
                         y=position[1] + int(scale_bar_conf["fontsize"]/2), 
                         s=label, 
                         color=scale_bar_conf["color"],
                         fontsize=scale_bar_conf["fontsize"], 
                         rotation=scale_bar_conf["rotation"], 
                         horizontalalignment="center", 
                         verticalalignment="top",
                         )
        elif scale_bar_conf["rotation"] == "vertical":
            self.ax.vlines(x=position[0],
                           ymin=position[1],
                           ymax=position[1] + bar_length, 
                           lw=scale_bar_conf["width"], 
                           color=scale_bar_conf["color"],
                           )
            self.ax.text(x=position[0] + int(scale_bar_conf["fontsize"]/2),
                         y=position[1] + int(bar_length/2),
                         s=label, 
                         color=scale_bar_conf["color"],
                         fontsize=scale_bar_conf["fontsize"], 
                         rotation=scale_bar_conf["rotation"], 
                         horizontalalignment="left", 
                         verticalalignment="center",
                         )

    def add_ek_axis(self):
        from configurations import ek_indicate_flag, ek_indicate_offset_XY, ek_indicate_fontsize_, ek_indicate_origin_offset_XY 
        for number in range(number_of_axis):
            ek_axis = eval("ek_axis_{}".format(number+1))
            ek_j_ = np.array(ek_axis["ek_mev"]) * coulomb_u * 1e6
            x_ = ek_to_x(ek_j_, ek_axis["mass"], ek_axis["charge"])
            y_ = np.zeros(len(x_))
            XY_axis_ = self.conv_xy_input(np.array((x_, y_)).T)
            XY_axis_[:, 1] = np.full(len(x_), ek_axis["origin"][0])
            tick_length = np.full(len(x_), ek_axis["linewidth"]*5)
#            self.ax.scatter(XY_axis_[:, 1], XY_axis_[:, 0], marker="+", s=100, c=ek_axis["color"])
#            self.ax.scatter(ek_axis["origin"][0], self.origin_YX[1], marker="+", s=100, c=ek_axis["color"])
            self.ax.errorbar(XY_axis_[:, 1], 
                             XY_axis_[:, 0], 
                             fmt="none", 
                             ecolor=ek_axis["color"], 
                             elinewidth=ek_axis["linewidth"], 
                             capsize=0, 
                             xerr=tick_length, 
                             yerr=tick_length,
                             )
            self.ax.errorbar(ek_axis["origin"][0],
                             self.origin_YX[1], 
                             fmt="none", 
                             ecolor=ek_axis["color"], 
                             elinewidth=ek_axis["linewidth"], 
                             capsize=0, 
                             xerr=tick_length[0], 
                             yerr=tick_length[0],
                             )
            self.ax.vlines(ek_axis["origin"][0],
                           0, self.origin_YX[1], 
                           color=ek_axis["color"], 
                           lw=ek_axis["linewidth"],
                           )
            if ek_indicate_flag:
                try:
                    _text_origin = ek_axis["name"] + " [MeV]"
                except KeyError:
                    _text_origin = "{:.0f}/{:.0f} [MeV]".format(ek_axis["charge"]/coulomb_u, ek_axis["mass"]/mass_u)
                _ek_origin_X = ek_axis["origin"][1] + ek_indicate_origin_offset_XY[1]
                _ek_origin_Y = ek_axis["origin"][0] + ek_indicate_origin_offset_XY[0]
                self.ax.annotate(_text_origin, xy=(_ek_origin_Y, _ek_origin_X), xycoords="data", color=ek_axis["color"], fontsize=ek_indicate_fontsize_)

                for _n, _ek_mev in enumerate(ek_axis["ek_mev"]):
                    _text = "{}".format(_ek_mev)
                    if flip_flag:   
                        _indicate_Y = XY_axis_[_n, 1]-ek_indicate_offset_XY[0]
                    else:
                        _indicate_Y = XY_axis_[_n, 1]+ek_indicate_offset_XY[0]
                    _indicate_X = XY_axis_[_n, 0]-ek_indicate_offset_XY[1]
                    self.ax.annotate(_text, xy=(_indicate_Y, _indicate_X), xycoords="data", color=ek_axis["color"], fontsize=ek_indicate_fontsize_)

    def init_parabola(self):
        length_params = len(img_scale_params)
        if length_params == 0:
            _norm = Normalize()
        else:
            if (length_params <= 2) & (img_scale_params[0] == "log"):
                _norm = LogNorm()
            elif (length_params > 2) & (img_scale_params[0] == "log"):
                _norm = LogNorm(img_scale_params[1], img_scale_params[2])
            elif (length_params > 2) & (img_scale_params[0] == "linear"):
                _norm = Normalize(img_scale_params[1], img_scale_params[2])
            else:
                print("Error!")
                print("Check the variable 'img_scale' in input")
                sys.exit()

        fr = self.ax.imshow(self.image, cmap=raw_color, norm=_norm)
        self.save_colorbar(fr)
            
        if decimate_image:
            self.ax.set_xlim(cut_left_top[0], cut_right_down[0])
            self.ax.set_ylim(cut_right_down[1], cut_left_top[1])
        else:
            self.ax.set_xlim(0, self.image.shape[1])
            self.ax.set_ylim(self.image.shape[0], 0)
        # self.ax.legend(bbox_to_anchor=(0, 0), loc="upper left")
        self.ax.tick_params(bottom=False, left=False, right=False, top=False)
        self.ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        if flip_parabola_flag:
            self.ax.invert_xaxis()

    def save_colorbar(self, _fr):
        print("saving colorbar")
        figcb = plt.figure(figsize=(1.7, 4.8))
        axcb = figcb.add_subplot()
        pos = axcb.get_position()
        axcb.set_position([pos.x0, pos.y0, 0.07, pos.height])
        cb = figcb.colorbar(_fr, cax=axcb)
#        cb.set_label("color scale")
        for ext in parabola_ext:
#            figcb.tight_layout()
            save_name = "/colorbar_" + self.name + "." + ext
            figcb.savefig(self.save_dir + save_name)
        

    def save_image(self):
        print("saving parabola...")
        # self.ax.scatter(self.origin_YX[0], self.origin_YX[1], marker='+', c='blue', s=50, label="origin")
        self.ax.errorbar(self.origin_YX[0], 
                         self.origin_YX[1], 
                         fmt='none', 
                         ecolor=origin_color, 
                         elinewidth=origin_size,
                         capsize=0,
                         xerr=origin_size*5,
                         yerr=origin_size*5,
                         label="origin",
                         )
        if scale_bar_flag_:
            self.add_scale()
        if ek_axis_on_parabola:
            self.add_ek_axis()
        if put_dot_flag:
            for YX_dot_ in dot_position:
                self.ax.scatter(YX_dot_[0], YX_dot_[1], facecolor="None", edgecolors="green", marker="o", s=50)

        for ext in parabola_ext:
            save_name = "/scatter_" + self.name + "." + ext
            self.fig.savefig(self.save_dir + save_name, bbox_inches='tight', pad_inches=0)

    def plot_xy(self, xy__):
        """
        for debug
        """
        xy_ = self.conv_xy_input(xy__)
        self.ax.scatter(xy_[:, 0], xy_[:, 1], c="yellow")


def core_get_count(image: np.ndarray, diff_XY_float: np.ndarray, XY_start: np.ndarray, Y_width: int):
    count_list_ = []
    x_list_ = []
    y_list_ = []
    a = diff_XY_float[1] / diff_XY_float[0]
    X_width = int(abs(diff_XY_float[0]))
    draw_pos_ = [np.nan, np.nan]
    draw_neg_ = [np.nan, np.nan]

    for l in range(X_width):
        # integration along parabola
        X_ = XY_start[0] - l
        Y_ = XY_start[1] - a * l
        for wid in np.linspace(0, Y_width, Y_width + 1):
            # integration along width
            Y_val_pos_ = int(Y_ + wid / np.sqrt(a ** 2 + 1))
            Y_val_neg_ = int(Y_ - wid / np.sqrt(a ** 2 + 1))
            X_val_pos_ = int(a * (Y_ - Y_val_pos_) + X_)
            X_val_neg_ = int(a * (Y_ - Y_val_neg_) + X_)
            count_list_.append(image.item(X_val_pos_, Y_val_pos_))
            x_list_.append(X_val_pos_)
            y_list_.append(Y_val_pos_)
            if not Y_val_pos_ == Y_val_neg_:
                count_list_.append(image.item(X_val_neg_, Y_val_neg_))
                x_list_.append(X_val_neg_)
                y_list_.append(Y_val_neg_)
            if wid == Y_width:
                draw_pos_ = [Y_val_pos_, X_val_pos_]
                draw_neg_ = [Y_val_neg_, X_val_neg_]

    return count_list_, x_list_, y_list_, draw_pos_, draw_neg_


class Fitting:
    """
    1st axis is horizontal axis
    2nd axis is vertical axis
    origin is at left-bottom
    y = ax^2
    [x, y]
    """

    def __init__(self, image: np.ndarray, origin_YX: list, save_dir: str):
        self.save_dir = save_dir
        # self.image_th = dc(image)
        image_th_XY = np.where(image > img_threshold)
        image_th_XY = rotation_matrix(image_th_XY[0], image_th_XY[1], init_rot_for_fitting, origin_YX[1], origin_YX[0])
        # self.image_th[np.where(self.image_th < img_threshold)] = 0

        self.th_xy = np.empty((len(image_th_XY[0]), 2))
        self.th_xy[:, 0] = -(image_th_XY[0] - origin_YX[1]) * res
        if flip_flag:
            self.th_xy[:, 1] = (image_th_XY[1] - origin_YX[0]) * res
        else:
            self.th_xy[:, 1] = -(image_th_XY[1] - origin_YX[0]) * res
        self.para_B = np.arange(np.min(self.th_xy[:, 0]), np.max(self.th_xy[:, 0]),
                                (np.max(self.th_xy[:, 0]) - np.min(self.th_xy[:, 0])) / 100)

        self.dec_th_BE = self.decimate(origin_YX)

    def decimate(self, origin_YX_):
        xray_th = exclude_r * res

        bottom_XY = rotation_matrix(bottom[0], bottom[1], init_rot_for_fitting, origin_YX_[0], origin_YX_[1])
        top_XY = rotation_matrix(top[0], top[1], init_rot_for_fitting, origin_YX_[0], origin_YX_[1])
        # bottom_XY = bottom
        # top_XY = top
        coef_bottom = abs(bottom_XY[0] - origin_YX_[0]) / abs(bottom_XY[1] - origin_YX_[1]) ** 2 / res
        coef_top = abs(top_XY[0] - origin_YX_[0]) / abs(top_XY[1] - origin_YX_[1]) ** 2 / res

        top_para_y = parabola(self.th_xy[:, 0], coef_top)
        bottom_para_y = parabola(self.th_xy[:, 0], coef_bottom)
        position = (top_para_y > self.th_xy[:, 1]) & (self.th_xy[:, 1] > bottom_para_y) & (xray_th < self.th_xy[:, 0])
        dec_th_xy = np.array((self.th_xy[position, 0], self.th_xy[position, 1])).T

        fig, ax = plt.subplots()
        ax.scatter(self.th_xy[:, 0], self.th_xy[:, 1], label="experimental", s=1)
        ax.plot(self.para_B, parabola(self.para_B, coef_top), linewidth=1, label="top (a={})".format(coef_top),
                color="red")
        ax.plot(self.para_B, parabola(self.para_B, coef_bottom), linewidth=1, label="bottom (a={})".format(coef_bottom),
                color="green")
        fig.legend()
        fig.savefig(self.save_dir + "/before_dec_th_xy.png")
        # fig.savefig(self.save_dir + "/before_dec_th_xy.{}".format(save_fig_ext))
        ax.cla()
        fig.clf()
        plt.close()
        del ax, fig

        return dec_th_xy

    def fitting(self):
        param_, cov_ = curve_fit(parabola, self.dec_th_BE[:, 0], self.dec_th_BE[:, 1])
        param = param_ * fit_ion_nmqc[2] * unit_mass / fit_ion_nmqc[1] / unit_charge
        initial = [1, np.pi / 180]
        param_rot_, cov_rot_ = curve_fit(rot_parabola, self.dec_th_BE[:, 0], self.dec_th_BE[:, 1], p0=initial)
        param_rot = param_rot_ * fit_ion_nmqc[2] * unit_mass / fit_ion_nmqc[1] / unit_charge

        yfit = parabola(self.para_B, param_[0])
        yfit_rot = rot_parabola(self.para_B, param_rot_[0], param_rot_[1])

        voltage = param[0] * coulomb_u * alpha / mass_u * B * B * d
        voltage_rot = param_rot[0] * coulomb_u * alpha / (1.0073 * mass_u) * B * B * d

        fig, ax = plt.subplots()
        ax.scatter(self.dec_th_BE[:, 0], self.dec_th_BE[:, 1], s=1, label="experiment")
        ax.plot(self.para_B, yfit, linewidth=1, label="fitting w/o rotation.", color="green")
        ax.plot(self.para_B, yfit_rot, linewidth=1, label="fitting w/ rotation", color="red")
        text_wo_rot = "w/o rot\n" + r"$a=$" + "{:.2f}\n".format(param[0]) + r"$V_{appl.}=$" + "{:.0f} [V]".format(voltage)
        ax.annotate(text_wo_rot, xy=(0.05, 0.65), xycoords="axes fraction")
        if flip_flag:
            angle_text = -param_rot[1] * 180 / np.pi
        else:
            angle_text = param_rot[1] * 180 / np.pi
        text_w_rot = "w/ rot\n" + r"$a=$" + "{:.2f}\n".format(param_rot[0]) + r"$\theta=$" + "{:.2f} [deg.]\n".format(
            angle_text) + r"$V_{appl.}=$" + "{:.0f} [V]".format(voltage_rot)
        ax.annotate(text_w_rot, xy=(0.05, 0.3), xycoords="axes fraction")
        text_fit_ion = "fitting ion : {}".format(fit_ion_nmqc[0])
        ax.annotate(text_fit_ion, xy=(0.05, 0.9), xycoords="axes fraction")

        fig.savefig(self.save_dir + "/fitting_init_a{}_{}deg.png".format(1, 1))
        # fig.savefig(self.save_dir + "/fitting_init_a{}_{}deg.{}".format(1, 1, save_fig_ext))
        ax.cla()
        fig.clf()
        plt.close()
        del fig, ax

        if flip_flag:
            return param_rot[0], -param_rot[1]
        else:
            return param_rot[0], param_rot[1]


class _Fitting:
    """    
    1st axis is horizontal axis
    2nd axis is vertical axis
    origin is at left-bottom
    y = ax^2
    [x, y]
    """

    def __init__(self, image: np.ndarray, origin_YX: list, save_dir: str):
        self.save_dir = save_dir
        # self.image_th = dc(image)
        image_th_XY = np.where(image > img_threshold)
        image_th_XY = rotation_matrix(image_th_XY[0], image_th_XY[1], init_rot_for_fitting, origin_YX[1], origin_YX[0])
        # self.image_th[np.where(self.image_th < img_threshold)] = 0

        self.th_xy = np.empty((len(image_th_XY[0]), 2))
        self.th_xy[:, 0] = -(image_th_XY[0] - origin_YX[1]) * res
        if flip_flag:
            self.th_xy[:, 1] = (image_th_XY[1] - origin_YX[0]) * res
        else:
            self.th_xy[:, 1] = -(image_th_XY[1] - origin_YX[0]) * res
        self.para_x = np.arange(np.min(self.th_xy[:, 0]), np.max(self.th_xy[:, 0]),
                                (np.max(self.th_xy[:, 0]) - np.min(self.th_xy[:, 0])) / 100)

        self.dec_th_xy = self.decimate(origin_YX)

    def decimate(self, origin_YX_):
        xray_th = exclude_r * res

        bottom_XY = rotation_matrix(bottom[0], bottom[1], init_rot_for_fitting, origin_YX_[0], origin_YX_[1])
        top_XY = rotation_matrix(top[0], top[1], init_rot_for_fitting, origin_YX_[0], origin_YX_[1])
        # bottom_XY = bottom
        # top_XY = top
        coef_bottom = abs(bottom_XY[0] - origin_YX_[0]) / abs(bottom_XY[1] - origin_YX_[1]) ** 2 / res
        coef_top = abs(top_XY[0] - origin_YX_[0]) / abs(top_XY[1] - origin_YX_[1]) ** 2 / res

        top_para_y = parabola(self.th_xy[:, 0], coef_top)
        bottom_para_y = parabola(self.th_xy[:, 0], coef_bottom)
        position = (top_para_y > self.th_xy[:, 1]) & (self.th_xy[:, 1] > bottom_para_y) & (xray_th < self.th_xy[:, 0])
        dec_th_xy = np.array((self.th_xy[position, 0], self.th_xy[position, 1])).T

        fig, ax = plt.subplots()
        ax.scatter(self.th_xy[:, 0], self.th_xy[:, 1], label="experimental", s=1)
        ax.plot(self.para_x, parabola(self.para_x, coef_top), linewidth=1, label="top (a={})".format(coef_top),
                color="red")
        ax.plot(self.para_x, parabola(self.para_x, coef_bottom), linewidth=1, label="bottom (a={})".format(coef_bottom),
                color="green")
        fig.legend()
        fig.savefig(self.save_dir + "/before_dec_th_xy.png")
        # fig.savefig(self.save_dir + "/before_dec_th_xy.{}".format(save_fig_ext))
        ax.cla()
        fig.clf()
        plt.close()
        del ax, fig

        return dec_th_xy

    def fitting(self):
        param_, cov_ = curve_fit(parabola, self.dec_th_xy[:, 0], self.dec_th_xy[:, 1])
        param = param_ * fit_ion_nmqc[2] * unit_mass / fit_ion_nmqc[1] / unit_charge
        initial = [1, np.pi / 180]
        param_rot_, cov_rot_ = curve_fit(rot_parabola, self.dec_th_xy[:, 0], self.dec_th_xy[:, 1], p0=initial)
        param_rot = param_rot_ * fit_ion_nmqc[2] * unit_mass / fit_ion_nmqc[1] / unit_charge

        yfit = parabola(self.para_x, param[0])
        yfit_rot = rot_parabola(self.para_x, param_rot[0], param_rot[1])

        voltage = param[0] * coulomb_u * alpha / mass_u * B * B * d
        voltage_rot = param_rot[0] * coulomb_u * alpha / (1.0073 * mass_u) * B * B * d

        fig, ax = plt.subplots()
        ax.scatter(self.dec_th_xy[:, 0], self.dec_th_xy[:, 1], s=1, label="experiment")
        ax.plot(self.para_x, yfit, linewidth=1, label="fitting w/o rotation.", color="green")
        ax.plot(self.para_x, yfit_rot, linewidth=1, label="fitting w/ rotation", color="red")
        text_wo_rot = "w/o rot\n" + r"$a=$" + "{:.2f}\n".format(param[0]) + r"$V_{appl.}=$" + "{:.0f} [V]".format(voltage)
        ax.annotate(text_wo_rot, xy=(0.05, 0.65), xycoords="axes fraction")
        if flip_flag:
            angle_text = -param_rot[1] * 180 / np.pi
        else:
            angle_text = param_rot[1] * 180 / np.pi
        text_w_rot = "w/ rot\n" + r"$a=$" + "{:.2f}\n".format(param_rot[0]) + r"$\theta=$" + "{:.2f} [deg.]\n".format(
            angle_text) + r"$V_{appl.}=$" + "{:.0f} [V]".format(voltage_rot)
        ax.annotate(text_w_rot, xy=(0.05, 0.3), xycoords="axes fraction")
        text_fit_ion = "fitting ion : {}".format(fit_ion_nmqc[0])
        ax.annotate(text_fit_ion, xy=(0.05, 0.9), xycoords="axes fraction")

        fig.savefig(self.save_dir + "/fitting_init_a{}_{}deg.png".format(1, 1))
        # fig.savefig(self.save_dir + "/fitting_init_a{}_{}deg.{}".format(1, 1, save_fig_ext))
        ax.cla()
        fig.clf()
        plt.close()
        del fig, ax

        if flip_flag:
            return param_rot[0], -param_rot[1]
        else:
            return param_rot[0], param_rot[1]

    def __fitting(self):
        param_, cov_ = curve_fit(parabola, self.dec_th_xy[:, 0], self.dec_th_xy[:, 1])
        param = param_ * fit_ion_nmqc[2] * unit_mass / fit_ion_nmqc[1] / unit_charge
        initial = [1, np.pi / 180]
        param_rot_, cov_rot_ = curve_fit(rot_parabola, self.dec_th_xy[:, 0], self.dec_th_xy[:, 1], p0=initial)
        param_rot = param_rot_ * fit_ion_nmqc[2] * unit_mass / fit_ion_nmqc[1] / unit_charge

        yfit = parabola(self.para_x, param[0])
        yfit_rot = rot_parabola(self.para_x, param_rot[0], param_rot[1])

        voltage = param[0] * coulomb_u * alpha / mass_u * B * B * d
        voltage_rot = param_rot[0] * coulomb_u * alpha / (1.0073 * mass_u) * B * B * d

        fig, ax = plt.subplots()
        ax.scatter(self.dec_th_BE[:, 0], self.dec_th_BE[:, 1], s=1, label="experiment")
        ax.plot(self.para_B, yfit, linewidth=1, label="fitting w/o rotation.", color="green")
        ax.plot(self.para_B, yfit_rot, linewidth=1, label="fitting w/ rotation", color="red")
        text_wo_rot = "w/o rot\n" + r"$a=$" + "{:.2f}\n".format(param[0]) + r"$V_{appl.}=$" + "{:.0f} [V]".format(voltage)
        ax.annotate(text_wo_rot, xy=(0.05, 0.65), xycoords="axes fraction")
        if flip_flag:
            angle_text = -param_rot[1] * 180 / np.pi
        else:
            angle_text = param_rot[1] * 180 / np.pi
        text_w_rot = "w/ rot\n" + r"$a=$" + "{:.2f}\n".format(param_rot[0]) + r"$\theta=$" + "{:.2f} [deg.]\n".format(
            angle_text) + r"$V_{appl.}=$" + "{:.0f} [V]".format(voltage_rot)
        ax.annotate(text_w_rot, xy=(0.05, 0.3), xycoords="axes fraction")
        text_fit_ion = "fitting ion : {}".format(fit_ion_nmqc[0])
        ax.annotate(text_fit_ion, xy=(0.05, 0.9), xycoords="axes fraction")

        fig.savefig(self.save_dir + "/fitting_init_a{}_{}deg.png".format(1, 1))
        # fig.savefig(self.save_dir + "/fitting_init_a{}_{}deg.{}".format(1, 1, save_fig_ext))
        ax.cla()
        fig.clf()
        plt.close()
        del fig, ax

        if flip_flag:
            return param_rot[0], -param_rot[1]
        else:
            return param_rot[0], param_rot[1]


class _Fitting:
    """
    - COORDINATES IN PIXEL -
      origin is at left-bottom
      X = aY^2
      [X, Y]
    - COORDINATES IN METRE -
      origin is at left-bottom
      E = aB^2
      [B, E]
    """

    def __init__(self, image: np.ndarray, origin_XY_: list, save_dir: str):
        self.save_dir = save_dir
        origin_BE = origin_XY_
        image_BE = np.where(image > img_threshold)
        image_BE: [np.ndarray, np.ndarray] = rotation_matrix(image_BE[0], image_BE[1], init_rot_for_fitting, origin_BE[1], origin_BE[0])

        self.th_BE = np.empty((len(image_BE[0]), 2))
        self.th_BE[:, 0] = (image_BE[0] - origin_BE[0]) * res
        if flip_flag:
            self.th_BE[:, 1] = -(image_BE[1] - origin_BE[1]) * res
        else:
            self.th_BE[:, 1] = (image_BE[1] - origin_BE[1]) * res
        # self.th_xy = np.array((image[:, 0] * res, image[:, 1] * res)).T
        self.para_B = np.arange(np.min(self.th_BE[:, 0]), np.max(self.th_BE[:, 0]),
                                (np.max(self.th_BE[:, 0]) - np.min(self.th_BE[:, 0])) / 100)

        self.dec_th_BE = self.decimate(origin_BE)

    def decimate(self, origin_BE_):

        xray_th = exclude_r * res

        bottom_BE: [float, float] = rotation_matrix(bottom[0], bottom[1], init_rot_for_fitting, origin_BE_[0], origin_BE_[1])
        top_BE: [float, float] = rotation_matrix(top[0], top[1], init_rot_for_fitting, origin_BE_[0], origin_BE_[1])

        coef_bottom = abs(bottom_BE[1] - origin_BE_[1]) / abs(bottom_BE[0] - origin_BE_[0]) ** 2 / res
        coef_top = abs(top_BE[1] - origin_BE_[1]) / abs(top_BE[0] - origin_BE_[0]) ** 2 / res

        # coef_bottom = abs(bottom_BE[0] - origin_YX_[0]) / abs(bottom_BE[1] - origin_YX_[1]) ** 2 / res
        # coef_top = abs(top_BE[0] - origin_YX_[0]) / abs(top_BE[1] - origin_YX_[1]) ** 2 / res

        top_para_E = parabola(self.th_BE[:, 0], coef_top)
        bottom_para_E = parabola(self.th_BE[:, 0], coef_bottom)
        position = (top_para_E > self.th_BE[:, 1]) & (self.th_BE[:, 1] > bottom_para_E) & (xray_th < self.th_BE[:, 0])
        dec_th_BE = np.array((self.th_BE[position, 0], self.th_BE[position, 1])).T

        fig, ax = plt.subplots()
        ax.scatter(self.th_BE[:, 0], self.th_BE[:, 1], label="experimental", s=1)
        ax.plot(self.para_B, parabola(self.para_B, coef_top), linewidth=1, label="top (a={})".format(coef_top),
                color="red")
        ax.plot(self.para_B, parabola(self.para_B, coef_bottom), linewidth=1, label="bottom (a={})".format(coef_bottom),
                color="green")
        fig.legend()
        fig.savefig(self.save_dir + "/before_dec_th_xy.png")
        # fig.savefig(self.save_dir + "/before_dec_th_xy.{}".format(save_fig_ext))
        ax.cla()
        fig.clf()
        plt.close()
        del ax, fig

        return dec_th_BE

    def fitting(self):
        param_, cov_ = curve_fit(parabola, self.dec_th_BE[:, 0], self.dec_th_BE[:, 1])
        param = param_ * fit_ion_nmqc[2] * unit_mass / fit_ion_nmqc[1] / unit_charge
        initial = [1, np.pi / 180]
        param_rot_, cov_rot_ = curve_fit(rot_parabola, self.dec_th_BE[:, 0], self.dec_th_BE[:, 1], p0=initial)
        param_rot = param_rot_ * fit_ion_nmqc[2] * unit_mass / fit_ion_nmqc[1] / unit_charge

        yfit = parabola(self.para_B, param_[0])
        yfit_rot = rot_parabola(self.para_B, param_rot_[0], param_rot_[1])

        voltage = param[0] * coulomb_u * alpha / mass_u * B * B * d
        voltage_rot = param_rot[0] * coulomb_u * alpha / (1.0073 * mass_u) * B * B * d

        fig, ax = plt.subplots()
        ax.scatter(self.dec_th_BE[:, 0], self.dec_th_BE[:, 1], s=1, label="experiment")
        ax.plot(self.para_B, yfit, linewidth=1, label="fitting w/o rotation.", color="green")
        ax.plot(self.para_B, yfit_rot, linewidth=1, label="fitting w/ rotation", color="red")
        text_wo_rot = "w/o rot\n" + r"$a=$" + "{:.2f}\n".format(param[0]) + r"$V_{appl.}=$" + "{:.0f} [V]".format(voltage)
        ax.annotate(text_wo_rot, xy=(0.05, 0.65), xycoords="axes fraction")
        if flip_flag:
            angle_text = -param_rot[1] * 180 / np.pi
        else:
            angle_text = param_rot[1] * 180 / np.pi
        text_w_rot = "w/ rot\n" + r"$a=$" + "{:.2f}\n".format(param_rot[0]) + r"$\theta=$" + "{:.2f} [deg.]\n".format(
            angle_text) + r"$V_{appl.}=$" + "{:.0f} [V]".format(voltage_rot)
        ax.annotate(text_w_rot, xy=(0.05, 0.3), xycoords="axes fraction")
        text_fit_ion = "fitting ion : {}".format(fit_ion_nmqc[0])
        ax.annotate(text_fit_ion, xy=(0.05, 0.9), xycoords="axes fraction")

        fig.savefig(self.save_dir + "/fitting_init_a{}_{}deg.png".format(1, 1))
        # fig.savefig(self.save_dir + "/fitting_init_a{}_{}deg.{}".format(1, 1, save_fig_ext))
        ax.cla()
        fig.clf()
        plt.close()
        del fig, ax

        if flip_flag:
            return param_rot[0], -param_rot[1]
        else:
            return param_rot[0], param_rot[1]


def define_coef_and_angle(image: Image):
    """
    execute fitting or read cache files
    :return: coef_
    """
    print("-----")

    log_file = image.save_dir + "/analysis.log"
    if cache_flag & os.path.exists(log_file):
        status = "log"
        print("Referring last used parameters from the log file.")
        with open(log_file, mode="r", encoding="utf-8") as f:
            log = f.readlines()[-5:]
            print("Date : {}".format(log[0].rstrip(os.linesep)))
            print("Status : {}".format(log[1].rstrip(os.linesep)))
            coef_ = float(log[2].rstrip(os.linesep))
            angle_ = float(log[3].rstrip(os.linesep))
    elif fit_flag:
        text_path = image.save_dir + "/fitting.txt"
        if os.path.exists(text_path) & fit_cache_flag:
            status = "fitting cache"
            date = datetime.datetime.fromtimestamp(os.path.getmtime(text_path))
            print("Fitting had already done before.")
            print("Last done date : {}".format(date))
            coef_, angle_ = np.genfromtxt(text_path, skip_header=1)
        else:
            status = "fit"
            print("fitting...")
            fit = Fitting(image.data, image.origin_YX, image.save_dir)
            coef_, angle_ = fit.fitting()
            np.savetxt(text_path, np.array((coef_, angle_)).T, header="coefficient [1/m]  angle [rad]")
            del fit
    else:
        text_path = image.save_dir + "/wo_fitting_parameters.txt"
        if os.path.exists(text_path) & wofit_cache_flag:
            status = "manual cache"
            date = datetime.datetime.fromtimestamp(os.path.getmtime(text_path))
            print("Analyze with last manual parameters used before.")
            print("Last done date : {}".format(date))
            coef_, angle_ = np.genfromtxt(text_path, skip_header=1)
        else:
            status = "manual"
            print("Analyze with manually input parameters.")
            angle_ = angle #* np.pi / 180
            from configurations import coefficient
            coef_ = coefficient
            np.savetxt(image.save_dir + "/wo_fitting_parameters.txt", np.array((coef_, angle_)),
                       header="coefficient [1/m]  angle [rad]")

    with open(log_file, mode="a", encoding="utf-8") as f:
        f.write(dt.now().strftime('%Y/%m/%d %H:%M:%S') + "\n")
        f.write(status + "\n")
        f.write("{:.18e}\n".format(coef_))
        f.write("{:.18e}\n".format(angle_))
        f.write("\n")

    image.angle = angle_
    print("coefficient is at {:.2e}.".format(coef_))
    print("angle is at {:.2f} degree.".format(angle_ * 180 / np.pi))
    return coef_
