#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import numpy as np
from scipy.optimize import fsolve
from copy import deepcopy as dc
from configurations import pn_term, c_0
from main import input_fname
exec("from {} import B, L, D".format(input_fname))


def parabola(x_, a_):
    y_ = a_ * np.power(x_, 2)
    return y_


def rot_parabola(x_, a_, theta_):
    """
    parabolic function with rotation
    :param x_:
    :param a_:
    :param theta_: angle [rad]
    :return:
    """
    sq_ = (np.cos(theta_))**2 - 4 * a_ * x_ * np.sin(theta_)
    y_pos_ = (np.cos(theta_) - a_ * x_ * np.sin(2*theta_) + np.sqrt(sq_))/(2 * a_ * (np.sin(theta_))**2)
    y_neg_ = (np.cos(theta_) - a_ * x_ * np.sin(2*theta_) - np.sqrt(sq_))/(2 * a_ * (np.sin(theta_))**2)

    if pn_term == "pos":
        return y_pos_
    elif pn_term == "neg":
        return y_neg_
    else:
        print("Error. Please check variable 'pn_term' in configurations.py.")
        print("'pn_term' should be 'pos' or 'neg'")
        sys.exit()


def ek_to_gamma(ek__, mass__):
    mc2__ = mass__ * c_0 ** 2
    gamma__ = 1 + ek__ / mc2__
    return gamma__


def gamma_to_ek(gamma__, mass__):
    mc2 = mass__ * c_0 ** 2
    ek__ = (gamma__ - 1) * mc2
    return ek__


def ek_to_x(ek_, mass_, charge_):
    gamma = ek_to_gamma(ek_, mass_)
    R = mass_ * c_0 * np.sqrt(np.power(gamma, 2) - 1) / charge_ / B
    x_ = R - np.sqrt(np.power(R, 2) - L**2) + L * D / np.sqrt(np.power(R, 2) - L**2)
    return x_


def x_to_ek(x_, mass_, charge_):
    larmor = np.array([fsolve(calc_larmor, x0=1000, args=x__) for x__ in x_])[:, 0]
    gamma = np.sqrt(1 + np.power(larmor * charge_ * B / mass_ / c_0, 2))
    ek_ = gamma_to_ek(gamma, mass_)
    return ek_


def calc_larmor(R, x_):
    """
    Formula of f(R,x)=C, used for calculate R in C=0 for each x
    :param R: larmor radius
    :param x_: coordinate on detector
    :return: C
    """
    a_ = (L ** 2 + 2 * L * D + np.power(x_, 2)) / 2
    return x_ * R ** 3 - a_ * R ** 2 - x_ * L ** 2 * R + (D ** 2 / 2 + a_) * L ** 2


def rotation_matrix(x_, y_, angle_, center_x_=0, center_y_=0):
    x__ = x_ - center_x_
    y__ = y_ - center_y_
    rot_x_ = x__ * np.cos(angle_) - y__ * np.sin(angle_) + center_x_
    rot_y_ = x__ * np.sin(angle_) + y__ * np.cos(angle_) + center_y_
    return rot_x_, rot_y_


def calc_stddev(datas):
    """
    Calculate standard deviation for any data
    :param datas:
    :return:
    """
    average = np.average(datas)
    std = np.std(datas)
    return average, std


def calc_stddev_old(datas):
    """
    Calculate standard deviation for any data
    :param datas:
    :return:
    """
    average = np.average(datas)
    variance = np.average(np.power(datas - average, 2))
    return average, np.sqrt(variance)


def calc_ek_stddev(x_averages_, x_stddevs__, mass_, charge_):
    x_stddevs_ = dc(x_stddevs__)
    min_stddevs_ = 0.00216 * x_averages_ + 4.79e-5  # minimum is set at the value of dl=300um
    x_stddevs_[np.where(x_stddevs_ < min_stddevs_)] = min_stddevs_[np.where(x_stddevs_ < min_stddevs_)]

    # np.savetxt("./x_ave_stdev5.txt", np.array((x_averages_, x_stddevs_)).T)

    ek_average_ = x_to_ek(x_averages_, mass_, charge_)
    ek_stddev_ = ek_average_ - x_to_ek(x_averages_ + x_stddevs_, mass_, charge_)
    return ek_stddev_



