#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import cv2, glob, re

file_names = [
              "./190726_3526.gel",
              ]
#file_names = glob.glob("./tps0deg*.tiff")

init_origin = [1712, 3512]  # 190726_3526.gel

def define_center(profile):
    """
    define center position of line, dot, etc.
    @param profile: line profile
    @type profile: np.ndarray, list
    @return: int
    """

    length = profile.__len__()
    profile = profile - np.max(profile) / 2

    posit_hm = [n for n in range(length - 1) if profile[n] * profile[n + 1] < 0]  # "hm" = half maximum
    temp_ave = np.average(posit_hm)
    pos_posit_hm = [n for n in posit_hm if n > temp_ave]
    neg_posit_hm = [n for n in posit_hm if n < temp_ave]
    center = int((np.average(pos_posit_hm) + np.average(neg_posit_hm) - length) / 2)

    return center


def scan_center(image):
    """
    Define the origin of the parabola
    @return: Tuple[int, int]
    """
    print("scanning center position...")
    scan_length: int = 100
    profile_x = np.zeros(scan_length, dtype=int)
    profile_y = np.zeros(scan_length, dtype=int)

    for n in range(scan_length):
        position = int(n - scan_length / 2)
        profile_x[n] = image.item(init_origin[1], init_origin[0] + position)
        profile_y[n] = image.item(init_origin[1] + position, init_origin[0])

    x_center = init_origin[0] + define_center(profile_x)
    y_center = init_origin[1] + define_center(profile_y)
    print("center is at [{}, {}]".format(x_center, y_center))
    return x_center, y_center


#centers = np.empty((file_names.__len__(), 2))
centers_x = []
centers_y = []
fw = open("center.txt", "w")

for n, f_name in enumerate(file_names):
    if re.findall("0009", f_name):
        continue
    elif re.findall("0020", f_name):
        continue
    elif re.findall("0005", f_name):
        continue
    elif re.findall("0018", f_name):
        continue
    elif re.findall("0012", f_name):
        continue

    print("----------------")
    print("{}".format(f_name))
    fw.write("{}\n".format(f_name))
    image: np.ndarray = cv2.imread(f_name, cv2.IMREAD_UNCHANGED)
    # scan_center(image)
#    centers[n, 0], centers[n, 1] = scan_center(image)
    _center = scan_center(image)
    centers_x.append(_center[0])
    centers_y.append(_center[1])
#    fw.write("{}\n".format(centers[n, :]))
    fw.write("[{}, {}]\n".format(_center[0], _center[1]))
    
    fw.write("\n")


#center_ave = [np.average(centers[:, 0]), np.average(centers[:, 1])]
center_ave = [np.average(centers_x), np.average(centers_y)]
print("")
print("average of center : [{}, {}]".format(center_ave[0], center_ave[1]))
fw.write("average\n")
fw.write("[{}, {}]\n".format(center_ave[0], center_ave[1]))

