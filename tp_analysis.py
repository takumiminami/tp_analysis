#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Not be used from version 3.2.0
"""

from input import *
from image import Image, define_coef_and_angle
from spectra import Spectra, get_value, Histogram
from configurations import scan_QM
import tqdm


def analysis():
    for file in file_names:
        print("-----")
        print("Starting to analyze {}".format(file))
        image = Image(file)
        coef = define_coef_and_angle(image)
        print("-----")
        for ions in ion_list:
            spectra = Spectra(ions[0], ions[1], ions[2], coef, ions[3])
            print("{} loop".format(spectra.name))
            get_value(spectra, image)
            # spectra.calc_fn(image.save_dir)
            spectra.save_result(image.save_dir)

        if scan_QM:
            print("-----")
            print("calculating histogram...")
            from configurations import scan_ion_list
            hist = Histogram(len(scan_ion_list))
            for n, ions in tqdm.tqdm(enumerate(scan_ion_list), total=len(scan_ion_list)):
                spectra = Spectra(ions[0], ions[1], ions[2], coef, ions[3])
                get_value(spectra, image)
                hist.calc_hist(n, spectra)
            hist.save_hist(image.save_dir)

        print("-----")
        del image
        print("")
