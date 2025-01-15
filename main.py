#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import __init__
import shutil, os, sys

args = sys.argv
global input_fname
try:
    input_fname_origin = args[1]
    input_fname, input_ext = os.path.splitext(input_fname_origin)
    if not input_ext == ".py":
        print("A file '{}' is not arrowed for the input file.".format(input_fname_origin)) 
        print("Input should be a python file.")
        sys.exit()
except IndexError:
    input_fname = "input"

if __name__ == "__main__":
    try:
        exec("from {} import *".format(input_fname))
    except ModuleNotFoundError:
        print("Not found the input file named '{}'.".format(input_fname))
        sys.exit()
    from image import Image, define_coef_and_angle
    from spectra import Spectra, get_value, get_qm_hist
    from configurations import scan_QM, no_spectra_flag

    for file in file_names:
        print("-------------------------------------------------")
        print("Starting to analyze {}".format(file))
        image = Image(file)
        coef = define_coef_and_angle(image)
        print("-----")
        for ions in ion_list:
            spectra = Spectra(ions[0], ions[1], ions[2], coef, ions[3])
            print("{} loop".format(spectra.name))
            get_value(spectra, image)
            # spectra.calc_fn(image.save_dir)
            if not no_spectra_flag:
                spectra.save_result(image.save_dir)

        if scan_QM:
            get_qm_hist(image, coef)

        print("-----")
        image.save_image()
        shutil.copy(input_fname + ".py", image.save_dir + "/previous_input.py")
        if not image.exist_normpy:
            shutil.copy(os.path.dirname(__file__) + "/normalize.py", image.save_dir + "/")
        if not image.exist_rcountpy:
            shutil.copy(os.path.dirname(__file__) + "/plt_r-count.py", image.save_dir + "/")
        print("")

    print("finished.")
