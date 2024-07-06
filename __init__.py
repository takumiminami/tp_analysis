#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = "Takumi Minami"
__year__ = "2024"
__date__ = "25 April"
__credits__ = ["Takumi Minami"]
__license__ = "Private Domain"
__version__ = "3.3.0"
__maintainer__ = "Takumi Minami"
__email__ = "takumi.minami@eie.eng.osaka-u.ac.jp"
__copyright__ = "Copyright (C), {}, {}".format(__year__, __author__)

print("-------------------------------------------------")
print("Thomson parabola spectrometer analyzer")
# print(__copyright__)
print("Version : {}".format(__version__))
print("Last update : {} {}".format(__date__, __year__))
print("Author : {}".format(__author__))
print("-------------------------------------------------")
print("")


import os, sys
sys.path.append(os.path.split(__file__)[0])
sys.path.append(os.getcwd())
# if __name__ == "tp":
#     from tp_analysis import analysis


"""
Updates
3.3.0   First version uploaded in Github
3.2.12  Modified the change in 3.2.11
3.2.11  Change the definition of the vertical value of spectra from sum to average
3.2.10  Apply to change the color scale of the parabola
3.2.9   Set origin_color in input.py to change the color of the origin indicator.
3.2.8   Apply plt_r-count.py.
        Change the plot configurations of sp_*.png files.
3.2.7   Add a function to save only parabola image.
        Change the output digit of spectra files to "%.10e" (default of numpy is "%.18e") .
        Change the configurations of output figures in normalize.py.
3.2.6   Change the name of variables in normalize.py.
3.2.5   Apply fitting with various species of ions
3.2.4   Change the definitions of variables in normalize.py.

3.2.3   Apply vertically scalebar.
3.2.2   Apply normalize.py.
3.2.1   Save x,y in meter.
3.2.0   Change the directory distribution.
        Save spectra as the default not distribution function. Omit the function to normalize.
3.1.13  Change the default font to Helvetica from TimesNewRoman.
3.1.12  Bug fix for saving images of fitting results.
3.1.11  Add function to draw ek axes on the parabola image.
        Change the details of scale bar in configurations.py.
3.1.10  Apply to save the data of Ek v.s. count (not normalized).
        Apply to save raw image of parabola w/o any scatters.
3.1.9   Add function to calculate N error and Ek error.
3.1.8   Add function to make scatter point which represents the position input manually.
3.1.7   Add function to decimate parabola image in specific size. 
        Change the standard position of parabola (from second quandrant to first). 
3.1.6   Add a function to export analysis parameters to "analysis.log" and refer it.
        Make it available to input "save_fig_ext" as a list.
        Add a function to insert scale bar into parabola output.
3.1.5   Export XY_mid in pixels of analysis position in "fn_.txt"
3.1.4   Add cache flag of w/o fitting analysis
        Bug fix for calculating Vappl in fitting
3.1.3   Add cache flag of QM histogram
        Apply log plot for QM histogram
3.1.2   Bug fix for saving histogram in txt
        Change the action of fitting cache as saving fitting parameter in txt always
3.1.1   Display the progress bar for histogram calculation
        Display fitting parameter as V in .png 
3.1     Add a class Histogram
3.0     Revised totally from 2.3.2
"""
