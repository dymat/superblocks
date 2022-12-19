"""
"""
import os
import sys
import pandas as pd
import numpy as np
import pprint
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import matplotlib.pyplot as plt
import matplotlib as mpl
def cm2inch(*tupl):
    """Convert input cm to inches (width, hight)
    """
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
    

def setBoxColors(bp, color_one, color_two):
    plt.setp(bp['boxes'][0], color=color_one)
    plt.setp(bp['caps'][0], color=color_one)
    plt.setp(bp['caps'][1], color=color_one)
    plt.setp(bp['whiskers'][0], color=color_one)
    plt.setp(bp['whiskers'][1], color=color_one)
    plt.setp(bp['fliers'][0], color=color_one)
    plt.setp(bp['medians'][0], color=color_one)

    plt.setp(bp['boxes'][1], color=color_two)
    plt.setp(bp['caps'][2], color=color_two)
    plt.setp(bp['caps'][3], color=color_two)
    plt.setp(bp['whiskers'][2], color=color_two)
    plt.setp(bp['whiskers'][3], color=color_two)
    plt.setp(bp['fliers'][1], color=color_two)
    plt.setp(bp['medians'][1], color=color_two)