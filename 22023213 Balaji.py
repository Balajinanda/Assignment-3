# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:28:06 2023

@author: Balaj
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def logistic(t, n0, g, t0):
    """
    Calculates the logistic function with scale factor n0 and growth rate g.

    Parameters:
        t (array-like): Independent variable values.
        n0 (float): Scale factor.
        g (float): Growth rate.
        t0 (float): Inflection point.

    Returns:
        array-like: Logistic function values.
    """
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f
