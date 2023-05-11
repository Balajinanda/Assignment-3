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

def plot_forecast(year, data, param, covar, predictions=None):
    """
    Plots the forecasted CO2 emissions along with the confidence \
        interval and predictions.

    Parameters:
        year (array-like): Years for the forecast.
        data (DataFrame): Data containing 'Year' and 'CO2 emissions' columns.
        param (array-like): Parameters for the logistic function.
        covar (array-like): Covariance matrix from curve_fit.
        predictions (array-like, optional): Predicted CO2 emissions values. \
            Default is None.

    Returns:
        None
    """
    forecast = logistic(year, *param)

    # Calculate confidence interval
    stderr = np.sqrt(np.diag(covar))
    conf_interval = 1.96 * stderr
    upper = logistic(year, *(param + conf_interval))
    lower = logistic(year, *(param - conf_interval))

    # Plot the result
    plt.figure()
    plt.plot(data["Year"], data["CO2 emissions"], label="CO2 emissions")
    plt.plot(year, forecast, label="Forecast")
    plt.fill_between(year, upper, lower, color='purple', alpha=0.2, \
                     label="95% Confidence Interval")
    if predictions is not None:
        plt.plot(year_pred, predictions, 'ro-', label='Predictions')
    plt.xlabel("Year")
    plt.ylabel("CO2 emissions")
    plt.title("Logistic forecast for China")
    plt.legend()
    plt.show()
