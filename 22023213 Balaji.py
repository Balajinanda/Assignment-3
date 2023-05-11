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
    
    # Load and preprocess the data for clustering
liquid = pd.read_csv("CO2 emissions from liquid fuel consumption (kt).csv", \
                     skiprows=4)
solid = pd.read_csv("CO2 emissions from solid fuel consumption (kt).csv", \
                    skiprows=4)

# Drop rows with NaN values in 2015
liquid = liquid[liquid["2015"].notna()]
solid = solid.dropna(subset=["2015"])

# Merge the datasets for 2015
liquid2015 = liquid[["Country Name", "Country Code", "2015"]].copy()
solid2015 = solid[["Country Name", "Country Code", "2015"]].copy()
df_2015 = pd.merge(liquid2015, solid2015, on="Country Name", how="outer")
df_2015 = df_2015.dropna()  # Drop entries with missing values
df_2015 = df_2015.rename(columns={"2015_x": "CO2 from liquid fuel", \
                                  "2015_y": "CO2 from solid fuel"})

# Perform clustering
df_cluster = df_2015[["CO2 from liquid fuel", "CO2 from solid fuel"]].copy()
scaler = StandardScaler()
df_cluster = scaler.fit_transform(df_cluster)

# Loop over the number of clusters and calculate silhouette scores
silhouette_scores = []
for ncluster in range(2, 10):
    kmeans = KMeans(n_clusters=ncluster)
    kmeans.fit(df_cluster)
    labels = kmeans.labels_
    score = skmet.silhouette_score(df_cluster, labels)
    silhouette_scores.append(score)
    print(f"Number of Clusters: {ncluster}, Silhouette Score: {score}")
    
# Plot the silhouette scores
plt.figure()
plt.plot(range(2, 10), silhouette_scores)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different Numbers of Clusters")
plt.show()

# Find the optimal number of clusters with the highest silhouette score
optimal_ncluster = np.argmax(silhouette_scores) + 2
print(f"\nOptimal Number of Clusters: {optimal_ncluster}")
