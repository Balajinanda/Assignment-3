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

#Set the number of clusters to 2
ncluster = 2
kmeans = KMeans(n_clusters=ncluster)
kmeans.fit(df_cluster)
labels = kmeans.labels_
cen = kmeans.cluster_centers_
xcen = cen[:, 0]
ycen = cen[:, 1]

#Plot the clusters and cluster centers
plt.figure(figsize=(6, 5))
cm = plt.cm.get_cmap('tab10')
for i, label in enumerate(np.unique(labels)):

    plt.scatter(df_cluster[labels == label, 0], \
                df_cluster[labels == label, 1], 10, \
                    label=f"Cluster {label}", cmap=cm, alpha=0.7)
plt.scatter(xcen, ycen, 45, "k", marker="d", label="Cluster centers")
plt.xlabel("CO2 from liquid fuel")
plt.ylabel("CO2 from solid fuel")
plt.title("Kmeans clustering")
plt.legend()
plt.show()

#Read and preprocess the data for CO2 emissions
CO2 = pd.read_csv("CO2 emissions (kt).csv", skiprows=4)
CO2 = CO2.set_index('Country Name', drop=True)
CO2 = CO2.loc[:, '1960':'2021']
CO2 = CO2.transpose()
CO2 = CO2.loc[:, 'China']
df = CO2.dropna(axis=0)

df_co2 = pd.DataFrame()
df_co2['Year'] = pd.DataFrame(df.index)
df_co2['CO2 emissions'] = pd.DataFrame(df.values)

#Convert year column to numeric
df_co2["Year"] = pd.to_numeric(df_co2["Year"])

#Fit the logistic function to the data
param, covar = curve_fit(logistic, df_co2["Year"], \
                         df_co2["CO2 emissions"], p0=(1.2e12, 0.03, 1990.0), \
                             maxfev=10000)

#Generate years for the forecast
year = np.arange(1960, 2031)

#Plot the forecast
plot_forecast(year, df_co2, param, covar)

# Generate predictions for the next 10 years
year_pred = np.arange(2022, 2031)
predictions = logistic(year_pred, *param)

# Plot the forecast and show predictions
plot_forecast(year, df_co2, param, covar, predictions)

# Print the predictions
print("Year\tPredicted CO2 emissions")
for year, prediction in zip(year_pred, predictions):
    print(f"{year}\t{prediction}")
