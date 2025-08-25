# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 18:28:23 2025

@author: max_s
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# Configure matplotlib for LaTeX-style text
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "mathptmx",
})

plt.rc('axes', labelsize='xx-large')
plt.rc('axes', titlesize='xx-large')
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')

path = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/dataframes6Networks'
#%%
# Load JSON data
with open(f'{path}/UFCdata.json', 'r') as file:
    data = json.load(file)
#%%
# Function to map metric names to descriptive labels
def get_label(metric_name):
    if "degree" in metric_name:
        base = "Degree"
    elif "clust" in metric_name:
        base = "Clustering"
    elif "bet" in metric_name:
        base = "Betweenness"
    elif "eigen" in metric_name:
        base = "Eigenvector"
    else:
        base = metric_name  # Default to the metric name if no match

    if "_un" in metric_name:
        return "undirected"
    elif "_w" in metric_name and "_dir" not in metric_name:
        return "undirected winners"
    elif "_l" in metric_name and "_dir" not in metric_name:
        return "undirected losers"
    elif "_dir" in metric_name and "_w" in metric_name:
        return "directed winners"
    elif "_dir" in metric_name and "_l" in metric_name:
        return "directed losers"
    elif "_dir" in metric_name:
        return "directed"
    else:
        return metric_name  # Fallback

# Function to plot histograms with aligned bins
def plot_histogram(metrics, title, xlabel, ylabel, aligned_bins, xticks_values):
    plt.figure(figsize=(8, 6))
#    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(xticks_values)  # Set x-ticks for alignment

    for metric_name in metrics:
        metric_list = data.get(metric_name, [])
        filtered_metric_list = [x for x in metric_list if not np.isnan(x)]
        filtered_metric_list = np.array(filtered_metric_list)
        plt.hist(
            filtered_metric_list, 
            bins=aligned_bins,  # Use aligned bins
            label=get_label(metric_name),  # Use descriptive label
            histtype='step', 
            linewidth=2.5, 
            alpha=1, 
            stacked=True, 
            weights=np.ones_like(filtered_metric_list) / len(filtered_metric_list)
        )

    plt.legend(fontsize=13)
    plt.show()

# Degree Metrics
deg_metrics = ["degree_un", "degree_w", "degree_l", "degree_dir", "degree_dir_w", "degree_dir_l"]
aligned_bins_deg = np.linspace(-1, 1, 17)  # 16 bins aligned to x-axis ticks
plot_histogram(deg_metrics, 'Histogram of Degrees', 'Pearson correlation coefficient value', 'frequency', aligned_bins_deg, np.linspace(-1, 1, 9))

# Clustering Metrics
clust_metrics = ["clust_un", "clust_w", "clust_l", "clust_dir", "clust_dir_w", "clust_dir_l"]
aligned_bins_clust = np.linspace(-1, 1, 13)  # 12 bins aligned to x-axis ticks
plot_histogram(clust_metrics, 'Histogram of Clustering Metrics', 'Pearson correlation coefficient value', 'frequency', aligned_bins_clust, np.linspace(-1, 1, 9))

# Betweenness Metrics
bet_metrics = ["bet_un", "bet_w", "bet_l", "bet_dir", "bet_dir_w", "bet_dir_l"]
aligned_bins_bet = np.linspace(-1, 1, 17)  # 16 bins aligned to x-axis ticks
plot_histogram(bet_metrics, 'Histogram of Betweenness Metrics', 'Pearson correlation coefficient value', 'frequency', aligned_bins_bet, np.linspace(-1, 1, 9))

# Eigenvector Metrics
eigen_metrics = ["eigen_un", "eigen_w", "eigen_l", "eigen_dir", "eigen_dir_w", "eigen_dir_l"]
aligned_bins_eigen = np.linspace(-1, 1, 17)  # 16 bins aligned to x-axis ticks
plot_histogram(eigen_metrics, 'Histogram of Eigenvector Metrics', 'Pearson correlation coefficient value', 'frequency', aligned_bins_eigen, np.linspace(-1, 1, 9))
