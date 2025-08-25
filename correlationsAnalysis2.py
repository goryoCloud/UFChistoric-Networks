# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 18:28:23 2025

@author: max_s
"""

import json
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

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

with open(f'{path}/UFCdata.json', 'r') as file:
    data = json.load(file)

#%%    
# List of metrics to plot
deg_metrics = ["degree_un", "degree_w", "degree_l", "degree_dir", "degree_dir_w", "degree_dir_l" ]

all_data = []
for metric_name in deg_metrics:
    metric_list = data.get(metric_name, [])
    filtered_metric_list = [x for x in metric_list if not np.isnan(x)]
    all_data.extend(filtered_metric_list)

# Create common bins based on the range of all data
common_bins = np.linspace(min(all_data), max(all_data), 16)

# Plot histograms with the common bins
plt.figure(figsize=(8, 6))
plt.title('Histogram of Degrees', fontsize=18)
plt.xlabel('Pearson correlation coefficient value', fontsize=18)
plt.ylabel('frequency', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.6)

for metric_name in deg_metrics:
    metric_list = data.get(metric_name, [])
    filtered_metric_list = [x for x in metric_list if not np.isnan(x)]
    filtered_metric_list = np.array(filtered_metric_list)
    plt.hist(
        filtered_metric_list, 
        bins=common_bins, 
        label=f'{metric_name}', 
        histtype='step', 
        linewidth=2.5, 
        alpha=1, 
        stacked=True, 
        weights=np.ones_like(filtered_metric_list) / len(filtered_metric_list)
    )

plt.legend(fontsize=18)
plt.show()
#%%
# List of clustering metrics to plot
clust_metrics = ["clust_un", "clust_w", "clust_l", "clust_dir", "clust_dir_w", "clust_dir_l"]

# Determine common bins for all clustering metrics
all_clust_data = []
for metric_name in clust_metrics:
    metric_list = data.get(metric_name, [])
    filtered_metric_list = [x for x in metric_list if not np.isnan(x)]
    all_clust_data.extend(filtered_metric_list)

# Create common bins based on the range of all clustering data
common_bins_clust = np.linspace(min(all_clust_data), max(all_clust_data), 12)

# Plot histograms for clustering metrics with common bins

plt.figure(figsize=(8, 6))
plt.title('Histogram of Clustering Metrics', fontsize=18)
plt.xlabel('Pearson correlation coefficient value', fontsize=18)
plt.ylabel('frequency', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(-1.25, 1.25)

for metric_name in clust_metrics:
    metric_list = data.get(metric_name, [])
    filtered_metric_list = [x for x in metric_list if not np.isnan(x)]
    filtered_metric_list = np.array(filtered_metric_list)
    plt.hist(
        filtered_metric_list, 
        bins=common_bins_clust, 
        label=f'{metric_name}', 
        histtype='step', 
        linewidth=2.5, 
        alpha=1, 
        stacked=True, 
        weights=np.ones_like(filtered_metric_list) / len(filtered_metric_list)
    )

plt.legend(fontsize=18)
plt.show()

#%%
# List of betweenness metrics to plot
bet_metrics = ["bet_un", "bet_w", "bet_l", "bet_dir", "bet_dir_w", "bet_dir_l"]

# Determine common bins for all betweenness metrics
all_bet_data = []
for metric_name in bet_metrics:
    metric_list = data.get(metric_name, [])
    filtered_metric_list = [x for x in metric_list if not np.isnan(x)]
    all_bet_data.extend(filtered_metric_list)

# Create common bins based on the range of all betweenness data
common_bins_bet = np.linspace(min(all_bet_data), max(all_bet_data), 16)

# Plot histograms for betweenness metrics with common bins
plt.figure(figsize=(8, 6))
plt.title('Histogram of Betweenness Metrics', fontsize=18)
plt.xlabel('Pearson correlation coefficient value', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.6)

for metric_name in bet_metrics:
    metric_list = data.get(metric_name, [])
    filtered_metric_list = [x for x in metric_list if not np.isnan(x)]
    filtered_metric_list = np.array(filtered_metric_list)
    plt.hist(
        filtered_metric_list, 
        bins=common_bins_bet, 
        label=f'{metric_name}', 
        histtype='step', 
        linewidth=2.5, 
        alpha=1, 
        stacked=True, 
        weights=np.ones_like(filtered_metric_list) / len(filtered_metric_list)
    )

plt.legend(fontsize=18)
plt.show()
#%%
# List of eigenvector metrics to plot
eigen_metrics = ["eigen_un", "eigen_w", "eigen_l", "eigen_dir", "eigen_dir_w", "eigen_dir_l"]

# Determine common bins for all eigenvector metrics
all_eigen_data = []
for metric_name in eigen_metrics:
    metric_list = data.get(metric_name, [])
    filtered_metric_list = [x for x in metric_list if not np.isnan(x)]
    all_eigen_data.extend(filtered_metric_list)

# Create common bins based on the range of all eigenvector data
common_bins_eigen = np.linspace(min(all_eigen_data), max(all_eigen_data), 16)

# Plot histograms for eigenvector metrics with common bins
plt.figure(figsize=(8, 6))
plt.title('Histogram of Eigenvector Metrics', fontsize=18)
plt.xlabel('Pearson correlation coefficient value', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.6)

for metric_name in eigen_metrics:
    metric_list = data.get(metric_name, [])
    filtered_metric_list = [x for x in metric_list if not np.isnan(x)]
    filtered_metric_list = np.array(filtered_metric_list)
    plt.hist(
        filtered_metric_list, 
        bins=common_bins_eigen, 
        label=f'{metric_name}', 
        histtype='step', 
        linewidth=2.5, 
        alpha=1, 
        stacked=True, 
        weights=np.ones_like(filtered_metric_list) / len(filtered_metric_list)
    )

plt.legend(fontsize=18)
plt.show()
