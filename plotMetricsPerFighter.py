# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:16:57 2025

@author: max_s
"""

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

# Configure matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "mathptmx",
})

plt.rc('axes', labelsize='x-large')
plt.rc('axes', titlesize='x-large')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

path = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/dataframes6Networks'

metrics_files = {
    "degree_un": f"{path}/degreeDf_un.csv",
    "clust_un": f"{path}/clustDf_un.csv",
    "bet_un": f"{path}/betDf_un.csv",
    "eigen_un": f"{path}/eigenDf_un.csv",
    "degree_w": f"{path}/degreeDf_un_winners.csv",
    "degree_l": f"{path}/degreeDf_un_loosers.csv",
    "clust_w": f"{path}/clustDf_un_winners.csv",
    "clust_l": f"{path}/clustDf_un_loosers.csv",
    "bet_w": f"{path}/betDf_un_winners.csv",
    "bet_l": f"{path}/betDf_un_loosers.csv",
    "eigen_w": f"{path}/eigenDf_un_winners.csv",
    "eigen_l": f"{path}/eigenDf_un_loosers.csv",
    "degree_dir": f"{path}/degreeDf_dir.csv",
    "clust_dir": f"{path}/clustDf_dir.csv",
    "bet_dir": f"{path}/betDf_dir.csv",
    "eigen_dir": f"{path}/eigenDf_dir.csv",
    "degree_dir_w": f"{path}/degreeDf_dir_winners.csv",
    "degree_dir_l": f"{path}/degreeDf_dir_loosers.csv",
    "clust_dir_w": f"{path}/clustDf_dir_winners.csv",
    "clust_dir_l": f"{path}/clustDf_dir_loosers.csv",
    "bet_dir_w": f"{path}/betDf_dir_winners.csv",
    "bet_dir_l": f"{path}/betDf_dir_loosers.csv",
    "eigen_dir_w": f"{path}/eigenDf_dir_winners.csv",
    "eigen_dir_l": f"{path}/eigenDf_dir_loosers.csv",
}

metrics_data = {
    name: pd.read_csv(filepath).set_index(pd.read_csv(filepath).columns[0]) 
    for name, filepath in metrics_files.items()
}

metrics_order = sorted(metrics_data.keys())

###########################################################################
# A helper to parse each key into a descriptive subplot title.
###########################################################################
def parse_title_from_key(key):
    # Identify which metric we have
    if key.startswith("degree"):
        metric = "Degree"
    elif key.startswith("clust"):
        metric = "Clustering"
    elif key.startswith("bet"):
        metric = "Betweenness"
    elif key.startswith("eigen"):
        metric = "Eigenvector"
    else:
        metric = key  # fallback if something else

    # Directed vs Undirected
    if "_dir" in key:
        graph_type = "directed"
    else:
        graph_type = "undirected"

    # Winners vs Losers
    if key.endswith("_w"):
        group = "winners"
    elif key.endswith("_l"):
        group = "losers"
    else:
        group = ""

    parts = [metric, graph_type]
    if group:
        parts.append(group)
    return " ".join(parts)

###########################################################################
# Plot setup
###########################################################################
fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(12, 17), sharex=False, sharey=False)
axes = axes.flatten()

fighters = ["Jon Jones", "Conor McGregor", "Charles Oliveira"]
colors   = ["red",      "blue",           "green"         ]

for i, key in enumerate(metrics_order):
    ax = axes[i]
    df = metrics_data[key]
    
    # Convert columns to datetime
    df.columns = pd.to_datetime(df.columns, errors='coerce')
    
    # Plot for each fighter
    for fighter, color in zip(fighters, colors):
        if fighter in df.index:
            y = df.loc[fighter]
            ax.plot(df.columns, y, label=fighter, color=color)
    
    # If this is a "degree" metric, start y-axis at 1
    if key.startswith("degree"):
        ax.set_ylim(1, None)
    else:
        ax.set_ylim(0, None)
        
    ax.set_xlim(pd.to_datetime("2007-01-01"), pd.to_datetime("2019-01-01"))
    
    # Fewer x-ticks: one tick every 2 years, labeled as YYYY
    ax.xaxis.set_major_locator(mdates.YearLocator(base=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Title and legend
    ax.set_title(parse_title_from_key(key), fontsize=15)
#    ax.legend(fontsize=9)
    ax.tick_params(axis='x', labelrotation=45)

# Hide any unused subplots if fewer than 24 metrics
for j in range(len(metrics_order), 24):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
