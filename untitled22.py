# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 18:22:57 2023

@author: max_s
"""

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

#%%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "mathptmx",})

plt.rc('axes', labelsize='x-large')
plt.rc('axes', titlesize='x-large')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')
#%%
data = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/data.csv')
#%%
def norm(x):
    normValues = []
    norma = sum(x)
    for i in x:
        normValue = i /norma
        normValues.append(normValue)
        normValuesA = np.array(normValues)
    normValues.clear()
    return (normValuesA)

def degree_histogram_directed(G, in_degree=False, out_degree=False):
    nodes = G.nodes()
    if in_degree:
        in_degree = dict(G.in_degree())
        degseq=[in_degree.get(k,0) for k in nodes]
    elif out_degree:
        out_degree = dict(G.out_degree())
        degseq=[out_degree.get(k,0) for k in nodes]
    else:
        degseq=[v for k, v in G.degree()]
    dmax=max(degseq)+1
    freq= [ 0 for d in range(dmax) ]
    for d in degseq:
        freq[d] += 1
    return freq

def fit_exponential_polyfit(x_data, y_data):
    # Take the logarithm of positive y_data, leave zero values as they are
    maskData = y_data > 0
    dataToFitX = x_data[maskData]
    dataToFitY = y_data[maskData]
    
    log_y_data = np.log(dataToFitY)

    # Fit a polynomial to the log-transformed data
    fit = np.poly1d(np.polyfit(dataToFitX, log_y_data, 1))
    t = np.linspace(min(dataToFitX), max(dataToFitX), 100)

    # Extract the fitted parameters
    y_fit = fit[1]*t + fit[0]

    return y_fit, t, fit[1]

def cum_deg_sum(graph):
    import collections
    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cs = np.cumsum(cnt)
    return deg, cs

def cumulative_prob(pmf, x):
    ps = [pmf[value] for value in pmf if value<=x]
    return np.sum(ps)

def average_degree_distribution(graph):
    degree_distribution = dict(graph.degree())

    # Calculate the average degree
    average_degree = sum(degree_distribution.values()) / len(degree_distribution)

    return average_degree

def graph_density(graph):
    density = nx.density(graph)

    return density

def average_clustering_coefficient(graph):
    avg_clustering = nx.average_clustering(graph)

    return avg_clustering

def gini(graph):
    degrees = dict(graph.degree())
    degree_values = sorted(degrees.values())
    n = len(degree_values)
    index = np.arange(1, n + 1)
    gini_coefficient = (np.sum((2 * index - n - 1) * degree_values)) / (n * np.sum(degree_values))
    
    return gini_coefficient

def compute_average_path_length(graph):
    avg_path_length = nx.average_shortest_path_length(graph)

    return avg_path_length
#%%
maleWeights = ['Bantamweight', 'Middleweight', 'Heavyweight', 'Lightweight', 'Welterweight', 'Flyweight', 'LightHeavyweight', 'Featherweight', 'CatchWeight', 'OpenWeight']
fighters = data.R_fighter.unique()

data['date'] = pd.to_datetime(data['date'])
winners_df = pd.DataFrame(columns=['Winner', 'date'])

for fight in range(0, len(data)):
    if data['weight_class'][fight] in maleWeights:
        date = data['date'][fight]
        win = data['Winner'][fight]
        if win == 'Red':
            winner_name = data["R_fighter"][fight]
        elif win == 'Blue':
            winner_name = data["B_fighter"][fight]
        winners_df = pd.concat([winners_df, pd.DataFrame({'Winner': [winner_name], 'date': [date]})], ignore_index= True)
    
#%%
fights = nx.Graph()

#def get_data_in_window(start_date, end_date):
#    # Assuming 'date' column is present in the DataFrame and needs to be converted to datetime
#    data['date'] = pd.to_datetime(data['date'])
    
    # Assuming 'date' should be set as the index
#    data.set_index('date', inplace=True)
    
    # Extract data within the specified date range
#    window_data = data.loc[start_date:end_date]
    
    # Reset index if needed
#    window_data.reset_index(inplace=True)
    
#    return window_data

def get_data_in_window(start_date, end_date):
    return winners_df.loc[start_date:end_date]

winners_df['date'] = pd.to_datetime(winners_df['date'])
winners_df = winners_df.sort_values('date')
winners_df = winners_df.set_index('date')
window_size = pd.DateOffset(years=4)
freq = pd.DateOffset(months=1)

start_date = winners_df.index.min()
end_date = start_date + window_size
result = []

deg = []
den = []
clust = []
gin = []
path = []
numFights = []
numFighters = []

dates = []

numWin = 0

#maxDate = pd.to_datetime(winners_df.index.max())
#%%
while end_date <= winners_df.index.max():
    fightCount = 0
    
    window_data = get_data_in_window(start_date, end_date)
    for i in range(0, len(window_data) - 1):
        fights.add_edge(window_data["Winner"][i], window_data["Winner"][i + 1])
    
    numWin = numWin + 1
    hist = nx.degree_histogram(fights)
    degrees = range(0, len(hist))
    
#    plt.title(f'Degree dist. fights window = {numWin}')
#    plt.plot(degrees, norm(hist), '-ko', ms = 7)
#    plt.yscale('log')
#    plt.xscale('log')
#    plt.xlabel('$k$')
#    plt.ylabel(r'$p(k)$')
#    plt.show()
    
    
#    concatenated_columns = pd.concat([window_data["R_fighter"], window_data["B_fighter"]])
#    num_unique_fighters = concatenated_columns.nunique()
    
    windowDates = start_date, end_date
    
    dates.append(windowDates)
    
    avDeg = average_degree_distribution(fights)
    avDens = graph_density(fights)
    avClust = average_clustering_coefficient(fights)
    locGin = gini(fights)
    locPathLenght = compute_average_path_length(fights)
    
    deg.append(avDeg)
    den.append(avDens)
    clust.append(avClust)
    gin.append(locGin)
    path.append(locPathLenght)
    
    fights.clear()
    
    start_date += freq
    end_date = start_date + window_size 
    
#%%
plt.title('Evolution average clustering coeff.')
plt.plot(clust, '-ro', ms = 5, lw = 3)
plt.xlabel('window index')
plt.ylabel(r'$\langle C  \rangle$')
plt.show()

plt.title('Evolution average degree dist.')
plt.plot(deg, '-bo', ms = 5, lw = 3)
plt.xlabel('window index')
plt.ylabel(r'$\langle p(k)  \rangle$')
plt.show()

plt.title('Evolution average density')
plt.plot(den, '-go', ms = 5, lw = 3)
plt.xlabel('window index')
plt.ylabel(r'$\langle density  \rangle$')
plt.show()

plt.title('Evolution gini coeff.')
plt.plot(gin, '-mo', ms = 5, lw = 3)
plt.xlabel('window index')
plt.ylabel(r'$\langle gini  \rangle$')
plt.show()

plt.title('Evolution path length.')
plt.plot(path, '-ko', ms = 5)
plt.xlabel('window index')
plt.ylabel(r'$\langle l  \rangle$')
plt.show()

plt.title('Diff. number of Fighters')
plt.plot(np.diff(numFighters), '-ko', ms = 5, lw = 3)
plt.xlabel('window index')
plt.ylabel(r'$N$')
plt.show()

plt.title('Diff. avg. deg dist.')
plt.plot(np.diff(deg), '-ko', ms = 5, lw = 3)
plt.xlabel('window index')
plt.ylabel(r'$\langle p(k_{i})  \rangle - \langle p(k_{i-1})  \rangle$')
plt.show()
