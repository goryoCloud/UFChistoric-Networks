# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:05:17 2023

@author: max_s
"""
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math

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
combined_fighters = pd.concat([data['R_fighter'], data['B_fighter']])
unique_fighters = combined_fighters.unique().tolist()

#%%

PPVdata = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/ufc_ppv_buys.csv')

PPVdata['date'] = pd.to_datetime(PPVdata[['Year', 'Month', 'Day']])
PPVdataSorted = PPVdata.sort_values(by='date')
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
    normDegrees = {}
    for key, value in degree_distribution.items():
        normDegrees[key] = value / len(graph.nodes())
    
    average_degree = sum(degree_distribution.values()) / len(degree_distribution)

    return average_degree, normDegrees

def graph_density(graph):
    density = nx.density(graph)

    return density

def average_clustering_coefficient(graph):
    avg_clustering = nx.average_clustering(graph)

    return avg_clustering

def node_clustering_coefficients(graph):
    clustering_coefficients = {}
    for node in graph.nodes():
        clustering_coefficients[node] = nx.clustering(graph, node)
    return clustering_coefficients

def gini(graph):
    degrees = dict(graph.degree())
    degree_values = sorted(degrees.values())
    n = len(degree_values)
    index = np.arange(1, n + 1)
    gini_coefficient = (np.sum((2 * index - n - 1) * degree_values)) / (n * np.sum(degree_values))
    
    return gini_coefficient

def average_path_length_largest_nw(graph):
    largest_component = max(nx.connected_components(graph), key=len)
    largest_subgraph = graph.subgraph(largest_component)
    avg_path_length = nx.average_shortest_path_length(largest_subgraph)

    return avg_path_length

#def pearsonCoeff(arrayX, arrayY):
#    return pearsonr(arrayX, arrayY)

def betweenness_largest_nw(graph):
    largest_component = max(nx.connected_components(graph), key=len)
    largest_subgraph = graph.subgraph(largest_component)
    betweenness_centrality = nx.betweenness_centrality(largest_subgraph)
    average_betweenness_centrality = sum(betweenness_centrality.values()) / len(betweenness_centrality)
    
    return average_betweenness_centrality, betweenness_centrality

def average_eigenvector_nw(graph):
    largest_component = max(nx.connected_components(graph), key=len)
    largest_subgraph = graph.subgraph(largest_component)
    eigenvector_centrality = nx.eigenvector_centrality(largest_subgraph, max_iter=5000)
    average_eigenvector_centrality = sum(eigenvector_centrality.values()) / len(eigenvector_centrality)

    return average_eigenvector_centrality, eigenvector_centrality
#%%
degreeEvol = pd.DataFrame({'fighter': unique_fighters})
clustEvol = pd.DataFrame({'fighter': unique_fighters})
betweenessEvol = pd.DataFrame({'fighter': unique_fighters})
eigenEvol = pd.DataFrame({'fighter': unique_fighters})

degreeEvol.set_index('fighter', inplace=True)
clustEvol.set_index('fighter', inplace=True)
betweenessEvol.set_index('fighter', inplace=True)
eigenEvol.set_index('fighter', inplace=True)
#%%
fights = nx.Graph()

def get_data_in_window(start_date, end_date):
    return data.loc[start_date:end_date]

def get_PPV_data_in_window(start_date, end_date):
    return PPVdataSorted.loc[start_date:end_date]

maleWeights = ['Bantamweight', 'Middleweight', 'Heavyweight', 'Lightweight', 'Welterweight', 'Flyweight', 'LightHeavyweight', 'Featherweight', 'CatchWeight', 'OpenWeight']
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')
data = data.set_index('date')
PPVdataSorted = PPVdataSorted.set_index('date')
window_size = pd.DateOffset(years = 2)
freq = pd.DateOffset(months=1)

start_date = data.index.min()
end_date = start_date + window_size
result = []

deg = []
den = []
clust = []
gin = []
pathLenght = []
betweeness = []
numFights = []
numFighters = []
eigenvector = []

dates = []
PPVsales = []

numWin = 0
while end_date <= data.index.max():
    numWin = numWin + 1
    
    fightCount = 0
    
    window_data = get_data_in_window(start_date, end_date)
    PPV_window_data = get_PPV_data_in_window(start_date, end_date)
    
    meanPPV = PPV_window_data['PPV'].mean()
#    if math.isnan(meanPPV):
#        PPVsales.append(0)
#    else:
#        PPVsales.append(meanPPV)
    PPVsales.append(meanPPV)   

    for fight in range(0, len(window_data)):
        if window_data['weight_class'][fight] in maleWeights:
            fights.add_edge(window_data["R_fighter"][fight], window_data["B_fighter"][fight])
            fightCount = fightCount + 1

    hist = nx.degree_histogram(fights)
    degrees = range(0, len(hist))

    concatenated_columns = pd.concat([window_data["R_fighter"], window_data["B_fighter"]])
    num_unique_fighters = concatenated_columns.nunique()
    
    windowDates = numWin, str(start_date), str(end_date)
    
    dates.append(windowDates)
    
    
    deg = average_degree_distribution(fights)[1]
    nodeClust = node_clustering_coefficients(fights)
    between = betweenness_largest_nw(fights)[1]
    eigen = average_eigenvector_nw(fights)[1]
    
    for name in deg:
        y = name
        x = numWin
        
        degreeEvol.at[y, x] = deg[y]
        
    for name in nodeClust:
        y = name
        x = numWin  
        
        clustEvol.at[y, x] = nodeClust[y]
        
    for name in between:
        y = name
        x = numWin
        
        betweenessEvol.at[y, x] = between[y]
        
    for name in eigen:
        y = name
        x = numWin
        
        eigenEvol.at[y, x] = eigen[y]
    
#    avDens = graph_density(fights)
#    avClust = average_clustering_coefficient(fights)
#    locGin = gini(fights)
#    locPathLenght = average_path_length_largest_nw(fights)
#    locBetweeness = betweenness_largest_nw(fights)
#    locEigenvector = average_eigenvector_nw(fights)
    
    fights.clear()
    print(f'Processing: Window {numWin} done...')
    start_date += freq
    end_date += freq
#%%
row_name = 'BJ Penn'

# Obtener la fila especificada
row_values = betweenessEvol.loc[row_name]

# Trazar los valores de la fila
plt.xlim(0, 301)
plt.xlabel('window index')
#plt.ylabel(r'$k$')
plt.title(f'{row_name} degree evolution')
row_values.plot(marker='o', linestyle='-')
#%%
path = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/dataframes2'
degreeEvol.to_csv(f'{path}/degreeEvol.csv', index=True)
clustEvol.to_csv(f'{path}/clustEvol.csv', index=True)
betweenessEvol.to_csv(f'{path}/betweenessEvol.csv', index=True)
eigenEvol.to_csv(f'{path}/eigenEvol.csv', index=True)