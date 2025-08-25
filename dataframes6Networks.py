# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:05:17 2023

@author: max_s
"""
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

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
PPVdata = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/ufc_ppv_buys.csv')
PPVdata['Month'] = pd.to_datetime(PPVdata['Month'])
PPVdataSorted = PPVdata.sort_values(by='Month')

PPVdata['date'] = pd.to_datetime(PPVdata[['Year', 'Month', 'Day']])
PPVdata = PPVdata.sort_values(by='date')

googleData = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/multiTimeline (1).csv')
googleData['Month'] = pd.to_datetime(googleData['Month'])
googleData = googleData.sort_values(by='Month')

# Define weight classes
maleWeights = ['Bantamweight', 'Middleweight', 'Heavyweight', 'Lightweight', 'Welterweight', 'Flyweight', 'LightHeavyweight', 'Featherweight', 'CatchWeight', 'OpenWeight']

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


def graph_density(graph, directed=True):

    if not isinstance(graph, (nx.Graph, nx.DiGraph)):
        raise ValueError("The graph must be a NetworkX Graph or DiGraph.")
    
    # Calculate density based on the directed argument
    if directed:
        if not isinstance(graph, nx.DiGraph):
            raise ValueError("The graph must be a directed graph (DiGraph) to calculate directed density.")
        return nx.density(graph)
    else:
        # Convert to undirected if necessary
        undirected_graph = graph.to_undirected()
        return nx.density(undirected_graph)

def gini(graph):
    degrees = dict(graph.degree())
    degree_values = sorted(degrees.values())
    n = len(degree_values)
    index = np.arange(1, n + 1)
    gini_coefficient = (np.sum((2 * index - n - 1) * degree_values)) / (n * np.sum(degree_values))
    
    return gini_coefficient

def average_path_length_largest_nw(graph, directed=True):
    if not isinstance(graph, (nx.Graph, nx.DiGraph)):
        raise ValueError("The graph must be a NetworkX Graph or DiGraph.")

    # Determine components based on graph type
    if directed:
        if not isinstance(graph, nx.DiGraph):
            raise ValueError("The graph must be a directed graph (DiGraph) when 'directed' is True.")
        components = list(nx.strongly_connected_components(graph))
    else:
        # Convert directed graph to undirected if necessary
        if isinstance(graph, nx.DiGraph):
            graph = graph.to_undirected()
        components = list(nx.connected_components(graph))

    # Calculate the average path length for each component
    avg_path_lengths = []
    for component in components:
        subgraph = graph.subgraph(component)
        try:
            avg_path_length = nx.average_shortest_path_length(subgraph)
            avg_path_lengths.append(avg_path_length)
        except nx.NetworkXError:
            # Skip components where the average path length cannot be calculated
            continue

    # Compute the overall average path length
    if avg_path_lengths:
        overall_avg_path_length = sum(avg_path_lengths) / len(avg_path_lengths)
    else:
        overall_avg_path_length = float('nan')  # No valid components

    return overall_avg_path_length

def pearsonCoeff(arrayX, arrayY):
    return pearsonr(arrayX, arrayY)

def betweenness_largest_nw(graph, directed=True):
    if directed:
        if not isinstance(graph, nx.DiGraph):
            raise ValueError("The graph must be a directed graph (DiGraph) when 'directed' is True.")
        # Calculate betweenness centrality for directed graphs
        return nx.betweenness_centrality(graph, normalized=True)
    else:
        if not isinstance(graph, (nx.Graph, nx.DiGraph)):
            raise ValueError("The graph must be an undirected graph (Graph) when 'directed' is False.")
        # Convert to undirected graph and calculate betweenness centrality
        undirected_graph = graph.to_undirected()
        return nx.betweenness_centrality(undirected_graph, normalized=True)

def average_eigenvector_nw(graph, directed=True):

    if not isinstance(graph, (nx.Graph, nx.DiGraph)):
        raise ValueError("The graph must be a NetworkX Graph or DiGraph.")
    
    # Convert to undirected if necessary
    if not directed:
        graph = graph.to_undirected()
    
    # Calculate eigenvector centrality for the entire graph
    try:
        return nx.eigenvector_centrality(graph, max_iter=5000)
    except nx.NetworkXException as e:
        raise ValueError(f"Eigenvector centrality could not be computed: {e}")

def average_clustering(graph, directed=True):
    if directed:
        if not isinstance(graph, nx.DiGraph):
            raise ValueError("The graph must be a directed graph (DiGraph) when 'directed' is True.")
        # Calculate out-clustering manually for directed graphs
        clustering = {}
        for node in graph.nodes:
            neighbors = set(graph.successors(node))
            if len(neighbors) < 2:
                clustering[node] = 0.0
            else:
                possible_links = len(neighbors) * (len(neighbors) - 1)
                actual_links = sum(1 for n1 in neighbors for n2 in neighbors if n1 != n2 and graph.has_edge(n1, n2))
                clustering[node] = actual_links / possible_links
        return clustering
    else:
        if not isinstance(graph, (nx.Graph, nx.DiGraph)):
            raise ValueError("The graph must be an undirected graph (Graph) when 'directed' is False.")
        # Calculate clustering using NetworkX for undirected graphs
        return nx.clustering(graph)


def average_degree(graph, directed=True):
    if directed:
        if not isinstance(graph, nx.DiGraph):
            raise ValueError("The graph must be a directed graph (DiGraph) when 'directed' is True.")
        return dict(graph.in_degree())
    else:
        if not isinstance(graph, (nx.Graph, nx.DiGraph)):
            raise ValueError("The graph must be an undirected or directed graph when 'directed' is False.")
        return dict(graph.degree())
#%%    
def get_data_in_window(start_date, end_date):
    return data.loc[start_date:end_date]

def get_PPV_data_in_window(start_date, end_date):
    return PPVdata.loc[start_date:end_date]

def get_google_averages(start_date, end_date):
    return googleData.loc[start_date:end_date]

#%% Create DataFrames
PPVDf = pd.DataFrame(columns = ['star_date', 'end_date','PPVmean', 'googleMean'])

degreeDf_un = pd.DataFrame()
degreeDf_dir = pd.DataFrame()

clustDf_un= pd.DataFrame()
clustDf_dir= pd.DataFrame()

betDf_un = pd.DataFrame()
betDf_dir = pd.DataFrame()

eigenDf_un = pd.DataFrame()
eigenDf_dir = pd.DataFrame()

degreeDf_un_w = pd.DataFrame()
degreeDf_un_l = pd.DataFrame()
        
clustDf_un_w = pd.DataFrame()
clustDf_un_l = pd.DataFrame()
        
betDf_un_w = pd.DataFrame()
betDf_un_l = pd.DataFrame()
        
eigenDf_un_w = pd.DataFrame()
eigenDf_un_l = pd.DataFrame()

degreeDf_dir_w = pd.DataFrame()
degreeDf_dir_l = pd.DataFrame()
        
clustDf_dir_w = pd.DataFrame()
clustDf_dir_l = pd.DataFrame()
        
betDf_dir_w = pd.DataFrame()
betDf_dir_l = pd.DataFrame()
        
eigenDf_dir_w = pd.DataFrame()
eigenDf_dir_l = pd.DataFrame()

#%% Iterating through windows
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')
data = data.set_index('date')
PPVdata = PPVdata.set_index('date')
googleData = googleData.set_index('Month')
window_size = pd.DateOffset(years = 2)
freq = pd.DateOffset(months=1)

start_date = data.index.min()
end_date = start_date + window_size

fights = nx.DiGraph()
winners = nx.DiGraph()
loosers = nx.DiGraph()
#%%
winIn = 0

while end_date <= data.index.max():
    print(f'{start_date} - {end_date}')
    
    window_data = get_data_in_window(start_date, end_date)
    PPV_window_data = get_PPV_data_in_window(start_date, end_date)
    PPVmean = PPV_window_data['PPV'].mean()
    
    google_window_data = get_google_averages(start_date, end_date)
    meanGoogle = google_window_data['ufc'].mean()
    
    PPVDf.loc[len(PPVDf)] = [start_date, end_date, PPVmean, meanGoogle]
    
    for _, row in window_data.iterrows():
        if row['weight_class'] in maleWeights:
            if row['R_fighter'] not in fights:
                fights.add_node(row['R_fighter'])
            if row['B_fighter'] not in fights:
                fights.add_node(row['B_fighter'])

    for _, row in window_data.iterrows():
        if row['weight_class'] in maleWeights:
            fights.add_edge(row['R_fighter'], row['B_fighter'])

    for _, row in window_data.iterrows():
        if row['weight_class'] in maleWeights:
            if row['R_fighter'] not in winners:
                winners.add_node(row['R_fighter'])
                loosers.add_node(row['R_fighter'])
            if row['B_fighter'] not in winners:
                winners.add_node(row['B_fighter'])
                loosers.add_node(row['B_fighter'])

    for _, row in window_data.iterrows():
        if row['weight_class'] in maleWeights:
            if row['Winner'].strip().lower() == 'red':
                winners.add_edge(row['R_fighter'], row['B_fighter'])
                loosers.add_edge(row['B_fighter'], row['R_fighter'])
            if row['Winner'].strip().lower() == 'blue':
                winners.add_edge(row['B_fighter'], row['R_fighter'])
                loosers.add_edge(row['R_fighter'], row['B_fighter'])
    
    print('***************************************************************************')
    print(f"Number of nodes in fights graph: {fights.number_of_nodes()}")
    print(f"Number of nodes in winners graph: {winners.number_of_nodes()}")
    print(f"Number of nodes in loosers graph: {loosers.number_of_nodes()}")
    print('---------------------------------------------------------------------------')
    print(f"Number of edges in fights graph: {fights.number_of_edges()}")
    print(f"Number of edges in winners graph: {winners.number_of_edges()}")
    print(f"Number of edges in loosers graph: {loosers.number_of_edges()}")
             
    
    deg_un = average_degree(fights, directed=False)
    clust_un = average_clustering(fights, directed=False)
    bet_un = betweenness_largest_nw(fights, directed=False)
    eigen_un = average_eigenvector_nw(fights, directed=False)
    
    deg_dir = average_degree(fights, directed=True)
    clust_dir = average_clustering(fights, directed=True)
    bet_dir = betweenness_largest_nw(fights, directed=True)
    eigen_dir = average_eigenvector_nw(fights, directed=True)
    
    for name, value in deg_un.items():
        degreeDf_un.loc[name, f'{start_date}'] = value
        
    for name, value in deg_dir.items():
        degreeDf_dir.loc[name, f'{start_date}'] = value
        
    for name, value in clust_un.items():
        clustDf_un.loc[name, f'{start_date}'] = value
        
    for name, value in clust_dir.items():
        clustDf_dir.loc[name, f'{start_date}'] = value
        
    for name, value in bet_un.items():
        betDf_un.loc[name, f'{start_date}'] = value
        
    for name, value in bet_dir.items():
        betDf_dir.loc[name, f'{start_date}'] = value
        
    for name, value in eigen_un.items():
        eigenDf_un.loc[name, f'{start_date}'] = value
        
    for name, value in eigen_dir.items():
        eigenDf_dir.loc[name, f'{start_date}'] = value
    
    deg_un_w = average_degree(winners, directed=False)
    clust_un_w = average_clustering(winners, directed=False)
    bet_un_w = betweenness_largest_nw(winners, directed=False)
    eigen_un_w = average_eigenvector_nw(winners, directed=False)
    
    deg_un_l = average_degree(loosers, directed=False)
    clust_un_l = average_clustering(loosers, directed=False)
    bet_un_l = betweenness_largest_nw(loosers, directed=False)
    eigen_un_l = average_eigenvector_nw(loosers, directed=False)
    
    for name, value in deg_un_w.items():
        degreeDf_un_w.loc[name, f'{start_date}'] = value
        
    for name, value in deg_un_l.items():
        degreeDf_un_l.loc[name, f'{start_date}'] = value
        
    for name, value in clust_un_w.items():
        clustDf_un_w.loc[name, f'{start_date}'] = value
        
    for name, value in clust_un_l.items():
        clustDf_un_l.loc[name, f'{start_date}'] = value
        
    for name, value in bet_un_w.items():
        betDf_un_w.loc[name, f'{start_date}'] = value
        
    for name, value in bet_un_l.items():
        betDf_un_l.loc[name, f'{start_date}'] = value
        
    for name, value in eigen_un_w.items():
        eigenDf_un_w.loc[name, f'{start_date}'] = value
        
    for name, value in eigen_un_l.items():
        eigenDf_un_l.loc[name, f'{start_date}'] = value
    
    deg_dir_w = average_degree(winners, directed=True)
    clust_dir_w = average_clustering(winners, directed=True)
    bet_dir_w = betweenness_largest_nw(winners, directed=True)
    eigen_dir_w = average_eigenvector_nw(winners, directed=True)
    
    deg_dir_l = average_degree(loosers, directed=True)
    clust_dir_l = average_clustering(loosers, directed=True)
    bet_dir_l = betweenness_largest_nw(loosers, directed=True)
    eigen_dir_l = average_eigenvector_nw(loosers, directed=True)
    
    for name, value in deg_dir_w.items():
        degreeDf_dir_w.loc[name, f'{start_date}'] = value
        
    for name, value in deg_dir_l.items():
        degreeDf_dir_l.loc[name, f'{start_date}'] = value
        
    for name, value in clust_dir_w.items():
        clustDf_dir_w.loc[name, f'{start_date}'] = value
        
    for name, value in clust_dir_l.items():
        clustDf_dir_l.loc[name, f'{start_date}'] = value
        
    for name, value in bet_dir_w.items():
        betDf_dir_w.loc[name, f'{start_date}'] = value
        
    for name, value in bet_dir_l.items():
        betDf_dir_l.loc[name, f'{start_date}'] = value
        
    for name, value in eigen_dir_w.items():
        eigenDf_dir_w.loc[name, f'{start_date}'] = value
        
    for name, value in eigen_dir_l.items():
        eigenDf_dir_l.loc[name, f'{start_date}'] = value
    
    winIn = winIn + 1
    
    fights.clear()
    winners.clear()
    loosers.clear()

    start_date += freq
    end_date += freq
    
#%%
path = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/dataframes6Networks'

PPVDf.to_csv(f'{path}/PPV_averages.csv', index=False)

# Undirected graphs
degreeDf_un.to_csv(f'{path}/degreeDf_un.csv', index=True)
clustDf_un.to_csv(f'{path}/clustDf_un.csv', index=True)
betDf_un.to_csv(f'{path}/betDf_un.csv', index=True)
eigenDf_un.to_csv(f'{path}/eigenDf_un.csv', index=True)

degreeDf_un_w.to_csv(f'{path}/degreeDf_un_winners.csv', index=True)
degreeDf_un_l.to_csv(f'{path}/degreeDf_un_loosers.csv', index=True)

clustDf_un_w.to_csv(f'{path}/clustDf_un_winners.csv', index=True)
clustDf_un_l.to_csv(f'{path}/clustDf_un_loosers.csv', index=True)

betDf_un_w.to_csv(f'{path}/betDf_un_winners.csv', index=True)
betDf_un_l.to_csv(f'{path}/betDf_un_loosers.csv', index=True)

eigenDf_un_w.to_csv(f'{path}/eigenDf_un_winners.csv', index=True)
eigenDf_un_l.to_csv(f'{path}/eigenDf_un_loosers.csv', index=True)

# Directed graphs
degreeDf_dir.to_csv(f'{path}/degreeDf_dir.csv', index=True)
clustDf_dir.to_csv(f'{path}/clustDf_dir.csv', index=True)
betDf_dir.to_csv(f'{path}/betDf_dir.csv', index=True)
eigenDf_dir.to_csv(f'{path}/eigenDf_dir.csv', index=True)

degreeDf_dir_w.to_csv(f'{path}/degreeDf_dir_winners.csv', index=True)
degreeDf_dir_l.to_csv(f'{path}/degreeDf_dir_loosers.csv', index=True)

clustDf_dir_w.to_csv(f'{path}/clustDf_dir_winners.csv', index=True)
clustDf_dir_l.to_csv(f'{path}/clustDf_dir_loosers.csv', index=True)

betDf_dir_w.to_csv(f'{path}/betDf_dir_winners.csv', index=True)
betDf_dir_l.to_csv(f'{path}/betDf_dir_loosers.csv', index=True)

eigenDf_dir_w.to_csv(f'{path}/eigenDf_dir_winners.csv', index=True)
eigenDf_dir_l.to_csv(f'{path}/eigenDf_dir_loosers.csv', index=True)
