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

    if not isinstance(graph, (nx.Graph, nx.DiGraph)):
        raise ValueError("The graph must be a NetworkX Graph or DiGraph.")
    
    # Handle directed and undirected graphs
    if directed:
        if not isinstance(graph, nx.DiGraph):
            raise ValueError("The graph must be a directed graph (DiGraph) when 'directed' is True.")
        betweenness_centrality = nx.betweenness_centrality(graph)
    else:
        undirected_graph = graph.to_undirected()
        betweenness_centrality = nx.betweenness_centrality(undirected_graph)

    # Calculate the average betweenness centrality
    avg_betweenness = sum(betweenness_centrality.values()) / len(betweenness_centrality)
    return avg_betweenness

def average_eigenvector_nw(graph, directed=True):

    if not isinstance(graph, (nx.Graph, nx.DiGraph)):
        raise ValueError("The graph must be a NetworkX Graph or DiGraph.")
    
    # Convert to undirected if necessary
    if not directed:
        graph = graph.to_undirected()
    
    # Calculate eigenvector centrality for the entire graph
    try:
        eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=5000)
    except nx.NetworkXException as e:
        raise ValueError(f"Eigenvector centrality could not be computed: {e}")
    
    # Calculate the average eigenvector centrality
    avg_eigenvector = sum(eigenvector_centrality.values()) / len(eigenvector_centrality)
    return avg_eigenvector

def average_clustering(graph, directed=True):

    if directed and not isinstance(graph, nx.DiGraph):
        raise ValueError("The graph must be a directed graph (DiGraph) when 'directed' is True.")
    
    if not directed and not isinstance(graph, (nx.Graph, nx.DiGraph)):
        raise ValueError("The graph must be an undirected graph (Graph) when 'directed' is False.")
    
    # For directed graphs: Use directed clustering
    if directed:
        clustering_dict = nx.clustering(graph.to_undirected())
    else:
        clustering_dict = nx.clustering(graph)
    
    # Compute average clustering
    num_nodes = len(graph.nodes())
    if num_nodes == 0:
        return 0.0  # No nodes, average clustering is 0

    avg_clustering = sum(clustering_dict.values()) / num_nodes
    return avg_clustering


def average_degree(graph, directed=True):

    if directed and not isinstance(graph, nx.DiGraph):
        raise ValueError("The graph must be a directed graph (DiGraph) when 'directed' is True.")
    
    if not directed and not isinstance(graph, (nx.Graph, nx.DiGraph)):
        raise ValueError("The graph must be an undirected graph (Graph) when 'directed' is False.")
    
    if directed:
        out_degrees = dict(graph.out_degree()).values()
        num_nodes = len(graph.nodes())
        
        if num_nodes == 0:
            return {"in_degree": 0.0}
        avg_out_degree = sum(out_degrees) / num_nodes
        
        return avg_out_degree
    else:
        # Undirected graph: calculate average degree
        degrees = dict(graph.degree()).values()
        num_nodes = len(graph.nodes())
        
        if num_nodes == 0:
            return {"degree": 0.0}
        
        avg_degree = sum(degrees) / num_nodes
        
        return avg_degree

#%%
fights = nx.DiGraph()

def get_data_in_window(start_date, end_date):
    return data.loc[start_date:end_date]

def get_PPV_data_in_window(start_date, end_date):
    return PPVdata.loc[start_date:end_date]

def get_google_averages(start_date, end_date):
    return googleData.loc[start_date:end_date]

maleWeights = ['Bantamweight', 'Middleweight', 'Heavyweight', 'Lightweight', 'Welterweight', 'Flyweight', 'LightHeavyweight', 'Featherweight', 'CatchWeight', 'OpenWeight']
data['date'] = pd.to_datetime(data['date'])
#%%
data = data.sort_values('date')
data = data.set_index('date')
PPVdata = PPVdata.set_index('date')
googleData = googleData.set_index('Month')
window_size = pd.DateOffset(years = 2)
freq = pd.DateOffset(months=1)

start_date = data.index.min()
end_date = start_date + window_size
result = []

dir_deg = []
dir_den = []
dir_clust = []
dir_gin = []
dir_pathLenght = []
dir_betweeness = []
dir_eigenvector = []

un_deg = []
un_den = []
un_clust = []
un_gin = []
un_pathLenght = []
un_betweeness = []
un_eigenvector = []

numFights = []
numFighters = []

dates = []
PPVsales = []
googleSearch = []

numWin = 0

#%%
while end_date <= data.index.max():
    fightCount = 0
    
    window_data = get_data_in_window(start_date, end_date)
    PPV_window_data = get_PPV_data_in_window(start_date, end_date)
    google_window_data = get_google_averages(start_date, end_date)
    
    meanPPV = PPV_window_data['PPV'].mean()
    PPVsales.append(meanPPV)   
    
    meanGoogle = google_window_data['ufc'].mean()
    googleSearch.append(meanGoogle)
    
    
    for fight in range(0, len(window_data)):

        if window_data['weight_class'][fight] in maleWeights:
            winner = window_data['Winner'][fight]  
            if winner == 'Red':
                fights.add_edge(window_data["R_fighter"][fight], window_data["B_fighter"][fight])
            elif winner == 'Blue':
                fights.add_edge(window_data["B_fighter"][fight], window_data["R_fighter"][fight])
            fightCount += 1

    numWin = numWin + 1
    print(numWin)
    hist = nx.degree_histogram(fights)
    degrees = range(0, len(hist))
    
    concatenated_columns = pd.concat([window_data["R_fighter"], window_data["B_fighter"]])
    num_unique_fighters = concatenated_columns.nunique()
    
    windowDates = numWin, str(start_date), str(end_date)
    
    dates.append(windowDates)
        
    dir_avDeg = average_degree(fights, directed=True)
    dir_avDens = graph_density(fights, directed=True)
    dir_avClust = average_clustering(fights, directed=True)
#    dir_locGin = gini(fights)
    dir_locPathLenght = average_path_length_largest_nw(fights, directed=True)
    dir_locBetweeness = betweenness_largest_nw(fights, directed=True)
    dir_locEigenvector = average_eigenvector_nw(fights, directed=True)
    
    un_avDeg = average_degree(fights, directed=False)
    un_avDens = graph_density(fights, directed=False)
    un_avClust = average_clustering(fights, directed=False)
#    un_locGin = gini(fights)
    un_locPathLenght = average_path_length_largest_nw(fights, directed=False)
    un_locBetweeness = betweenness_largest_nw(fights, directed=False)
    un_locEigenvector = average_eigenvector_nw(fights, directed=False)
    
    dir_deg.append(dir_avDeg)
    dir_den.append(dir_avDens)
    dir_clust.append(dir_avClust)
#    dir_gin.append(dir_locGin)
    dir_pathLenght.append(dir_locPathLenght)
    dir_betweeness.append(dir_locBetweeness)
    dir_eigenvector.append(dir_locEigenvector)
    
    un_deg.append(un_avDeg)
    un_den.append(un_avDens)
    un_clust.append(un_avClust)
#    un_gin.append(un_locGin)
    un_pathLenght.append(un_locPathLenght)
    un_betweeness.append(un_locBetweeness)
    un_eigenvector.append(un_locEigenvector)
    
    fights.clear()
    
    start_date += freq
    end_date += freq
    
#%%
factor = 100/(np.nanmax(np.array(PPVsales)))
scaledPPV = np.array(PPVsales)*factor
PPVsales = np.array(PPVsales)
PPVsales = PPVsales*factor
savingPath = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/paperImages/0networkFightsPPV/GoogleInterest/'
#%%
plt.rc('xtick', labelsize='x-large')
fig, ax1 = plt.subplots()
ax1.plot(dir_clust, '.-r', ms = 6, lw  = 1.5, label=r'directed network')
ax1.plot(un_clust, 'x-b', ms = 4, lw  = 1.5, label=r'undirected network')
ax1.set_xlabel('window index')
ax1.set_ylabel(r'$\langle C  \rangle$', color='k')
ax1.tick_params('y', colors='k')
ax1.set_ylim(0)
# Create the second plot with the right y-axis
#ax2 = ax1.twinx()
#ax2.plot(PPVsales,'-k', ms = 1, lw = 1.5, alpha = 0.7)
#ax2.set_ylabel('Google interest', color='k')
#ax2.tick_params('y', colors='k')

#plt.title('Evolution average clustering coeff.')
plt.xlim(0, len(dir_clust))
#ax2.set_ylim(np.nanmin(PPVsales))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#plt.savefig(f'{savingPath}clustering.png', dpi = 1500)
plt.legend()
plt.show()
#%%
plt.rc('xtick', labelsize='x-large')
fig, ax1 = plt.subplots()
ax1.plot(dir_deg, '.-r', ms = 6, lw  = 1.5, label=r'directed network')
ax1.plot(un_deg, 'x-b', ms = 4, lw  = 1.5, label=r'undirected network')
ax1.set_xlabel('window index')
ax1.set_ylabel(r'$\langle k \rangle$', color='k')
ax1.tick_params('y', colors='k')
ax1.set_ylim(0)
# Create the second plot with the right y-axis
#ax2 = ax1.twinx()
#ax2.plot(PPVsales,'-k', ms = 1, lw = 1.5, alpha = 0.7)
#ax2.set_ylabel('Google interest', color='k')
#ax2.tick_params('y', colors='k')

#plt.title('Evolution average clustering coeff.')
plt.xlim(0, len(dir_deg))
#ax2.set_ylim(np.nanmin(PPVsales))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#plt.savefig(f'{savingPath}clustering.png', dpi = 1500)
plt.legend()
plt.show()
#%%
plt.rc('xtick', labelsize='x-large')
fig, ax1 = plt.subplots()
ax1.plot(dir_den, '.-r', ms = 6, lw  = 1.5, label=r'directed network')
ax1.plot(un_den, 'x-b', ms = 4, lw  = 1.5, label=r'undirected network')
ax1.set_xlabel('window index')
ax1.set_ylabel(r'$D$', color='k')
ax1.tick_params('y', colors='k')
ax1.set_ylim(0)
# Create the second plot with the right y-axis
#ax2 = ax1.twinx()
#ax2.plot(PPVsales,'-k', ms = 1, lw = 1.5, alpha = 0.7)
#ax2.set_ylabel('Google interest', color='k')
#ax2.tick_params('y', colors='k')

#plt.title('Evolution average clustering coeff.')
plt.xlim(0, len(dir_deg))
#ax2.set_ylim(np.nanmin(PPVsales))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#plt.savefig(f'{savingPath}clustering.png', dpi = 1500)
plt.legend()
plt.show()
#%%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(dir_pathLenght, '.-r', ms=6, lw=2, label='directed network')
axes[0].set_xlabel('window index')
axes[0].set_ylabel(r'$\langle l \rangle$', color='k')
axes[0].set_ylim(0)
axes[0].set_xlim(0, len(dir_betweeness) - 1)
axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
axes[0].legend(fontsize = 10)

# Plot for undirected network
axes[1].plot(un_pathLenght, 'x-b', ms=6, lw=2, label='undirected network')
axes[1].set_xlabel('window index')
axes[1].set_ylabel(r'$\langle l \rangle$', color='k')
axes[1].set_xlim(0, len(un_betweeness) - 1)
axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
axes[1].legend(fontsize = 10)
plt.show()
#%%
fig, ax1 = plt.subplots()
ax1.plot(numFights, color='orange', lw  = 2, label=r'$\langle C  \rangle$')
ax1.set_xlabel('window index')
ax1.set_ylabel(r'$N_{fights}$', color='orange')
ax1.tick_params('y', colors='orange')
ax1.set_ylim(0)

# Create the second plot with the right y-axis
ax2 = ax1.twinx()
ax2.plot(PPVsales,'-k', ms = 1, lw = 1.5, alpha = 0.7)
ax2.set_ylabel('Google interest', color='k')
ax2.tick_params('y', colors='k')

ax2.autoscale()

# Set the right y-axis in scientific notation
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax2.set_ylim(52857)
ax2.set_ylim(np.nanmin(PPVsales))


#plt.title('Number of fights.')
#plt.xlim(0, len(clust))
#plt.savefig(f'{savingPath}nFights.png', dpi = 1500)
plt.show()
#%%
fig, ax1 = plt.subplots()
ax1.plot(numFighters, color='olive', lw  = 2, label=r'$\langle C  \rangle$')
ax1.set_xlabel('window index')
ax1.set_ylabel(r'$N_{fighters}$', color='olive')
ax1.tick_params('y', colors='olive')
ax1.set_ylim(0)
# Create the second plot with the right y-axis
ax2 = ax1.twinx()
ax2.plot(PPVsales,'-k', ms = 1, lw = 1.5, alpha = 0.7)
ax2.set_ylabel('Google interest', color='k')
ax2.tick_params('y', colors='k')

#plt.title('Number of fights.')
#plt.xlim(0, len(clust))
ax2.set_ylim(np.nanmin(PPVsales))
#ax2.set_ylim(52857)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#plt.savefig(f'{savingPath}nFighters.png', dpi = 1500)
plt.show()

#%%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(dir_betweeness, '.-r', ms=6, lw=2, label='directed network')
axes[0].set_xlabel('window index')
axes[0].set_ylabel(r'$\langle b \rangle$', color='k')
axes[0].set_ylim(0)
axes[0].set_xlim(0, len(dir_betweeness) - 1)
axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
axes[0].legend(fontsize = 13)

# Plot for undirected network
axes[1].plot(un_betweeness, 'x-b', ms=6, lw=2, label='undirected network')
axes[1].set_xlabel('window index')
axes[1].set_ylabel(r'$\langle b \rangle$', color='k')
axes[1].set_xlim(0, len(un_betweeness) - 1)
axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
axes[1].legend(fontsize = 13)
#%%
plt.rc('xtick', labelsize='x-large')
fig, ax1 = plt.subplots()
ax1.plot(dir_eigenvector, '.-r', ms = 6, lw  = 1.5, label=r'directed network')
ax1.plot(un_eigenvector, 'x-b', ms = 4, lw  = 1.5, label=r'undirected network')
ax1.set_xlabel('window index')
ax1.set_ylabel(r'$\lambda$', color='k')
ax1.tick_params('y', colors='k')
ax1.set_ylim(0)
# Create the second plot with the right y-axis
#ax2 = ax1.twinx()
#ax2.plot(PPVsales,'-k', ms = 1, lw = 1.5, alpha = 0.7)
#ax2.set_ylabel('Google interest', color='k')
#ax2.tick_params('y', colors='k')

#plt.title('Evolution average clustering coeff.')
plt.xlim(0, len(dir_deg))
#ax2.set_ylim(np.nanmin(PPVsales))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#plt.savefig(f'{savingPath}clustering.png', dpi = 1500)
plt.legend()
plt.show()

#%% GOOGLE CORRELATIONS
corr_deg_dir = pearsonr(googleSearch[94:300], dir_deg[94:300])
corr_deg_un = pearsonr(googleSearch[94:300], un_deg[94:300])

corr_clust_dir = pearsonr(googleSearch[94:300], dir_clust[94:300])
corr_clust_un = pearsonr(googleSearch[94:300], un_clust[94:300])

corr_den_dir = pearsonr(googleSearch[94:300], dir_den[94:300])
corr_den_un = pearsonr(googleSearch[94:300], un_den[94:300])

corr_pl_dir = pearsonr(googleSearch[94:300], dir_pathLenght[94:300])
corr_pl_un = pearsonr(googleSearch[94:300], un_pathLenght[94:300])

corr_bet_dir = pearsonr(googleSearch[94:300], dir_betweeness[94:300])
corr_bet_un = pearsonr(googleSearch[94:300], un_betweeness[94:300])

corr_eig_dir = pearsonr(googleSearch[94:300], dir_eigenvector[94:300])
corr_eig_un = pearsonr(googleSearch[94:300], un_eigenvector[94:300])

#%%

xCorr = [r'$\langle C\rangle$', r'$\langle k \rangle$', r'$D$', r'$\langle l \rangle$', r'$\langle b \rangle$', '$\lambda$']
dir_yCorr = [corr_clust_dir[0], corr_deg_dir[0], corr_den_dir[0], corr_pl_dir[0], corr_bet_dir[0], corr_eig_dir[0]]
dir_err = [corr_clust_dir[1], corr_deg_dir[1], corr_den_dir[1], corr_pl_dir[1], corr_bet_dir[1], corr_eig_dir[1]]

un_yCorr = [corr_clust_un[0], corr_deg_un[0], corr_den_un[0], corr_pl_un[0], corr_bet_un[0], corr_eig_un[0]]
un_err = [corr_clust_un[1], corr_deg_un[1], corr_den_un[1], corr_pl_un[1], corr_bet_un[1], corr_eig_un[1]]

x_positions = np.arange(len(xCorr))
plt.errorbar(x_positions, dir_yCorr, yerr=dir_err, ms = 8, fmt='s', color = 'r', label = 'directed network')
plt.errorbar(x_positions, un_yCorr, yerr=un_err, ms = 6, fmt='o', color = 'b', label = 'undirected network')

#plt.title('correlations')
plt.xticks(x_positions, xCorr)
plt.ylim(-1,1)
plt.axhline(0, color='k')
plt.legend()

plt.ylabel('Pearson correlation coefficient')
plt.show()
#plt.savefig(f'{savingPath}correlations.png', dpi = 1500)
#plt.xlabel('metric')
#%% GOOGLE CORRELATIONS
corr_deg_dir_PPV = pearsonr(PPVsales[67:286], dir_deg[67:286])
corr_deg_un_PPV = pearsonr(PPVsales[67:286], un_deg[67:286])

corr_clust_dir_PPV = pearsonr(PPVsales[67:286], dir_clust[67:286])
corr_clust_un_PPV = pearsonr(PPVsales[67:286], un_clust[67:286])

corr_den_dir_PPV = pearsonr(PPVsales[67:286], dir_den[67:286])
corr_den_un_PPV = pearsonr(PPVsales[67:286], un_den[67:286])

corr_pl_dir_PPV = pearsonr(PPVsales[67:286], dir_pathLenght[67:286])
corr_pl_un_PPV = pearsonr(PPVsales[67:286], un_pathLenght[67:286])

corr_bet_dir_PPV = pearsonr(PPVsales[67:286], dir_betweeness[67:286])
corr_bet_un_PPV = pearsonr(PPVsales[67:286], un_betweeness[67:286])

corr_eig_dir_PPV = pearsonr(PPVsales[67:286], dir_eigenvector[67:286])
corr_eig_un_PPV = pearsonr(PPVsales[67:286], un_eigenvector[67:286])

#%%

xCorr = [r'$\langle C\rangle$', r'$\langle k \rangle$', r'$D$', r'$\langle l \rangle$', r'$\langle b \rangle$', '$\lambda$']
dir_yCorr_PPV = [corr_clust_dir_PPV[0], corr_deg_dir_PPV[0], corr_den_dir_PPV[0], corr_pl_dir_PPV[0], corr_bet_dir_PPV[0], corr_eig_dir_PPV[0]]
dir_err_PPV = [corr_clust_dir_PPV[1], corr_deg_dir_PPV[1], corr_den_dir_PPV[1], corr_pl_dir_PPV[1], corr_bet_dir_PPV[1], corr_eig_dir_PPV[1]]

un_yCorr_PPV = [corr_clust_un_PPV[0], corr_deg_un_PPV[0], corr_den_un_PPV[0], corr_pl_un_PPV[0], corr_bet_un_PPV[0], corr_eig_un_PPV[0]]
un_err_PPV = [corr_clust_un_PPV[1], corr_deg_un_PPV[1], corr_den_un_PPV[1], corr_pl_un_PPV[1], corr_bet_un_PPV[1], corr_eig_un_PPV[1]]

x_positions = np.arange(len(xCorr))
plt.errorbar(x_positions, dir_yCorr_PPV, yerr=dir_err_PPV, ms = 8, fmt='s', color = 'r', label = 'directed network')
plt.errorbar(x_positions, un_yCorr_PPV, yerr=un_err_PPV, ms = 6, fmt='o', color = 'b', label = 'undirected network')

#plt.title('correlations')
plt.xticks(x_positions, xCorr)
plt.ylim(-1,1)
plt.axhline(0, color='k')
plt.legend()

plt.ylabel('Pearson correlation coefficient')
plt.show()
#plt.savefig(f'{savingPath}correlations.png', dpi = 1500)
#plt.xlabel('metric')