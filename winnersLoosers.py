"""
Created on Sun Dec 24 18:22:57 2023

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
maleWeights = ['Bantamweight', 'Middleweight', 'Heavyweight', 'Lightweight', 'Welterweight', 'Flyweight', 'LightHeavyweight', 'Featherweight', 'CatchWeight', 'OpenWeight']
fighters = data.R_fighter.unique()

data['date'] = pd.to_datetime(data['date'])
winners_df = pd.DataFrame(columns=['Winner', 'date'])
loosers_df = pd.DataFrame(columns=['Looser', 'date'])

for fight in range(0, len(data)):
    if data['weight_class'][fight] in maleWeights:
        date = data['date'][fight]
        win = data['Winner'][fight]
        if win == 'Red':
            winner_name = data["R_fighter"][fight]
        elif win == 'Blue':
            winner_name = data["B_fighter"][fight]
        winners_df = pd.concat([winners_df, pd.DataFrame({'Winner': [winner_name], 'date': [date]})], ignore_index= True)
        
for fight in range(0, len(data)):
    if data['weight_class'][fight] in maleWeights:
        date = data['date'][fight]
        win = data['Winner'][fight]
        if win == 'Red':
            looser_name = data["B_fighter"][fight]
        elif win == 'Blue':
            looser_name = data["R_fighter"][fight]
        loosers_df = pd.concat([loosers_df, pd.DataFrame({'Looser': [looser_name], 'date': [date]})], ignore_index= True)
    
#%%
winners = nx.DiGraph()
loosers = nx.DiGraph()

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

w_dir_deg = []
w_dir_den = []
w_dir_clust = []
w_dir_gin = []
w_dir_pathLenght = []
w_dir_betweeness = []
w_dir_eigenvector = []

w_un_deg = []
w_un_den = []
w_un_clust = []
w_un_gin = []
w_un_pathLenght = []
w_un_betweeness = []
w_un_eigenvector = []

l_dir_deg = []
l_dir_den = []
l_dir_clust = []
l_dir_gin = []
l_dir_pathLenght = []
l_dir_betweeness = []
l_dir_eigenvector = []

l_un_deg = []
l_un_den = []
l_un_clust = []
l_un_gin = []
l_un_pathLenght = []
l_un_betweeness = []
l_un_eigenvector = []


numFights = []
numFighters = []

dates = []
PPVsales = []
googleSearch = []

numWin = 0

#%%
winners_df['date'] = pd.to_datetime(winners_df['date'])
winners_df = winners_df.sort_values('date')
winners_df = winners_df.set_index('date')
window_size = pd.DateOffset(years = 2)
freq = pd.DateOffset(months=1)

start_date = winners_df.index.min()
end_date = start_date + window_size
result = []
dates = []
numWin = 0

def get_data_in_window_winners(start_date, end_date):
    return winners_df.loc[start_date:end_date]

while end_date <= winners_df.index.max():
    fightCount = 0
    
    window_data = get_data_in_window_winners(start_date, end_date)
    for i in range(0, len(window_data) - 1):
        winners.add_edge(window_data["Winner"][i], window_data["Winner"][i + 1])   
    
    google_window_data = get_google_averages(start_date, end_date)
    meanGoogle = google_window_data['ufc'].mean()
    googleSearch.append(meanGoogle)
    
    PPV_window_data = get_PPV_data_in_window(start_date, end_date)
    meanPPV = PPV_window_data['PPV'].mean()
    PPVsales.append(meanPPV)
    numWin = numWin + 1
    print(numWin)

    windowDates = start_date, end_date
    dates.append(windowDates)
    
    dir_avDeg = average_degree(winners, directed=True)
    dir_avDens = graph_density(winners, directed=True)
    dir_avClust = average_clustering(winners, directed=True)
    dir_locPathLenght = average_path_length_largest_nw(winners, directed=True)
    dir_locBetweeness = betweenness_largest_nw(winners, directed=True)
    dir_locEigenvector = average_eigenvector_nw(winners, directed=True)
    
    un_avDeg = average_degree(winners, directed=False)
    un_avDens = graph_density(winners, directed=False)
    un_avClust = average_clustering(winners, directed=False)
    un_locPathLenght = average_path_length_largest_nw(winners, directed=False)
    un_locBetweeness = betweenness_largest_nw(winners, directed=False)
    un_locEigenvector = average_eigenvector_nw(winners, directed=False)
    
    w_dir_deg.append(dir_avDeg)
    w_dir_den.append(dir_avDens)
    w_dir_clust.append(dir_avClust)
    w_dir_pathLenght.append(dir_locPathLenght)
    w_dir_betweeness.append(dir_locBetweeness)
    w_dir_eigenvector.append(dir_locEigenvector)
    
    w_un_deg.append(un_avDeg)
    w_un_den.append(un_avDens)
    w_un_clust.append(un_avClust)
    w_un_pathLenght.append(un_locPathLenght)
    w_un_betweeness.append(un_locBetweeness)
    w_un_eigenvector.append(un_locEigenvector)
    
    winners.clear()
    
    start_date += freq
    end_date = start_date + window_size 
    
#%%
def get_data_in_window_loosers(start_date, end_date):
    return loosers_df.loc[start_date:end_date]

loosers_df['date'] = pd.to_datetime(loosers_df['date'])
loosers_df = loosers_df.sort_values('date')
loosers_df = loosers_df.set_index('date')
window_size = pd.DateOffset(years=2)
freq = pd.DateOffset(months=1)

start_date = loosers_df.index.min()
end_date = start_date + window_size

numWin = 0

while end_date <= loosers_df.index.max():
    fightCount = 0
    
    window_data = get_data_in_window_loosers(start_date, end_date)
    for i in range(0, len(window_data) - 1):
        loosers.add_edge(window_data["Looser"][i], window_data["Looser"][i + 1])
    
    numWin = numWin + 1
    print(numWin)

    dir_avDeg = average_degree(loosers, directed=True)
    dir_avDens = graph_density(loosers, directed=True)
    dir_avClust = average_clustering(loosers, directed=True)
    dir_locPathLenght = average_path_length_largest_nw(loosers, directed=True)
    dir_locBetweeness = betweenness_largest_nw(loosers, directed=True)
    dir_locEigenvector = average_eigenvector_nw(loosers, directed=True)
    
    un_avDeg = average_degree(loosers, directed=False)
    un_avDens = graph_density(loosers, directed=False)
    un_avClust = average_clustering(loosers, directed=False)
    un_locPathLenght = average_path_length_largest_nw(loosers, directed=False)
    un_locBetweeness = betweenness_largest_nw(loosers, directed=False)
    un_locEigenvector = average_eigenvector_nw(loosers, directed=False)
    
    l_dir_deg.append(dir_avDeg)
    l_dir_den.append(dir_avDens)
    l_dir_clust.append(dir_avClust)
    l_dir_pathLenght.append(dir_locPathLenght)
    l_dir_betweeness.append(dir_locBetweeness)
    l_dir_eigenvector.append(dir_locEigenvector)
    
    l_un_deg.append(un_avDeg)
    l_un_den.append(un_avDens)
    l_un_clust.append(un_avClust)
    l_un_pathLenght.append(un_locPathLenght)
    l_un_betweeness.append(un_locBetweeness)
    l_un_eigenvector.append(un_locEigenvector)
    
    loosers.clear()
    
    start_date += freq
    end_date = start_date + window_size
#%%
factor = 100/(np.nanmax(np.array(PPVsales)))
scaledPPV = np.array(PPVsales)*factor
PPVsales = np.array(PPVsales)
PPVsales = PPVsales*factor
savingPath = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/paperImages/0networkFightsPPV/GoogleInterest/'
#%%
plt.rc('xtick', labelsize='x-large')
fig, ax1 = plt.subplots()
ax1.plot(w_dir_deg, '.-r', ms = 10, lw  = 2, markevery = 10, label=r'winners directed network')
ax1.plot(l_dir_deg, 'x-', color = 'orange', ms = 9, lw  = 2,markevery = 10, label=r'loosers directed network')
ax1.plot(w_un_deg, 's-b', ms = 6, lw  = 2,markevery = 10, label=r'winners undirected network')
ax1.plot(l_un_deg,  '^-', ms = 9, color = 'green', lw  = 2, markevery = 10, label=r'loosers undirected network')
ax1.set_xlabel('window index')
ax1.set_ylabel(r'$\langle k  \rangle$', color='k')
ax1.tick_params('y', colors='k')
ax1.set_ylim(0)
# Create the second plot with the right y-axis
#ax2 = ax1.twinx()
#ax2.plot(PPVsales,'-k', ms = 1, lw = 1.5, alpha = 0.7)
#ax2.set_ylabel('Google interest', color='k')
#ax2.tick_params('y', colors='k')

#plt.title('Evolution average clustering coeff.')
plt.xlim(0, len(w_dir_clust))
plt.ylim(0,6)
#ax2.set_ylim(np.nanmin(PPVsales))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#plt.savefig(f'{savingPath}clustering.png', dpi = 1500)
plt.legend(fontsize = 8.2)
plt.show()
#%%
plt.rc('xtick', labelsize='x-large')
fig, ax1 = plt.subplots()
ax1.plot(w_dir_clust, '.-r', ms = 8, lw  = 2,markevery = 3, label=r'winners directed network')
ax1.plot(l_dir_clust, 'x-', color = 'orange', ms = 8, lw  = 2, markevery = 10,label=r'loosers directed network')
ax1.plot(w_un_clust, 's-b', ms = 6, lw  = 2,markevery = 10, label=r'winners undirected network')
ax1.plot(l_un_clust,  '^-', ms = 6, color = 'green', lw  = 2, markevery = 10,label=r'loosers undirected network')
ax1.set_xlabel('window index')
ax1.set_ylabel(r'$\langle C  \rangle$', color='k')
ax1.tick_params('y', colors='k')
ax1.set_ylim(-0.03, 0.5)
# Create the second plot with the right y-axis
#ax2 = ax1.twinx()
#ax2.plot(PPVsales,'-k', ms = 1, lw = 1.5, alpha = 0.7)
#ax2.set_ylabel('Google interest', color='k')
#ax2.tick_params('y', colors='k')

#plt.title('Evolution average clustering coeff.')
plt.xlim(0, len(w_dir_clust))
#ax2.set_ylim(np.nanmin(PPVsales))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#plt.savefig(f'{savingPath}clustering.png', dpi = 1500)
plt.legend(fontsize = 11)
plt.show()
#%%
plt.rc('xtick', labelsize='x-large')
fig, ax1 = plt.subplots()
ax1.plot(w_dir_den, '.-r', ms = 10, lw  = 2,markevery =7, label=r'winners directed network')
ax1.plot(l_dir_den, 'x-', color = 'orange', ms = 7, lw  = 2,markevery = 10, label=r'loosers directed network')
ax1.plot(w_un_den, 's-b', ms = 5, lw  = 2,markevery = 10, label=r'winners undirected network')
ax1.plot(l_un_den,  '^-', ms = 6, color = 'green', lw  = 2,markevery = 10, label=r'loosers undirected network')
ax1.set_xlabel('window index')
ax1.set_ylabel(r'$D$', color='k')
ax1.tick_params('y', colors='k')
ax1.set_ylim(0, 0.25)
# Create the second plot with the right y-axis
#ax2 = ax1.twinx()
#ax2.plot(PPVsales,'-k', ms = 1, lw = 1.5, alpha = 0.7)
#ax2.set_ylabel('Google interest', color='k')
#ax2.tick_params('y', colors='k')

#plt.title('Evolution average clustering coeff.')
plt.xlim(0, len(w_dir_clust))
#ax2.set_ylim(np.nanmin(PPVsales))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#plt.savefig(f'{savingPath}clustering.png', dpi = 1500)
plt.legend(fontsize = 11)
plt.show()
#%%
import numpy as np

def moving_average(data, window_size=30):

    # Ensure data is a NumPy array
    data = np.array(data)

    # Initialize arrays to hold the results
    smoothed_data = []
    dispersion = []

    # Loop over the data to calculate the moving window stats
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        smoothed_data.append(np.mean(window))
        dispersion.append(np.std(window))

    # Pad the results to match the original data length
    padding = (len(data) - len(smoothed_data)) // 2
    smoothed_data = np.pad(smoothed_data, (padding, len(data) - len(smoothed_data) - padding), 'constant', constant_values=np.nan)
    dispersion = np.pad(dispersion, (padding, len(data) - len(dispersion) - padding), 'constant', constant_values=np.nan)

    return smoothed_data, dispersion

#%%
plt.rc('xtick', labelsize='x-large')
fig, axs = plt.subplots(nrows=2, ncols=2)
plt.tight_layout(pad = 1.5)

axs[0,0].plot(np.arange(0, len(moving_average(l_dir_pathLenght)[0])), moving_average(l_dir_pathLenght)[0], color = 'orange', alpha = 1, label=r'loosers directed network')
axs[0,0].errorbar(np.arange(0, len(moving_average(l_dir_pathLenght)[0])), moving_average(l_dir_pathLenght)[0], yerr = moving_average(l_dir_pathLenght)[1], errorevery = 1, color = 'orange', alpha = 0.3)
print(np.nanmean(moving_average(l_dir_pathLenght)[1]))
axs[0,0].set_xlabel('window index')
axs[0,0].set_ylabel(r'$\langle l \rangle$')
axs[0,0].set_xlim(0,300)
axs[0,0].set_ylim(0,10)
axs[0,0].text(0.95, 0.07, fr'$\langle \epsilon \rangle = ${np.nanmean(moving_average(l_dir_pathLenght)[1]):.2f}', fontsize=15, transform=axs[0,0].transAxes, horizontalalignment='right')

axs[0,1].plot(np.arange(0, len(moving_average(w_dir_pathLenght)[0])), moving_average(w_dir_pathLenght)[0],color = 'r', alpha = 1, label=r'winnwers directed network')
axs[0,1].errorbar(np.arange(0, len(moving_average(w_dir_pathLenght)[0])), moving_average(w_dir_pathLenght)[0], yerr = moving_average(w_dir_pathLenght)[1],errorevery = 1,color = 'r', alpha = 0.3)
axs[0,1].set_xlabel('window index')
axs[0,1].set_ylabel(r'$\langle l \rangle$')
axs[0,1].set_xlim(0,300)
axs[0,1].set_ylim(0,10)
axs[0,1].text(0.95, 0.07, fr'$\langle \epsilon \rangle = ${np.nanmean(moving_average(w_dir_pathLenght)[1]):.2f}', fontsize=15, transform=axs[0,1].transAxes, horizontalalignment='right')


axs[1,0].plot(np.arange(0, len(moving_average(l_un_pathLenght)[0])), moving_average(l_un_pathLenght)[0],color = 'g', alpha = 1, label=r'loosers undirected network')
axs[1,0].errorbar(np.arange(0, len(moving_average(l_un_pathLenght)[0])), moving_average(l_un_pathLenght)[0], yerr = moving_average(l_un_pathLenght)[1],errorevery = 1,color = 'g', alpha = 0.3)
axs[1,0].set_xlabel('window index')
axs[1,0].set_ylabel(r'$\langle l \rangle$')
axs[1,0].set_xlim(0,300)
axs[1,0].set_ylim(0,10)
axs[1,0].text(0.95, 0.07, fr'$\langle \epsilon \rangle = ${np.nanmean(moving_average(l_un_pathLenght)[1]):.2f}', fontsize=15, transform=axs[1,0].transAxes, horizontalalignment='right')
print(np.nanmean(moving_average(l_un_pathLenght)[1]))

axs[1,1].plot(np.arange(0, len(moving_average(w_un_pathLenght)[0])), moving_average(w_un_pathLenght)[0], color = 'blue', alpha = 1, label=r'winners undirected network')
axs[1,1].errorbar(np.arange(0, len(moving_average(w_un_pathLenght)[0])), moving_average(w_un_pathLenght)[0], yerr = moving_average(w_un_pathLenght)[1],errorevery = 1,color = 'blue', alpha = 0.3)
axs[1,1].set_xlabel('window index')
axs[1,1].set_ylabel(r'$\langle l \rangle$')
axs[1,1].set_xlim(0,300)
axs[1,1].set_ylim(0,10)
axs[1,1].text(0.95, 0.07, fr'$\langle \epsilon \rangle = ${np.nanmean(moving_average(w_un_pathLenght)[1]):.2f}', fontsize=15, transform=axs[1,1].transAxes, horizontalalignment='right')
print(np.nanmean(moving_average(w_un_pathLenght)[1]))

#plt.legend(fontsize = 8)

#plt.xlabel('window index')
#plt.ylabel(r'$\langle l \rangle$', color='k')
plt.xlim(0)
#%%
plt.rc('xtick', labelsize='x-large')
fig, ax1 = plt.subplots()

ax1.plot(l_dir_betweeness, 'x-', color = 'orange', ms = 7, lw  = 2,markevery = 3, label=r'loosers directed network')
ax1.plot(w_dir_betweeness, '.-r', ms = 10, lw  = 2,markevery =7, label=r'winners directed network')
ax1.plot(w_un_betweeness, 's-b', ms = 5, lw  = 2,markevery = 10, label=r'winners undirected network')
ax1.plot(l_un_betweeness,  '^-', ms = 6, color = 'green', lw  = 2,markevery = 10, label=r'loosers undirected network')
ax1.set_xlabel('window index')
ax1.set_ylabel(r'$\langle b \rangle$', color='k')
ax1.tick_params('y', colors='k')
ax1.set_ylim(0, 0.25)
# Create the second plot with the right y-axis
#ax2 = ax1.twinx()
#ax2.plot(PPVsales,'-k', ms = 1, lw = 1.5, alpha = 0.7)
#ax2.set_ylabel('Google interest', color='k')
#ax2.tick_params('y', colors='k')

#plt.title('Evolution average clustering coeff.')
plt.xlim(0, len(w_dir_clust))
#ax2.set_ylim(np.nanmin(PPVsales))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#plt.savefig(f'{savingPath}clustering.png', dpi = 1500)
plt.legend(fontsize = 11)
plt.show()
#%%

plt.rc('xtick', labelsize='x-large')
fig, ax1 = plt.subplots()
ax1.plot(w_dir_eigenvector, '.-r', ms = 10, lw  = 2,markevery =7, label=r'winners directed network')
ax1.plot(l_dir_eigenvector, 'x-', color = 'orange', ms = 7, lw  = 2,markevery = 10, label=r'loosers directed network')
ax1.plot(w_un_eigenvector, 's-b', ms = 5, lw  = 2,markevery = 10, label=r'winners undirected network')
ax1.plot(l_un_eigenvector,  '^-', ms = 6, color = 'green', lw  = 2,markevery = 10, label=r'loosers undirected network')
ax1.set_xlabel('window index')
ax1.set_ylabel(r'$\langle \lambda \rangle$', color='k')
ax1.tick_params('y', colors='k')
ax1.set_ylim(0, 0.2)
# Create the second plot with the right y-axis
#ax2 = ax1.twinx()
#ax2.plot(PPVsales,'-k', ms = 1, lw = 1.5, alpha = 0.7)
#ax2.set_ylabel('Google interest', color='k')
#ax2.tick_params('y', colors='k')

#plt.title('Evolution average clustering coeff.')
plt.xlim(0, len(w_dir_clust))
#ax2.set_ylim(np.nanmin(PPVsales))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#plt.savefig(f'{savingPath}clustering.png', dpi = 1500)
plt.legend(fontsize = 11)
plt.show()

#%%
corr_w_deg_dir = pearsonr(googleSearch[94:300], w_dir_deg[94:300])
corr_w_deg_un = pearsonr(googleSearch[94:300], w_un_deg[94:300])
corr_l_deg_dir = pearsonr(googleSearch[94:300], l_dir_deg[94:300])
corr_l_deg_un = pearsonr(googleSearch[94:300], l_un_deg[94:300])

corr_w_clust_dir = pearsonr(googleSearch[94:300], w_dir_clust[94:300])
corr_w_clust_un = pearsonr(googleSearch[94:300], w_un_clust[94:300])
corr_l_clust_dir = pearsonr(googleSearch[94:300], l_dir_clust[94:300])
corr_l_clust_un = pearsonr(googleSearch[94:300], l_un_clust[94:300])

corr_w_den_dir = pearsonr(googleSearch[94:300], w_dir_den[94:300])
corr_w_den_un = pearsonr(googleSearch[94:300], w_un_den[94:300])
corr_l_den_dir = pearsonr(googleSearch[94:300], l_dir_den[94:300])
corr_l_den_un = pearsonr(googleSearch[94:300], l_un_den[94:300])

corr_w_pl_dir = pearsonr(googleSearch[94:300], w_dir_pathLenght[94:300])
corr_w_pl_un = pearsonr(googleSearch[94:300], w_un_pathLenght[94:300])
corr_l_pl_dir = pearsonr(googleSearch[94:300], l_dir_pathLenght[94:300])
corr_l_pl_un = pearsonr(googleSearch[94:300], l_un_pathLenght[94:300])

corr_w_bet_dir = pearsonr(googleSearch[94:300], w_dir_betweeness[94:300])
corr_w_bet_un = pearsonr(googleSearch[94:300], w_un_betweeness[94:300])
corr_l_bet_dir = pearsonr(googleSearch[94:300], l_dir_betweeness[94:300])
corr_l_bet_un = pearsonr(googleSearch[94:300], l_un_betweeness[94:300])

corr_w_eig_dir = pearsonr(googleSearch[94:300], w_dir_eigenvector[94:300])
corr_w_eig_un = pearsonr(googleSearch[94:300], w_un_eigenvector[94:300])
corr_l_eig_dir = pearsonr(googleSearch[94:300], l_dir_eigenvector[94:300])
corr_l_eig_un = pearsonr(googleSearch[94:300], l_un_eigenvector[94:300])


xCorr = [r'$\langle C\rangle$', r'$\langle k \rangle$', r'$D$', '$l$', '$b$', '$\lambda$']

dir_win_corr = [corr_w_clust_dir[0], corr_w_deg_dir[0], corr_w_den_dir[0], corr_w_pl_dir[0], corr_w_bet_dir[0], corr_w_eig_dir[0]]
dir_win_err = [corr_w_clust_dir[1], corr_w_deg_dir[1], corr_w_den_dir[1], corr_w_pl_dir[1], corr_w_bet_dir[1], corr_w_eig_dir[1]]

un_win_corr = [corr_w_clust_un[0], corr_w_deg_un[0], corr_w_den_un[0], corr_w_pl_un[0], corr_w_bet_un[0], corr_w_eig_un[0]]
un_win_err = [corr_w_clust_un[1], corr_w_deg_un[1], corr_w_den_un[1], corr_w_pl_un[1], corr_w_bet_un[1], corr_w_eig_un[1]]

dir_los_corr = [corr_l_clust_dir[0], corr_l_deg_dir[0], corr_l_den_dir[0], corr_l_pl_dir[0], corr_l_bet_dir[0], corr_l_eig_dir[0]]
dir_los_err = [corr_l_clust_dir[1], corr_l_deg_dir[1], corr_l_den_dir[1], corr_l_pl_dir[1], corr_l_bet_dir[1], corr_l_eig_dir[1]]

un_los_corr = [corr_l_clust_un[0], corr_l_deg_un[0], corr_l_den_un[0], corr_l_pl_un[0], corr_l_bet_un[0], corr_l_eig_un[0]]
un_los_err = [corr_l_clust_un[1], corr_l_deg_un[1], corr_l_den_un[1], corr_l_pl_un[1], corr_l_bet_un[1], corr_l_eig_un[1]]

x_positions = np.arange(len(xCorr))

plt.errorbar(x_positions, un_win_corr, yerr=un_win_err, fmt='s', color = 'b', ms = 10, alpha = 1, label = 'winners undireced network')
plt.errorbar(x_positions, dir_win_corr, yerr=dir_win_err, fmt='o', color = 'r', ms = 8, alpha = 1, label = 'winners directed network')
plt.errorbar(x_positions, un_los_corr, yerr=un_los_err, fmt='^', color = 'g', ms = 10, alpha = 1, label = 'loosers undirected network')
plt.errorbar(x_positions, dir_los_corr, yerr=dir_los_err, fmt='x', color = 'orange', ms = 10, alpha = 1, label = 'loosers directed network')

#plt.title('correlations')
plt.xticks(x_positions, xCorr)
plt.ylim(-1,1)
plt.axhline(0, color='k')
plt.legend(fontsize = 7.5)
plt.ylabel('Pearson correlation coefficient')
plt.show()

#%%
corr_w_deg_dir = pearsonr(PPVsales[67:286], w_dir_deg[67:286])
corr_w_deg_un = pearsonr(PPVsales[67:286], w_un_deg[67:286])
corr_l_deg_dir = pearsonr(PPVsales[67:286], l_dir_deg[67:286])
corr_l_deg_un = pearsonr(PPVsales[67:286], l_un_deg[67:286])

corr_w_clust_dir = pearsonr(PPVsales[67:286], w_dir_clust[67:286])
corr_w_clust_un = pearsonr(PPVsales[67:286], w_un_clust[67:286])
corr_l_clust_dir = pearsonr(PPVsales[67:286], l_dir_clust[67:286])
corr_l_clust_un = pearsonr(PPVsales[67:286], l_un_clust[67:286])

corr_w_den_dir = pearsonr(PPVsales[67:286], w_dir_den[67:286])
corr_w_den_un = pearsonr(PPVsales[67:286], w_un_den[67:286])
corr_l_den_dir = pearsonr(PPVsales[67:286], l_dir_den[67:286])
corr_l_den_un = pearsonr(PPVsales[67:286], l_un_den[67:286])

corr_w_pl_dir = pearsonr(PPVsales[67:286], w_dir_pathLenght[67:286])
corr_w_pl_un = pearsonr(PPVsales[67:286], w_un_pathLenght[67:286])
corr_l_pl_dir = pearsonr(PPVsales[67:286], l_dir_pathLenght[67:286])
corr_l_pl_un = pearsonr(PPVsales[67:286], l_un_pathLenght[67:286])

corr_w_bet_dir = pearsonr(PPVsales[67:286], w_dir_betweeness[67:286])
corr_w_bet_un = pearsonr(PPVsales[67:286], w_un_betweeness[67:286])
corr_l_bet_dir = pearsonr(PPVsales[67:286], l_dir_betweeness[67:286])
corr_l_bet_un = pearsonr(PPVsales[67:286], l_un_betweeness[67:286])

corr_w_eig_dir = pearsonr(PPVsales[67:286], w_dir_eigenvector[67:286])
corr_w_eig_un = pearsonr(PPVsales[67:286], w_un_eigenvector[67:286])
corr_l_eig_dir = pearsonr(PPVsales[67:286], l_dir_eigenvector[67:286])
corr_l_eig_un = pearsonr(PPVsales[67:286], l_un_eigenvector[67:286])


xCorr = [r'$\langle C\rangle$', r'$\langle k \rangle$', '$D$', '$l$', '$b$', '$\lambda$']

dir_win_corr = [corr_w_clust_dir[0], corr_w_deg_dir[0], corr_w_den_dir[0], corr_w_pl_dir[0], corr_w_bet_dir[0], corr_w_eig_dir[0]]
dir_win_err = [corr_w_clust_dir[1], corr_w_deg_dir[1], corr_w_den_dir[1], corr_w_pl_dir[1], corr_w_bet_dir[1], corr_w_eig_dir[1]]

un_win_corr = [corr_w_clust_un[0], corr_w_deg_un[0], corr_w_den_un[0], corr_w_pl_un[0], corr_w_bet_un[0], corr_w_eig_un[0]]
un_win_err = [corr_w_clust_un[1], corr_w_deg_un[1], corr_w_den_un[1], corr_w_pl_un[1], corr_w_bet_un[1], corr_w_eig_un[1]]

dir_los_corr = [corr_l_clust_dir[0], corr_l_deg_dir[0], corr_l_den_dir[0], corr_l_pl_dir[0], corr_l_bet_dir[0], corr_l_eig_dir[0]]
dir_los_err = [corr_l_clust_dir[1], corr_l_deg_dir[1], corr_l_den_dir[1], corr_l_pl_dir[1], corr_l_bet_dir[1], corr_l_eig_dir[1]]

un_los_corr = [corr_l_clust_un[0], corr_l_deg_un[0], corr_l_den_un[0], corr_l_pl_un[0], corr_l_bet_un[0], corr_l_eig_un[0]]
un_los_err = [corr_l_clust_un[1], corr_l_deg_un[1], corr_l_den_un[1], corr_l_pl_un[1], corr_l_bet_un[1], corr_l_eig_un[1]]

x_positions = np.arange(len(xCorr))

plt.errorbar(x_positions, un_win_corr, yerr=un_win_err, fmt='s', color = 'b', ms = 10, alpha = 1, label = 'winners undireced network')
plt.errorbar(x_positions, dir_win_corr, yerr=dir_win_err, fmt='o', color = 'r', ms = 8, alpha = 1, label = 'winners directed network')
plt.errorbar(x_positions, un_los_corr, yerr=un_los_err, fmt='^', color = 'g', ms = 10, alpha = 1, label = 'loosers undirected network')
plt.errorbar(x_positions, dir_los_corr, yerr=dir_los_err, fmt='x', color = 'orange', ms = 10, alpha = 1, label = 'loosers directed network')

#plt.title('correlations')
plt.xticks(x_positions, xCorr)
plt.ylim(-1,1)
plt.axhline(0, color='k')
plt.legend(fontsize = 7.5)
plt.ylabel('Pearson correlation coefficient')



