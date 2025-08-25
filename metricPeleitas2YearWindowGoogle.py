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
PPVdata = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/multiTimeline (1).csv')

PPVdata['Month'] = pd.to_datetime(PPVdata['Month'])
PPVdataSorted = PPVdata.sort_values(by='Month')
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

def average_path_length_largest_nw(graph):
    largest_component = max(nx.connected_components(graph), key=len)
    largest_subgraph = graph.subgraph(largest_component)
    avg_path_length = nx.average_shortest_path_length(largest_subgraph)

    return avg_path_length

def pearsonCoeff(arrayX, arrayY):
    return pearsonr(arrayX, arrayY)

def betweenness_largest_nw(graph):
    largest_component = max(nx.connected_components(graph), key=len)
    largest_subgraph = graph.subgraph(largest_component)
    betweenness_centrality = nx.betweenness_centrality(largest_subgraph)
    average_betweenness_centrality = sum(betweenness_centrality.values()) / len(betweenness_centrality)
    
    return average_betweenness_centrality

def average_eigenvector_nw(graph):
    largest_component = max(nx.connected_components(graph), key=len)
    largest_subgraph = graph.subgraph(largest_component)
    eigenvector_centrality = nx.eigenvector_centrality(largest_subgraph, max_iter=5000)
    average_eigenvector_centrality = sum(eigenvector_centrality.values()) / len(eigenvector_centrality)

    return average_eigenvector_centrality
#%%
fights = nx.Graph()

def get_data_in_window(start_date, end_date):
    return data.loc[start_date:end_date]

def get_PPV_data_in_window(start_date, end_date):
    return PPVdataSorted.loc[start_date:end_date]

maleWeights = ['Bantamweight', 'Middleweight', 'Heavyweight', 'Lightweight', 'Welterweight', 'Flyweight', 'LightHeavyweight', 'Featherweight', 'CatchWeight', 'OpenWeight']
data['date'] = pd.to_datetime(data['date'])
#%%
data = data.sort_values('date')
data = data.set_index('date')
PPVdataSorted = PPVdataSorted.set_index('Month')
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
    fightCount = 0
    
    window_data = get_data_in_window(start_date, end_date)
    PPV_window_data = get_PPV_data_in_window(start_date, end_date)
    
    meanPPV = PPV_window_data['ufc'].mean()
#    if math.isnan(meanPPV):
#        PPVsales.append(0)
#    else:
#        PPVsales.append(meanPPV)
    PPVsales.append(meanPPV)   

    for fight in range(0, len(window_data)):
        
        if window_data['weight_class'][fight] in maleWeights:
            fights.add_edge(window_data["R_fighter"][fight], window_data["B_fighter"][fight])
            fightCount = fightCount + 1

    numWin = numWin + 1
    print(numWin)
    hist = nx.degree_histogram(fights)
    degrees = range(0, len(hist))
    
#    plt.title(f'Degree dist. fights window = {numWin}')
#    plt.plot(degrees, norm(hist), '-ko', ms = 7)
#    plt.yscale('log')
#    plt.xscale('log')
#    plt.xlabel('$k$')
#    plt.ylabel(r'$p(k)$')
#    plt.show()
    
    
    concatenated_columns = pd.concat([window_data["R_fighter"], window_data["B_fighter"]])
    num_unique_fighters = concatenated_columns.nunique()
    
    windowDates = numWin, str(start_date), str(end_date)
    
    dates.append(windowDates)
    
    avDeg = average_degree_distribution(fights)
    avDens = graph_density(fights)
    avClust = average_clustering_coefficient(fights)
    locGin = gini(fights)
    locPathLenght = average_path_length_largest_nw(fights)
    locBetweeness = betweenness_largest_nw(fights)
    locEigenvector = average_eigenvector_nw(fights)

    deg.append(avDeg)
    den.append(avDens)
    clust.append(avClust)
    gin.append(locGin)
    numFights.append(fightCount)
    numFighters.append(num_unique_fighters)
    pathLenght.append(locPathLenght)
    betweeness.append(locBetweeness)
    eigenvector.append(locEigenvector)
    
    fights.clear()
    
    start_date += freq
    end_date += freq
    
#%%
factor = 100/(np.nanmax(np.array(PPVsales)))
scaledPPV = np.array(PPVsales)*factor
PPVsales = np.array(PPVsales)
PPVsales = PPVsales*factor
#%%
savingPath = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/paperImages/0networkFightsPPV/GoogleInterest/'
#%%
plt.rc('xtick', labelsize='x-large')
fig, ax1 = plt.subplots()
ax1.plot(clust, color='red', lw  = 2, label=r'$\langle C  \rangle$')
ax1.set_xlabel('window index')
ax1.set_ylabel(r'$\langle C  \rangle$', color='red')
ax1.tick_params('y', colors='red')
ax1.set_ylim(0)
# Create the second plot with the right y-axis
ax2 = ax1.twinx()
ax2.plot(PPVsales,'-k', ms = 1, lw = 1.5, alpha = 0.7)
ax2.set_ylabel('Google interest', color='k')
ax2.tick_params('y', colors='k')

#plt.title('Evolution average clustering coeff.')
plt.xlim(0, len(clust))
ax2.set_ylim(np.nanmin(PPVsales))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#plt.savefig(f'{savingPath}clustering.png', dpi = 1500)
plt.show()
#%%
fig, ax1 = plt.subplots()
ax1.plot(deg, color='blue', lw  = 2, label=r'$\langle C  \rangle$')
ax1.set_xlabel('window index')
ax1.set_ylabel(r'$\langle k  \rangle$', color='blue')
ax1.tick_params('y', colors='blue')
#ax1.set_ylim(0)
# Create the second plot with the right y-axis
ax2 = ax1.twinx()
ax2.plot(PPVsales,'-k', ms = 1, lw = 1.5, alpha = 0.7)
ax2.set_ylabel('Google interest', color='k')
ax2.tick_params('y', colors='k')

#plt.title('Evolution average degree dist.')
plt.xlim(0, len(clust))
#ax2.set_ylim(52857)
ax2.set_ylim(np.nanmin(PPVsales))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#plt.savefig(f'{savingPath}degree.png', dpi = 1500)
plt.show()
#%%
fig, ax1 = plt.subplots()
ax1.plot(den, color='green', lw  = 2, label=r'$D$')
ax1.set_xlabel('window index')
ax1.set_ylabel(r'$\langle g \rangle$', color='green')
ax1.tick_params('y', colors='green')
ax1.set_ylim(0)
# Create the second plot with the right y-axis
ax2 = ax1.twinx()
ax2.plot(PPVsales,'-k', ms = 1, lw = 1.5, alpha = 0.7)
ax2.set_ylabel('Google interest', color='k')
ax2.tick_params('y', colors='k')

#plt.title('Evolution norm. degree distribution average')
plt.xlim(0, len(clust))
ax2.set_ylim(np.nanmin(PPVsales))
#ax2.set_ylim(52857)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#plt.savefig(f'{savingPath}normDegree.png', dpi = 1500)
plt.show()
#%%
fig, ax1 = plt.subplots()
ax1.plot(gin, color='m', lw  = 2, label=r'$\langle C  \rangle$')
ax1.set_xlabel('window index')
ax1.set_ylabel(r'$ G[p(k)] $', color='m')
ax1.tick_params('y', colors='m')
#ax1.set_ylim(0)
# Create the second plot with the right y-axis
ax2 = ax1.twinx()
ax2.plot(PPVsales,'-k', ms = 1, lw = 1.5, alpha = 0.7)
ax2.set_ylabel('Google interest', color='k')
ax2.tick_params('y', colors='k')

#plt.title('Evolution gini coeff.')
plt.xlim(0, len(clust))
ax2.set_ylim(np.nanmin(PPVsales))
#ax2.set_ylim(52857)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#plt.savefig(f'{savingPath}gini.png', dpi = 1500)
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
plt.xlim(0, len(clust))
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
plt.xlim(0, len(clust))
ax2.set_ylim(np.nanmin(PPVsales))
#ax2.set_ylim(52857)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#plt.savefig(f'{savingPath}nFighters.png', dpi = 1500)
plt.show()
#%%
fig, ax1 = plt.subplots()
ax1.plot(pathLenght, color='teal', lw  = 2, label=r'$\langle C  \rangle$')
ax1.set_xlabel('window index')
ax1.set_ylabel(r'$l$', color='teal')
ax1.tick_params('y', colors='teal')
ax1.set_ylim(0)
# Create the second plot with the right y-axis
ax2 = ax1.twinx()
ax2.plot(PPVsales,'-k', ms = 1, lw = 1.5, alpha = 0.7)
ax2.set_ylabel('Google interest', color='k')
ax2.tick_params('y', colors='k')

#plt.title('Path lenght')
plt.xlim(0, len(clust))
#ax2.set_ylim(52857)
ax2.set_ylim(np.nanmin(PPVsales))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#plt.savefig(f'{savingPath}path.png', dpi = 1500)
plt.show()
#%%
fig, ax1 = plt.subplots()
ax1.plot(betweeness, color='darkblue', lw  = 2, label=r'$b$')
ax1.set_xlabel('window index')
ax1.set_ylabel(r'$b$', color='darkblue')
ax1.tick_params('y', colors='darkblue')
ax1.set_ylim(0)
# Create the second plot with the right y-axis
ax2 = ax1.twinx()
ax2.plot(PPVsales,'-k', ms = 1, lw = 1.5, alpha = 0.7)
ax2.set_ylabel('Google interest', color='k')
ax2.tick_params('y', colors='k')

#plt.title('Betweeness')
plt.xlim(0, len(clust))
#ax2.set_ylim(52857)
ax2.set_ylim(np.nanmin(PPVsales))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#plt.savefig(f'{savingPath}betweeness.png', dpi = 1500)
plt.show()
#%%
fig, ax1 = plt.subplots()
ax1.plot(eigenvector, color='darkgreen', lw  = 2, label=r'$b$')
ax1.set_xlabel('window index')
ax1.set_ylabel(r'$\lambda$', color='darkgreen')
ax1.tick_params('y', colors='darkgreen')
ax1.set_ylim(0)
# Create the second plot with the right y-axis
ax2 = ax1.twinx()
ax2.plot(PPVsales,'-k', ms = 1, lw = 1.5, alpha = 0.7)
ax2.set_ylabel('Google interest', color='k')
ax2.tick_params('y', colors='k')

#plt.title('Eigenvector centrality')
plt.xlim(0, len(clust))
#ax2.set_ylim(52857)
ax2.set_ylim(np.nanmin(PPVsales))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#plt.savefig(f'{savingPath}eigen.png', dpi = 1500)
plt.show()
#%%
from scipy.stats import pearsonr
plt.rc('xtick', labelsize='medium')
PPVnonZero = PPVsales[94:300]
#-----------------------------------------------------
clustPPV =  clust[94:300]
corrClust = pearsonr(PPVnonZero, clustPPV)

degPPV = deg[94:300]
corrDeg = pearsonr(PPVnonZero, degPPV)

denPPV = den[94:300]
corrDen = pearsonr(PPVnonZero, denPPV)

ginPPV = gin[94:300]
corrGin = pearsonr(PPVnonZero, ginPPV)

nFightsPPV = numFights[94:300]
corrNFights = pearsonr(PPVnonZero, nFightsPPV)

nFightersPPV = numFighters[94:300]
corrNFighters = pearsonr(PPVnonZero, nFightersPPV)

lenghtPPV = pathLenght[94:300]
corrPath = pearsonr(PPVnonZero, lenghtPPV)

betPPV = betweeness[94:300]
corrBet = pearsonr(PPVnonZero, betPPV)

eigenvectorPPV = eigenvector[94:300]
corrEigen = pearsonr(PPVnonZero, eigenvectorPPV)

xCorr = [r'$\langle C\rangle$', r'$\langle k \rangle$', r'$D$', r'$ G[p(k)] $', r'$N_{fights}$', r'$N_{fighters}$', '$l$', '$b$', '$\lambda$']
yCorr = [corrClust[0], corrDeg[0], corrDen[0], corrGin[0], corrNFights[0], corrNFighters[0], corrPath[0], corrBet[0], corrEigen[0]]
err = [corrClust[1], corrDeg[1], corrDen[1], corrGin[1], corrNFights[1], corrNFighters[1], corrPath[1], corrBet[1], corrEigen[1]]

x_positions = np.arange(len(xCorr))
plt.errorbar(x_positions, yCorr, yerr=err, fmt='o', color = 'g')

#plt.title('correlations')
plt.xticks(x_positions, xCorr)
plt.ylim(-1,1)
plt.axhline(0, color='k')

plt.ylabel('Pearson correlation coefficient')
#plt.savefig(f'{savingPath}correlations.png', dpi = 1500)
#plt.xlabel('metric')
#%%

xCorr = [r'$\langle C\rangle$', r'$\langle k \rangle$', r'$\langle g \rangle$', r'$ G[p(k)] $', '$l$', '$b$', '$\lambda$']
yCorr = [corrClust[0], corrDeg[0], corrDen[0], corrGin[0], corrPath[0], corrBet[0], corrEigen[0]]
err = [corrClust[1], corrDeg[1], corrDen[1], corrGin[1], corrPath[1], corrBet[1], corrEigen[1]]

x_positions = np.arange(len(xCorr))
plt.errorbar(x_positions, yCorr, yerr=err, fmt='o', color = 'g')

plt.title('Correlations network 1')
plt.xticks(x_positions, xCorr)
plt.ylim(-1,1)
plt.axhline(0, color='k')

plt.ylabel('Pearson correlation coefficient')

#%%
from scipy.stats import pearsonr
plt.rc('xtick', labelsize='medium')
PPVnonZero = PPVsales[94:300]
#-----------------------------------------------------
clustPPV =  clust[94:300]
corrClust = pearsonr(PPVnonZero, clustPPV)

degPPV = deg[94:300]
corrDeg = pearsonr(PPVnonZero, degPPV)

denPPV = den[94:300]
corrDen = pearsonr(PPVnonZero, denPPV)

ginPPV = gin[94:300]
corrGin = pearsonr(PPVnonZero, ginPPV)

nFightsPPV = numFights[94:300]
corrNFights = pearsonr(PPVnonZero, nFightsPPV)

nFightersPPV = numFighters[94:300]
corrNFighters = pearsonr(PPVnonZero, nFightersPPV)

lenghtPPV = pathLenght[94:300]
corrPath = pearsonr(PPVnonZero, lenghtPPV)

betPPV = betweeness[94:300]
corrBet = pearsonr(PPVnonZero, betPPV)

eigenvectorPPV = eigenvector[94:300]
corrEigen = pearsonr(PPVnonZero, eigenvectorPPV)

xCorr = [r'$\langle C\rangle$', r'$\langle k \rangle$', r'$D$', r'$ G[p(k)] $', '$l$', '$b$', '$\lambda$']
yCorr = [corrClust[0], corrDeg[0], corrDen[0], corrGin[0], corrPath[0], corrBet[0], corrEigen[0]]
err = [corrClust[1], corrDeg[1], corrDen[1], corrGin[1], corrPath[1], corrBet[1], corrEigen[1]]

x_positions = np.arange(len(xCorr))
plt.errorbar(x_positions, yCorr, yerr=err, fmt='s', color = 'k')

#plt.title('correlations')
plt.xticks(x_positions, xCorr)
plt.ylim(-1,1)
plt.axhline(0, color='k')

plt.ylabel('Pearson correlation coefficient')
#plt.savefig(f'{savingPath}correlations.png', dpi = 1500)
#plt.xlabel('metric')