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

data = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/data.csv')
#%% DIRECTED GRAPH
fighters = data.R_fighter.unique()
fightsWinners = nx.DiGraph()

maleWeights = ['Bantamweight', 'Middleweight', 'Heavyweight', 'Lightweight', 'Welterweight', 'Flyweight', 'LightHeavyweight', 'Featherweight', 'CatchWeight', 'OpenWeight']

for i in fighters:
    fightsWinners.add_node(i)

for fight in range(0, len(data)):
    win = data['Winner'][fight]
    if win == 'Red':
        fightsWinners.add_edge(data["R_fighter"][fight], data["B_fighter"][fight])
    elif win == 'Blue':
        fightsWinners.add_edge(data["B_fighter"][fight], data["R_fighter"][fight])
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
    # Calculate degree distribution for each node
    degree_distribution = dict(graph.degree())

    # Calculate the average degree
    average_degree = sum(degree_distribution.values()) / len(degree_distribution)

    return average_degree

def graph_density(graph):
    # Calculate the density of the graph
    density = nx.density(graph)

    return density

def average_clustering_coefficient(graph):
    # Calculate the average clustering coefficient
    avg_clustering = nx.average_clustering(graph)

    return avg_clustering

#%%
plt.rc('font', family='serif')
plt.rc('axes', labelsize='x-large')
plt.rc('axes', titlesize='x-large')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

inDegree = degree_histogram_directed(fightsWinners, in_degree=True)
outDegree = degree_histogram_directed(fightsWinners, out_degree=True)

inDegreeA = np.array(norm(inDegree))
outDegreeA = np.array(norm(outDegree))
#%%
inFit = fit_exponential_polyfit(np.arange(len(inDegreeA)),inDegreeA)
outFit = fit_exponential_polyfit(np.arange(len(outDegreeA)),outDegreeA)
#%%
plt.plot(range(len(inDegreeA)), inDegreeA, 'sk', fillstyle = 'none', ms = 5, lw = 0.5, label = rf'in-degree $\gamma = ${-inFit[2]:.3f}')
plt.plot(inFit[1], math.e**inFit[0], 'k')
plt.plot(range(len(outDegreeA)), outDegreeA, '^k',fillstyle = 'none' , ms = 5, lw = 0.5, label = rf'out-degree $\gamma = ${-outFit[2]:.3f}')
plt.plot(outFit[1], math.e**outFit[0], 'k')
plt.legend(fontsize = 15)
plt.yscale('log')
#plt.xscale('log')
plt.xlabel('k')
plt.ylabel('p(k)')
plt.show()      
#%%

fighters1 = data.R_fighter.unique()
fighters2 = data.R_fighter.unique()

fightersComb = np.concatenate([fighters1, fighters2])
fighters = np.unique(fightersComb)

#%%
fights = nx.Graph()

#for i in fighters:
#    fights.add_node(i)
    
#%%
for fight in range(0, len(data)):
    if data['weight_class'][fight] in maleWeights:
        fights.add_edge(data["R_fighter"][fight], data["B_fighter"][fight])

    
# avg_clust_coeff= 0.06383993303854954
#%%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "mathptmx",})

plt.rc('axes', labelsize='x-large')
plt.rc('axes', titlesize='x-large')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

degDist = nx.degree_histogram(fights)
degrees = np.arange(0, len(degDist))

degDistA = np.array(norm(degDist))
fit = fit_exponential_polyfit(np.arange(len(degDistA)),degDistA)

plt.plot(degrees, norm(degDist), 'ok', fillstyle = 'none', ms = 5, lw = 1, label = rf'UFC: undirected $\gamma = ${-fit[2]:.3f}')
plt.plot(fit[1], math.e**fit[0], '-k')
plt.plot(range(len(inDegreeA)), inDegreeA, 'sk', fillstyle = 'none', ms = 5, lw = 0.5, label = rf'UFC: in-degree $\gamma = ${-inFit[2]:.3f}')
plt.plot(inFit[1], math.e**inFit[0], 'k')
plt.plot(range(len(outDegreeA)), outDegreeA, '^k',fillstyle = 'none' , ms = 5, lw = 0.5, label = rf'UFC: out-degree $\gamma = ${-outFit[2]:.3f}')
plt.plot(outFit[1], math.e**outFit[0], 'k')
plt.yscale('log')
#plt.xscale('log')
plt.legend()
plt.xlabel('$k$')
plt.ylabel('$p(k)$')
plt.show()


#%%% CUM FIGHTS UNDIRECTED
cumSumPeleita = cum_deg_sum(fights)
degDistPeleitaY = np.array(norm(cumSumPeleita[1]))
degDistPeleitaX = np.array(cumSumPeleita[0])
fitPeleita = fit_exponential_polyfit(degDistPeleitaX, degDistPeleitaY)

#%%
plt.plot(fitPeleita[1], math.e**fitPeleita[0], 'r')
plt.plot(cumSumPeleita[0], norm(cumSumPeleita[1]), 'ko', ms = 7,  label = rf'UFC $\gamma = ${-fitPeleita[2]:.3f}')

plt.yscale('log')
plt.xlabel('$k$')
plt.ylabel('$p(k)$')
plt.legend(fontsize = 13)
plt.show()
#%%
#plot_options = {"node_size": 2, "with_labels": False, "width": 0.3}
#pos = nx.spring_layout(fights, iterations=70, seed=1721)
#fig, ax = plt.subplots(figsize=(15, 9))
#ax.axis("off")
#nx.draw_networkx(fights, node_color = 'r', pos=pos, ax=ax, **plot_options)   
#Un alumno que se saca entre un 0 y 4 perdió y sobre eso ganó. 

#Icializar desde una distribución real la probabilidad de ganar y meter los 
#nodos desde un baño termal

#langostas, factor de ganancia inicializado desde un temalizado

