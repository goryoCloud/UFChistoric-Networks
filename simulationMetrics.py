import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import random as rn
import copy as cp

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

def norm(x):
    normValues = []
    norma = sum(x)
    for i in x:
        normValue = i /norma
        normValues.append(normValue)
        normValuesA = np.array(normValues)
    normValues.clear()
    return (normValuesA)

def fit_exponential_polyfit(x_data, y_data):
    # Take the logarithm of positive y_data, leave zero values as they are
    maskData = y_data > 0
    dataToFitX = x_data[maskData]
    dataToFitY = y_data[maskData]
    
    log_y_data = np.log(dataToFitY)

    # Fit a polynomial to the log-transformed data
    fit = np.poly1d(np.polyfit(dataToFitX, log_y_data, 1))
    t = np.linspace(min(dataToFitX), max(dataToFitX), 25)

    # Extract the fitted parameters
    y_fit = fit[1]*t + fit[0]

    return y_fit, t, fit[1]

def generate_random_pairs(lista):
    random_pairs = []
    used_numbers = set()

    while len(lista) >= 2:
        num1 = rn.choice(lista)
        lista.remove(num1)
        num2 = rn.choice(lista)
        lista.remove(num2)
        random_pairs.append((num1, num2))

    return random_pairs

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
x = np.arange(1, 54)
probDist = np.e**(x*-0.110784)
probDist = np.insert(probDist, 0, 1)
probLong = 1 - probDist
#%% NETWORK INITIALIZATION
numstep = 150
windowSize = 40
nNodes = 250

initialNodes = np.arange(1, nNodes + 1)
fightsPerNight = 2

lifespan = pd.DataFrame({'fighter': initialNodes})
lifespan['nFights'] = 0
#%%
fights = nx.Graph()
fightsWindows = nx.Graph()

for i in lifespan['fighter']:
    fights.add_node(i)
#%%
lifespan.set_index('fighter', inplace=True)
#%%
deg = []
den = []
clust = []
gin = []
pathLenght = []
betweeness = []
numFights = []
numFighters = []
eigenvector = []

#%%
step = 0
nFighters = []
for step in range(0, numstep):
    step = step + 1
    fighterList = list(fights.nodes())
    random_nodes = rn.sample(fighterList, 2 * fightsPerNight)
    usedNodes = cp.deepcopy(random_nodes)
    randomPairs = generate_random_pairs(random_nodes)

    for pair in randomPairs:
        fights.add_edge(pair[0], pair[1])
        fightsWindows.add_edge(pair[0], pair[1])
        
        lifespan.loc[pair[0],'nFights'] = lifespan.loc[pair[0],'nFights'] + 1
        lifespan.loc[pair[1],'nFights'] = lifespan.loc[pair[1],'nFights'] + 1
    print('----------------------------------------------------------')    
    for used in usedNodes:
        lifetime = lifespan.loc[used,'nFights']
        longTest = probLong[lifetime]
        rand = rn.uniform(0, 1)
        if rand < longTest:
            fights.remove_node(used)
    
    if step%windowSize == 0:
        avDeg = average_degree_distribution(fightsWindows)
        avDens = graph_density(fightsWindows)
        avClust = average_clustering_coefficient(fightsWindows)
        locGin = gini(fightsWindows)
        locPathLenght = average_path_length_largest_nw(fightsWindows)
        locBetweeness = betweenness_largest_nw(fightsWindows)
        locEigenvector = average_eigenvector_nw(fightsWindows)

        deg.append(avDeg)
        den.append(avDens)
        clust.append(avClust)
        gin.append(locGin)
        pathLenght.append(locPathLenght)
        betweeness.append(locBetweeness)
        eigenvector.append(locEigenvector)

    fightsWindows.clear()
    print(f'step {step}: {len(fights.nodes())}') 
#    nFighters.append(len(fights.nodes()))
    nFighters.append(len(usedNodes))
    
plt.plot(nFighters, lw = 3)
plt.xlabel(r'$N_{nights}$')
plt.ylabel(r'$N_{fighters}$')
plt.xlim(0, numstep - 1)
plt.show()

lifespan.hist('nFights')

#%%
plt.title('Evolution average clustering coeff.')
plt.plot(clust, '-ro', ms = 1, lw = 2)
plt.xlabel('window index')
plt.ylabel(r'$\langle C  \rangle$')
plt.xlim(0, len(clust))
plt.show()

plt.title('Evolution average degree dist.')
plt.plot(deg, '-bo', ms = 1, lw = 2)
plt.xlabel('window index')
plt.ylabel(r'$\langle p(k)  \rangle$')
plt.xlim(0, len(clust))
plt.show()

plt.title('Evolution norm. degree distribution average')
plt.plot(den, '-go', ms = 1, lw = 2)
plt.xlabel('window index')
plt.ylabel(r'$\langle p(k)  \rangle/N$')
plt.xlim(0, len(clust))
plt.show()

plt.title('Evolution gini coeff.')
plt.plot(gin, '-mo', ms = 1, lw = 2)
plt.xlabel('window index')
plt.ylabel(r'$ G[p(k)] $')
plt.xlim(0, len(clust))
plt.show()

plt.title('Number of Fights.')
plt.plot(numFights, '-ko', ms = 1, lw = 2)
plt.xlabel('window index')
plt.ylabel(r'$N$')
plt.xlim(0, len(clust))
plt.show()

#plt.title('Evolution path length.')
#plt.plot(path, '-ko', ms = 5)
#plt.xlabel('window index')
#plt.ylabel(r'$\langle l  \rangle$')
#plt.show()

plt.title('Number of Fighters')
plt.plot(numFighters, '-ko', ms = 1, lw = 2)
plt.xlabel('window index')
plt.ylabel(r'$N$')
plt.xlim(0, len(clust))
plt.show()

plt.title('Diff. number of Fighters')
plt.plot(np.diff(numFighters), '-ko', ms = 1, lw = 1)
plt.xlabel('window index')
plt.ylabel(r'$N_i - N_{i-1}$')
plt.xlim(0, len(clust))
plt.show()

plt.title('Diff. avg. deg dist.')
plt.plot(np.diff(deg), '-bo', ms = 1, lw = 1)
plt.xlabel('window index')
plt.ylabel(r'$\langle p(k_{i})  \rangle - \langle p(k_{i-1})  \rangle$')
plt.xlim(0, len(clust))
plt.show()
        
        

    

    
    
    
    







