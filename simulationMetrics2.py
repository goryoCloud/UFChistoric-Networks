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
#def create_network(data, start_night, window_size):

#    end_night = start_night + window_size - 1
#    window_data = data[(data['Fight_Night'] >= start_night) & (data['Fight_Night'] <= end_night)]
    
#    G = nx.Graph()
    
#    for index, row in window_data.iterrows():
#        G.add_node(row['Fighter1'])
#        G.add_node(row['Fighter2'])
#        G.add_edge(row['Fighter1'], row['Fighter2'], fight_night=row['Fight_Night'], winner=row['Winner'])
        
#    avDeg = average_degree_distribution(G)
#    print(avDeg)
#    avDens = graph_density(G)
#    print(avDens)
#    avClust = average_clustering_coefficient(G)
#    print(avClust)
#    print('-------------------------------------------------------')
#    locGin = gini(G)
#    locPathLenght = average_path_length_largest_nw(G)
#    locBetweeness = betweenness_largest_nw(G)
#    locEigenvector = average_eigenvector_nw(G)

#    deg.append(avDeg)
#    den.append(avDens)
#    clust.append(avClust)
#    gin.append(locGin)
#    pathLenght.append(locPathLenght)
#    betweeness.append(locBetweeness)
#    eigenvector.append(locEigenvector)
#    return G
#%%
file_path = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/fights6.txt'
data = pd.read_csv(file_path, header=None, names=['Fight_Night', 'Fighter1', 'Fighter2', 'Winner'])

data = data.set_index('Fight_Night')

def get_data_in_window(start_date, end_date):
    return data.loc[start_date:end_date]
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

numWin = 0
fightCount = 0
nIters = 0

window_size = 30
start = data.index.min()
end = start + window_size
freq = data.shift(1)

#%%  
fightNetwork = nx.Graph()

while end <= data.index.max():
    nIters = nIters + 1
#    print(nIters)
    window_data = get_data_in_window(start, end)
#    for fight in range( len(window_data)):
    for row in window_data.itertuples(index=False, name='row_data'):
        
        fightNetwork.add_edge(row[0], row[1])
        fightCount = fightCount + 1
        
    avDeg = average_degree_distribution(fightNetwork)
    avDens = graph_density(fightNetwork)
    avClust = average_clustering_coefficient(fightNetwork)
    locGin = gini(fightNetwork)
    locPathLenght = average_path_length_largest_nw(fightNetwork)
    locBetweeness = betweenness_largest_nw(fightNetwork)
    locEigenvector = average_eigenvector_nw(fightNetwork)

    deg.append(avDeg)
    den.append(avDens)
    clust.append(avClust)
    gin.append(locGin)
    pathLenght.append(locPathLenght)
    betweeness.append(locBetweeness)
    eigenvector.append(locEigenvector)
    
    
    
    fightNetwork.clear()
    
    start = start + 1
    end = end + 1    
print(len(deg))    
#%%


#def analyze_temporal_network(data, start, end, window_size, step):
#    current_start = start
#    while current_start + window_size - 1 <= end:
#        G = create_network(data, current_start, window_size)
        
#        avDeg = average_degree_distribution(G)
#        print(avDeg)
#        avDens = graph_density(G)
#        print(avDens)
#        avClust = average_clustering_coefficient(G)
#        print(avClust)
#        print('-------------------------------------------------------')
#        locGin = gini(G)
#        locPathLenght = average_path_length_largest_nw(G)
#        locBetweeness = betweenness_largest_nw(G)
#        locEigenvector = average_eigenvector_nw(G)

#        deg.append(avDeg)
#        den.append(avDens)
#        clust.append(avClust)
#        gin.append(locGin)
#        pathLenght.append(locPathLenght)
#        betweeness.append(locBetweeness)
#        eigenvector.append(locEigenvector)
#        G.clear()
#        current_start += step
#%%
# Load the data


# Example: Analyze from night 1 to 100, window of 5 nights, moving the window by 3 nights each step
#analyze_temporal_network(data, 1, 100, 10, 1)
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

plt.title('Evolution path lenght')
plt.plot(pathLenght, '-o', color = 'teal', ms = 1, lw = 2)
plt.xlabel('window index')
plt.ylabel(r'$ l $')
plt.xlim(0, len(clust))
plt.show()

plt.title('Evolution bet. centrality')
plt.plot(betweeness, '-o', color='darkblue', ms = 1, lw = 2)
plt.xlabel('window index')
plt.ylabel(r'$b$')
plt.xlim(0, len(clust))
plt.show()

plt.title('Evolution eigenvector centrality')
plt.plot(eigenvector, '-o', color='darkgreen', ms = 1, lw = 2)
plt.xlabel('window index')
plt.ylabel(r'$\lambda$')
plt.xlim(0, len(clust))
plt.show()

