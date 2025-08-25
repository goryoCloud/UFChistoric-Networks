import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
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

#file_path = 'fights3.txt'
#data = pd.read_csv(file_path, header=None, names=['Fight_Night', 'Fighter1', 'Fighter2', 'Winner'])

#data = data.set_index('Fight_Night')

#%%
import os

def create_next_simulation_folder(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)


    existing_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    numeric_folders = [int(f) for f in existing_folders if f.isdigit()]
    
    if numeric_folders:
        largest_number = max(numeric_folders)

    else:
        largest_number = 0
        
    new_folder_number = largest_number + 1
    new_folder_name = os.path.join(base_path, str(new_folder_number))
    os.makedirs(new_folder_name)


    print(f"Created new folder: {new_folder_name}")
    return new_folder_name

base_directory = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/simulationResults/'

new_folder = create_next_simulation_folder(base_directory)
#%%
x = np.arange(1, 54)
probDist = np.e**(x*-0.110784)
probDist = np.insert(probDist, 0, 1)
probLong = 1 - probDist
#%% NETWORK INITIALIZATION
numNights = 200
nNodes = 1000
fightsPerNight = 9

fighter = np.arange(0, nNodes)
fightPool = fighter
nFights = numNights * fightsPerNight

lifespan = pd.DataFrame({'fighter': fighter})
lifespan['nFights'] = 0

fights = pd.DataFrame(columns = ['night', 'fighter_A', 'fighter_B', 'winner'])
#%%
night = 0
nFighters = []
totalNumberFights = 0

for night in range(0, numNights):
    night = night + 1
#    print(f'night: {night}')
    
    fighterList = list(fightPool)
    random_nodes = rn.sample(fighterList, 2 * fightsPerNight)
    usedNodes = cp.deepcopy(random_nodes)
    randomPairs = generate_random_pairs(random_nodes)

    for pair in randomPairs:
        totalNumberFights = totalNumberFights + 1
        lifespan.loc[pair[0],'nFights'] = lifespan.loc[pair[0],'nFights'] + 1
        lifespan.loc[pair[1],'nFights'] = lifespan.loc[pair[1],'nFights'] + 1
        
        winner = rn.choice(pair)
        if winner == pair[0]:
            fighterWin = 'fighter_A'
#            print(f'fighter_A won, {winner}')            
        if winner == pair[1]:
            fighterWin = 'fighter_B'
#            print(f'fighter_B won, {winner}')
        
        new_row = {'night': f'{night}', 'fighter_A': f'{pair[0]}', 'fighter_B': f'{pair[1]}', 'winner': f'{fighterWin}'}
        fights = pd.concat([fights, pd.DataFrame([new_row])], ignore_index=True)

    for used in usedNodes:
        lifetime = lifespan.loc[used,'nFights']
        longTest = probLong[lifetime]
        
        rand = rn.uniform(0, 1)
        if rand < longTest:
            fightPool = fightPool[fightPool != used]
#            print(f'{used} has been removed!')
#    print('----------------------------------------------------------') 
    nFighters.append(len(fightPool))
#    fightsWindows.clear()
#    print(f'step {step}: {len(fights.nodes())}') 
#    nFighters.append(len(fights.nodes()))
#    nFighters.append(len(usedNodes))

fights.to_csv(f'{new_folder}/fights.cvs', index=False, header = False)
print('fights.cvs file created')
#%%
data = pd.read_csv(f'{new_folder}/fights.cvs', header=None, names=['Fight_Night', 'Fighter1', 'Fighter2', 'Winner'])
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
plt.plot(deg)     
#%%

with open(os.path.join(new_folder, "degreeDist.dat"), "w") as file:
    file.write(f"{deg}")
with open(os.path.join(new_folder, "density.dat"), "w") as file:
    file.write(f"{den}")
with open(os.path.join(new_folder, "clustering.dat"), "w") as file:
    file.write(f"{clust}")
with open(os.path.join(new_folder, "ginni.dat"), "w") as file:
    file.write(f"{gin}")
with open(os.path.join(new_folder, "pathLenght.dat"), "w") as file:
    file.write(f"{pathLenght}")
with open(os.path.join(new_folder, "betweeness.dat"), "w") as file:
    file.write(f"{betweeness}")
with open(os.path.join(new_folder, "eigenvector.dat"), "w") as file:
    file.write(f"{eigenvector}")


