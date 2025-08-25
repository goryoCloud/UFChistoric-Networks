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
path = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/dataframes/'

degreeEvol = pd.read_csv(f'{path}degreeEvol.csv', index_col= 'fighter')
clustEvol = pd.read_csv(f'{path}clustEvol.csv', index_col= 'fighter')
betweenessEvol = pd.read_csv(f'{path}betweenessEvol.csv', index_col= 'fighter')
eigenEvol = pd.read_csv(f'{path}eigenEvol.csv', index_col= 'fighter')

#%%

fighterArr = ['Conor McGregor', 'Georges St-Pierre', 'Khabib Nurmagomedov', 'Jon Jones', 'Demetrious Johnson']

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
axs[0, 0].set_title('Degree Evolution')
axs[1, 0].set_title('Betweenness Centrality Evolution')
axs[0, 1].set_title('Clustering Coefficient Evolution')
axs[1, 1].set_title('Eigenvector Centrality Evolution')

axs[0, 0].set_xlabel('window index')
axs[1, 0].set_xlabel('window index')
axs[0, 1].set_xlabel('window index')
axs[1, 1].set_xlabel('window index')


for name in fighterArr:
    deg = np.array(degreeEvol.loc[name])
    clust = np.array(clustEvol.loc[name])
    bet = np.array(betweenessEvol.loc[name])
    eigen = np.array(eigenEvol.loc[name])
    
    axs[0, 0].plot(deg, label=name)
    axs[0,0].legend()
    axs[0, 1].plot(clust, label=name)
    axs[0,1].legend()
    axs[1, 0].plot(bet, label=name)
    axs[1,0].legend()
    axs[1, 1].plot(eigen, label=name)
    axs[1,1].legend()
    
plt.tight_layout()
plt.show()    
