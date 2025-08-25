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
path = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/dataframes'
dataDeg = pd.read_csv(f'{path}/degreeEvol.csv')
dataClust = pd.read_csv(f'{path}/clustEvol.csv')
dataBet = pd.read_csv(f'{path}/betweenessEvol.csv')
dataEigen = pd.read_csv(f'{path}/eigenEvol.csv')

dataDeg.set_index('fighter', inplace=True)
dataClust.set_index('fighter', inplace=True)
dataBet.set_index('fighter', inplace=True)
dataEigen.set_index('fighter', inplace=True)

#%%
list_fighters = ['Conor McGregor','Anderson Silva', 'Jon Jones', 'Georges St-Pierre', 'Khabib Nurmagomedov']

#list_fighters = ['Conor McGregor','Anderson Silva', 'Jon Jones', 'Georges St-Pierre', 'Stipe Miocic', 'Daniel Cormier', 'Demetrious Johnson', 'Khabib Nurmagomedov']
fig, axs = plt.subplots(2,2, figsize = (9,6))
plt.tight_layout(pad=3)

for fighter in list_fighters:
    deg = dataDeg.loc[fighter].tolist()
    clust = dataClust.loc[fighter].tolist()
    bet = dataBet.loc[fighter].tolist()
    eigen = dataEigen.loc[fighter].tolist()

    
    axs[0,0].plot(deg, label = f'{fighter}')

    axs[0,1].plot(clust, label = f'{fighter}')

    axs[1,0].plot(bet, label = f'{fighter}')

    axs[1,1].plot(eigen, label = f'{fighter}')
    

axs[0,0].set_title('Norm. degree evolution')
axs[0,0].set_ylabel(r'$k$')
axs[0,0].set_xlabel('window index')
axs[0,0].set_ylim(0)
axs[0,0].set_xlim(0, 301)
axs[0,0].legend()

axs[0,1].set_title('Clustering coeff. evolution')
axs[0,1].set_ylabel(r'$C$')
axs[0,1].set_xlabel('window index')
axs[0,1].set_ylim(0)
axs[0,1].set_xlim(0,301)
#axs[0,1].legend()

axs[1,0].set_title('Betweeness centrality evolution')
axs[1,0].set_ylabel(r'$B$')
axs[1,0].set_xlabel('window index')
axs[1,0].set_ylim(0)
axs[1,0].set_xlim(0,301)
#axs[1,0].legend()

axs[1,1].set_title('Eigenvector centrality evolution')
axs[1,1].set_ylabel(r'$\lambda$')
axs[1,1].set_xlabel('window index')
axs[1,1].set_ylim(0)
axs[1,1].set_xlim(0, 301)
#axs[1,1].legend()
plt.show()    
