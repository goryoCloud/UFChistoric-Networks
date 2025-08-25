# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 21:03:25 2023

@author: max_s
"""

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math

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
    t = np.linspace(min(dataToFitX), max(dataToFitX), 100)

    # Extract the fitted parameters
    y_fit = fit[1]*t + fit[0]

    return y_fit, t, fit[1]


data = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/data.csv')
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
    fights.add_edge(data["R_fighter"][fight], data["B_fighter"][fight])
    
# avg_clust_coeff= 0.06383993303854954
#%%
degDist = nx.degree_histogram(fights)
degrees = np.arange(0, len(degDist))

degDistA = np.array(norm(degDist))
fit = fit_exponential_polyfit(np.arange(len(degDistA)),degDistA)

plt.plot(degrees, norm(degDist), 'ok', ms = 5, lw = 0.5)
plt.plot(fit[1], math.e**fit[0], '-k', lw = 1, label = rf'$\gamma = ${-fit[2]:.3f}')
plt.yscale('log')
#plt.xscale('log')
plt.legend()
plt.xlabel('k')
plt.ylabel('p(k)')
plt.show()
#%% GINI COEFF
list_degrees = fights.degree()
dfDeg = pd.DataFrame(list_degrees)

sortedDeg = dfDeg.sort_values(1)