# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:38:12 2023

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
    t = np.linspace(min(dataToFitX), max(dataToFitX), 25)

    # Extract the fitted parameters
    y_fit = fit[1]*t + fit[0]

    return y_fit, t, fit[1]
#%%
data = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/data.csv')
#%%
plt.rc('font', family='serif')
plt.rc('axes', labelsize='x-large')
plt.rc('axes', titlesize='x-large')
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='x-large')

weights = data['weight_class'].value_counts()
weightsLabels = data['weight_class'].unique()

distA = np.array(weights)
fit = fit_exponential_polyfit(np.arange(len(distA)),distA)

plt.bar(weightsLabels, norm(weights), color ='maroon', width = 0.7)
plt.plot(fit[1], norm(math.e**fit[0]), '-x', color = 'teal', label = rf'exponential fit $\gamma = ${-fit[2]:.3f}')
plt.xticks(rotation=70)
plt.legend()
plt.yscale('log')
plt.ylabel('N of athletes')
plt.xlabel('weight category')
plt.show()

plt.bar(weightsLabels, weights, color ='maroon', width = 0.7)
plt.plot(fit[1], math.e**fit[0], '-x', color = 'teal', label = rf'exponential fit $\gamma = ${-fit[2]:.3f}')
plt.xticks(rotation=70)
plt.legend()
plt.yscale('log')
plt.ylabel('N of athletes')
plt.xlabel('weight category')
plt.show()