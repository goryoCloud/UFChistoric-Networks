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
#%%
R_fighters = data['R_fighter']
B_fighters = data['R_fighter']

fighterData = pd.concat((R_fighters, B_fighters))

counts = fighterData.value_counts()
#%%
from collections import Counter

value_counts = Counter(counts)
sorted_values = sorted(value_counts.items(), key=lambda x: x[0], reverse=True)
#%%
histogramNumFights = []
for i in sorted_values:
    a = i[0]
    b = i[1]
    toAppend = a, b
    histogramNumFights.append(toAppend)
    
histogramNumFightsA = np.array(histogramNumFights)
fit = fit_exponential_polyfit(histogramNumFightsA[:, 0], norm(histogramNumFightsA[:, 1]))

plt.plot(fit[1], norm(math.e**fit[0]), '-', color = 'r', label = rf'exponential fit $\gamma = ${-fit[2]:.6f}')
plt.plot(histogramNumFightsA[:, 0], norm(histogramNumFightsA[:, 1]), 'ko')
plt.xlabel('number of disputed fights')
plt.legend(fontsize = 15)
plt.ylabel('PDF')
plt.yscale('log')
#%%
