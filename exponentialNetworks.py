# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:42:05 2023

@author: max_s
"""

import networkx as nx
import math
import matplotlib.pyplot as plt
import numpy as np

distribution = []

for i in np.arange(0,2.7, 0.4):
    value = 2000* math.e**(-2.286 * i)
    distribution.append(value)
#%%
plt.rc('font', family='serif')
plt.rc('axes', labelsize='x-large')
plt.rc('axes', titlesize='x-large')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')
 
plt.plot(distribution, 'ok')
plt.yscale('log')

#%%



