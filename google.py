# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:38:55 2024

@author: max_s
"""

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "mathptmx",})

plt.rc('axes', labelsize='x-large')
plt.rc('axes', titlesize='x-large')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large') 

path = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/multiTimeline (1).csv'

data = pd.read_csv(path)

data['Month'] = pd.to_datetime(data['Month'])

plt.plot(data['Month'], data['ufc'], lw = 2, c = 'b')
plt.xlabel('date')
plt.ylabel('Google interest')
plt.ylim(0)
plt.xlim(min(data['Month']), max(data['Month']))