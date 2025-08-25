# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 22:16:57 2023

@author: max_s
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

G = nx.erdos_renyi_graph(1000, 0.5)

listDegrees = []
for k in range(0, len(G)):

    degreeNode = G.degree(k)
    if degreeNode > 0:
        print(degreeNode)
        listDegrees.append(degreeNode)

listDegreesA = np.array(listDegrees)
listDegreesANorm = listDegreesA/len(G.nodes())

avg = sum(listDegreesANorm)/ len(G.nodes())
print(avg)

#%%
a = nx.density(G)
print(a)
