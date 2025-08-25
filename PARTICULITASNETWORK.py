# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 15:41:09 2023

@author: max_s
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rn
import math
import networkx as nx
import itertools as it
#%%
plt.rc('font', family='serif')
plt.rc('axes', labelsize='large')
plt.rc('axes', titlesize='large')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')

#%%
NPARTS = 500
PARTSPOS = np.empty((NPARTS, 5))
EVOLUTION = np.empty((NPARTS, 5))
PARTRADIUS = 0.01 
TIMESTEP = 0.01
NSTEPS = 400
STEP = 0
BOXSIZE = 1

G = nx.Graph()

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

def bc(pos): 
    if pos >= BOXSIZE:
        pos = pos - BOXSIZE
    elif pos <= 0:
        pos = pos + BOXSIZE
    return pos

def dynamic(ids, xPos, yPos, v, theta):
#    NEWPOSX = xPos + v*np.cos(theta) + 0.1*np.random.normal(0.0, np.sqrt(TIMESTEP))
#    NEWPOSY = yPos + v*np.sin(theta) + 0.1*np.random.normal(0.0, np.sqrt(TIMESTEP))
    
    NEWPOSX = xPos + 0.1*np.random.normal(0.0, np.sqrt(TIMESTEP))
    NEWPOSY = yPos + 0.1*np.random.normal(0.0, np.sqrt(TIMESTEP))

    BOUNDARYTESTX = bc(NEWPOSX)
    BOUNDARYTESTY = bc(NEWPOSY)
    
    return int(ids), BOUNDARYTESTX, BOUNDARYTESTY, v, theta

def evaluateCol(x1, y1, x2, y2, radius):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance < 2*radius

def plotGraph(graph, timestep, step, color):
    degrees = range(len(nx.degree_histogram(graph)))
    hist = plt.plot(degrees, norm(nx.degree_histogram(graph)), '-o', color = color, ms = 2.5, lw = 1, label = rf'$\tau=$ {step*timestep}')
#    plt.xlim(1, )
    plt.ylim(0.01, )
    plt.ylabel('p(k)')
    plt.xlabel('k')
#    plt.yscale('log')
#    plt.ylabel('counts')
#    plt.title(rf'$\tau=$ {STEP*TIMESTEP}')

combinations = np.array(list(it.combinations(np.arange(0, NPARTS), 2)))
#%%
for i in range(0, len(PARTSPOS)):
    xPos = rn.uniform(0,1)
    yPos = rn.uniform(0,1)
    v0 = 0.01*rn.uniform(0,1)
    theta = 2*np.pi*rn.uniform(0,1)
    
    PARTSPOS[i] = i, xPos, yPos, v0, theta
#%%
colors = ["#1984c5", "#22a7f0", "#63bff0", "#a7d5ed", "#e2e2e2", "#e1a692", "#de6e56", "#e14b31", "#c23728"]
COLORCOUNTER = 0
OLDPOS = np.array(PARTSPOS)
COLLISIONCOUNT = 0
while STEP < NSTEPS:
    for j in range(0, len(PARTSPOS)):
        PART = OLDPOS[j]
        EVOLVEDPART = dynamic(int(PART[0]), PART[1], PART[2], PART[3], PART[4])
        EVOLUTION[j] = EVOLVEDPART
        
    for par in combinations:
        if evaluateCol(EVOLUTION[par[0],1], EVOLUTION[par[0],2], EVOLUTION[par[1],1], EVOLUTION[par[1],2], PARTRADIUS):
            COLLISIONCOUNT = COLLISIONCOUNT + 1
            G.add_edge(par[0], par[1])
#            print(f'collisions = {COLLISIONCOUNT}')
            
        
    OLDPOS = EVOLUTION
#    plt.scatter(EVOLUTION[:,1], EVOLUTION[:,2], color = 'k') 
#    plt.xlim(0, BOXSIZE)
#    plt.ylim(0, BOXSIZE)
#    plt.xlabel('X')
#    plt.ylabel('Y')
#    plt.title(rf'$\tau=$ {STEP*TIMESTEP}')
#    plt.show()
    STEP = STEP + 1
    if STEP%50 == 0:
        COLORCOUNTER = COLORCOUNTER + 1
        plotGraph(G, TIMESTEP, STEP, colors[COLORCOUNTER])
plt.legend()
plt.show()
#%%
plotGraph(G)