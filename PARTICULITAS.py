import numpy as np
import matplotlib.pyplot as plt
import random as rn
import math
import itertools as it
#%%
plt.rc('font', family='serif')
plt.rc('axes', labelsize='large')
plt.rc('axes', titlesize='large')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')

#%%
NPARTS = 50
PARTSPOS = np.empty((NPARTS, 5))
EVOLUTION = np.empty((NPARTS, 5))
PARTRADIUS = 0.01 
TIMESTEP = 0.01
NSTEPS = 20
STEP = 0
BOXSIZE = 1

#%%
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

combinations = np.array(list(it.combinations(np.arange(0, NPARTS), 2)))
#%%
for i in range(0, len(PARTSPOS)):
    xPos = rn.uniform(0,1)
    yPos = rn.uniform(0,1)
    v0 = 0.01*rn.uniform(0,1)
    theta = 2*np.pi*rn.uniform(0,1)
    
    PARTSPOS[i] = i, xPos, yPos, v0, theta
#%%

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
            print(f'collisions = {COLLISIONCOUNT}')
            
        
    OLDPOS = EVOLUTION
    plt.scatter(EVOLUTION[:,1], EVOLUTION[:,2], color = 'k') 
    plt.xlim(0, BOXSIZE)
    plt.ylim(0, BOXSIZE)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(rf'$\tau=$ {STEP*TIMESTEP}')
    plt.show()
    STEP = STEP + 1

