import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rn
import copy as cp

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

fixedAggregation = 5

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

def generate_random_pairs(lista):
    random_pairs = []
    used_numbers = set()

    while len(lista) >= 2:
        num1 = rn.choice(lista)
        lista.remove(num1)
        num2 = rn.choice(lista)
        lista.remove(num2)
        random_pairs.append((num1, num2))

    return random_pairs

#%%
x = np.arange(1, 54)
probDist = np.e**(x*-0.110784)
probDist = np.insert(probDist, 0, 1)
probLong = 1 - probDist
#%% NETWORK INITIALIZATION
numNights = 200
nNodes = 1000
fightsPerNight = 9

fighter = np.arange(0, nNodes)
fightPool = fighter
nFights = numNights * fightsPerNight

lifespan = pd.DataFrame({'fighter': fighter})
lifespan['nFights'] = 0

fights = pd.DataFrame(columns = ['night', 'fighter_A', 'fighter_B', 'winner'])
#%%
night = 0
nFighters = []
totalNumberFights = 0

for night in range(0, numNights):
    night = night + 1
    print(f'night: {night}')
    
    fighterList = list(fightPool)
    random_nodes = rn.sample(fighterList, 2 * fightsPerNight)
    usedNodes = cp.deepcopy(random_nodes)
    randomPairs = generate_random_pairs(random_nodes)

    for pair in randomPairs:
        totalNumberFights = totalNumberFights + 1
        lifespan.loc[pair[0],'nFights'] = lifespan.loc[pair[0],'nFights'] + 1
        lifespan.loc[pair[1],'nFights'] = lifespan.loc[pair[1],'nFights'] + 1
        
        winner = rn.choice(pair)
        if winner == pair[0]:
            fighterWin = 'fighter_A'
#            print(f'fighter_A won, {winner}')            
        if winner == pair[1]:
            fighterWin = 'fighter_B'
#            print(f'fighter_B won, {winner}')
        
        new_row = {'night': f'{night}', 'fighter_A': f'{pair[0]}', 'fighter_B': f'{pair[1]}', 'winner': f'{fighterWin}'}
#        fights = fights.concat(new_row, ignore_index=True)
        fights = pd.concat([fights, pd.DataFrame([new_row])], ignore_index=True)
   
    removed_nodes = 0
    for used in usedNodes:
        lifetime = lifespan.loc[used,'nFights']
        longTest = probLong[lifetime]
        
        rand = rn.uniform(0, 1)
        if rand < longTest:
            fightPool = fightPool[fightPool != used]
            removed_nodes += 1

    new_nodes = np.arange(nNodes, nNodes + fixedAggregation)
    nNodes += fixedAggregation
    fightPool = np.append(fightPool, new_nodes)
    lifespan = pd.concat([lifespan, pd.DataFrame({'fighter': new_nodes, 'nFights': 0})], ignore_index=True)
    nFighters.append(len(fightPool))


print(totalNumberFights)
#fights.to_csv('fights6.txt', index=False)
#lifespan.to_csv('lifespan.txt', index=False)

plt.plot(nFighters, c = 'm',lw = 4)
plt.xlabel(r'$N_{night}$')
plt.ylabel(r'$N_{fighters}$')
plt.savefig('C:/Users/max_s/Desktop/test.pdf', format='pdf', dpi = 2000)
plt.xlim(0, numNights - 1)

plt.show()

