import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import pearsonr

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
pathDataframes = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/dataframes3/betweenessEvol.csv'
pathFighter = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/fightersTrends/'
fighterNamesPath = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/dataframes3/fighter_names.csv'
#%%
def pearsonCoeff(arrayX, arrayY):
    return pearsonr(arrayX, arrayY)

def fighterTrend(fighter):
    pathFighter = f'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/pound-for-pound/PFP_history/{fighter}_timeseries.csv'
    data = pd.read_csv(pathFighter)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    result = data.rolling(window='730D').mean().resample('MS').first()
    result.reset_index(inplace=True)
    
    return result

degree_evol_data = pd.read_csv(pathDataframes, index_col='fighter')
    
def fighterMetrics(fighter_name):
    if fighter_name in degree_evol_data.index:
        fighter_data = degree_evol_data.loc[fighter_name].dropna()
        fighter_data.index = pd.to_datetime(fighter_data.index, format='%Y-%m')
        fighter_data_df = fighter_data.reset_index()
        fighter_data_df.columns = ['date', f'{fighter_name}']
#        fighter_data_df['fighter'] = fighter_name
#        fighter_data_df = fighter_data_df[['date', 'fighter', 'degree']]
        
        return fighter_data_df
    else:
        return f"No data available for {fighter_name}"
#%%
fighterNames = pd.read_csv(fighterNamesPath)
#fighterNames = ['Conor McGregor']
degCorrelation = []
for fighter in fighterNames['fighter']:
    print(fighter)
    try:
        deg = fighterMetrics(fighter)
        google = fighterTrend(fighter)
    
        if google is None or deg is None:
            print(f"Skipping {fighter} due to missing data or 'date' column.")
            continue

        merged_data = pd.merge(google, deg, on='date', how='left')
        merged_data = merged_data.fillna(0)
        pearson = pearsonCoeff(merged_data['rank'], merged_data[f'{fighter}'])[0]
            
        new_value = fighter, pearson
        degCorrelation.append(new_value)
    
    except Exception as e:
        print(f"Error processing {fighter}: {e}")
        continue

#%%
import csv
output_csv_path = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/pound-for-pound/PFP_correlations/betEvolCorrelation.csv'

# Write the list to a CSV file
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(degCorrelation)

    

