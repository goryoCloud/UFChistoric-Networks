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
path = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/dataframes6Networks'

degreeDf_un = pd.read_csv(f'{path}/degreeDf_un.csv')
clustDf_un = pd.read_csv(f'{path}/clustDf_un.csv')
betDf_un = pd.read_csv(f'{path}/betDf_un.csv')
eigenDf_un = pd.read_csv(f'{path}/eigenDf_un.csv')

degreeDf_un_w = pd.read_csv(f'{path}/degreeDf_un_winners.csv')
degreeDf_un_l = pd.read_csv(f'{path}/degreeDf_un_loosers.csv')

clustDf_un_w = pd.read_csv(f'{path}/clustDf_un_winners.csv')
clustDf_un_l = pd.read_csv(f'{path}/clustDf_un_loosers.csv')

betDf_un_w = pd.read_csv(f'{path}/betDf_un_winners.csv')
betDf_un_l = pd.read_csv(f'{path}/betDf_un_loosers.csv')

eigenDf_un_w = pd.read_csv(f'{path}/eigenDf_un_winners.csv')
eigenDf_un_l = pd.read_csv(f'{path}/eigenDf_un_loosers.csv')

# Directed graphs
degreeDf_dir = pd.read_csv(f'{path}/degreeDf_dir.csv')
clustDf_dir = pd.read_csv(f'{path}/clustDf_dir.csv')
betDf_dir = pd.read_csv(f'{path}/betDf_dir.csv')
eigenDf_dir = pd.read_csv(f'{path}/eigenDf_dir.csv')

degreeDf_dir_w = pd.read_csv(f'{path}/degreeDf_dir_winners.csv')
degreeDf_dir_l = pd.read_csv(f'{path}/degreeDf_dir_loosers.csv')

clustDf_dir_w = pd.read_csv(f'{path}/clustDf_dir_winners.csv')
clustDf_dir_l = pd.read_csv(f'{path}/clustDf_dir_loosers.csv')

betDf_dir_w = pd.read_csv(f'{path}/betDf_dir_winners.csv')
betDf_dir_l = pd.read_csv(f'{path}/betDf_dir_loosers.csv')

eigenDf_dir_w = pd.read_csv(f'{path}/eigenDf_dir_winners.csv')
eigenDf_dir_l = pd.read_csv(f'{path}/eigenDf_dir_loosers.csv')


pathFighter = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/fightersTrends/'
fighterNamesPath = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/dataframes3/fighter_names.csv'

PPVdata = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/dataframes6Networks/PPV_averages.csv')
PPVdata['star_date'] = pd.to_datetime(PPVdata['star_date'])
PPVdata['end_date'] = pd.to_datetime(PPVdata['end_date'])
#%%
def get_data_in_window(start_date, end_date, pathFighter):
    data = pd.read_csv(pathFighter)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')
    data.set_index('date')
    
    return data.loc[start_date:end_date]

def get_data_in_trends(start_date, end_date, pathFighter):
    data = pd.read_csv(pathFighter)
    data['date'] = pd.to_datetime(data['date'])  
    data = data.sort_values('date')
    
    data.set_index('date', inplace=True)  
    return data.loc[start_date:end_date]

def calculate_pearson_correlation(result_df, trend_df):
    result_df['Date'] = pd.to_datetime(result_df['Date'])
    trend_df['star_date'] = pd.to_datetime(trend_df['star_date'])
    
    merged_df = pd.merge(result_df, trend_df, left_on='Date', right_on='star_date')
    merged_df = merged_df.dropna(subset=['Value', 'trend_mean'])

    correlation, p_value = pearsonr(merged_df['Value'], merged_df['trend_mean'])
    
    return correlation, p_value

def fighterTrendTimeseries(fighter):
    trend_results = []

    pathFighter = f'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/fightersTrends/{fighter}.csv'

    for dateIndex in range(len(PPVdata['star_date'])):
        start = pd.to_datetime(PPVdata['star_date'][dateIndex])
        end = pd.to_datetime(PPVdata['end_date'][dateIndex])

        fighterInfo = get_data_in_trends(start, end, pathFighter)
        trendMean = fighterInfo[f'{fighter}'].mean()
        trend_results.append({'star_date': start, 'trend_mean': trendMean})
        
    trendData = pd.DataFrame(trend_results)    
    return trendData

def fighterMetricsCorrelations(fighter_name):
    
    trends = fighterTrendTimeseries(fighter_name)
    
    degree_un = degreeDf_un.set_index(degreeDf_un.columns[0])
    clust_un = clustDf_un.set_index(clustDf_un.columns[0])
    bet_un = betDf_un.set_index(betDf_un.columns[0])
    eigen_un = eigenDf_un.set_index(eigenDf_un.columns[0])
    degree_w = degreeDf_un_w.set_index(degreeDf_un_w.columns[0])
    degree_l = degreeDf_un_l.set_index(degreeDf_un_l.columns[0])
    clust_w = clustDf_un_w.set_index(clustDf_un_w.columns[0])
    clust_l = clustDf_un_l.set_index(clustDf_un_l.columns[0])
    bet_w = betDf_un_w.set_index(betDf_un_w.columns[0])
    bet_l = betDf_un_l.set_index(betDf_un_l.columns[0])
    eigen_w = eigenDf_un_w.set_index(eigenDf_un_w.columns[0])
    eigen_l = eigenDf_un_l.set_index(eigenDf_un_l.columns[0])
    degree_dir = degreeDf_dir.set_index(degreeDf_dir.columns[0])
    clust_dir = clustDf_dir.set_index(clustDf_dir.columns[0])
    bet_dir = betDf_dir.set_index(betDf_dir.columns[0])
    eigen_dir = eigenDf_dir.set_index(eigenDf_dir.columns[0])
    degree_dir_w = degreeDf_dir_w.set_index(degreeDf_dir_w.columns[0])
    degree_dir_l = degreeDf_dir_l.set_index(degreeDf_dir_l.columns[0])
    clust_dir_w = clustDf_dir_w.set_index(clustDf_dir_w.columns[0])
    clust_dir_l = clustDf_dir_l.set_index(clustDf_dir_l.columns[0])
    bet_dir_w = betDf_dir_w.set_index(betDf_dir_w.columns[0])
    bet_dir_l = betDf_dir_l.set_index(betDf_dir_l.columns[0])
    eigen_dir_w = eigenDf_dir_w.set_index(eigenDf_dir_w.columns[0])
    eigen_dir_l = eigenDf_dir_l.set_index(eigenDf_dir_l.columns[0])
    
    dataframes = ["degree_un", "clust_un", "bet_un", "eigen_un", "degree_w", "degree_l",
                  "clust_w", "clust_l", "bet_w", "bet_l", "eigen_w", "eigen_l",
                  "degree_dir", "clust_dir", "bet_dir", "eigen_dir", "degree_dir_w",
                  "degree_dir_l", "clust_dir_w", "clust_dir_l", "bet_dir_w", "bet_dir_l",
                  "eigen_dir_w", "eigen_dir_l"]
    
    correlations = []
    
    for dataframeName in dataframes:    
        df = eval(dataframeName)  # Dynamically get the dataframe object
        if fighter_name in df.index:
            row = df.loc[fighter_name]
            result = row.dropna().reset_index()
            result.columns = ["Date", "Value"]

        # Ensure both columns are datetime type
            result['Date'] = pd.to_datetime(result['Date'])
            trends['star_date'] = pd.to_datetime(trends['star_date'])

        # Merge to ensure alignment of dates
            merged_df = pd.merge(result, trends, left_on='Date', right_on='star_date', how='inner')

        # Drop rows with NaN values
            merged_df = merged_df.dropna(subset=['Value', 'trend_mean'])

        # Calculate correlation only for aligned data
            correlation, _ = pearsonr(merged_df['Value'], merged_df['trend_mean'])
        
        # Append to correlations list with correct structure
            correlations.append({"Metric": dataframeName, "Correlation": correlation})

# Ensure correlations list is converted to DataFrame properly
    correlation_df = pd.DataFrame(correlations)
    return correlation_df

#%%
fighterNames = pd.read_csv(fighterNamesPath)

degree_un_list = []
clust_un_list = []
bet_un_list = []
eigen_un_list = []
degree_w_list = []
degree_l_list = []
clust_w_list = []
clust_l_list = []
bet_w_list = []
bet_l_list = []
eigen_w_list = []
eigen_l_list = []
degree_dir_list = []
clust_dir_list = []
bet_dir_list = []
eigen_dir_list = []
degree_dir_w_list = []
degree_dir_l_list = []
clust_dir_w_list = []
clust_dir_l_list = []
bet_dir_w_list = []
bet_dir_l_list = []
eigen_dir_w_list = []
eigen_dir_l_list = []

error = 0
metric_to_list = {
    "degree_un": degree_un_list,
    "clust_un": clust_un_list,
    "bet_un": bet_un_list,
    "eigen_un": eigen_un_list,
    "degree_w": degree_w_list,
    "degree_l": degree_l_list,
    "clust_w": clust_w_list,
    "clust_l": clust_l_list,
    "bet_w": bet_w_list,
    "bet_l": bet_l_list,
    "eigen_w": eigen_w_list,
    "eigen_l": eigen_l_list,
    "degree_dir": degree_dir_list,
    "clust_dir": clust_dir_list,
    "bet_dir": bet_dir_list,
    "eigen_dir": eigen_dir_list,
    "degree_dir_w": degree_dir_w_list,
    "degree_dir_l": degree_dir_l_list,
    "clust_dir_w": clust_dir_w_list,
    "clust_dir_l": clust_dir_l_list,
    "bet_dir_w": bet_dir_w_list,
    "bet_dir_l": bet_dir_l_list,
    "eigen_dir_w": eigen_dir_w_list,
    "eigen_dir_l": eigen_dir_l_list,
}


for fighterName in fighterNames['fighter']:
    print("-------------------------------------------------------------------------------------------")
    print(f"Processing: {fighterName}")
    try:
        pearsonMetrics = fighterMetricsCorrelations(fighterName)
        
        for metric, metric_list in metric_to_list.items():
            print(f"(a) Processing {metric}")
            metric_value = pearsonMetrics.loc[pearsonMetrics['Metric'] == metric, 'Correlation'].values
            
            if len(metric_value) >= 0:
                metric_list.append(metric_value[0])
            print(f"(b) Processed {metric}")    

    except Exception as e:
        print(f"(*) Skipping {metric} due to error {e}")

#%%
for metric_name, metric_list in metric_to_list.items():
    plt.figure(figsize=(8, 6))  # Create a new figure for each histogram
    plt.hist(metric_list, bins=21, edgecolor='black', alpha=0.7, density = True)
    plt.title(f'Histogram of {metric_name}', fontsize=16)
    plt.xlabel('Pearson coefficient value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.xlim(-1,1)
    plt.ylim(1)
#    plt.savefig(f'{metric_name}_histogram.png')  # Save each histogram as a separate file
    plt.show()
