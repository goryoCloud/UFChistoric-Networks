import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Configure matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "mathptmx",
})

plt.rc('axes', labelsize='x-large')
plt.rc('axes', titlesize='x-large')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

# Define paths
path = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/dataframes6Networks'
pathFighter = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/fightersTrends/'
fighterNamesPath = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/dataframes3/fighter_names.csv'

# Load data
PPVdata = pd.read_csv(f'{path}/PPV_averages.csv')
PPVdata['star_date'] = pd.to_datetime(PPVdata['star_date'])
PPVdata['end_date'] = pd.to_datetime(PPVdata['end_date'])

# Load metrics data
metrics_files = {
    "degree_un": f"{path}/degreeDf_un.csv",
    "clust_un": f"{path}/clustDf_un.csv",
    "bet_un": f"{path}/betDf_un.csv",
    "eigen_un": f"{path}/eigenDf_un.csv",
    "degree_w": f"{path}/degreeDf_un_winners.csv",
    "degree_l": f"{path}/degreeDf_un_loosers.csv",
    "clust_w": f"{path}/clustDf_un_winners.csv",
    "clust_l": f"{path}/clustDf_un_loosers.csv",
    "bet_w": f"{path}/betDf_un_winners.csv",
    "bet_l": f"{path}/betDf_un_loosers.csv",
    "eigen_w": f"{path}/eigenDf_un_winners.csv",
    "eigen_l": f"{path}/eigenDf_un_loosers.csv",
    "degree_dir": f"{path}/degreeDf_dir.csv",
    "clust_dir": f"{path}/clustDf_dir.csv",
    "bet_dir": f"{path}/betDf_dir.csv",
    "eigen_dir": f"{path}/eigenDf_dir.csv",
    "degree_dir_w": f"{path}/degreeDf_dir_winners.csv",
    "degree_dir_l": f"{path}/degreeDf_dir_loosers.csv",
    "clust_dir_w": f"{path}/clustDf_dir_winners.csv",
    "clust_dir_l": f"{path}/clustDf_dir_loosers.csv",
    "bet_dir_w": f"{path}/betDf_dir_winners.csv",
    "bet_dir_l": f"{path}/betDf_dir_loosers.csv",
    "eigen_dir_w": f"{path}/eigenDf_dir_winners.csv",
    "eigen_dir_l": f"{path}/eigenDf_dir_loosers.csv",
}

metrics_data = {name: pd.read_csv(filepath).set_index(pd.read_csv(filepath).columns[0]) for name, filepath in metrics_files.items()}

# Functions
def get_data_in_trends(start_date, end_date, pathFighter):
    data = pd.read_csv(pathFighter)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')
    data.set_index('date', inplace=True)
    return data.loc[start_date:end_date]

def fighterTrendTimeseries(fighter):
    trend_results = []
    pathFighter = f'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/fightersTrends/{fighter}.csv'

    for dateIndex in range(len(PPVdata['star_date'])):
        start = PPVdata['star_date'][dateIndex]
        end = PPVdata['end_date'][dateIndex]

        fighterInfo = get_data_in_trends(start, end, pathFighter)
        trendMean = fighterInfo[f'{fighter}'].mean()
        trend_results.append({'star_date': start, 'trend_mean': trendMean})

    trendData = pd.DataFrame(trend_results)
    return trendData

def fighterMetricsCorrelations(fighter_name):
    trends = fighterTrendTimeseries(fighter_name)
    correlations = []

    for metric_name, df in metrics_data.items():
        try:
            if fighter_name in df.index:
                row = df.loc[fighter_name].dropna().reset_index()
                row.columns = ["Date", "Value"]
                row['Date'] = pd.to_datetime(row['Date'])

                trends['star_date'] = pd.to_datetime(trends['star_date'])
                merged_df = pd.merge(row, trends, left_on='Date', right_on='star_date', how='inner')
                merged_df = merged_df.dropna(subset=['Value', 'trend_mean'])

                correlation, _ = pearsonr(merged_df['Value'], merged_df['trend_mean'])
                correlations.append({"Metric": metric_name, "Correlation": correlation})
            else:
                correlations.append({"Metric": metric_name, "Correlation": np.nan})
        except Exception as e:
            print(f"Error processing metric {metric_name} for fighter {fighter_name}: {e}")
            correlations.append({"Metric": metric_name, "Correlation": np.nan})

    return pd.DataFrame(correlations)

# Main processing
fighterNames = pd.read_csv(fighterNamesPath)
metric_to_list = {name: [] for name in metrics_data.keys()}
nFighter = 1

for fighterName in fighterNames['fighter']:
    print('----------------------------------------------------------------------------------------')
    print(f"Processing: {fighterName} ({nFighter}/{len(fighterNames)})")
    try:
        pearsonMetrics = fighterMetricsCorrelations(fighterName)

        for metric, metric_list in metric_to_list.items():
            try:
                print(f"(a) Processing {metric}")
                metric_value = pearsonMetrics.loc[pearsonMetrics['Metric'] == metric, 'Correlation'].values
                metric_list.append(metric_value[0] if len(metric_value) > 0 else np.nan)
                print(f"(b) Processed {metric}")
            except Exception as e:
                print(f"(!) Error processing {metric}")
                metric_list.append(np.nan)

    except Exception as e:
        print(f"Error processing fighter {fighterName}: {e}")

    nFighter += 1
#%%
# Visualization
for metric_name, metric_list in metric_to_list.items():
    # Remove NaN values
    filtered_metric_list = [x for x in metric_list if not np.isnan(x)]
    plt.figure(figsize=(8, 6))
    plt.hist(filtered_metric_list, bins=15, edgecolor='black', alpha=0.7, density=True)
    plt.title(f'Histogram of {metric_name}', fontsize=16)
    plt.xlabel('Pearson correlation coeff. Value', fontsize=14)
    plt.ylabel('frequency (normalized)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.xlim(-1, 1)
    plt.show()
    
#%%
import json

with open(f'{path}/UFCdata.json', 'w') as file:
    json.dump(metric_to_list, file, indent=4)
