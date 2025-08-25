import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
#%%
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
#%%
path = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/dataframes6Networks'

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
#%%
def get_top_for_20_fighters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame indexed by fighters, with date columns,
    return exactly one row per fighter (the highest value), 
    until we have 20 distinct fighters in total.
    """
    # 1) Collapse into a Series ((fighter, date) -> value), dropping NaNs
    stacked = df.stack(dropna=True)
    # 2) Sort in descending order
    sorted_series = stacked.sort_values(ascending=False)
    
    seen_fighters = set()
    rows = []
    
    # 3) Iterate over sorted values
    for (fighter, date), value in sorted_series.items():
        # Only add the FIRST time we see each fighter
        if fighter not in seen_fighters:
            rows.append((fighter, date, value))
            seen_fighters.add(fighter)
            
            # Stop if we've reached 20 distinct fighters
            if len(seen_fighters) == 20:
                break
    
    # 4) Convert to DataFrame
    result_df = pd.DataFrame(rows, columns=["Fighter", "Date", "Value"])
    return result_df


# Example usage:
results_dict = {}

for key, df_sub in metrics_data.items():
    top_df = get_top_for_20_fighters(df_sub)
    results_dict[key] = top_df
#%%

sum_results = {}  # new dictionary to hold (key -> df of sums)

for name, df_sub in metrics_data.items():
    # 1) Sum across columns (axis=1)
    #    skipna=True ensures that NaNs don't propagate.
    sum_series = df_sub.sum(axis=1, skipna=True)
    
    # 2) Convert this Series into a DataFrame:
    #    sum_series.index is the fighter name
    #    sum_series.values is the numeric sum
    sum_df = pd.DataFrame({
        "Fighter": sum_series.index,
        "SumValue": sum_series.values
    })
    
    # 3) Store the result in our new dictionary
    sum_results[name] = sum_df

#%%
top20_sums_dict = {}  # will store a "top 10" DataFrame per dictionary key

for name, df_sub in metrics_data.items():
    # 1) Sum across columns (i.e., across all dates)
    sum_series = df_sub.sum(axis=1, skipna=True)
    
    # 2) Extract the top 10 sums
    top20 = sum_series.nlargest(20)
    
    # 3) Convert to DataFrame for readability
    top20_df = pd.DataFrame({
        "Fighter": top20.index,
        "SumValue": top20.values
    })
    
    # Store this top-10 DataFrame in a dictionary
    top20_sums_dict[name] = top20_df
    
#%%
pp_results = {}  # new dict to hold (key -> df of per-point values)

for name, df_sub in metrics_data.items():
    # sum across columns (ignoring NaN)
    sum_series = df_sub.sum(axis=1, skipna=True)
    # count how many non-NaN entries each fighter has
    count_series = df_sub.count(axis=1)
    
    # compute "per-point" = sum / count (for fighters with count > 0)
    per_point_series = sum_series / count_series
    
    # convert to DataFrame
    pp_df = pd.DataFrame({
        "Fighter": per_point_series.index,
        "PerPointValue": per_point_series.values
    })
    
    pp_results[name] = pp_df

# 2) Get top 20 fighters by "PerPointValue" (descending)
top20_pp_dict = {}

for name, df_sub in metrics_data.items():
    # sum + count
    sum_series = df_sub.sum(axis=1, skipna=True)
    count_series = df_sub.count(axis=1)
    
    # per-point
    per_point_series = sum_series / count_series
    
    # pick top 20
    top20 = per_point_series.nlargest(20)
    
    top20_df = pd.DataFrame({
        "Fighter": top20.index,
        "PerPointValue": top20.values
    })
    top20_pp_dict[name] = top20_df