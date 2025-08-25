import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ----------------------------------------------------------------------
# 1) SETUP & DATA IMPORT
# ----------------------------------------------------------------------

# Configure matplotlib (optional)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "mathptmx",
})
plt.rc('axes', labelsize='x-large')
plt.rc('axes', titlesize='x-large')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

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

# Read all CSVs into a dictionary of DataFrames
metrics_data = {}
for name, filepath in metrics_files.items():
    df_temp = pd.read_csv(filepath)
    df_temp.set_index(df_temp.columns[0], inplace=True)  # set the first column as index (fighters)
    metrics_data[name] = df_temp

# ----------------------------------------------------------------------
# 2) DEFINE A HELPER FUNCTION TO FILTER COLUMNS UP TO 2003-04-01
# ----------------------------------------------------------------------

def filter_until_2003_04_01(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame whose columns are date strings like 'YYYY-MM-DD HH:MM:SS',
    return a new DataFrame with only the columns up to and including 2003-04-01.
    """
    # Convert columns to datetime
    col_dates = pd.to_datetime(df.columns, errors='coerce')
    # Create a boolean mask: True if date <= 2003-04-01
    mask = col_dates <= pd.to_datetime("2003-04-01")

    # Filter columns; use .loc to select columns
    df_filtered = df.loc[:, mask]
    return df_filtered

# ----------------------------------------------------------------------
# 3) BUILD A NEW DICTIONARY "metrics_data_2003" FOR FILTERED DATA
# ----------------------------------------------------------------------

metrics_data_2003 = {}
for name, df_sub in metrics_data.items():
    df_filtered = filter_until_2003_04_01(df_sub)
    metrics_data_2003[name] = df_filtered

# ----------------------------------------------------------------------
# 4) FUNCTIONS & ANALYSES (SAME AS BEFORE, APPLIED TO FILTERED DATA)
# ----------------------------------------------------------------------

def get_top_for_20_fighters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame indexed by fighters, with date columns,
    return exactly one row per fighter (the highest value), 
    until we have 20 distinct fighters in total.
    """
    # Collapse into a Series ((fighter, date) -> value), dropping NaNs
    stacked = df.stack(dropna=True)
    # Sort in descending order
    sorted_series = stacked.sort_values(ascending=False)
    
    seen_fighters = set()
    rows = []
    
    # Collect the highest value row for each new fighter until 20 fighters are reached
    for (fighter, date), value in sorted_series.items():
        if fighter not in seen_fighters:
            rows.append((fighter, date, value))
            seen_fighters.add(fighter)
            
            # Stop once we have 20 unique fighters
            if len(seen_fighters) == 20:
                break
    
    result_df = pd.DataFrame(rows, columns=["Fighter", "Date", "Value"])
    return result_df

# ----------------------------------------------------------------------
# 4A) GET TOP 20 FIGHTERS (ONE ROW/FIGHTER) FROM FILTERED DATA
# ----------------------------------------------------------------------

results_dict_2003 = {}
for key, df_sub in metrics_data_2003.items():
    top_df = get_top_for_20_fighters(df_sub)
    results_dict_2003[key] = top_df

# ----------------------------------------------------------------------
# 4B) SUM VALUES ACROSS COLUMNS FOR EACH FIGHTER, STORE IN sum_results_2003
# ----------------------------------------------------------------------

sum_results_2003 = {}
for name, df_sub in metrics_data_2003.items():
    sum_series = df_sub.sum(axis=1, skipna=True)
    sum_df = pd.DataFrame({
        "Fighter": sum_series.index,
        "SumValue": sum_series.values
    })
    sum_results_2003[name] = sum_df

# ----------------------------------------------------------------------
# 4C) GET TOP 20 FIGHTERS BY SUM (FILTERED DATA)
# ----------------------------------------------------------------------

top20_sums_dict_2003 = {}
for name, df_sub in metrics_data_2003.items():
    sum_series = df_sub.sum(axis=1, skipna=True)
    top20 = sum_series.nlargest(20)
    top20_df = pd.DataFrame({
        "Fighter": top20.index,
        "SumValue": top20.values
    })
    top20_sums_dict_2003[name] = top20_df

# ----------------------------------------------------------------------
# 4D) PER-POINT (SUM / COUNT) AND TOP 20 FOR FILTERED DATA
# ----------------------------------------------------------------------

pp_results_2003 = {}
for name, df_sub in metrics_data_2003.items():
    sum_series = df_sub.sum(axis=1, skipna=True)
    count_series = df_sub.count(axis=1)
    per_point_series = sum_series / count_series  # could be NaN if count is 0
    
    pp_df = pd.DataFrame({
        "Fighter": per_point_series.index,
        "PerPointValue": per_point_series.values
    })
    pp_results_2003[name] = pp_df

top20_pp_dict_2003 = {}
for name, df_sub in metrics_data_2003.items():
    sum_series = df_sub.sum(axis=1, skipna=True)
    count_series = df_sub.count(axis=1)
    per_point_series = sum_series / count_series
    
    top20 = per_point_series.nlargest(20)
    top20_df = pd.DataFrame({
        "Fighter": top20.index,
        "PerPointValue": top20.values
    })
    top20_pp_dict_2003[name] = top20_df

# ----------------------------------------------------------------------
# DONE: At this point, you have new dictionaries:
#   results_dict_2003       : top 20 fighters (one row per fighter, by highest value)
#   sum_results_2003        : sum across columns for each fighter
#   top20_sums_dict_2003    : top 20 fighters by sum
#   pp_results_2003         : per-point values for each fighter
#   top20_pp_dict_2003      : top 20 fighters by per-point
# all of which are restricted to columns (dates) <= 2003-04-01.
# ----------------------------------------------------------------------
