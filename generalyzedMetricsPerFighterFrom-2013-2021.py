import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ------------------------------------------------------------------------------
# 1) SETUP & DATA IMPORT
# ------------------------------------------------------------------------------
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
#%%
P4Punique = ['Alexander Gustafsson',
 'Alexander Volkanovski',
 'Anderson Silva',
 'Anthony Pettis',
 'Benson Henderson',
 'Cain Velasquez',
 'Chris Weidman',
 'Cody Garbrandt',
 'Conor McGregor',
 'Dan Henderson',
 'Daniel Cormier',
 'Demetrious Johnson',
 'Dominick Cruz',
 'Dustin Poirier',
 'Eddie Alvarez',
 'Fabricio Werdum',
 'Frankie Edgar',
 'Georges St. Pierre',
 'Gilbert Melendez',
 'Henry Cejudo',
 'Israel Adesanya',
 'Johny Hendricks',
 'Jon Jones',
 'José Aldo',
 'Jéssica Andrade',
 'Kamaru Usman',
 'Khabib Nurmagomedov',
 'Luke Rockhold',
 'Max Holloway',
 'Michael Bisping',
 'Rafael Dos Anjos',
 'Renan Barao',
 'Robbie Lawler',
 'Robert Whittaker',
 'Stipe Miocic',
 'TJ Dillashaw',
 'Tony Ferguson',
 'Tyron Woodley',
 'Urijah Faber',
 'Vitor Belfort',
 'Yoel Romero']
#%%

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
    # etc.
}

# Read all CSVs into a dictionary of DataFrames
metrics_data = {}
for name, filepath in metrics_files.items():
    df_temp = pd.read_csv(filepath)
    # set the first column as the index (fighter names)
    df_temp.set_index(df_temp.columns[0], inplace=True)
    metrics_data[name] = df_temp

# ------------------------------------------------------------------------------
# 2) DEFINE HELPER FUNCTION TO FILTER COLUMNS (DATES) BETWEEN 2013-02-01 AND 2021-03-31
# ------------------------------------------------------------------------------
def filter_2013feb_to_2021mar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame whose columns are date strings like 'YYYY-MM-DD HH:MM:SS',
    returns a new DataFrame with only the columns from 2013-02-01 through 2021-03-31.
    """
    # Convert columns to datetime
    col_dates = pd.to_datetime(df.columns, errors='coerce')
    # Create a boolean mask for columns >= 2013-02-01 and <= 2021-03-31
    start_date = pd.to_datetime("2013-02-01")
    end_date = pd.to_datetime("2021-03-31")
    mask = (col_dates >= start_date) & (col_dates <= end_date)
    
    # Filter columns using .loc
    df_filtered = df.loc[:, mask]
    return df_filtered

# ------------------------------------------------------------------------------
# 3) BUILD A NEW DICTIONARY "metrics_data_2013feb_2021mar" FOR FILTERED DATA
# ------------------------------------------------------------------------------
metrics_data_2013feb_2021mar = {}
for name, df_sub in metrics_data.items():
    df_filtered = filter_2013feb_to_2021mar(df_sub)
    metrics_data_2013feb_2021mar[name] = df_filtered

# ------------------------------------------------------------------------------
# 4) FUNCTIONS & REPEAT ANALYSIS ON FILTERED DATA
# ------------------------------------------------------------------------------
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
            if len(seen_fighters) == 41:
                break
    
    result_df = pd.DataFrame(rows, columns=["Fighter", "Date", "Value"])
    return result_df

# (4A) GET TOP 20 FIGHTERS (ONE ROW/FIGHTER) FROM FILTERED DATA
results_dict_2013feb_2021mar = {}
for key, df_sub in metrics_data_2013feb_2021mar.items():
    top_df = get_top_for_20_fighters(df_sub)
    results_dict_2013feb_2021mar[key] = top_df

# (4B) SUM VALUES ACROSS COLUMNS FOR EACH FIGHTER
sum_results_2013feb_2021mar = {}
for name, df_sub in metrics_data_2013feb_2021mar.items():
    sum_series = df_sub.sum(axis=1, skipna=True)
    sum_df = pd.DataFrame({
        "Fighter": sum_series.index,
        "SumValue": sum_series.values
    })
    sum_results_2013feb_2021mar[name] = sum_df

# (4C) GET TOP 20 FIGHTERS BY SUM
top20_sums_dict_2013feb_2021mar = {}
for name, df_sub in metrics_data_2013feb_2021mar.items():
    sum_series = df_sub.sum(axis=1, skipna=True)
    top20 = sum_series.nlargest(41)
    top20_df = pd.DataFrame({
        "Fighter": top20.index,
        "SumValue": top20.values
    })
    top20_sums_dict_2013feb_2021mar[name] = top20_df

# (4D) PER-POINT (SUM / COUNT) AND TOP 20
pp_results_2013feb_2021mar = {}
for name, df_sub in metrics_data_2013feb_2021mar.items():
    sum_series = df_sub.sum(axis=1, skipna=True)
    count_series = df_sub.count(axis=1)
    per_point_series = sum_series / count_series  # could be NaN if count is 0
    
    pp_df = pd.DataFrame({
        "Fighter": per_point_series.index,
        "PerPointValue": per_point_series.values
    })
    pp_results_2013feb_2021mar[name] = pp_df

top20_pp_dict_2013feb_2021mar = {}
for name, df_sub in metrics_data_2013feb_2021mar.items():
    sum_series = df_sub.sum(axis=1, skipna=True)
    count_series = df_sub.count(axis=1)
    per_point_series = sum_series / count_series
    
    top20 = per_point_series.nlargest(41)
    top20_df = pd.DataFrame({
        "Fighter": top20.index,
        "PerPointValue": top20.values
    })
    top20_pp_dict_2013feb_2021mar[name] = top20_df

# ------------------------------------------------------------------------------
# DONE: At this point, you have new dictionaries (for dates between 2013-02-01 and 2021-03-31):
#
#   results_dict_2013feb_2021mar      : top 20 fighters (one row per fighter, by highest value)
#   sum_results_2013feb_2021mar       : sum across columns for each fighter
#   top20_sums_dict_2013feb_2021mar   : top 20 fighters by sum
#   pp_results_2013feb_2021mar        : per-point values for each fighter
#   top20_pp_dict_2013feb_2021mar     : top 20 fighters by per-point
#
# ------------------------------------------------------------------------------

#%%
