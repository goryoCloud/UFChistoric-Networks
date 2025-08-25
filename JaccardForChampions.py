import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# -------------------------------------------------------------------
# 1) SETUP
# -------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "mathptmx",
})
plt.rc('axes', labelsize='x-large')
plt.rc('axes', titlesize='x-large')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

# Path to your CSV files
path = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/dataframes6Networks'

P4Punique = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/pound-for-pound/ranked0_unique_fighters.csv')
P4Punique = P4Punique['Fighter'].to_numpy()
#%%
# Convert P4P list to lowercase + strip, then to a set
p4p_set = {name.lower().strip() for name in P4Punique}

# -------------------------------------------------------------------
# 2) READ DATA INTO A DICTIONARY OF DATAFRAMES
# -------------------------------------------------------------------
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
    # etc. If you have more, list them here
}

metrics_data = {}
for name, filepath in metrics_files.items():
    df_temp = pd.read_csv(filepath)
    # set the first column as the index (fighter names)
    df_temp.set_index(df_temp.columns[0], inplace=True)
    metrics_data[name] = df_temp

# -------------------------------------------------------------------
# 3) FILTER COLUMNS FOR 2013-02-01 TO 2021-03-31
# -------------------------------------------------------------------
def filter_2013feb_to_2021mar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only columns (dates) between 2013-02-01 and 2021-03-31.
    """
    col_dates = pd.to_datetime(df.columns, errors='coerce')
    start_date = pd.to_datetime("2013-02-01")
    end_date = pd.to_datetime("2021-03-31")
    mask = (col_dates >= start_date) & (col_dates <= end_date)
    df_filtered = df.loc[:, mask]
    return df_filtered

# Build a new dictionary with filtered columns
metrics_data_2013feb_2021mar = {}
for name, df_sub in metrics_data.items():
    df_filtered = filter_2013feb_to_2021mar(df_sub)
    metrics_data_2013feb_2021mar[name] = df_filtered

# -------------------------------------------------------------------
# 4) ANALYSES
# -------------------------------------------------------------------

def get_top_for_20_fighters(df: pd.DataFrame, max_fighters=len(p4p_set)) -> pd.DataFrame:
    """
    Return exactly one row per fighter (the highest value), 
    until we have `max_fighters` distinct fighters in total.
    """
    stacked = df.stack(dropna=True)   # ((fighter, date) -> value)
    sorted_series = stacked.sort_values(ascending=False)
    
    seen_fighters = set()
    rows = []
    
    for (fighter, date), value in sorted_series.items():
        if fighter not in seen_fighters:
            rows.append((fighter, date, value))
            seen_fighters.add(fighter)
            if len(seen_fighters) == max_fighters:
                break
    
    result_df = pd.DataFrame(rows, columns=["Fighter", "Date", "Value"])
    return result_df

# (A) results_dict_2013feb_2021mar: top 20 fighters by highest value
results_dict_2013feb_2021mar = {}
for key, df_sub in metrics_data_2013feb_2021mar.items():
    top_df = get_top_for_20_fighters(df_sub, max_fighters=len(p4p_set))  
    # ^ You mentioned 61 in your code, change as needed
    results_dict_2013feb_2021mar[key] = top_df

# (B) sum_results_2013feb_2021mar: sum across columns for each fighter
sum_results_2013feb_2021mar = {}
for name, df_sub in metrics_data_2013feb_2021mar.items():
    sum_series = df_sub.sum(axis=1, skipna=True)
    sum_df = pd.DataFrame({
        "Fighter": sum_series.index,
        "SumValue": sum_series.values
    })
    sum_results_2013feb_2021mar[name] = sum_df

# (C) top20_sums_dict_2013feb_2021mar: top 20 by sum
top20_sums_dict_2013feb_2021mar = {}
for name, df_sub in metrics_data_2013feb_2021mar.items():
    sum_series = df_sub.sum(axis=1, skipna=True)
    topN = sum_series.nlargest(len(p4p_set))  # again, your code used 61
    topN_df = pd.DataFrame({
        "Fighter": topN.index,
        "SumValue": topN.values
    })
    top20_sums_dict_2013feb_2021mar[name] = topN_df

# (D) per-point (sum / count)
pp_results_2013feb_2021mar = {}
for name, df_sub in metrics_data_2013feb_2021mar.items():
    sum_series = df_sub.sum(axis=1, skipna=True)
    count_series = df_sub.count(axis=1)
    per_point_series = sum_series / count_series  # can be NaN if count=0
    
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
    
    topN = per_point_series.nlargest(len(p4p_set))
    topN_df = pd.DataFrame({
        "Fighter": topN.index,
        "PerPointValue": topN.values
    })
    top20_pp_dict_2013feb_2021mar[name] = topN_df

# -------------------------------------------------------------------
# 5) JACCARD COEFFICIENT vs. P4P SET
# -------------------------------------------------------------------
def jaccard_coefficient(set_a, set_b):
    """Compute the Jaccard coefficient = |A ∩ B| / |A ∪ B|."""
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if len(union) else 0.0

# Because fighter names might differ in capitalization,
# let's convert them all to lowercase + strip before making sets.

def get_fighter_set_lower(df: pd.DataFrame, fighter_col="Fighter") -> set:
    """
    Return a set of fighters from df[fighter_col], all in lowercase and stripped.
    """
    return set(s.lower().strip() for s in df[fighter_col].dropna())

# (A) Jaccard for results_dict_2013feb_2021mar
jaccard_results_for_results_dict = {}

for metric_name, df_sub in results_dict_2013feb_2021mar.items():
    # create a set of fighters (in lowercase)
    fighters_set = get_fighter_set_lower(df_sub, fighter_col="Fighter")
    # compute Jaccard with p4p_set
    jc = jaccard_coefficient(fighters_set, p4p_set)
    jaccard_results_for_results_dict[metric_name] = jc

# (B) Jaccard for top20_pp_dict_2013feb_2021mar
jaccard_results_for_top20pp = {}

for metric_name, df_sub in top20_pp_dict_2013feb_2021mar.items():
    fighters_set = get_fighter_set_lower(df_sub, fighter_col="Fighter")
    jc = jaccard_coefficient(fighters_set, p4p_set)
    jaccard_results_for_top20pp[metric_name] = jc

# -------------------------------------------------------------------
# 6) INSPECT RESULTS
# -------------------------------------------------------------------
print("=== Jaccard Coefficients for results_dict_2013feb_2021mar ===")
for metric, val in jaccard_results_for_results_dict.items():
    print(f"{metric}: {val:.3f}")

print("\n=== Jaccard Coefficients for top20_pp_dict_2013feb_2021mar ===")
for metric, val in jaccard_results_for_top20pp.items():
    print(f"{metric}: {val:.3f}")

# The above prints a numeric Jaccard for each metric (0.0 ~ no overlap, 1.0 ~ perfect overlap).
# -------------------------------------------------------------------
