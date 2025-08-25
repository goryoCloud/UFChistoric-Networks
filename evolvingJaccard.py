# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 04:00:17 2025

@author: max_s
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1) IMPORT & SETUP
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

# Path to your CSV files for the metrics
path = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/dataframes6Networks'

# Your dictionary of wide DataFrames (rows=fighters, columns=dates)
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

# The CSV for official UFC rankings (columns: date,weightclass,fighter,rank).
# Example structure:
#   date,weightclass,fighter,rank
#   2013-02-04,Pound-for-Pound,Anderson Silva,1
#   ...
rankings_csv = "C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/pound-for-pound/rankings_history.csv"

# ------------------------------------------------------------------------------
# 2) READ & STACK THE WIDE METRIC DATAFRAMES
# ------------------------------------------------------------------------------

def read_and_stack_wide_df(filepath: str) -> pd.DataFrame:
    """
    Reads a CSV with row 0 as fighter names, columns as dates,
    and stacks it into a long format DataFrame with columns:
      ["Fighter", "Date", "Value"]
    """
    df_wide = pd.read_csv(filepath)
    # first column is fighter name
    df_wide.set_index(df_wide.columns[0], inplace=True)

    # columns are date strings. We'll keep them as strings, but let's rename them as needed.
    # .stack() => Series with MultiIndex (fighter, date)
    stacked = df_wide.stack(dropna=True)
    # Convert the MultiIndex to columns
    # stacked is a Series; we convert to DataFrame
    df_long = stacked.reset_index()
    # By default: level_0=fighter, level_1=the old column name, 0=the value
    df_long.columns = ["Fighter", "Date", "Value"]

    return df_long

def filter_dates_2013_2021(df: pd.DataFrame, date_col="Date") -> pd.DataFrame:
    """
    Given a DataFrame with 'Date' column in string form,
    keep only rows with Date in [2013-02-01 .. 2021-03-31].
    """
    # Convert the date col to datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    start_date = pd.to_datetime("2013-02-01")
    end_date   = pd.to_datetime("2021-03-31")
    mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    return df.loc[mask].copy()

# We'll build a dictionary: stacked_data_2013_2021[metric_name] = long DataFrame
stacked_data_2013_2021 = {}

for metric_name, filepath in metrics_files.items():
    df_long = read_and_stack_wide_df(filepath)
    df_filtered = filter_dates_2013_2021(df_long, date_col="Date")
    stacked_data_2013_2021[metric_name] = df_filtered

# ------------------------------------------------------------------------------
# 3) READ RANKINGS & DEFINE 2-YEAR MONTHLY WINDOWS
# ------------------------------------------------------------------------------

df_rank = pd.read_csv(rankings_csv, parse_dates=["date"])

start_date = pd.to_datetime("2013-02-01")
end_date   = pd.to_datetime("2021-03-31")

def window_end(d: pd.Timestamp) -> pd.Timestamp:
    """Return the end date for a 2-year window starting at d (minus 1 day)."""
    return d + pd.DateOffset(years=2) - pd.Timedelta(days=1)

# Build a list of monthly starts
window_starts = []
current = start_date
while current <= end_date:
    window_starts.append(current)
    current = current + pd.DateOffset(months=1)  # shift by 1 month

# The main weight classes for rank=1 or rank=1..15
WEIGHT_CLASSES = [
    "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
    "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight"
]

def extract_sets_for_window(df: pd.DataFrame, start_d: pd.Timestamp, end_d: pd.Timestamp):
    """
    From the rankings DataFrame, filter rows in [start_d, end_d].
    Return 3 sets:
      1) set_p4p   (all Pound-for-Pound, any rank)
      2) set_rank1 (rank=1 in the main weight classes)
      3) set_rank_1_15 (rank in [1..15] in the main weight classes)
    """
    mask = (df["date"] >= start_d) & (df["date"] <= end_d)
    df_window = df.loc[mask].copy()

    # standardize fighter names
    df_window["fighter_lower"] = df_window["fighter"].str.lower().str.strip()

    # P4P
    df_p4p = df_window[df_window["weightclass"] == "Pound-for-Pound"]
    set_p4p = set(df_p4p["fighter_lower"].unique())

    # rank=1 in main classes
    df_rank1 = df_window[
        (df_window["weightclass"].isin(WEIGHT_CLASSES)) &
        (df_window["rank"] == 1)
    ]
    set_rank1 = set(df_rank1["fighter_lower"].unique())

    # rank=1..15 in main classes
    df_rank_1_15 = df_window[
        (df_window["weightclass"].isin(WEIGHT_CLASSES)) &
        (df_window["rank"] >= 1) &
        (df_window["rank"] <= 15)
    ]
    set_rank_1_15 = set(df_rank_1_15["fighter_lower"].unique())

    return set_p4p, set_rank1, set_rank_1_15

# ------------------------------------------------------------------------------
# 4) DEFINING HELPER FUNCTIONS TO PICK "TOP X FIGHTERS" FOR A GIVEN DATE
#    & COMPUTE JACCARD
# ------------------------------------------------------------------------------
def jaccard_coefficient(set_a, set_b):
    inter = set_a & set_b
    union = set_a | set_b
    return len(inter) / len(union) if len(union) else 0.0

def get_fighter_set_lower(df: pd.DataFrame, fighter_col="Fighter") -> set:
    """Return a set of fighters from df[fighter_col], all in lowercase + stripped."""
    return set(s.lower().strip() for s in df[fighter_col].dropna())

def closest_date_in_metrics(df: pd.DataFrame, target_date: pd.Timestamp, date_col="Date"):
    """
    Each row in df has a 'Date'. We find which row's 'Date' is closest to `target_date`.
    Return that date. If df is empty, return None.
    """
    if df.empty:
        return None
    dt_series = pd.to_datetime(df[date_col], errors='coerce')
    diffs = (dt_series - target_date).abs()
    min_idx = diffs.idxmin()
    return df.loc[min_idx, date_col]

def pick_top_x_for_date(
    df: pd.DataFrame,
    chosen_date: pd.Timestamp,
    date_col="Date",
    value_col="Value",
    fighter_col="Fighter",
    x=10
) -> set:
    """
    From df with columns [Fighter, Date, Value],
    filter rows to chosen_date,
    sort descending by Value,
    pick top X fighters,
    return as a set (lowercased).
    """
    if chosen_date is None:
        return set()

    df_filtered = df[df[date_col] == chosen_date].copy()
    df_sorted = df_filtered.sort_values(value_col, ascending=False)
    top_x = df_sorted.head(x)
    return get_fighter_set_lower(top_x, fighter_col=fighter_col)

# ------------------------------------------------------------------------------
# 5) MAIN ANALYSIS: FOR EACH 2-YEAR MONTHLY WINDOW,
#    EXTRACT THE 3 RANKING SETS, FIND THE CLOSEST DATE IN EACH METRIC,
#    PICK TOP-X, AND COMPUTE JACCARD
# ------------------------------------------------------------------------------
all_results = []

for start_d in window_starts:
    w_end = window_end(start_d)
    if start_d > end_date:
        break

    # Extract sets from the ranking data
    set_p4p, set_r1, set_r1_15 = extract_sets_for_window(df_rank, start_d, w_end)
    ranking_sets = {
        "p4p_any": set_p4p,
        "rank1": set_r1,
        "rank1_15": set_r1_15
    }

    for rset_name, rank_set in ranking_sets.items():
        set_size = len(rank_set)
        if set_size == 0:
            continue  # skip if no fighters in that window

        # Evaluate each metric in stacked_data_2013_2021
        for metric_name, df_metrics_long in stacked_data_2013_2021.items():
            # 1) find the date in df_metrics_long closest to start_d
            chosen_date = closest_date_in_metrics(df_metrics_long, start_d, date_col="Date")
            # 2) pick top <set_size> for that date
            top_x_set = pick_top_x_for_date(
                df=df_metrics_long,
                chosen_date=chosen_date,
                date_col="Date",
                value_col="Value",
                fighter_col="Fighter",
                x=set_size
            )
            # 3) Jaccard
            jc = jaccard_coefficient(rank_set, top_x_set)

            all_results.append({
                "window_start": start_d,
                "window_end": w_end,
                "ranking_set": rset_name,
                "ranking_size": set_size,
                "metric_name": metric_name,
                "closest_date": chosen_date,
                "jaccard": jc
            })

# ------------------------------------------------------------------------------
# 6) INSPECT OR SAVE THE RESULTS
# ------------------------------------------------------------------------------
df_all_results = pd.DataFrame(all_results)
print(df_all_results.head(20))

# Maybe save to CSV
df_all_results.to_csv("jaccard_results_rolling2year_monthly.csv", index=False)
