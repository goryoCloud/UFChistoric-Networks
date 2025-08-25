# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:27:34 2024

@author: max_s
"""

import pandas as pd


path ='C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/pound-for-pound/rankings_history.csv'
fighterNamesPath = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/dataframes3/fighter_names.csv'

rankings_df = pd.read_csv(path)
fighter_names_df = pd.read_csv(fighterNamesPath)

pound_for_pound_df = rankings_df[rankings_df['weightclass'] == 'Pound-for-Pound']
fighter_names = fighter_names_df['fighter'].tolist()

for fighter in fighter_names:
    # Filter the pound-for-pound dataframe for the current fighter
    fighter_df = pound_for_pound_df[pound_for_pound_df['fighter'] == fighter]
    
    # If the dataframe is not empty, save it as a CSV
    if not fighter_df.empty:
        # Select only the date and rank columns
        fighter_df = fighter_df[['date', 'rank']]
        
        # Define the path to save the CSV file
        output_path = f'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/pound-for-pound/PFP_history/{fighter}_timeseries.csv'
        
        # Save the dataframe to a CSV file
        fighter_df.to_csv(output_path, index=False)

print("Timeseries CSV files have been created for each fighter with pound-for-pound information.")
