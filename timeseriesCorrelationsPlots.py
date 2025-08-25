# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 22:25:04 2024

@author: max_s
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path_degree = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/pound-for-pound/PFP_correlations/degEvolCorrelation.csv'
df = pd.read_csv(file_path_degree)

# Display the first few rows of the dataframe to understand its structure
print(df.head())

# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(df['pearson'], bins=7, edgecolor='k', alpha=0.7)
plt.title('Pearson Correlation Coefficients (Pound-for-pound vs. Degree)')
plt.xlabel('Pearson Correlation')
plt.ylabel('Frequency')
plt.show()
#%%
file_path_clust = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/pound-for-pound/PFP_correlations/clustEvolCorrelation.csv'
df = pd.read_csv(file_path_clust)

# Display the first few rows of the dataframe to understand its structure
print(df.head())

# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(df['pearson'], bins=7, edgecolor='k', alpha=0.7)
plt.title('Pearson Correlation Coefficients (Pound-for-pound vs. Clustering coefficient)')
plt.xlabel('Pearson Correlation')
plt.ylabel('Frequency')
plt.show()
#%%
file_path_bet = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/pound-for-pound/PFP_correlations/betEvolCorrelation.csv'
df = pd.read_csv(file_path_bet)

# Display the first few rows of the dataframe to understand its structure
print(df.head())

# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(df['pearson'], bins=5, edgecolor='k', alpha=0.7)
plt.title('Pearson Correlation Coefficients (Pound-for-pound vs. Betweenness centrality)')
plt.xlabel('Pearson Correlation')
plt.ylabel('Frequency')
plt.show()
#%%
file_path_bet = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/pound-for-pound/PFP_correlations/eigenEvolCorrelation.csv'
df = pd.read_csv(file_path_bet)

# Display the first few rows of the dataframe to understand its structure
print(df.head())

# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(df['pearson'], bins=5, edgecolor='k', alpha=0.7)
plt.title('Pearson Correlation Coefficients (Pound-for-pound vs. Eigenvector centrality)')
plt.xlabel('Pearson Correlation')
plt.ylabel('Frequency')
plt.show()