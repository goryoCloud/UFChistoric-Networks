# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:05:17 2023

@author: max_s
"""
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math

#%%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "mathptmx",})

plt.rc('axes', labelsize='xx-large')
plt.rc('axes', titlesize='xx-large')
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
#%%
data = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/data.csv')
googleData = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/multiTimeline (1).csv')
PPVdata = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/ufc_ppv_buys.csv')

googleData['Month'] = pd.to_datetime(googleData['Month'])

PPVdata['date'] = pd.to_datetime(PPVdata[['Year', 'Month', 'Day']])
PPVdata['month'] = PPVdata['date'].dt.to_period('M')
monthly_avg_ppv = PPVdata.groupby('month')['PPV'].mean().reset_index()

# Convert 'month' from Period to datetime for plotting
monthly_avg_ppv['month'] = monthly_avg_ppv['month'].dt.to_timestamp()
#%%
fig, ax1 = plt.subplots(figsize = (6,4))
ax1.plot(monthly_avg_ppv['month'], monthly_avg_ppv['PPV'], 's--r', markersize = 4, lw = 1)
#plt.plot(googleData['Month'], googleData['ufc']), alpha =
ax1.set_xlabel('date')
ax1.set_ylabel('PPV sales', color = 'r')
ax1.set_ylim(0)
ax1.tick_params('y', colors='r')
ax1.set_xlim(monthly_avg_ppv['month'][0], googleData['Month'].iloc[-1])
#plt.title('Average PPV Buys per Month')
#plt.show()
ax2 = ax1.twinx()
ax2.plot(googleData['Month'], googleData['ufc'], 'x--b', lw = 1)
ax2.set_ylabel(r'Google interest for \textit{UFC} ', color = 'b')
ax2.tick_params('y', colors='b')
plt.tight_layout()
#plt.savefig('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/paperImages/googleVSppv.eps', dpi = 5000)