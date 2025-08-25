# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 00:15:42 2023

@author: max_s
"""

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math

data = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/data.csv')
#%%
fighters = data.R_fighter.unique()
