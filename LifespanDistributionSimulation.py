# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:30:32 2024

@author: max_s
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/lifespan.txt')

data.hist(column = 'nFights', bins = 11)
plt.yscale('log')
