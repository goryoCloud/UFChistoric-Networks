# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 16:04:10 2024

@author: max_s
"""

import pandas as pd
import numpy as np

path = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/dataframes'

data = pd.read_csv(f'{path}/eigenEvol.csv')
names = data['fighter']

names.to_csv(f'{path}/fighterNames.csv', index = False)