# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 16:19:33 2024

@author: max_s
"""

from pytrends.request import TrendReq
import pandas as pd
import time

savingPath = '/home/msilva/UFC/fightersTrends/'
namesPath = '/home/msilva/UFC/fightersTrends/fighterNames.csv'

fightersNames = pd.read_csv(namesPath)
pytrends = TrendReq(hl='en-US', tz=360)

def get_trends_data(event):
    for attempt in range(5):
        try:
            pytrends.build_payload([event], cat=0, timeframe='all', geo='', gprop='')
            data = pytrends.interest_over_time()
            if not data.empty:
                data = data.drop(labels=['isPartial'], axis='columns')
            return data
        except Exception as e:
            if "429" in str(e):
                print(f"Too many requests for event '{event}'. Retrying in 60 seconds...")
                time.sleep(60)
            else:
                print(f"An error occurred: {e}")
                break
    return pd.DataFrame()

for fighter in fightersNames['fighter']:
    print(fighter)
    trends_data = get_trends_data(fighter)
    trends_data.to_csv(f'{savingPath}/{fighter}.csv')
    time.sleep(60)

