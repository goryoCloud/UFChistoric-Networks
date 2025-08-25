import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def count_fights_per_year(file_path):
    data = pd.read_csv(file_path)
    male_weights = ['Bantamweight', 'Middleweight', 'Heavyweight', 'Lightweight', 'Welterweight', 
                    'Flyweight', 'Light Heavyweight', 'Featherweight', 'Catch Weight', 'Open Weight']
    filtered_data = data[data['weight_class'].isin(male_weights)]
    
    # Convert 'date' to datetime and set as index
    filtered_data['date'] = pd.to_datetime(filtered_data['date'])
    filtered_data.set_index('date', inplace=True)
    
    # Group by year and count the number of fights, then calculate the mean of these counts per year
    yearly_fights = filtered_data.resample('Y').size()
    return yearly_fights

file_path = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/data.csv'  # Change this to your file's path
yearly_fights_data = count_fights_per_year(file_path)

average_fights_per_year = yearly_fights_data.mean()
print("Average number of fights per year:", average_fights_per_year)

plt.figure(figsize=(10, 5))
yearly_fights_data.plot()
plt.title('Total Fights Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Fights')
plt.grid(True)
plt.show()