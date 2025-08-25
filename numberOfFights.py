import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def count_fights_per_night(file_path):
    # Load the dataset from the specified file path
    data = pd.read_csv(file_path)
    
    # Define the list of male weight categories to include
    male_weights = ['Bantamweight', 'Middleweight', 'Heavyweight', 'Lightweight', 'Welterweight', 
                    'Flyweight', 'Light Heavyweight', 'Featherweight', 'Catch Weight', 'Open Weight']
    
    # Filter the dataset to include only the specified weight classes
    filtered_data = data[data['weight_class'].isin(male_weights)]
    
    # Group by 'date' and count the number of fights per night across all specified weight classes
    total_fights_per_night = filtered_data.groupby('date').size().reset_index(name='total_count')
    
    # Return the DataFrame containing the count of fights per night
    return total_fights_per_night

# Specify the path to your CSV file
file_path = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/data.csv'  # Change this to the path of your CSV file

# Call the function and print the results
result = count_fights_per_night(file_path)
#%%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "mathptmx",})

plt.rc('axes', labelsize='x-large')
plt.rc('axes', titlesize='x-large')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

plt.plot(result['total_count'], color = 'k', linestyle='-', lw = 0.5)
plt.ylabel(r'$N_{fights}$')
plt.xlabel('event')
plt.show()

print(np.mean(result['total_count']))
#%%
result['date'] = pd.to_datetime(result['date'])
result.set_index('date', inplace=True)

# Calculate the moving average
window_size = 10  # You can adjust the window size to your preference
result['smoothed_count'] = result['total_count'].rolling(window=window_size, center=True).mean()
plt.plot(result['smoothed_count'], color = 'k', linestyle='-', lw = 2)
plt.ylabel(r'$N_{fights}$')
plt.xlabel('date')
plt.show()
#%%
