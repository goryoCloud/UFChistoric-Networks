import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/jaccard_results_rolling2year_monthly.csv')

# Ensure the data types are consistent for plotting
data['closest_date'] = pd.to_datetime(data['closest_date'])

# Get unique ranking sets
ranking_sets = data['ranking_set'].unique()

# Loop through each ranking set and create a plot
for ranking_set in ranking_sets:
    # Filter data for the current ranking set
    subset = data[data['ranking_set'] == ranking_set]

    # Initialize the plot
    plt.figure(figsize=(12, 8))

    # Plot each metric_name as a separate line
    for metric_name in subset['metric_name'].unique():
        metric_data = subset[subset['metric_name'] == metric_name]
        plt.plot(metric_data['closest_date'], metric_data['jaccard'], label=metric_name)

    # Customize the plot
    plt.title(f'Metric Evolution for Ranking Set: {ranking_set}', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Jaccard Metric', fontsize=14)
    plt.legend(title='Metric Name', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the plot as a file
#    plt.savefig(f'{ranking_set}_metrics_evolution.png')

print("Plots created and saved as PNG files.")