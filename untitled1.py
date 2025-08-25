import matplotlib.pyplot as plt
import numpy as np

# Example data
data = np.random.normal(0, 1, 1000)

# Create the histogram
plt.figure(figsize=(8, 6))
plt.hist(data, bins=20, density=True, edgecolor='black', alpha=0.7)

# Add labels and title
plt.title('Normalized Histogram', fontsize=16)
plt.xlabel('Value', fontsize=14)
plt.ylabel('Frequency (normalized)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

# Display the plot
plt.show()