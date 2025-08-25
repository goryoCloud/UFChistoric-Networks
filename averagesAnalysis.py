import ast
import matplotlib.pyplot as plt

# Define the path to the file
file_path = 'average_and_std_degreeDist.dat'

# Read the content of the file
with open(file_path, 'r') as file:
    content = file.readlines()

# Extracting the means and standard deviations from the content
means_str = content[1]
std_devs_str = content[3]

# Converting the string representations of lists into actual Python lists
means = ast.literal_eval(means_str.strip())
std_devs = ast.literal_eval(std_devs_str.strip())

# Printing the arrays to verify
print("Means:", means)
print("Standard Deviations:", std_devs)

plt.plot(means)