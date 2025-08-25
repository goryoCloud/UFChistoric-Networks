import ast
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "mathptmx",
})

plt.rc('axes', labelsize='x-large')
plt.rc('axes', titlesize='x-large')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

# Define the path to the file
file_path = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/averages_and_std_devsV8.dat'

# Define a mapping from file types to y-axis labels
ylabel_dict = {
    'degreeDist': r'$\langle k \rangle$',
    'pathLenght': r'$\langle l \rangle$',
    'ginni': r'$G[p(k)]$',
    'eigenvector': r'$\lambda$',
    'density': r'$\langle g \rangle$',
    'clustering': r'$\langle C\rangle$',
    'betweeness': 'b'
}

# Read the content of the file
with open(file_path, 'r') as file:
    content = file.read()

# Initialize dictionaries to hold the means and standard deviations for each type
means_dict = {}
std_devs_dict = {}

# Parse the file content to extract means and standard deviations
lines = content.split('\n')
for i in range(0, len(lines), 5):
    if i + 3 < len(lines):
        file_type = lines[i].split()[0]
        means_str = lines[i+1]
        std_devs_str = lines[i+3]
        
        means = ast.literal_eval(means_str.strip())
        std_devs = ast.literal_eval(std_devs_str.strip())
        
        means_dict[file_type] = means
        std_devs_dict[file_type] = std_devs

# Plotting the results
for file_type, means in means_dict.items():
    std_devs = std_devs_dict[file_type]
#    plt.figure(figsize=(10, 5))
    
    # Plot the data line with alpha = 1
    plt.plot(range(len(means)), means, label=f'{file_type}', color = 'b', lw = 3.5, alpha=1)
    
    # Plot the error bars with alpha = 0.3
    plt.errorbar(range(len(means)), means, yerr=std_devs, alpha=0.5, ecolor='b', capsize=3)
    
    plt.title(f'{file_type} means')
    plt.xlabel('window index')
    plt.ylabel(ylabel_dict.get(file_type, 'Value'))
    plt.xlim(0, len(means))
    plt.show()