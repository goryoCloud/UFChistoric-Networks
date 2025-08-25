import os
import numpy as np

def read_degree_dist(file_path):
    with open(file_path, 'r') as file:
        data = file.read().strip()
        values = [float(x) for x in data.strip('[]').split(',')]
        return values

def main():
    base_path = "/home/msilva/UFC/simulation/"
    all_degree_dists = []

    # Find all degreeDist.dat files in numerically named folders
    for folder in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path) and folder.isdigit():
            file_path = os.path.join(folder_path, 'degreeDist.dat')
            if os.path.exists(file_path):
                degree_dist = read_degree_dist(file_path)
                all_degree_dists.append(degree_dist)

    if not all_degree_dists:
        print("No degreeDist.dat files found.")
        return

    # Convert the list of lists to a numpy array for easier averaging
    all_degree_dists = np.array(all_degree_dists)
    
    # Compute the average for each position
    averages = np.mean(all_degree_dists, axis=0)

    # Save the averages to a new file
    output_file_path = os.path.join(base_path, 'average_degreeDist.dat')
    with open(output_file_path, 'w') as output_file:
        output_file.write(str(list(averages)))

    print(f"Averages saved to {output_file_path}")