import json
import numpy as np
import matplotlib.pyplot as plt

# Configure matplotlib for LaTeX-style text
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "mathptmx",
})

plt.rc('axes', labelsize=25)
plt.rc('axes', titlesize=25)
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)

path = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/UFCDATA/dataframes6Networks'

# Load JSON data
with open(f'{path}/UFCdata.json', 'r') as file:
    data = json.load(file)

# Function to map metric names to descriptive labels
def get_label(metric_name):
    if "degree" in metric_name:
        base = "Degree"
    elif "clust" in metric_name:
        base = "Clustering"
    elif "bet" in metric_name:
        base = "Betweenness"
    elif "eigen" in metric_name:
        base = "Eigenvector"
    else:
        base = metric_name

    if "_un" in metric_name:
        return "undirected"
    elif "_w" in metric_name and "_dir" not in metric_name:
        return "und. winners"
    elif "_l" in metric_name and "_dir" not in metric_name:
        return "und. losers"
    elif "_dir" in metric_name and "_w" in metric_name:
        return "dir. winners"
    elif "_dir" in metric_name and "_l" in metric_name:
        return "dir. losers"
    elif "_dir" in metric_name:
        return "directed"
    else:
        return metric_name

# Function to plot histograms with aligned bins
def plot_histogram(ax, metrics, xlabel, ylabel, aligned_bins, xticks_values, colors):
    ax.set_xlim(-1.5, 1.5)
    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xticks(xticks_values)

    for metric_name, color in zip(metrics, colors):
        metric_list = data.get(metric_name, [])
        filtered_metric_list = [x for x in metric_list if not np.isnan(x)]
        filtered_metric_list = np.array(filtered_metric_list)
        linestyle = 'dashed' if "_dir_l" in metric_name or "_l" in metric_name else 'solid'  # Dashed for directed losers
        ax.hist(
            filtered_metric_list,
            bins=aligned_bins,
            label=get_label(metric_name),
            histtype='step',
            linestyle=linestyle,
            linewidth=2.5,
            alpha=1,
            stacked=True,
            color=color,
            weights=np.ones_like(filtered_metric_list) / len(filtered_metric_list)
        )
    ax.legend(fontsize=20)

# Define metrics and bins for each plot

deg_metrics = ["degree_un", "degree_w", "degree_l", "degree_dir", "degree_dir_w", "degree_dir_l"]
aligned_bins_deg = np.linspace(-1, 1, 17)

deg_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']

clust_metrics = ["clust_un", "clust_w", "clust_l", "clust_dir", "clust_dir_w", "clust_dir_l"]
aligned_bins_clust = np.linspace(-1, 1, 13)

clust_colors = ['magenta', 'brown', 'olive', 'pink', 'teal', 'gray']

bet_metrics = ["bet_un", "bet_w", "bet_l", "bet_dir", "bet_dir_w", "bet_dir_l"]
aligned_bins_bet = np.linspace(-1, 1, 17)

bet_colors = ['gold', 'indigo', 'lime', 'navy', 'coral', 'darkred']

eigen_metrics = ["eigen_un", "eigen_w", "eigen_l", "eigen_dir", "eigen_dir_w", "eigen_dir_l"]
aligned_bins_eigen = np.linspace(-1, 1, 17)

eigen_colors = ['black', 'violet', 'deepskyblue', 'chocolate', 'chartreuse', 'crimson']

# Create 2x2 layout for plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot each metric in a subplot
plot_histogram(axes[0, 0], deg_metrics, 'Pearson correlation coefficient value', 'frequency', aligned_bins_deg, np.linspace(-1, 1, 5), bet_colors)
#axes[0, 0].set_title("Histogram of Degrees")

plot_histogram(axes[0, 1], clust_metrics, 'Pearson correlation coefficient value', 'frequency', aligned_bins_clust, np.linspace(-1, 1, 5), bet_colors)
#axes[0, 1].set_title("Histogram of Clustering Metrics")

plot_histogram(axes[1, 0], bet_metrics, 'Pearson correlation coefficient value', 'frequency', aligned_bins_bet, np.linspace(-1, 1, 5), bet_colors)
#axes[1, 0].set_title("Histogram of Betweenness Metrics")

plot_histogram(axes[1, 1], eigen_metrics, 'Pearson correlation coefficient value', 'frequency', aligned_bins_eigen, np.linspace(-1, 1, 5), bet_colors)
#axes[1, 1].set_title("Histogram of Eigenvector Metrics")


plt.tight_layout()
plt.show()
