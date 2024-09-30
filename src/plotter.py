import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_csv_files(directory):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    data_frames = []

    for file in csv_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        data_frames.append(df)
    
    return data_frames


def compute_mean_variance(data_frames, column):
    all_data = pd.concat([df[column] for df in data_frames], axis=1)

    mean_values = all_data.mean(axis=1)
    variance_values = all_data.var(axis=1)
    
    return mean_values, variance_values

def plot_mean_variance(mean_values, variance_values, metric):
    plt.figure(figsize=(10, 6))

    cmap = plt.get_cmap('viridis')
    color = cmap(0.5)


    plt.plot(mean_values, label=metric, color=color)

    upper_bound = mean_values + np.sqrt(variance_values)
    lower_bound = mean_values - np.sqrt(variance_values)
    plt.fill_between(mean_values.index, lower_bound, upper_bound, color=color, alpha=0.2)

    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel(metric)
    plt.title(metric)

    # Add legend
    # plt.legend()

    # Show plot
    plt.savefig(f"charts/{metric}.pdf")



charts_dir = 'charts/'
Path(charts_dir).mkdir(parents=True, exist_ok=True)
data_frames = read_csv_files('data/')
    
# Compute mean and variance for column 'A'
mean_values, variance_values = compute_mean_variance(data_frames, column='episode_reward_mean')

# Plot the mean and variance
plot_mean_variance(mean_values, variance_values, 'episode_reward_mean')