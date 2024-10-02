import os
import glob
from pathlib import Path
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def read_csv_files(directory, experiment):

    csv_files = glob.glob(f"{directory}*{experiment}*.csv")
    data_frames = []

    for file in csv_files:
        df = pd.read_csv(file)
        data_frames.append(df)
    
    return data_frames


def compute_mean_variance(data_frames, column):
    all_data = pd.concat([df[column] for df in data_frames], axis=1)

    mean_values = all_data.mean(axis=1)
    variance_values = all_data.var(axis=1)
    
    return mean_values, variance_values


def plot(data, metric):
    plt.figure(figsize=(12, 8))

    colors_v = sns.color_palette("colorblind", 10) 

    for i, (mean, variance, exp) in enumerate(data):
        plt.plot(mean, label=exp, color=colors_v[i])
        upper_bound = mean + np.sqrt(variance)
        lower_bound = mean - np.sqrt(variance)
        plt.fill_between(mean.index, lower_bound, upper_bound, color=colors_v[i], alpha=0.2)

    plt.xlabel('Episode time')
    plt.ylabel(metric)
    plt.title(metric)
    plt.legend()
    plt.savefig(f"charts/{metric}.pdf")



charts_dir = 'charts/'
Path(charts_dir).mkdir(parents=True, exist_ok=True)

experiments = ['CTDE', 'NN-averaging', 'NN-consensus', 'experience-sharing']

data_reward = []
data_ep_len = []

for experiment in experiments: 
    dataframes = read_csv_files('data/', experiment)
    mean_values, variance_values = compute_mean_variance(dataframes, column='episode_reward_mean')
    data_reward.append((mean_values, variance_values, experiment))
    mean_values, variance_values = compute_mean_variance(dataframes, column='episode_len_mean')
    data_ep_len.append((mean_values, variance_values, experiment))

plot(data_reward, 'episode_reward_mean')
plot(data_ep_len, 'episode_len_mean')