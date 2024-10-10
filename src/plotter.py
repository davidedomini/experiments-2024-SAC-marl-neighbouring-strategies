import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utils.metrics import load_json_as_dict

def read_csv_files(directory, experiment):
    csv_files = glob.glob(f"{directory}*{experiment}-seed*.csv")
    print(f"For {experiment} opened {len(csv_files)} files")
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


def beautify_metric(metric):
    if metric == 'episode_reward_mean':
        return 'Mean Episode Reward'
    elif metric == 'episode_len_mean':
        return 'Mean Episode Length'
    else:
        return 'Unknown metric'


def beautify_label(label):
    if label == 'CTDE':
        return 'CTDE' 
    elif label == 'DTDE':
        return 'DTDE'
    elif label == 'NN-averaging':
        return 'NN Averaging'
    elif label == 'NN-consensus':
        return 'NN Consensus'
    elif label == 'experience-sharing':
        return 'Experience Sharing'
    else:
        return 'Unknown label'


def plot(data, metric):
    plt.figure(figsize=(10, 6))
    viridis = plt.colormaps['viridis']
    indexes = np.linspace(0.1, 0.9, len(data))
    for i, (mean, variance, exp) in enumerate(data):
        plt.plot(mean, label=beautify_label(exp), color=viridis(indexes[i]))
        upper_bound = mean + np.sqrt(variance)
        lower_bound = mean - np.sqrt(variance)
        plt.fill_between(mean.index, lower_bound, upper_bound, color=viridis(indexes[i]), alpha=0.2)
    plt.xlabel('Episode')
    m = beautify_metric(metric)
    plt.ylabel(m)
    # plt.title(m)
    plt.legend()
    plt.savefig(f"charts/{metric}.pdf")
    plt.close()


def plot_mean_time_comparison_methods():
    
    for n_agents in [4, 8, 16, 32]:
        # plt.figure(figsize=(40, 30))
        data_ctde = load_json_as_dict(f"data/eval/collect_the_items-algo=DQN&method=CTDE_v3_2908--agents={n_agents}&items=30&spawn_area=500")
        data_nn_averaging = load_json_as_dict(f"data/eval/collect_the_items-algo=DQN&method=DTDE[02]_v2_2908--agents={n_agents}&items=30&spawn_area=500")
        data_nn_consensus = load_json_as_dict(f"data/eval/collect_the_items-algo=DQN&method=DTDE[03]_v2_2908--agents={n_agents}&items=30&spawn_area=500")
        data_experience_sharing = load_json_as_dict(f"data/eval/collect_the_items-algo=DQN&method=DTDE[05]_v2_2908--agents={n_agents}&items=30&spawn_area=500")
        data_dtde = load_json_as_dict(f"data/eval/collect_the_items-algo=DQN&method=DTDE[01]_v0--agents={n_agents}&items=30&spawn_area=500")

        data = [(data_ctde, 'CTDE'), (data_dtde, 'DTDE'), (data_nn_averaging, 'NN Averaging'), (data_nn_consensus, 'NN Consensus'), (data_experience_sharing, 'Experience Sharing')]

        fig, ax = plt.subplots()
        viridis = plt.colormaps['viridis']
        indexes = np.linspace(0.1, 0.9, len(data))

        for i, (data, experiment) in enumerate(data):
            ax.boxplot(data,
            positions=[i],
            notch=True,
            label=experiment,
            widths=0.5,
            patch_artist=True, boxprops={"facecolor": viridis(indexes[i])},)
        plt.legend(loc='upper right')
        plt.ylabel('Mean Episode Length')
        plt.title(f'{n_agents} Agents')
        plt.tight_layout()
        plt.savefig(f"charts/mean-time-comparison-{n_agents}-agents.pdf")
        plt.close()


def plot_communication_overhead():
    nbrs = np.array([i for i in range(1, 11)])
    nn_weight = 2.80 # MB
    trajectory_weight = 0.0891 # 297 (one trajectory step weight [byte]) * 300 (max episode length) = 89100 byte
    
    viridis = plt.colormaps['viridis']
    indexes = np.linspace(0.1, 0.9, 3)

    nn_consensus = np.full_like(nbrs, nn_weight)
    nn_averaging = nbrs * nn_weight + nbrs * 6.4e-5
    experience_sharing = nbrs * trajectory_weight

    plt.figure(figsize=(10, 6))

    plt.plot(nbrs, nn_consensus, label='NN Consensus', color=viridis(indexes[0]))
    plt.plot(nbrs, nn_averaging, label='NN veraging', color=viridis(indexes[1]))
    plt.plot(nbrs, experience_sharing, label='Experience Sharing', color=viridis(indexes[2]))

    # Add labels and title
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Weight (MB)')
    plt.title('Communication overhead')
    plt.legend()
    plt.savefig(f"charts/scalability.pdf")
    plt.close()


matplotlib.rcParams.update({'axes.titlesize': 20})
matplotlib.rcParams.update({'axes.labelsize': 18})
matplotlib.rcParams.update({'xtick.labelsize': 15})
matplotlib.rcParams.update({'ytick.labelsize': 15})

charts_dir = 'charts/'
Path(charts_dir).mkdir(parents=True, exist_ok=True)

# experiments = ['CTDE', 'DTDE', 'NN-averaging', 'NN-consensus', 'experience-sharing']
experiments = ['CTDE', 'NN-averaging']
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
plot_mean_time_comparison_methods()
plot_mean_time_comparison_n_agents()

plot_communication_overhead()