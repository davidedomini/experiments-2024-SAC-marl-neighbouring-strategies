import json
from matplotlib import pyplot as plt

def save_dict_as_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
        
def load_json_as_dict(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        return data

def plot_metrics(metrics, key_a, key_b=None):
    if key_b == None:
        plt.plot(metrics[key_a], linestyle='-', color='r', label=f"{key_a}")
        plt.legend()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 5))
        ax1.plot(metrics[key_a], linestyle='-', color='r', label=f"{key_a}")
        ax1.legend()
        ax2.plot(metrics[key_b], linestyle='-', color='y', label=f"{key_b}")
        ax2.legend()

def compare_metrics(metric_a, metric_b, key_a, key_b=None):
    if key_b == None:
        plt.plot(metric_a[key_a], linestyle='-', color='r', label=f"{key_a} algo A")
        plt.plot(metric_b[key_a], linestyle='-', color='b', label=f"{key_a} algo B")
        plt.legend([f"{key_a} algo A", f"{key_a} algo B"])
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 5))
        ax1.plot(metric_a[key_a], linestyle='-', color='r', label=f"{key_a} algo A")
        ax1.plot(metric_b[key_a], linestyle='-', color='b', label=f"{key_a} algo B")
        ax1.legend([f"{key_a} algo A", f"{key_a} algo B"])
        ax2.plot(metric_a[key_b], linestyle='-', color='r', label=f"{key_b} algo A")
        ax2.plot(metric_b[key_b], linestyle='-', color='b', label=f"{key_b} algo B")
        ax2.legend([f"{key_b} algo A", f"{key_b} algo B"])

def compare_metrics(metrics, key_a, key_b=None):
    if key_b == None:
        legend = []
        for metric_name, dataset in metrics.items():
            plt.plot(dataset[key_a], linestyle='-', label=f"{key_a} {metric_name}")
            legend.append(f"{key_a} {metric_name}")
        plt.legend(legend)
    else:
        ax1_legend = []
        ax2_legend = []
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 5))
        for metric_name, dataset in metrics.items():
            ax1.plot(dataset[key_a], linestyle='-', label=f"{metric_name}")
            ax2.plot(dataset[key_b], linestyle='-', label=f"{metric_name}")
            ax1_legend.append(f"{metric_name}")
            ax2_legend.append(f"{metric_name}")
        ax1.legend(ax1_legend)
        ax1.title.set_text(key_a)
        ax2.legend(ax2_legend)
        ax2.title.set_text(key_b)
    
    plt.save("charts/independent-learning-vs-CTDE.pdf")