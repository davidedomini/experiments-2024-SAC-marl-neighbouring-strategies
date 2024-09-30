import os
from pathlib import Path


from training.DTDE import DTDE

if __name__ == "__main__":

    results_dir = 'data/'
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    for seed in range(10):
        DTDE(seed)
    print("Finished")