from training.DTDE import DTDE
from pathlib import Path
import time
import os

if __name__ == "__main__":

    results_dir = 'data/'
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    for seed in range(10):
        print(f"Start of training with seed {seed}")
        start = time.time()
        DTDE(seed)
        end = time.time()
        print(f"End of training with seed {seed}")
        print(f"Took {end - start} seconds")
        print("-------------------------------------------------")
    print("Finished")