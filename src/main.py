from training.nn_averaging import DTDE_nn_averaging
from training.nn_consensus import DTDE_nn_consensus
from training.nn_weigh_averaging import DTDE_nn_weigh_averaging
from training.experience_sharing import DTDE_experience_sharing
from training.mappo import PPO

from training.DTDE import DTDE
from pathlib import Path
import time
import os

if __name__ == "__main__":
    iteration=50
    results_dir = 'data/'
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    for seed in range(1, 10):
        print(f"Start of training with seed {seed}")
        start = time.time()
        PPO(seed, iteration)
        # DTDE(seed)
        #DTDE_nn_averaging(seed, iteration)
        #DTDE_nn_consensus(seed, iteration)
        #DTDE_nn_weigh_averaging(seed, iteration)
        #DTDE_experience_sharing(seed)
        end = time.time()
        print(f"End of training with seed {seed}")
        print(f"Took {end - start} seconds")
        print("-------------------------------------------------")
    print("Finished")