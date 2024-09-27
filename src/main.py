from utils.vectors import Vector2D
from utils.canvas import CanvasWithBorders
from utils.algo_utils import (save_algo, load_algo, disable_exploration, enable_exploration, keep_best_policy_only, compute_performance)
from utils.metrics import (save_dict_as_json, load_json_as_dict, plot_metrics, compare_metrics)
from utils.simulations import (simulate_episode, simulate_random_episode, dqn_result_format, simulate_episode_multipolicy)
from utils.dictionary import mean_dict
from environment_configuration import EnvironmentConfiguration
from collect_the_items import RenderableCollectTheItems


from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiDiscrete
from gymnasium.spaces.utils import flatten, flatten_space
from ray.tune.registry import register_env
from IPython.display import clear_output
from ipycanvas import Canvas, hold_canvas
from typing import Set
import random as rnd
import numpy as np
import math
import ray

from training.DTDE import DTDE

if __name__ == "__main__":

    DTDE()
    print("Finished")