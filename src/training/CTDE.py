from utils.vectors import Vector2D
from utils.canvas import CanvasWithBorders
from utils.algo_utils import (save_algo, load_algo)
from utils.metrics import (save_dict_as_json, load_json_as_dict, plot_metrics, compare_metrics)
from utils.simulations import (simulate_episode, simulate_random_episode, dqn_result_format)

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiDiscrete
from gymnasium.spaces.utils import flatten, flatten_space
from typing import Set
import random as rnd
import numpy as np
import math
import ray

def CTDE(seed, training_iterations=50):
    env_config = EnvironmentConfiguration(
    n_agents = 4,
    n_items = 10,
    spawn_area = 100,
    max_steps=300,
    agent_range = 5,
    visible_nbrs = 3,
    visible_items = 3,
    memory_size=3,
    movement_sensitivity=5,
    speed_sensitivity=5)

    env = RenderableCollectTheItems(env_config)

    register_env("collect_the_items?algo=DQN&method=CTDE", lambda _: RenderableCollectTheItems(env_config))

    algo = (DQNConfig()
        .training(
            gamma=0.95,
            lr=0.001,
            train_batch_size=32,
            n_step=1,
            item_network_update_freq=500,
            double_q=True,
            dueling=True)
        #.env_runners(num_env_runners=1)
        #.resources(num_gpus=0)  
        .debugging(seed=seed)
        .environment("collect_the_items?algo=DQN&method=CTDE")
    ).build()


    data = pd.DataFrame(columns=['Iteration','episode_reward_mean', 'episode_len_mean'])

    out = ""
    for i in range(training_iterations):
        result = algo.train()
        data = pd.concat([data, pd.DataFrame([
                {'Iteration': i,
                'episode_reward_mean': result['env_runners']['episode_reward_mean'], 
                'episode_len_mean': result['env_runners']['episode_len_mean']
                }])])

    data.to_csv(f'data/results-CTDE-seed_{seed}.csv', index=False)