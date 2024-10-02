from ray.rllib.algorithms.callbacks import DefaultCallbacks

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
import pandas as pd
import os
import copy

n_fixed_nbrs = 1
n_agents = 4

def get_nbrs(agent, n_neighbours):
    agent_id = int(agent.split("-")[1])
    return [f"agent-{(agent_id+i)%n_agents}" for i in range(1, n_neighbours+1)]
    
class SyncPolicyCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        local_worker = algorithm.workers.local_worker()
        policies = local_worker.policy_map 

        mean_episode_reward = result["sampler_results"]["policy_reward_mean"]

        new_policies = {}
        for agent in policies.keys():
            best_policy_in_nn = agent
            for nbr in get_nbrs(agent, n_fixed_nbrs):
                if mean_episode_reward[nbr] < mean_episode_reward[best_policy_in_nn]:
                    best_policy_in_nn = agent
            new_policies[agent] = copy.deepcopy(
                policies[best_policy_in_nn].get_weights())        
            
        for agent in policies.keys():
            policies[agent].set_weights(new_policies[agent])

        algorithm.workers.sync_weights()


def DTDE_nn_consensus(seed, training_iterations=2):
    
    env_config = EnvironmentConfiguration(
        n_agents = n_agents,
        n_items = 10,
        spawn_area = 100,
        max_steps=300,
        agent_range = 5,
        visible_nbrs = 3,
        visible_items = 3,
        memory_size=3,
        movement_sensitivity=5,
        speed_sensitivity=5
    )

    register_env("collect_the_items?algo=DQN&method=DTDE", lambda _: RenderableCollectTheItems(env_config))

    policies = {f"agent-{i}": (None, None, None, {}) for i in range(n_agents)}


    algo = (DQNConfig()
        .training(
            gamma=0.95,
            lr=0.001,
            train_batch_size=32,
            n_step=1,
            # item_network_update_freq=500,
            double_q=True,
            dueling=True,
            replay_buffer_config={"type": "MultiAgentPrioritizedReplayBuffer"})
        .debugging(seed=seed)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=(lambda agentId, *args, **kwargs: agentId),
        )
        .callbacks(SyncPolicyCallback)
        .environment("collect_the_items?algo=DQN&method=DTDE")
    ).build()

    data = pd.DataFrame(columns=['Iteration','episode_reward_mean', 'episode_len_mean'])
    

    for i in range(training_iterations):
        result = algo.train()
        data = pd.concat([data, pd.DataFrame([
            {'Iteration': i,
            'episode_reward_mean': result['sampler_results']['episode_reward_mean'], 
            'episode_len_mean': result['sampler_results']['episode_len_mean']
            }])])

    data.to_csv(f'data/results-DTDE-NN-consensus-seed_{seed}.csv', index=False)