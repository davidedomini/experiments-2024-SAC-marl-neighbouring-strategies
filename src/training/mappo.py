from ray.rllib.algorithms.callbacks import DefaultCallbacks

from utils.vectors import Vector2D
from utils.canvas import CanvasWithBorders
from utils.algo_utils import (save_algo, load_algo, disable_exploration, enable_exploration, keep_best_policy_only, compute_performance)
from utils.metrics import (save_dict_as_json, load_json_as_dict, plot_metrics, compare_metrics)
from utils.simulations import (simulate_episode, simulate_random_episode, dqn_result_format, simulate_episode_multipolicy)
from utils.dictionary import mean_dict
from environment_configuration import EnvironmentConfiguration
from collect_the_items import RenderableCollectTheItems


from ray.rllib.algorithms.ppo import PPOConfig
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

n_agents = 4

def get_nbrs(agent, n_neighbours):
    agent_id = int(agent.split("-")[1])
    return [f"agent-{(agent_id+i)%n_agents}" for i in range(1, n_neighbours+1)]


def PPO(seed, training_iterations=2):
    
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

    register_env("collect_the_items?algo=PPO&method=ppo", lambda _: RenderableCollectTheItems(env_config))

    policies = {f"agent-{i}": (None, None, None, {}) for i in range(n_agents)}


    algo = (PPOConfig()
        .debugging(seed=seed)
        .env_runners(
            num_env_runners=2,
            num_envs_per_env_runner=4,
        )
        .training(
            train_batch_size_per_learner=1024,
            lr=0.0002 * 1 ** 0.5,
            gamma=0.95,
            lambda_=0.5,
            kl_coeff=0.5,
            clip_param=0.1,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            grad_clip=100.0,
            grad_clip_by="global_norm",
            # num_epochs=8,
        )
        #.multi_agent(
        #    policies=policies,
        #    policy_mapping_fn=(lambda agentId, *args, **kwargs: agentId),
        #)
        .environment("collect_the_items?algo=PPO&method=ppo")
    ).build()

    data = pd.DataFrame(columns=['Iteration','episode_reward_mean', 'episode_len_mean'])
    

    for i in range(training_iterations):
        result = algo.train()
        print(f"Iteration {i}: {result['sampler_results']['episode_reward_mean']}")
        
        data = pd.concat([data, pd.DataFrame([
            {'Iteration': i,
            'episode_reward_mean': result['sampler_results']['episode_reward_mean'], 
            'episode_len_mean': result['sampler_results']['episode_len_mean']
            }])])

    data.to_csv(f'data/results-ppo_{seed}.csv', index=False)