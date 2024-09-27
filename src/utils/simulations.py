import time
import numpy as np
import random as rnd

def simulate_episode(env, policy, steps, sleep_between_frames=0.3, print_info=False, print_action=False, print_reward=False, print_ob=False,):
    obs, _ = env.reset()
    # env.render()
    last_frame = time.time()
    for i in range(steps):
        if print_ob:
            print(f"obs: ", obs)
        
        actions = {agent: policy.compute_single_action(obs[agent]) for agent in obs.keys()} #policy.compute_actions(obs)
        obs, reward, terminated, _, infos = env.step(actions)
        time.sleep(max(0, sleep_between_frames - (time.time() - last_frame)))
        last_frame = time.time()
        # env.render()
        
        if print_info:
            print(f"info: ", infos)
        if print_action: 
            print(f"action: ", actions)
        if print_reward:
            print(f"reward: ", reward, "\n")

        if terminated["__all__"]:
            break

def simulate_episode_multipolicy(env, algo, steps, seed=None, sleep_between_frames=0.3, print_info=False, print_action=False, print_reward=False, print_ob=False):
    if(seed == None):
        obs, _ = env.reset()
    else:   
        obs, _ = env.reset(seed)
    # env.render()
    last_frame = time.time()
    for i in range(steps):
        if print_ob:
            print(f"obs: ", obs)

        actions = {}
        for agent in obs.keys():
            actions[agent] = algo.compute_single_action(obs[agent], policy_id=agent)
        obs, reward, terminated, _, infos = env.step(actions)
        time.sleep(max(0, sleep_between_frames - (time.time() - last_frame)))
        last_frame = time.time()
        # env.render()
        
        if print_info:
            print(f"info: ", infos)
        if print_action: 
            print(f"action: ", actions)
        if print_reward:
            print(f"reward: ", reward, "\n")

        if terminated["__all__"]:
            break

def simulate_random_episode(env, steps, sleep_between_frames=0.3, print_info=True):
    obs, _ = env.reset()
    # env.render()
    action_space = env.action_space
    last_frame = time.time()
    for i in range(steps):
        if print_info:
            print(f"obs: ", obs)
        actions = {agent: action_space.sample() for agent in obs.keys()}
        obs, reward, _, _, _ = env.step(actions)
        time.sleep(max(0, sleep_between_frames - (time.time() - last_frame)))
        last_frame = time.time()
        # env.render()
        if print_info:
            print(f"action: ", actions)
            print(f"reward: ", reward, "\n")

def ppo_result_format_v2(result):
    return (f"iteration [{result['training_iteration']}] => " +
          f"episode_reward_mean: {result['env_runners']['episode_reward_mean']}, " +
          f"episode_len_mean: {result['env_runners']['episode_len_mean']}, ")

def sac_result_format_v2(result):
    return (f"iteration [{result['training_iteration']}] => " +
          f"episode_reward_mean: {result['env_runners']['episode_reward_mean']}, " +
          f"episode_len_mean: {result['env_runners']['episode_len_mean']}")
          
def dqn_result_format_v2(result):
    return (f"iteration [{result['training_iteration']}] => " +
        f"episode_reward_mean: {result['env_runners']['episode_reward_mean']}, " +
        f"episode_len_mean: {result['env_runners']['episode_len_mean']}")

def ppo_result_format(result):
    return (f"iteration [{result['training_iteration']}] => " +
          f"episode_reward_mean: {result['sampler_results']['episode_reward_mean']}, " +
          f"episode_len_mean: {result['sampler_results']['episode_len_mean']}, ")

def sac_result_format(result):
    return (f"iteration [{result['training_iteration']}] => " +
          f"episode_reward_mean: {result['sampler_results']['episode_reward_mean']}, " +
          f"episode_len_mean: {result['sampler_results']['episode_len_mean']}")
          
def dqn_result_format(result):
    return (f"iteration [{result['training_iteration']}] => " +
        f"episode_reward_mean: {result['sampler_results']['episode_reward_mean']}, " +
        f"episode_len_mean: {result['sampler_results']['episode_len_mean']}")
