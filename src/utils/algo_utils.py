import os
from ray.rllib.algorithms.algorithm import Algorithm
import copy
    
def save_algo(algo, name):
    base_dir = os.path.join(os.getcwd(), "algos")
    subfolder_path = os.path.join(base_dir, name)
    os.makedirs(subfolder_path, exist_ok=True)
    path_to_checkpoint  = algo.save(subfolder_path)
    print(f"An Algorithm checkpoint has been created inside directory: '{path_to_checkpoint}'.")

def load_algo(name):
    base_dir = os.path.join(os.getcwd(), "algos")
    subfolder_path = os.path.join(base_dir, name)
    if not os.path.exists(subfolder_path):
        raise FileNotFoundError(f"The specified subfolder '{subfolder_path}' does not exist.")
    
    return Algorithm.from_checkpoint(subfolder_path)

def disable_exploration(algo):
    policies = algo.workers.local_worker().policy_map
    for policy_id in policies.keys():
        policies[policy_id].config["explore"] = False

def enable_exploration(algo):
    policies = algo.workers.local_worker().policy_map
    for policy_id in policies.keys():
        policies[policy_id].config["explore"] = True

def compute_performance(env, algo, policy, n_episodes):
    tot = 0.0
    for seed in range(n_episodes):
        i = 0
        terminated = {"__all__": False}
        while i < env.max_steps and not terminated['__all__']:
            obs, _ = env.reset(seed=seed)
            actions = algo.compute_actions(obs, policy_id=policy)
            obs, rew, terminated, _, _ = env.step(actions)
            tot += sum(rew.values())
            i = i+1
        tot += i
    return tot/n_episodes

def keep_best_policy_only(algo, env):
    policies = algo.workers.local_worker().policy_map
    if len(policies) == 1:
        return

    print("choosing the best policy...")
    disable_exploration(algo)
    performances = {}
    for policy_id in policies.keys():
        performances[policy_id] = compute_performance(env, algo, policy_id, 5)
        print(f"{policy_id}'s mean reward -> {performances[policy_id]}") 

    best_policy = 'agent-0'
    for policy_id, performance in performances.items():
        if performance > performances[best_policy]:
            best_policy = policy_id

    print("creating the new policy...")
    algo.add_policy(
        policy_id = "default_policy",
        policy_cls = type(policies['agent-0']),
    )

    policies["default_policy"].set_weights(copy.deepcopy(policies[best_policy].get_weights()))
    policies["default_policy"].config["explore"] = False

    for policy_id in policies.keys():
        if policy_id != "default_policy":
            del policies[policy_id]
    algo.workers.sync_weights()

    print("done")