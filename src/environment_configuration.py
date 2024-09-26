class EnvironmentConfiguration: 
    def __init__(self, n_agents, n_items, agent_range, movement_sensitivity, speed_sensitivity, spawn_area=100, visible_nbrs=1, visible_items=1, max_steps=None, memory_size=1):
        # parameters that shouldn't affect the agents' behaviour
        self.n_agents = n_agents
        self.n_items = n_items
        self.spawn_area = spawn_area
        self.max_steps = max_steps
        # parameters that affect the agents' behavious
        self.agent_range = agent_range
        # parameters that affect the observation space
        self.visible_nbrs = visible_nbrs
        self.visible_items = visible_items
        self.memory_size = memory_size
        # parameters that affect the action space
        self.movement_sensitivity = movement_sensitivity
        self.speed_sensitivity = speed_sensitivity

    def __deepcopy__(self, memo):
        return EnvironmentConfiguration(
            self.n_agents, 
            self.n_items, 
            self.agent_range, 
            self.movement_sensitivity, 
            self.speed_sensitivity, 
            self.spawn_area, 
            self.visible_nbrs, 
            self.visible_items, 
            self.max_steps, 
            self.memory_size
        )