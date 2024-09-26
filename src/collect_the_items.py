class CollectTheItems(MultiAgentEnv):
    canvas = None
    CANVAS_WIDTH, CANVAS_HEIGHT = 300.0, 300.0

    def __init__(self, config: EnvironmentConfiguration):
        assert config.n_agents > config.visible_nbrs
        assert config.movement_sensitivity % 2 == 1

        self.n_agents = config.n_agents
        self.n_items = config.n_items
        self.spawn_area = config.spawn_area
        self.max_steps = config.max_steps
        self.agent_range = config.agent_range
        self.visible_nbrs = config.visible_nbrs
        self.visible_items = config.visible_items
        self.memory_size = config.memory_size
        self.movement_sensitivity = config.movement_sensitivity
        self.speed_sensitivity = config.speed_sensitivity
        self.step_penalty = 1

        self.agents_ids = ['agent-' + str(i) for i in range(self.n_agents)]
        self.observation_space = self.observation_space('agent-0')
        self.action_space = self.action_space('agent-0')

    def unflatten_observation_space(self, agent):
        direction = Box(low=-1, high=1, shape=(2,1), dtype=np.float32)
        distance = Box(low=-np.inf, high=np.inf, shape=(1,1), dtype=np.float32)

        nbrs = Dict({f"nbr-{i}": Dict({'direction': direction, 'distance': distance}) for i in range(self.visible_nbrs)})
        items = Dict({f"item-{i}": Dict({'direction': direction, 'distance': distance}) for i in range(self.visible_items)})

        time_t_obs = Dict({"nbrs": nbrs, "items": items})

        return Dict({f"t[-{t}]": time_t_obs for t in range(0, self.memory_size)})

    def observation_space(self, agent):
        return flatten_space(self.unflatten_observation_space(agent))

    def __continuous_action(self, discrete_action):
        action_tuple = (discrete_action // (self.movement_sensitivity*self.speed_sensitivity), 
                        (discrete_action % (self.movement_sensitivity*self.speed_sensitivity)) // (self.speed_sensitivity), 
                        discrete_action % self.speed_sensitivity)

        return [(2*(action_tuple[0] / (self.movement_sensitivity-1))-1),
                (2*(action_tuple[1] / (self.movement_sensitivity-1))-1),
                (action_tuple[2]) / float(self.speed_sensitivity-1)]

    def action_space(self, agent):
        """
        direction_x = Discrete(self.movement_sensitivity)#Box(low=-1.0, high=1.0, shape=(2,1), dtype=np.float32)
        direction_y = Discrete(self.movement_sensitivity)
        speed = Discrete(self.speed_sensitivity)#Box(0.0, 1.0, dtype=np.float32)
        return Tuple([direction_x, direction_y, speed])
        """
        return Discrete(self.movement_sensitivity * self.movement_sensitivity * self.speed_sensitivity)
    
    def __get_time_t_observation(self, agent):
        nbrs_distance_vectors = [Vector2D.distance_vector(self.agents_pos[agent], self.agents_pos[nbr])  
                            for nbr in self.__get_n_closest_neighbours(agent, self.visible_nbrs)]

        items_distance_vectors = [Vector2D.distance_vector(self.agents_pos[agent], self.items_pos[item])  
                            for item in self.__get_n_closest_items(agent, self.visible_items)]

        nbrs = {
            f"nbr-{i}": {
                "direction": Vector2D.unit_vector(nbrs_distance_vectors[i]).to_np_array(),
                "distance": np.log(1 + Vector2D.norm(nbrs_distance_vectors[i])) #1 - np.exp(-alpha * x)
            }
            for i in range(len(nbrs_distance_vectors))
        }
    
        items = {
            f"item-{i}": {
                "direction": Vector2D.unit_vector(items_distance_vectors[i]).to_np_array(),
                "distance": np.log(1 + Vector2D.norm(items_distance_vectors[i])) #1 - np.exp(-alpha * x)
            }
            for i in range(len(items_distance_vectors))
        }
        
        for i in range(len(items_distance_vectors), self.visible_items):
            items[f"item-{i}"] = {
                "direction": np.array([0,0], dtype=np.int32),
                "distance": -1 #1 - np.exp(-alpha * x)
            }

        obs = {
            "nbrs": nbrs,
            "items": items
        }

        return obs

    def __get_observation(self, agent):
        if len(self.observation_cache[agent]) == 0:
            self.observation_cache[agent] = [self.__get_time_t_observation(agent)]*self.memory_size
        else:
            self.observation_cache[agent] = [self.__get_time_t_observation(agent)] + self.observation_cache[agent]
            self.observation_cache[agent].pop()

        obs = {
            f"t[-{t}]": self.observation_cache[agent][t]
            for t in range(0, self.memory_size)
        }

        return flatten(self.unflatten_observation_space(agent), obs)

    def rgb_to_hex(self, r, g, b):
        return f'#{r:02x}{g:02x}{b:02x}'

    def __get_local_reward(self, agent, action):
        # reward_1: small bonus if the agent collects an item
        reward_1 = +5 if agent in self.collectors else 0

        # reward_2: malus if the agent collides with another agent 
        reward_2= sum([-2 if Vector2D.distance(self.agents_pos[agent], self.agents_pos[nbr]) < self.agent_range*2 else 0 for nbr in self.__get_other_agents(agent)])

        # reward_3: -1 at each step
        reward_3 = -self.step_penalty

        # reward_4: positive reward if the agent moves toward the closest items, negative otherwise
        distance_diff = ([Vector2D.distance(self.agent_old_pos[agent], self.items_pos[item]) -
                    Vector2D.distance(self.agents_pos[agent], self.items_pos[item])
            for item in self.closest_items[agent]])
        
        reward_4 = max(distance_diff) if len(distance_diff) > 0 else 0

        # reward_5: positive reward when increasing the distance from the neighbours
        distance_diff = [Vector2D.distance(self.agents_pos[agent], self.agents_pos[nbr]) -
                         Vector2D.distance(self.agent_old_pos[agent], self.agent_old_pos[nbr])
                         for nbr in self.__get_n_closest_neighbours(agent, self.visible_nbrs)]
        
        reward_5 = np.mean(distance_diff)/2.0 if len(distance_diff) > 0 else 0

        # reward_6: positive reward if an old visible item has been collected
        old_visible_item_collected = [1 if not(item in self.items_pos.keys()) else 0 for item in self.old_visible_items[agent]]
        self.old_visible_items[agent] = self.__get_n_closest_items(agent, self.visible_items)

        reward_6 = sum(old_visible_item_collected) * 100

        self.info[agent] = {"info": {f"r2: {reward_2}, r3: {reward_3}, r4: {reward_4} , r5: {reward_5}, r6: {reward_6}"}}
        return  reward_3 + reward_4*3 + reward_2 + reward_5 + reward_6

    def __get_global_reward(self):
        return 0#self.global_reward * 100
    
    def __get_other_agents(self, agent):
        return [other for other in self.agents_ids if other != agent]

    def __get_n_closest_neighbours(self, agent, n=1):
        distances = {other: Vector2D.distance(self.agents_pos[agent], self.agents_pos[other]) for other in self.__get_other_agents(agent)}
        return [neighbour[0] for neighbour in sorted(list(distances.items()), key=lambda d: d[1])[:n]]
        # return {neighbour[0]: neighbour[1] for neighbour in sorted(list(dst.items()), key=lambda d: d[0])[:n]}

    def __get_n_closest_items(self, agent, n=1):
        n = min(n, len(self.items_pos.keys()))
        distances = {item: Vector2D.distance(self.agents_pos[agent], pos) for item, pos in self.items_pos.items()}
        self.closest_items[agent] = [item[0] for item in sorted(list(distances.items()), key=lambda d: d[1])[:n]]
        return self.closest_items[agent]

    def __update_agent_position(self, agent, action):
        unit_movement = Vector2D(action[0], action[1])
        self.agent_old_pos[agent] = self.agents_pos[agent]
        self.agents_pos[agent] = Vector2D.sum(self.agents_pos[agent], Vector2D.mul(unit_movement, action[2]))

    def __collect_items(self):
        self.collectors = []
        uncollected_items = {}
        for item, item_pos in self.items_pos.items():
            collected = False
            for agent in self.agents_pos.values():
                if Vector2D.distance(item_pos, agent) < self.agent_range:
                    collected = True
                    self.collectors.append(agent)
            if not collected:
                uncollected_items[item] = item_pos
        self.items_pos = uncollected_items

    def increase_step_penalty(self, inc):
        self.step_penalty += inc       

    def __collect_items_and_compute_global_reward(self):
        old_uncollected_items = len(self.items_pos.keys())
        self.__collect_items()
        updated_uncollected_items = len(self.items_pos.keys())
        self.global_reward = old_uncollected_items - updated_uncollected_items

    def reset(self, seed=None, options=None):
        if seed is not None:
            rnd.seed(seed)
            np.random.seed(seed)

        self.steps = 0
        self.agents_pos = {agent: Vector2D.get_random_point(max_x=self.spawn_area, max_y=self.spawn_area) for agent in self.agents_ids}
        self.agent_old_pos = dict(self.agents_pos)
        self.items_pos = {f"item-{i}": Vector2D.get_random_point(max_x=self.spawn_area, max_y=self.spawn_area) for i in range(self.n_items)}
        self.collectors = []
        self.closest_items = {}
        self.old_visible_items = {agent: self.__get_n_closest_items(agent, self.visible_items) for agent in self.agents_ids}
        self.info = {}
        self.observation_cache = {agent: [] for agent in self.agents_ids}
        return {agent: self.__get_observation(agent) for agent in self.agents_ids}, {}
     
    def step(self, actions):
        self.steps += 1
        observations, rewards, terminated, truncated, infos = {}, {}, {}, {}, {}

        for agent, action in actions.items():
            self.__update_agent_position(agent, self.__continuous_action(action))

        self.__collect_items_and_compute_global_reward()

        for agent, action in actions.items():
            observations[agent] = self.__get_observation(agent)
            rewards[agent] = self.__get_local_reward(agent, self.__continuous_action(action)) + self.__get_global_reward()
            terminated[agent] = False
            truncated[agent] = False
            infos[agent] = self.info[agent]

        truncated['__all__'] = False
        if len(self.items_pos.keys()) == 0:
            terminated['__all__'] = True
        elif self.max_steps != None and self.steps == self.max_steps:
            terminated['__all__'] = True
        else:
            terminated['__all__'] = False
        
        if terminated['__all__'] == True:
            for key in terminated.keys():
                terminated[key] = True

        return observations, rewards, terminated, truncated, infos
     
    def rgb_to_hex(self, r, g, b):
        return f'#{r:02x}{g:02x}{b:02x}'

    def render(self):
        pass

    def get_agent_ids(self):
       return self.agents


class RenderableCollectTheItems(CollectTheItems):
    def __init__(self, config: EnvironmentConfiguration):
        super().__init__(config)
        self.agent_colors = {agent: self.rgb_to_hex(rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255)) for agent in self.agents_ids}

        self.unit = self.CANVAS_WIDTH/float(self.spawn_area)
        self.render_size_agent = max(self.unit,1)
        self.render_size_agent_range = self.unit*self.agent_range

        self.reset()

    def __position_in_frame(self, position_in_env):
        return [((self.spawn_area-position_in_env[0])/self.spawn_area)*self.CANVAS_WIDTH,
                        ((self.spawn_area-position_in_env[1])/self.spawn_area)*self.CANVAS_HEIGHT,]

    def render(self):
        with hold_canvas():
            if self.canvas == None:
                self.canvas = CanvasWithBorders(width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT)
                display(self.canvas)

            self.canvas.clear()

            self.canvas.fill_style = "red"
            for item in self.items_pos.values():
                self.canvas.draw_circle(pos=self.__position_in_frame(item.to_np_array()), 
                                        radius=2, 
                                        fill_color="red")

            for agent in self.agents_ids:
                self.canvas.draw_circle(pos=self.__position_in_frame(self.agents_pos[agent].to_np_array()), 
                                        radius=self.render_size_agent/2.0, 
                                        fill_color=self.agent_colors[agent],
                                        border_color="black")

                self.canvas.draw_circle(pos=self.__position_in_frame(self.agents_pos[agent].to_np_array()), 
                                        radius=self.render_size_agent_range, 
                                        border_color="red")