import functools
import numpy as np
import copy

import gymnasium
from gymnasium.spaces import Discrete, MultiDiscrete, Box

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, grid_x=6, grid_y=6, ddl=2, packet_arrival_probability=0.8, success_transmission_probability=0.8,
                 n_obs_neighbors=1, max_iter=50, render_mode=None):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces
        These attributes should not be changed after initialization.
        """
        self.n_agents = grid_x * grid_y
        self.agents = ["player_" + str(r) for r in range(self.n_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(self.n_agents)))
        )

        self.grid_x = grid_x
        self.grid_y = grid_y
        self.ddl = ddl
        self.p = packet_arrival_probability
        self.q = success_transmission_probability
        self.n_obs_nghbr = n_obs_neighbors
        self.max_iter = max_iter
        self.render_mode = render_mode

        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    Box(low=0, high=2, shape=(self.ddl, (self.n_obs_nghbr * 2 + 1) ** 2), dtype=np.float32)
                ]
                * self.n_agents,
            )
        )
        self.action_spaces = dict(
            zip(self.agents, [gymnasium.spaces.Discrete(5)] * self.n_agents)
        )
        # each agent is initialized to have no package deadlines
        self.state = np.full((self.ddl, self.grid_x + 2 * self.n_obs_nghbr, self.grid_y + 2 * self.n_obs_nghbr),
                             fill_value=2, dtype=np.float32)
        self.state[:, self.n_obs_nghbr:self.grid_x+self.n_obs_nghbr, self.n_obs_nghbr:self.grid_y+self.n_obs_nghbr] = 0
        self.actions = np.zeros((self.grid_x + 2 * self.n_obs_nghbr, self.grid_y + 2 * self.n_obs_nghbr)).astype(
            int)  # reset initial actions

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        string = "Current state: \n"
        for i, agent in enumerate(self.agents):
            string += f"Agent {i}: action = {self.actions[i + self.n_obs_nghbr]}, " \
                      f"state = {self.state[i + self.n_obs_nghbr]}\n"
        return string

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, return_info=False, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        # each agent has 0.5 probability to receive a message at the initialization
        self.state[:, self.n_obs_nghbr:self.grid_x + self.n_obs_nghbr, self.n_obs_nghbr:self.grid_y + self.n_obs_nghbr] = \
            np.random.choice(2, size=(self.ddl, self.grid_x, self.grid_y))
        self.actions = np.zeros((self.grid_x + 2 * self.n_obs_nghbr, self.grid_y + 2 * self.n_obs_nghbr)).astype(
            int)  # reset initial actions

        observations = {
            self.agents[i]: self.state[:, i // self.grid_y: i // self.grid_y + self.n_obs_nghbr * 2 + 1,
                            i % self.grid_y: i % self.grid_y + self.n_obs_nghbr * 2 + 1].reshape(self.ddl, -1)
            for i in range(self.n_agents)
        }

        if not return_info:
            return observations
        else:
            infos = {}
            return observations, infos

    def access_point_mapping(self, i, j, agent_action):
        if agent_action == 0:
            return None, None
        if agent_action == 1:  # transit to the up and left access point
            agent_access_point_x = i - 1
            agent_access_point_y = j - 1
        elif agent_action == 2:  # down and left
            agent_access_point_x = i
            agent_access_point_y = j - 1
        elif agent_action == 3:  # up and right
            agent_access_point_x = i - 1
            agent_access_point_y = j
        elif agent_action == 4:  # down and right
            agent_access_point_x = i
            agent_access_point_y = j
        else:
            raise ValueError(f'agent_action = {agent_action} is not defined!')
        if agent_access_point_x < 0 or agent_access_point_x >= self.grid_x - 1 or agent_access_point_y < 0 or \
                agent_access_point_y >= self.grid_y - 1:
            return None, None
        else:
            return agent_access_point_x, agent_access_point_y

    def check_transmission_fail(self, agent_access_point_x, agent_access_point_y, access_point_profile):
        if agent_access_point_x is None or agent_access_point_y is None:
            return True
        return access_point_profile[agent_access_point_x, agent_access_point_y] != 1

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            return {}, {}, {}, {}, {}

        # Given all agent's action, create a profile record the packet information at each access point
        access_point_profile = np.zeros((self.grid_x - 1, self.grid_y - 1))
        for agent_id, agent in enumerate(self.agents):
            agent_access_point_x, agent_access_point_y = self.access_point_mapping(agent_id // self.grid_y,
                                                                                   agent_id % self.grid_y,
                                                                                   actions[agent])
            if agent_access_point_x is not None and agent_access_point_y is not None:
                access_point_profile[agent_access_point_x, agent_access_point_y] += 1

        # initialize the next_state and reward
        rewards = {agent: 0 for agent in self.agents}

        # Iterate over all agents to update the state and obtain the reward
        for agent_id, agent in enumerate(self.agents):
            i = agent_id // self.grid_y
            j = agent_id % self.grid_y
            agent_action = actions[agent]
            agent_access_point_x, agent_access_point_y = self.access_point_mapping(i, j, agent_action)
            # if the conditions in the first three rows of Table 1 are satisfied:
            if not (actions[agent] == 0 or \
                    self.state[:, self.n_obs_nghbr + i, self.n_obs_nghbr + j].max() == 0 or \
                    (self.state[:, self.n_obs_nghbr + i, self.n_obs_nghbr + j].max() == 1 and \
                     self.check_transmission_fail(agent_access_point_x, agent_access_point_y, access_point_profile))):
                # find the most left "1" in the agent's state
                idx = 0
                while self.state[idx, self.n_obs_nghbr + i, self.n_obs_nghbr + j] == 0:
                    idx += 1
                # Flip left most “1”
                if np.random.rand(1) <= self.q:
                    self.state[idx, self.n_obs_nghbr + i, self.n_obs_nghbr + j] = 0
                    rewards[agent] = 1

        # Left shift and append Bernoulli
        self.state[:-1, self.n_obs_nghbr:self.grid_x+self.n_obs_nghbr, self.n_obs_nghbr:self.grid_y+self.n_obs_nghbr] = \
            copy.deepcopy(self.state[1:, self.n_obs_nghbr:self.grid_x+self.n_obs_nghbr, self.n_obs_nghbr:self.grid_y+self.n_obs_nghbr])
        self.state[-1, self.n_obs_nghbr:self.grid_x+self.n_obs_nghbr, self.n_obs_nghbr:self.grid_y+self.n_obs_nghbr] = \
            np.random.rand(self.grid_x, self.grid_y) <= self.p

        terminations = {agent: False for agent in self.agents}

        self.num_moves += 1
        env_truncation = self.num_moves >= self.max_iter
        truncations = {agent: env_truncation for agent in self.agents}

        # current observation is just the other player's most recent action
        observations = {
            self.agents[i]: self.state[:, i // self.grid_y: i // self.grid_y + self.n_obs_nghbr * 2 + 1,
                            i % self.grid_y: i % self.grid_y + self.n_obs_nghbr * 2 + 1].reshape(self.ddl, -1)
            for i in range(self.n_agents)
        }

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {}

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos


if __name__ == "__main__":
    myenv = parallel_env()
    myenv.reset()
    actions = {}
    for agent in range(len(myenv.agents)):
        actions[agent] = 1
    myenv.step(actions)
