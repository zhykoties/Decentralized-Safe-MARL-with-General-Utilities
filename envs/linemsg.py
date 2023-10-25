import functools
import numpy as np

import gymnasium
from gymnasium.spaces import Discrete, MultiDiscrete, Box

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

ROCK = 0
PAPER = 1
SCISSORS = 2
NONE = 3
MOVES = ["ROCK", "PAPER", "SCISSORS", "None"]
NUM_ITERS = 100
REWARD_MAP = {
    (ROCK, ROCK): (0, 0),
    (ROCK, PAPER): (-1, 1),
    (ROCK, SCISSORS): (1, -1),
    (PAPER, ROCK): (1, -1),
    (PAPER, PAPER): (0, 0),
    (PAPER, SCISSORS): (-1, 1),
    (SCISSORS, ROCK): (-1, 1),
    (SCISSORS, PAPER): (1, -1),
    (SCISSORS, SCISSORS): (0, 0),
}


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

    def __init__(self, num_agents=10, n_obs_neighbors=1, max_iter=50, render_mode=None):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces
        These attributes should not be changed after initialization.
        """
        self.possible_agents = ["player_" + str(r) for r in range(num_agents)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.n_agents = num_agents
        self.n_obs_nghbr = n_obs_neighbors
        self.max_iter = max_iter
        self.render_mode = render_mode

        self.state = np.ones(self.n_agents + self.n_obs_nghbr * 2).astype(int)
        self.state[:self.n_obs_nghbr] = 2
        self.state[-self.n_obs_nghbr:] = 2
        self.actions = np.zeros(self.n_agents + self.n_obs_nghbr * 2).astype(int) + 2

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # three states each: [action: [2: missing/no neighbor, 1: action 1, 0: action 0], state: [...]]
        return Box(0, 3, shape=(self.n_obs_neighbors * 2 + 1, 2))

    @functools.lru_cache(maxsize=None)
    def action_space(self):
        return Discrete(2)

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
        self.state = np.ones(self.n_agents + self.n_obs_nghbr * 2).astype(int)
        self.state[:self.n_obs_nghbr] = 2
        self.state[-self.n_obs_nghbr:] = 2
        self.actions = np.zeros(self.n_agents + self.n_obs_nghbr * 2).astype(int) + 2

        observations = {
            # self.agents[i]: np.stack([self.actions[i:i + self.n_obs_nghbr * 2 + 1],
            #                           self.state[i:i + self.n_obs_nghbr * 2 + 1]], axis=1)
            self.agents[i]: self.state[i:i + self.n_obs_nghbr * 2 + 1]
            for i in range(len(self.agents))
        }

        if not return_info:
            return observations
        else:
            infos = dict(zip(self.agents,
                             [{'pos_y': self.state[i + self.n_obs_nghbr]} for i, _ in enumerate(self.agents)]))
            return observations, infos

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
            self.agents = []
            return {}, {}, {}, {}, {}

        # print('step: ', self.num_moves)
        # print('state: ', self.state[self.n_obs_nghbr:-self.n_obs_nghbr])

        self.actions[self.n_obs_nghbr:-self.n_obs_nghbr] = np.stack([actions[a] for a in actions], axis=0)
        # print('actions: ', self.actions)
        new_state_agent0 = self.state[self.n_obs_nghbr + 1]
        new_state_agent_last = self.actions[-self.n_obs_nghbr - 1]
        new_state = ((self.state[self.n_obs_nghbr + 2:-self.n_obs_nghbr] + self.actions[self.n_obs_nghbr + 1:-self.n_obs_nghbr-1]) == 2) * 1.0
        # print('new_state: ', new_state)
        ran_mask = ((1 - self.state[self.n_obs_nghbr + 2:-self.n_obs_nghbr]) + self.actions[self.n_obs_nghbr + 1:-self.n_obs_nghbr-1]) == 2
        # print('ran_mask: ', ran_mask)
        random_prob = np.random.rand(self.n_agents - 2)
        # print('ran prob: ', random_prob)
        # print('random_prob[ran_mask]: ', random_prob[ran_mask])
        # print('new_state[ran_mask][random_prob[ran_mask] < 0.8]: ', new_state[ran_mask][random_prob[ran_mask] < 0.8])
        new_state[ran_mask & (random_prob < 0.8)] = 1
        self.state[self.n_obs_nghbr + 1:-self.n_obs_nghbr - 1] = new_state
        self.state[self.n_obs_nghbr] = new_state_agent0
        self.state[-self.n_obs_nghbr-1] = new_state_agent_last
        # print('new state final: ', self.state[self.n_obs_nghbr:-self.n_obs_nghbr])

        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards_np = (self.state[self.n_obs_nghbr:-self.n_obs_nghbr] == 1) * 0.1

        # print('rewards_np before: ', rewards_np)
        rewards_np[0] = rewards_np[0] * 10
        rewards = {agent: rewards_np[i] for i, agent in enumerate(self.agents)}

        terminations = {agent: False for agent in self.agents}

        self.num_moves += 1
        env_truncation = self.num_moves >= self.max_iter
        truncations = {agent: env_truncation for agent in self.agents}

        # current observation is just the other player's most recent action
        observations = {
            # self.agents[i]: np.stack([self.actions[i:i + self.n_obs_nghbr * 2 + 1],
            #                           self.state[i:i + self.n_obs_nghbr * 2 + 1]], axis=1)
            self.agents[i]: self.state[i:i + self.n_obs_nghbr * 2 + 1]
            for i in range(len(self.agents))
        }

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = dict(zip(self.agents, [{'pos_y': self.state[i + self.n_obs_nghbr]} for i, _ in enumerate(self.agents)]))

        if env_truncation:
            self.agents = []

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos
