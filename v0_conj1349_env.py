from enum import Enum
import gymnasium as gym # type: ignore
from gymnasium import spaces # type: ignore
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
#import pygame # type: ignore
import math
import networkx as nx # type: ignore
import numpy as np # type: ignore
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

register(
    id = "conj1349-v0",
    entry_point = "v0_conj1349_env:conj1349env",
)


N = 5

class Actions(Enum):
    delete = 0
    connect = 1


class conj1349env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, nodes=N):
        self.size = nodes  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Discrete(math.floor(0.5 * nodes * (nodes - 1))), # which EDGE the agent is at (this is linear game mode)
                "matrix": spaces.MultiBinary([nodes, nodes]), #adjacency matrix
            }
        )

        # connect or delete
        self.action_space = spaces.Discrete(2)

        
        #The following dictionary maps abstract actions from `self.action_space` to 
        #the direction we will walk in if that action is taken.
        #i.e. 0 corresponds to "right", 1 to "up" etc.
        
        self._action_to_direction = {
            Actions.delete.value: 0,
            Actions.connect.value: 1,
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        
        #If human-rendering is used, `self.window` will be a reference
        #to the window that we draw to. `self.clock` will be a clock that is used
        #to ensure that the environment is rendered at the correct framerate in
        #human-mode. They will remain `None` until human-mode is used for the
        #first time.
        
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "matrix": self._matrix_location}
    
    def _get_info(self):
        return {"rounds left": math.floor(0.5 * N * (N - 1)) - 1 - self._agent_location - 1}

    def reset(self, seed=None, options=None, nodes=N):
        super().reset(seed=seed)
        self._agent_location = 0
        self._matrix_location = []
        for i in range(nodes):
            r = []
            for j in range(nodes):
                if i == j:
                    r.append(0)
                else:
                    r.append(1)
            self._matrix_location.append(r)

        observation = self._get_obs()
        info = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action, nodes=N):
        k = self._agent_location
        l = math.floor(0.5 + math.sqrt(2*k + 0.25))
        m = k - math.floor(0.5*l*(l-1))
        self._matrix_location[l][m] = self._action_to_direction[int(action)]
        self._matrix_location[m][l] = self._action_to_direction[int(action)]

        self._agent_location = self._agent_location + 1

        terminated = (self._agent_location == math.floor(0.5 * nodes * (nodes - 1)) - 1)

        self.graph = nx.from_numpy_array(np.array(self._matrix_location))

        if nx.is_connected(self.graph):
            #alpha = 0.09*(nodes+1)/(nodes+4)
            alpha = 0
            avglen = nx.average_shortest_path_length(self.graph)
            edges = self.graph.number_of_edges()
            reward = ((2*(nodes-2)*alpha/(nodes+1)+1)*(2/nodes)) - (3*alpha*avglen/(nodes+1)) - (2*(1-alpha)*edges/nodes/(nodes-1))
        else:
            reward = -6


        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        print('NEW EPISODE')
        for i in range(0, N):
            for j in range(0, N):
                print(self._matrix_location[i][j], end=' ')
            print()
        print()
        print()
        print()
        print()

if __name__ == "__main__":
    env = gym.make('conj1349-v0', render_mode = 'human')
    print("begin checking stuff")
    check_env(env.unwrapped)
    print("end checking stuff")