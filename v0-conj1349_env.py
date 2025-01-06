from enum import Enum
import gymnasium as gym # type: ignore
from gymnasium import spaces # type: ignore
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import pygame # type: ignore
import math
import networkx as nx # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt #type: ignore
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #type: ignore

register(
    id = "conj1349-v0",
    entry_point = "v0-conj1349_env:conj1349env",
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

    """def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }"""

    def reset(self, seed=None, options=None, nodes=N):
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

        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()
        """

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action, nodes=N):
        k = self._agent_location
        l = math.floor(0.5 + math.sqrt(2*k + 0.25))
        m = k - math.floor(0.5*l*(l-1))
        self._matrix_location[l][m] = self._action_to_direction[action]
        self._matrix_location[m][l] = self._action_to_direction[action]

        terminated = (self._agent_location == math.floor(0.5 * nodes * (nodes - 1)) - 1)

        graph = nx.from_numpy_matrix(self._matrix_location)

        if nx.is_connected(graph):
            alpha = 0.99*(nodes+1)/(nodes+4)
            avglen = nx.average_shortest_path_length(graph)
            edges = graph.number_of_edges()
            reward = ((2*(nodes-2)*alpha/(nodes+1)+1)*(2/nodes)) - (3*alpha*avglen/(nodes+1)) - (2*(1-alpha)*edges/nodes/(nodes-1))
        else:
            reward = -6


        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False

        """        
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
        """

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        #pix_square_size = (
        #    self.window_size / self.size
        #)  # The size of a single grid square in pixels

        graph = nx.from_numpy_matrix(self._matrix_location)

        plt.clf()
        pos = nx.circular_layout(graph)
        edge_colors = ['gray']
        nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color=edge_colors, node_size=500, font_size=10)
        #u, v = current
        #if graph.has_edge(u, v):
        #    nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], edge_color='red', width=2.0)
        #else:
        #    nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], edge_color='red', style='dashed', width=2.0)
        plt.title("Current Graph")
        canvas.draw()


        """
        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )"""

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
"""
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
"""