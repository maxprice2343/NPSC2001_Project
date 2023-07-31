"""
Defines the environment that the RL agent interacts with.
Code adapted from https://github.com/Farama-Foundation/gym-examples.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

WINDOW_SIZE = 512
DISTANCE = 1

REWARD_RECEIVED = 10
REWARD_TOWARDS = 1
REWARD_AWAY = -1
REWARD_MISSED = -10

INACTIVE = np.array([-1, -1])

WHITE = (255,255,255)
RED = (255,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)

class NetworkEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10, num_nodes=4, max_steps=50):
        self.size = size
        self.num_nodes = num_nodes
        self.window_size = WINDOW_SIZE
        self.max_steps = max_steps

        # Observations consist of agent location, node locations, and currently
        # active node
        self.observation_space = spaces.Box(-1, self.size-1, (4,), dtype=int)
        # Agent can move up, down, left, or right
        self.action_space = spaces.Discrete(4)

        # Ensures the provided render mode is None (i.e., environment is not to
        # be rendered) or part of the array defining valid render types
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
    
    def reset(self, seed=None, options=None):
        """
        Initializes a new episode of the environment
        """
        # Seeds the random number generator
        super().reset(seed=seed)

        # Randomly generates the agent location in the form [x,y]
        self._agent_location = self._generate_random_position()

        # Randomly generates node locations until they are not equal to the
        # agent location or other node locations
        self._node_locations = np.empty((self.num_nodes, 2), dtype=int)
        for i in range(self.num_nodes):
            self._node_locations[i] = self._generate_random_position(
                excludes=np.concatenate((self._agent_location[np.newaxis, :], self._node_locations[:i]), axis=0)
            )

        # Selects a random node to become active
        self._active = self._node_locations[np.random.randint(0, self.num_nodes)]
        # Stores the distance between the agent and the previously active node
        self.prev_distance = self._get_distance(self._active, self._agent_location)

        self.reward = 0
        self.step_count = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        Takes in an action and computes the state of the environment after
        the action is applied
        """
        reward = 0
        terminated = False
        direction = self._action_to_direction(action)

        # Moves the agent
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # If the agent has moved closer to the active node since the last step
        # then it receives a small positive reward, otherwise it receives a small
        # negative award
        current_distance = self._get_distance(self._active, self._agent_location)
        if current_distance < self.prev_distance:
            reward += REWARD_TOWARDS
        else:
            reward += REWARD_AWAY
        self.prev_distance = current_distance

        # Checks whether the agent is within range of the active node and
        # adds to the reward if it is
        if self._get_distance(self._active, self._agent_location) <= DISTANCE:
            reward += REWARD_RECEIVED
            terminated = True

        self.step_count += 1
        # If the episode isn't terminated yet
        if not terminated:
            # If the maximum number of steps has been reached then the episode
            # is terminated
            if self.step_count == self.max_steps:
                terminated = True
                self.reward += REWARD_MISSED

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def close(self):
        """
        Closes remaining environment resources
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def render(self):
        """Used by simulate_trained_model.py to convert the environment in
        to an rgb_array for MoviePy to create a video"""
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """
        Creates a frame (using pygame) representing the state of the
        environment.
        If the render mode is human, the window is updated with the frame.
        """
        # Initializes pygame, display, and clock if they haven't been already
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Initializes the Surface and fills it with white
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(WHITE)
        # pix_square_size is the size of each cell in pixels
        pix_square_size = (self.window_size / self.size)

        # Drawing the nodes
        for i, node in enumerate(self._node_locations):
            # Colours the active node green and the others red
            colour = GREEN if np.all(np.equal(node, self._active)) else RED

            pygame.draw.rect(
                canvas,
                colour,
                pygame.Rect(
                    (pix_square_size * self._node_locations[i][0],
                     pix_square_size * self._node_locations[i][1]),
                    (pix_square_size, pix_square_size)
                )
            )

        # Drawing the agent
        pygame.draw.circle(
            canvas,
            BLUE,
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3
        )

        # Adding grid lines
        for i in range(self.size + 1):
            # Horizontal lines
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * i),
                (self.window_size, pix_square_size * i),
                width=3
            )
            # Vertical lines
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * i, 0),
                (pix_square_size * i, self.window_size),
                width=3
            )
        
        if self.render_mode == "human":
            # Copies the canvas to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # Keeps rendering at the predfined framerate
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)),
                axes=(1, 0, 2)
            )

    def _generate_random_position(self, excludes=np.array([])):
        """
        Generates a random position (x,y) within the range 0 - self.size - 1.
        Argument is a list of positions that the generated position should not
        be equal to.
        """
        location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # While the generated location is in the excludes list, a new location
        # is generated
        while self._in(location, excludes):
            location = self.np_random.integers(0, self.size, size=2, dtype=int)

        return location

    def _in(self, x: np.ndarray, y: np.ndarray):
        """
        Checks whether a numpy array x is an element of a numpy array y
        """
        res = False
        # Ensures that the elements of y are the same shape as x
        if y.ndim == x.ndim + 1:
            if y.shape[y.ndim-1] == x.shape[x.ndim-1]:
                res = False
                idx = 0
                # Iterates through the elements of y and compares them to x
                while idx < y.shape[0] and res == False:
                    sub = y[idx]
                    res = np.all(np.equal(sub, x))
                    idx += 1
        return res

    def _get_obs(self):
        """
        Returns the environment observations
        """
        return np.concatenate((self._agent_location, self._active))
    
    def _get_info(self):
        """
        Returns a list containing the distances between the agent and all nodes
        """
        return {}  

    @staticmethod
    def _get_distance(location1, location2):
        """
        Returns the distance between two locations
        """
        return np.linalg.norm(location1 - location2)

    @staticmethod
    def _action_to_direction(action):
        """
        Maps an action (integer from 0-3) to a direction (represented using a
        1D numpy array with 2 elements)
        """
        match action:
            case 0:
                direction = np.array([1,0])
            case 1:
                direction = np.array([0,1])
            case 2:
                direction = np.array([-1,0])
            case 3:
                direction = np.array([0,-1])

        return direction            
        