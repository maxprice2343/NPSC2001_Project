import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

WINDOW_SIZE = 512
DISTANCE = 1
REWARD_RECEIVED = 5
REWARD_MISSED = -10
REWARD_STEP = -1

WHITE = (255,255,255)
RED = (255,0,0)
BLUE = (0,0,255)

class NetworkEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10, num_nodes=4):
        self.size = size
        self.num_nodes = num_nodes
        self.window_size = WINDOW_SIZE

        # Observations consist of agent location, node locations, and currently
        # active node
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "nodes": spaces.Sequence(spaces.Box(0, size - 1, shape=(2,), dtype=int)),
                "active": spaces.MultiBinary(num_nodes)
            }
        )
        # Agent can move up, down, left, or right
        self.action_space = spaces.Discrete(4)

        # Ensures the provided render mode is None (i.e., environment is not to
        # be rendered) or part of the array defining valid render types
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
    
    # Initiates a new episode.
    def reset(self, seed=None):
        # Seeds the random number generator
        super().reset(seed=seed)

        # Randomly generates the agent location in the form [x,y]
        self._agent_location = self._generate_random_position()

        # Randomly generates node locations until they are not equal to the
        # agent location or other node locations
        self._node_locations = []
        for i in range(self.num_nodes):
            self._node_locations[i] = self._generate_random_position(
                excludes=(self._agent_location + self._node_locations[:i])
            )

        self._count = 0

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    # Takes an action and computes the state of the environment after the
    # action
    def step(self, action):
        direction = self._action_to_direction(action)
        reward = 0

        # Moves the agent
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # Checks whether the agent is within range of any of the nodes and
        # increments the count variable if so
        for i in range(self.num_nodes):
            if self._get_distance(i) <= DISTANCE:
                self._count = self._count + 1
                reward += REWARD_RECEIVED
        terminated = self._count == self.num_nodes
        reward += REWARD_STEP

        observation = self._get_obs()
        info = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _render_frame(self):
        # Initializes pygame, display, and clock if they haven't been already
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(WHITE)
        # The size of each cell in pixels
        pix_square_size = (self.window_size / self.size)

        # Drawing the nodes
        for i in range(self.num_nodes):
            pygame.draw.rect(
                canvas,
                RED,
                pygame.Rect(
                    pix_square_size * self._node_locations[i],
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

        # Add grid lines
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

    def close(self):
        """
        Closes remaining environment resources
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _generate_random_position(self, excludes=[]):
        """
        Generates a random position (x,y) within the range 0 - self.size - 1.
        Argument is a list of positions that the generated position should not
        be equal to.
        """
        location = self.np_random.integers(0, self.size, size=2, dtype=int)

        while location in excludes:
            location = self.np_random.integers(0, self.size, size=2, dtype=int)
        
        return location

    def _get_obs(self):
        """
        Returns the environment observations
        """
        return {
            "agent": self._agent_location,
            "nodes": self._node_locations,
            "active": self._active_nodes
        }
    
    def _get_info(self):
        """
        Returns a list containing the distances between the agent and all nodes
        """
        return [self._get_distance(i) for i in range(self.num_nodes)]

    def _get_distance(self, node_number):
        """
        Returns the distance between the agent and a given node
        """
        return np.linalg.norm(
            self._agent_location - self._node_locations[node_number]
        )

    def _action_to_direction(self, action):
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
        