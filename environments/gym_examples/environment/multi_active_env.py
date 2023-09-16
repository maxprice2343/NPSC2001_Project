"""
An environment based on the timed network.
This network can have multiple nodes active at the same time.
Active nodes may have different activation periods.
The agent needs to prioritize reaching the node that will deactivate the
soonest.
Code adapted from https://github.com/Farama-Foundation/gym-examples.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from .node import node
from gym_examples.environment.network_env import NetworkEnv

WINDOW_SIZE = 512
DISTANCE = 1

REWARD_RECEIVED = 10
REWARD_TOWARDS = 1
REWARD_AWAY = -5
REWARD_MISSED = -10

WHITE = (255,255,255)
RED = (255,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)
BLACK= (0,0,0)

class MultiActiveEnv(NetworkEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, *, render_mode=None, size=10, num_nodes, max_steps, min_range, max_range):
        self.size = size
        self.num_nodes = num_nodes
        self.window_size = WINDOW_SIZE
        self.max_steps = max_steps
        self.min_range = min_range
        self.max_range = max_range

        # Defines the observation space for the environment
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(-1, self.size-1, (2,), dtype=int),
            "node_information": spaces.Box(
                low=np.array([0, 0, min_range, 0]),
                max=np.array([size, size, max_range, 1]),
                shape=(num_nodes, 4),
                dtype=int
            )
        })
        # Defines the action space for the environment (moving up, down, left, or right)
        self.action_space = spaces.Discrete(4)

        self.nodes = []

        # Ensures the provided render mode is None (i.e., environment is not to
        # be rendered) or part of the array defining valid render types
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # window and clock are used for rendering the environment
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        """
        Initializes a new episode of the environment
        """
        # Seeds the random number generator
        super().reset(seed=seed)

        # Generates the random agent location
        self._agent_location = self._generate_random_location()

        generated_locations = [self._agent_location]
        for i in range(self.num_nodes):
            new_location = self._generate_random_location(generated_locations)
            generated_locations.append(new_location)
            range = self.np_random.integers(self.min_range, self.max_range)
            self.nodes[i] = node(new_location, range)

        # Selects the active node randomly
        self.active = self.nodes[np.random.randint(0, self.nodes.count)]
        
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()
    
    def step(self, action):
        """
        Takes an action and computes the state of the environment after
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
        current_distance = self._get_distance(self._active.location, self._agent_location)
        if current_distance < self.prev_distance:
            reward += REWARD_TOWARDS
        else:
            reward += REWARD_AWAY
        self.prev_distance = current_distance

        # Checks whether the agent is within range of the active node and
        # adds to the reward if it is
        if self.active.within_range(self._agent_location):
            reward += REWARD_RECEIVED
            terminated = True
        
        self.step_count += 1

        # If the episode isn't terminated yet
        if not terminated:
            if self.step_count == self.max_steps:
                terminated = True
                reward += REWARD_MISSED
        
        if self.render_mode == "human":
            self._render_frame()
        
        return self._get_obs(), reward, terminated, False, self._get_info()

    def _get_obs(self):
        """
        Returns the environment observations
        """
        return {
            "agent": self._agent_location,
            "node_information": self.nodes_to_array(self.nodes)
        }
    
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
        for i, node in enumerate(self.nodes):
            # Colours the active node green and the others red
            colour = GREEN if np.all(np.equal(node.location, self._active.location)) else RED

            pygame.draw.rect(
                canvas,
                colour,
                pygame.Rect(
                    (pix_square_size * node.location[0],
                     pix_square_size * node.location[1]),
                    (pix_square_size, pix_square_size)
                )
            )

            # Draws circles around active nodes to indicate their range
            if node.active:
                pygame.draw.circle(
                    canvas,
                    BLACK,
                    node.location,
                    node.range,
                    width=1
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