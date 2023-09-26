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
from gym_environments.environments.network_env import NetworkEnv

WINDOW_SIZE = 512
DISTANCE = 1

REWARD_RECEIVED = 5
REWARD_TOWARDS = 1
REWARD_AWAY = -1
REWARD_MISSED = -5

WHITE = (255,255,255)
RED = (255,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)
BLACK= (0,0,0)
ORANGE = (255, 165, 0)

NUM_NODES = 5

class MultiActiveEnv(NetworkEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, *, render_mode=None, size=10, num_nodes=NUM_NODES, max_steps,
                 min_range, max_range, min_time, max_time):
        self.size = size
        self.num_nodes = num_nodes
        self.window_size = WINDOW_SIZE
        self.max_steps = max_steps
        self.min_range = min_range
        self.max_range = max_range
        self.min_time = min_time
        self.max_time = max_time
        self.current_active = 0

        # Defines the observation space for the environment
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, self.size, (2,), dtype=int),
            "node_information": spaces.Box(
                low=0,
                high=size*size,
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
        # Seeds the ranedom number generator
        super(NetworkEnv, self).reset(seed=seed)

        # Generates the random agent location
        self._agent_location = self._generate_random_position()

        # Generates the nodes
        self.nodes = []
        # Array stores the locations of the agent and nodes to prevent repeats
        generated_locations = np.ndarray((self.num_nodes + 1, 2), dtype=int)
        # For each node, generates a location, range, and time
        for i in range(self.num_nodes):
            new_location = self._generate_random_position(generated_locations)
            generated_locations[i + 1] = new_location
            new_range = self.np_random.integers(self.min_range, self.max_range + 1)
            new_time = self.np_random.integers(self.min_time, self.max_time + 1)
            new_node = node(new_location, new_range, new_time)
            new_node.activate()
            self.nodes.append(new_node)

        # All nodes are active at the start
        self.current_active = self.num_nodes
        self.num_missed = 0

        # Gets the initial distances between the agent and all nodes
        self.previous_distances = self.check_distances()

        # The number of steps in the episode
        self.step_count = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()
    
    def step(self, action):
        """
        Takes an action and computes the state of the environment after
        the action is applied
        """
        # Reward for the current step
        reward = 0
        terminated = False
        direction = self._action_to_direction(action)

        # Moves the agent
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # If the agent has moved closer to the active nodes since the last step
        # then it receives a small positive reward, otherwise it receives a small
        # negative award
        current_distances = self.check_distances()
        for i in range(current_distances.size):
            if current_distances[i] < self.previous_distances[i]:
                reward += REWARD_TOWARDS
            else:
                reward += REWARD_AWAY
        self.previous_distances = current_distances

        for node in self.nodes:
            # Checks if any of the active nodes have been missed (i.e., the agent
            # hasn't reached them in time)
            if node.state == node.ACTIVE:
                # Checks whether the agent is within range of any active nodes and
                # adds to the reward if it is
                if node.within_range(self._agent_location):
                    reward += REWARD_RECEIVED
                    node.deactivate()
                    self.current_active -= 1
                elif node.countdown == 0:
                    node.miss()
                    self.current_active -= 1
                    self.num_missed += 1
                    reward += REWARD_MISSED
                else:
                    node.countdown -= 1

        if self.current_active == 0:
            terminated = True

        self.step_count += 1

        # If the episode has reached the maximum number of steps, rewards are
        # deducted based on the number of remaining active nodes
        if not terminated:
            if self.step_count == self.max_steps:
                terminated = True

        # Renders the environment frame
        if self.render_mode == "human":
            self._render_frame()
        
        print(f"Reward = {reward}")
        return self._get_obs(), reward, terminated, False, self._get_info()

    def _get_obs(self):
        """
        Returns the environment observations
        """
        obs = {
            "agent": self._agent_location,
            "node_information": self.nodes_to_array()
        }
        return obs
    
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
            colour = GREEN if node.state == node.ACTIVE else RED
            if node.state == node.ACTIVE:
                colour = GREEN
            elif node.state == node.INACTIVE:
                colour = RED
            else:
                colour = ORANGE

            pygame.draw.rect(
                canvas,
                colour,
                pygame.Rect(
                    (pix_square_size * node.location[0],
                     pix_square_size * node.location[1]),
                    (pix_square_size, pix_square_size)
                )
            )

            # Draws circles aeround active nodes to indicate their range
            if node.state == node.ACTIVE:
                pygame.draw.circle(
                    canvas,
                    BLACK,
                    (node.location + 0.5) * pix_square_size,
                    node.range * pix_square_size + 0.5 * pix_square_size,
                    width=2
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

    def check_distances(self):
        distances = np.ndarray(shape=(self.current_active))
        idx = 0
        for node in self.nodes:
            if node.state == node.ACTIVE:
                distances[idx] = self._get_distance(self._agent_location, node.location)
                idx += 1
        return distances

    def nodes_to_array(self):
        arr = np.ndarray(shape=(self.num_nodes, 4), dtype=int)
        for idx, node in enumerate(self.nodes):
            arr[idx][0] = node.location[0]
            arr[idx][1] = node.location[1]
            arr[idx][2] = node.countdown
            arr[idx][3] = node.state
        return arr