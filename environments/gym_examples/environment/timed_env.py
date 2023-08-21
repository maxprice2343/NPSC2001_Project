"""
Defines the environment that the RL agent interacts with.
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

class TimedEnv(NetworkEnv):
    def __init__(self, render_mode=None, size=10, num_nodes=1, max_steps=50, range=1, time=50):
        super().__init__(render_mode=render_mode, size=size, num_nodes=num_nodes, max_steps=max_steps, range=range, time=time)

    def reset(self, seed=None, options=None):
        """
        Initializes a new episode of the environment
        """
        # Seeds the random number generator
        super(NetworkEnv, self).reset(seed=seed)

        # Randomly generates the agent location in the form [x,y]
        self._agent_location = self._generate_random_position()

        # Randomly generates node locations until they are not equal to the
        # agent location or other node locations
        node_locations = np.zeros(shape=(self.num_nodes, 2), dtype=int)
        self.nodes = []
        for i in range(self.num_nodes):
            node_locations[i] = self._generate_random_position(
                excludes=np.concatenate((self._agent_location[np.newaxis, :], node_locations[:i]), axis=0)
            )
            new_node = node(node_locations[i], self.range, self.time)
            self.nodes.append(new_node)
        
        for n in self.nodes:
            n.time = (np.random.rand() * options["range"]) + options["min_time"]

        # Selects a random node to become active
        self._active = self.nodes[np.random.randint(0, self.num_nodes)]
        # Stores the distance between the agent and the previously active node
        self.prev_distance = self._get_distance(self._active.location, self._agent_location)

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
        current_distance = self._get_distance(self._active.location, self._agent_location)
        if current_distance < self.prev_distance:
            reward += REWARD_TOWARDS
        else:
            reward += REWARD_AWAY
        self.prev_distance = current_distance

        # Checks whether the agent is within range of the active node and
        # adds to the reward if it is
        if self._get_distance(self._active.location, self._agent_location) <= DISTANCE:
            reward += REWARD_RECEIVED
            self._active = self.nodes[np.random.randint(0, self.num_nodes)]

        self.step_count += 1
        # If the episode isn't terminated yet
        if not terminated:
            # If the maximum number of steps has been reached then the episode
            # is terminated
            if self.step_count == self._active.time:
                terminated = True
                reward += REWARD_MISSED

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info