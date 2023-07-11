"""
Defines the dqn_agent class that creates the neural network.
Adapted from https://aleksandarhaber.com/deep-q-networks-dqn-in-python-from-scratch-by-using-openai-gym-and-tensorflow-reinforcement-learning-tutorial/
"""

import numpy as np
import tensorflow as tf
import gymnasium as gym
import gym_examples
from replay_buffer import transition, replay_buffer

UPDATE_TARGET_NETWORK_PERIOD = 100

class dqn_agent:
    """
    
    """
    def __init__(self, env_name, state_dimension, action_dimension, gamma,
                 epsilon, num_episodes, replay_buffer_size, batch_size):
        self.env = gym.make(env_name)
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes

        self.state_dimension = state_dimension
        self.action_dimension = action_dimension

        self.replay_buffer = replay_buffer(replay_buffer_size, batch_size)

        self.update_target_network_period = UPDATE_TARGET_NETWORK_PERIOD
        self.update_target_network_count = 0

        self.main_network = self.create_network()
        self.target_network = self.create_network()
        self.target_network.set_weights(self.main_network.get_weights())

        self.sum_rewards_episode = []
        self.actions_append = []