import gymnasium as gym
from dqn_agent import dqn_agent
from replay_buffer import replay_buffer
import gym_examples

GAMMA = 1
EPSILON = 0.1
NUM_EPISODES = 1000
REPLAY_BUFFER_SIZE = 300
BATCH_SIZE = 100

agent = dqn_agent(
    "Network-v0",
    4,
    4,
    GAMMA,
    EPSILON,
    NUM_EPISODES,
    REPLAY_BUFFER_SIZE,
    BATCH_SIZE
)
agent.training_episodes()
print(agent.sum_rewards_episode)

agent.main_network.summary()
agent.main_network.save("trained_model_temp.h5")