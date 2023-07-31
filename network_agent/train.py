import gymnasium as gym
from dqn_agent import dqn_agent
from replay_buffer import replay_buffer
import gym_examples

GAMMA = 1
EPSILON = 0.1
NUM_EPISODES = 500
REPLAY_BUFFER_SIZE = 300
BATCH_SIZE = 100
STATE_DIMENSION = 4
ACTION_DIMENSION = 4

env = gym.make("Network-v0", size=5, render_mode="human")

agent = dqn_agent(
    env,
    STATE_DIMENSION,
    ACTION_DIMENSION,
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