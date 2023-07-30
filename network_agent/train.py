import gymnasium as gym
from dqn_agent import dqn_agent
from replay_buffer import replay_buffer
import gym_examples

GAMMA = 1
EPSILON = 0.1
NUM_EPISODES = 100
REPLAY_BUFFER_SIZE = 300
BATCH_SIZE = 100

env = gym.make("Network-v0", size=5, render_mode="human")

agent = dqn_agent(
    env,
    4,
    2,
    GAMMA,
    EPSILON,
    NUM_EPISODES,
    REPLAY_BUFFER_SIZE,
    BATCH_SIZE
)
agent.training_episodes()
print(agent.sum_rewards_episode)

agent.target_network.summary()
agent.target_network.save("trained_model_temp.h5")