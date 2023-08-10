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
try:
    agent.training_episodes()
except KeyboardInterrupt:
    print("Training interrupted")

agent.main_network.summary()

# Prompts the user if they want to save the model or not
save = input("Do you want to save the model (y/n)?")

# Repeats until the user inputs a valid option
while(not(save == "y" or save == "n")):
    save = input("Do you want to save the model (y/n)?")

# If the user wants to save the model, prompts them to enter a file path
# to save it to
if(save == "y"):
    path = input("Enter file path:")
    agent.main_network.save(path)