import numpy as np
import tensorflow as tf
import gymnasium as gym
from dqn_agent import dqn_agent
import pathlib

# Path to the trained model
model_path = pathlib.Path("trained_model_temp.h5")

# Loads the model from the file
model = tf.keras.models.load_model(
    model_path,
    custom_objects={'loss':dqn_agent.loss}
)

# Creates the environment
env = gym.make("Network-v0", size=5, render_mode="rgb_array")

# Initializes the environment
current_state, _ = env.reset()

# Sets up the environment to record
video_length = 400
env = gym.wrappers.RecordVideo(env, 'stored_video', video_length=video_length)

# Steps until the environment is in a terminal state
for _ in range(10):
    terminal_state = False
    sum_obtained_rewards = 0
    while not terminal_state:
        # Uses the model to generate q values based on the current state
        q_values = model.predict(current_state[None, :])

        # Selects the optimal action based on the q values
        action = np.random.choice(
            np.where(q_values[0,:]==np.max(q_values[0,:]))[0]
        )
        current_state, reward, terminal_state, _, _ = env.step(action)
        sum_obtained_rewards += reward
    current_state, _ = env.reset()

env.close()